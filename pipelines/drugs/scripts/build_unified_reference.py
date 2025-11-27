#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the unified drug reference dataset.

This script:
1. Loads all source datasets into DuckDB
2. Normalizes generics (strip salts, handle synonyms)
3. Extracts form-route validity mapping with provenance
4. Builds unified dataset with explosion logic
5. Exports as parquet

Output files:
- unified_drug_reference.parquet: Main reference (drugbank_id × atc_code × form × route)
- form_route_validity.parquet: Valid form-route combinations with provenance
- generics_lookup.parquet: Generic names with synonyms and salts
- brands_lookup.parquet: Brand → generic mapping
- mixtures_lookup.parquet: Mixture component combinations

Usage:
    python -m pipelines.drugs.scripts.build_unified_reference
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"

# Use environment variable for inputs if set (for GitIgnored data)
PIPELINE_INPUTS_DIR = Path(os.environ.get("PIPELINE_INPUTS_DIR", INPUTS_DIR))
PIPELINE_OUTPUTS_DIR = Path(os.environ.get("PIPELINE_OUTPUTS_DIR", OUTPUTS_DIR))


def _find_latest_file(pattern: str, directory: Path) -> Optional[Path]:
    """Find the latest file matching a glob pattern."""
    matches = list(directory.glob(pattern))
    if not matches:
        return None
    # Prefer parquet over csv
    parquet_matches = [m for m in matches if m.suffix == ".parquet"]
    if parquet_matches:
        return max(parquet_matches, key=lambda p: p.stat().st_mtime)
    return max(matches, key=lambda p: p.stat().st_mtime)


def _load_csv_or_parquet(path: Path) -> pd.DataFrame:
    """Load a file as DataFrame, preferring parquet."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_unified_reference(
    inputs_dir: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Build the unified drug reference dataset.
    
    Returns dict with paths to output files.
    """
    inputs_dir = inputs_dir or PIPELINE_INPUTS_DIR
    outputs_dir = outputs_dir or PIPELINE_OUTPUTS_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("Building Unified Drug Reference Dataset")
        print("=" * 60)
    
    # Create in-memory DuckDB connection
    con = duckdb.connect(":memory:")
    
    # =========================================================================
    # STEP 1: Load all source datasets
    # =========================================================================
    if verbose:
        print("\n[Step 1] Loading source datasets...")
    
    # DrugBank Generics
    drugbank_generics_path = inputs_dir / "drugbank_generics_master.csv"
    if drugbank_generics_path.exists():
        con.execute(f"""
            CREATE TABLE drugbank_generics AS 
            SELECT * FROM read_csv_auto('{drugbank_generics_path}')
        """)
        count = con.execute("SELECT COUNT(*) FROM drugbank_generics").fetchone()[0]
        if verbose:
            print(f"  - drugbank_generics: {count:,} rows")
    
    # DrugBank Mixtures
    drugbank_mixtures_path = inputs_dir / "drugbank_mixtures_master.csv"
    if drugbank_mixtures_path.exists():
        con.execute(f"""
            CREATE TABLE drugbank_mixtures AS 
            SELECT * FROM read_csv_auto('{drugbank_mixtures_path}')
        """)
        count = con.execute("SELECT COUNT(*) FROM drugbank_mixtures").fetchone()[0]
        if verbose:
            print(f"  - drugbank_mixtures: {count:,} rows")
    
    # DrugBank Brands
    drugbank_brands_path = inputs_dir / "drugbank_brands_master.csv"
    if drugbank_brands_path.exists():
        con.execute(f"""
            CREATE TABLE drugbank_brands AS 
            SELECT * FROM read_csv_auto('{drugbank_brands_path}')
        """)
        count = con.execute("SELECT COUNT(*) FROM drugbank_brands").fetchone()[0]
        if verbose:
            print(f"  - drugbank_brands: {count:,} rows")
    
    # DrugBank Products
    drugbank_products_path = inputs_dir / "drugbank_products_export.csv"
    if drugbank_products_path.exists():
        con.execute(f"""
            CREATE TABLE drugbank_products AS 
            SELECT * FROM read_csv_auto('{drugbank_products_path}')
        """)
        count = con.execute("SELECT COUNT(*) FROM drugbank_products").fetchone()[0]
        if verbose:
            print(f"  - drugbank_products: {count:,} rows")
    
    # DrugBank Salts
    drugbank_salts_path = inputs_dir / "drugbank_salts_master.csv"
    if drugbank_salts_path.exists():
        con.execute(f"""
            CREATE TABLE drugbank_salts AS 
            SELECT * FROM read_csv_auto('{drugbank_salts_path}')
        """)
        count = con.execute("SELECT COUNT(*) FROM drugbank_salts").fetchone()[0]
        if verbose:
            print(f"  - drugbank_salts: {count:,} rows")
    
    # Pure Salts
    pure_salts_path = inputs_dir / "drugbank_pure_salts.csv"
    if pure_salts_path.exists():
        con.execute(f"""
            CREATE TABLE pure_salts AS 
            SELECT * FROM read_csv_auto('{pure_salts_path}')
        """)
        count = con.execute("SELECT COUNT(*) FROM pure_salts").fetchone()[0]
        if verbose:
            print(f"  - pure_salts: {count:,} rows")
    
    # Salt Suffixes
    salt_suffixes_path = inputs_dir / "drugbank_salt_suffixes.csv"
    if salt_suffixes_path.exists():
        con.execute(f"""
            CREATE TABLE salt_suffixes AS 
            SELECT * FROM read_csv_auto('{salt_suffixes_path}')
        """)
        count = con.execute("SELECT COUNT(*) FROM salt_suffixes").fetchone()[0]
        if verbose:
            print(f"  - salt_suffixes: {count:,} rows")
    
    # PNF Lexicon
    pnf_path = inputs_dir / "pnf_lexicon.csv"
    if not pnf_path.exists():
        pnf_path = inputs_dir / "pnf_prepared.csv"
    if pnf_path.exists():
        con.execute(f"""
            CREATE TABLE pnf AS 
            SELECT * FROM read_csv_auto('{pnf_path}')
        """)
        count = con.execute("SELECT COUNT(*) FROM pnf").fetchone()[0]
        if verbose:
            print(f"  - pnf: {count:,} rows")
    
    # WHO ATC
    who_path = _find_latest_file("who_atc_*", inputs_dir)
    if who_path and who_path.exists():
        if who_path.suffix == ".parquet":
            con.execute(f"""
                CREATE TABLE who_atc AS 
                SELECT * FROM read_parquet('{who_path}')
            """)
        else:
            con.execute(f"""
                CREATE TABLE who_atc AS 
                SELECT * FROM read_csv_auto('{who_path}')
            """)
        count = con.execute("SELECT COUNT(*) FROM who_atc").fetchone()[0]
        if verbose:
            print(f"  - who_atc: {count:,} rows")
    
    # FDA Drug Brands
    fda_drug_path = _find_latest_file("fda_drug_*.parquet", inputs_dir)
    if not fda_drug_path:
        fda_drug_path = _find_latest_file("fda_drug_*.csv", inputs_dir)
    if fda_drug_path and fda_drug_path.exists():
        if fda_drug_path.suffix == ".parquet":
            con.execute(f"""
                CREATE TABLE fda_drug AS 
                SELECT * FROM read_parquet('{fda_drug_path}')
            """)
        else:
            con.execute(f"""
                CREATE TABLE fda_drug AS 
                SELECT * FROM read_csv_auto('{fda_drug_path}')
            """)
        count = con.execute("SELECT COUNT(*) FROM fda_drug").fetchone()[0]
        if verbose:
            print(f"  - fda_drug: {count:,} rows")
    
    # =========================================================================
    # STEP 2: Extract form-route validity mapping
    # =========================================================================
    if verbose:
        print("\n[Step 2] Extracting form-route validity mapping...")
    
    # Collect form-route combinations from all sources
    form_route_queries = []
    
    # From PNF
    try:
        form_route_queries.append("""
            SELECT DISTINCT
                UPPER(TRIM(form_token)) as form,
                UPPER(TRIM(route_allowed)) as route,
                'pnf' as source,
                generic_id as example_id
            FROM pnf
            WHERE form_token IS NOT NULL AND form_token != ''
              AND route_allowed IS NOT NULL AND route_allowed != ''
        """)
    except:
        pass
    
    # From DrugBank Products
    try:
        form_route_queries.append("""
            SELECT DISTINCT
                UPPER(TRIM(dosage_form)) as form,
                UPPER(TRIM(route)) as route,
                'drugbank_products' as source,
                drugbank_id as example_id
            FROM drugbank_products
            WHERE dosage_form IS NOT NULL AND dosage_form != ''
              AND route IS NOT NULL AND route != ''
        """)
    except:
        pass
    
    # From FDA Drug
    try:
        form_route_queries.append("""
            SELECT DISTINCT
                UPPER(TRIM(dosage_form)) as form,
                UPPER(TRIM(route)) as route,
                'fda_drug' as source,
                registration_number as example_id
            FROM fda_drug
            WHERE dosage_form IS NOT NULL AND dosage_form != ''
              AND route IS NOT NULL AND route != ''
        """)
    except:
        pass
    
    if form_route_queries:
        combined_query = " UNION ALL ".join(form_route_queries)
        form_route_df = con.execute(f"""
            SELECT form, route, source, example_id
            FROM ({combined_query})
            WHERE form IS NOT NULL AND route IS NOT NULL
              AND form != '' AND route != ''
        """).fetchdf()
        
        if verbose:
            print(f"  - Total form-route combinations: {len(form_route_df):,}")
        
        # Save form-route validity
        form_route_out = outputs_dir / "form_route_validity.parquet"
        form_route_df.to_parquet(form_route_out, index=False)
        form_route_df.to_csv(outputs_dir / "form_route_validity.csv", index=False)
        if verbose:
            print(f"  - Saved: {form_route_out}")
    
    # =========================================================================
    # STEP 3: Build generics lookup with synonyms
    # =========================================================================
    if verbose:
        print("\n[Step 3] Building generics lookup...")
    
    # Get unique generics from DrugBank
    generics_df = con.execute("""
        SELECT DISTINCT
            drugbank_id,
            UPPER(TRIM(canonical_generic_name)) as generic_name,
            atc_code,
            'drugbank' as source
        FROM drugbank_generics
        WHERE drugbank_id IS NOT NULL
          AND canonical_generic_name IS NOT NULL
          AND canonical_generic_name != ''
    """).fetchdf()
    
    if verbose:
        print(f"  - DrugBank generics: {len(generics_df):,}")
    
    # Add WHO generics
    try:
        who_generics = con.execute("""
            SELECT DISTINCT
                NULL as drugbank_id,
                UPPER(TRIM(atc_name)) as generic_name,
                atc_code,
                'who' as source
            FROM who_atc
            WHERE atc_name IS NOT NULL AND atc_name != ''
        """).fetchdf()
        generics_df = pd.concat([generics_df, who_generics], ignore_index=True)
        if verbose:
            print(f"  - Added WHO generics: {len(who_generics):,}")
    except:
        pass
    
    # Add PNF generics
    try:
        pnf_generics = con.execute("""
            SELECT DISTINCT
                NULL as drugbank_id,
                UPPER(TRIM(generic_name)) as generic_name,
                atc_code,
                'pnf' as source
            FROM pnf
            WHERE generic_name IS NOT NULL AND generic_name != ''
        """).fetchdf()
        generics_df = pd.concat([generics_df, pnf_generics], ignore_index=True)
        if verbose:
            print(f"  - Added PNF generics: {len(pnf_generics):,}")
    except:
        pass
    
    # Deduplicate by generic_name, keeping first drugbank_id and aggregating sources
    generics_agg = generics_df.groupby("generic_name", as_index=False).agg({
        "drugbank_id": "first",
        "atc_code": lambda x: "|".join(sorted(set(str(v) for v in x if pd.notna(v) and str(v).strip()))),
        "source": lambda x: "|".join(sorted(set(str(v) for v in x if pd.notna(v)))),
    })
    
    if verbose:
        print(f"  - Deduplicated generics: {len(generics_agg):,}")
    
    # Save generics lookup
    generics_out = outputs_dir / "generics_lookup.parquet"
    generics_agg.to_parquet(generics_out, index=False)
    generics_agg.to_csv(outputs_dir / "generics_lookup.csv", index=False)
    if verbose:
        print(f"  - Saved: {generics_out}")
    
    # =========================================================================
    # STEP 4: Build brands lookup
    # =========================================================================
    if verbose:
        print("\n[Step 4] Building brands lookup...")
    
    brands_parts = []
    
    # From DrugBank brands
    try:
        db_brands = con.execute("""
            SELECT DISTINCT
                UPPER(TRIM(brand_name_normalized)) as brand_name,
                drugbank_id,
                UPPER(TRIM(canonical_generic_name)) as generic_name,
                'drugbank' as source
            FROM drugbank_brands
            WHERE brand_name_normalized IS NOT NULL AND brand_name_normalized != ''
        """).fetchdf()
        brands_parts.append(db_brands)
        if verbose:
            print(f"  - DrugBank brands: {len(db_brands):,}")
    except:
        pass
    
    # From FDA drug
    try:
        fda_brands = con.execute("""
            SELECT DISTINCT
                UPPER(TRIM(brand_name)) as brand_name,
                NULL as drugbank_id,
                UPPER(TRIM(generic_name)) as generic_name,
                'fda_drug' as source
            FROM fda_drug
            WHERE brand_name IS NOT NULL AND brand_name != ''
        """).fetchdf()
        brands_parts.append(fda_brands)
        if verbose:
            print(f"  - FDA brands: {len(fda_brands):,}")
    except:
        pass
    
    if brands_parts:
        brands_df = pd.concat(brands_parts, ignore_index=True)
        
        # Deduplicate by brand_name
        brands_agg = brands_df.groupby("brand_name", as_index=False).agg({
            "drugbank_id": "first",
            "generic_name": "first",
            "source": lambda x: "|".join(sorted(set(str(v) for v in x if pd.notna(v)))),
        })
        
        if verbose:
            print(f"  - Deduplicated brands: {len(brands_agg):,}")
        
        # Save brands lookup
        brands_out = outputs_dir / "brands_lookup.parquet"
        brands_agg.to_parquet(brands_out, index=False)
        brands_agg.to_csv(outputs_dir / "brands_lookup.csv", index=False)
        if verbose:
            print(f"  - Saved: {brands_out}")
    
    # =========================================================================
    # STEP 5: Build mixtures lookup
    # =========================================================================
    if verbose:
        print("\n[Step 5] Building mixtures lookup...")
    
    try:
        mixtures_df = con.execute("""
            SELECT DISTINCT
                mixture_drugbank_id as drugbank_id,
                UPPER(TRIM(mixture_name)) as mixture_name,
                UPPER(TRIM(ingredient_components)) as components,
                component_drugbank_ids,
                'drugbank' as source
            FROM drugbank_mixtures
            WHERE mixture_drugbank_id IS NOT NULL
        """).fetchdf()
        
        if verbose:
            print(f"  - Mixtures: {len(mixtures_df):,}")
        
        # Create component key for lookup
        def make_component_key(components: str) -> str:
            if pd.isna(components) or not components:
                return ""
            parts = [p.strip() for p in components.split(";") if p.strip()]
            return "||".join(sorted(parts))
        
        mixtures_df["component_key"] = mixtures_df["components"].apply(make_component_key)
        
        # Save mixtures lookup
        mixtures_out = outputs_dir / "mixtures_lookup.parquet"
        mixtures_df.to_parquet(mixtures_out, index=False)
        mixtures_df.to_csv(outputs_dir / "mixtures_lookup.csv", index=False)
        if verbose:
            print(f"  - Saved: {mixtures_out}")
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not build mixtures lookup: {e}")
    
    # =========================================================================
    # STEP 6: Build unified reference dataset
    # =========================================================================
    if verbose:
        print("\n[Step 6] Building unified reference dataset...")
    
    # Start with DrugBank generics as base
    # Explode by: drugbank_id × atc_code × form × route (only valid combos)
    
    # Get base generics with forms and routes from products
    try:
        unified_df = con.execute("""
            WITH base AS (
                SELECT DISTINCT
                    g.drugbank_id,
                    g.atc_code,
                    UPPER(TRIM(g.canonical_generic_name)) as generic_name,
                    UPPER(TRIM(p.dosage_form)) as form,
                    UPPER(TRIM(p.route)) as route,
                    p.strength as dose
                FROM drugbank_generics g
                LEFT JOIN drugbank_products p ON g.drugbank_id = p.drugbank_id
                WHERE g.drugbank_id IS NOT NULL
                  AND g.canonical_generic_name IS NOT NULL
            )
            SELECT 
                drugbank_id,
                atc_code,
                generic_name,
                form,
                route,
                STRING_AGG(DISTINCT dose, '|') as doses
            FROM base
            WHERE form IS NOT NULL AND form != ''
              AND route IS NOT NULL AND route != ''
            GROUP BY drugbank_id, atc_code, generic_name, form, route
        """).fetchdf()
        
        if verbose:
            print(f"  - Base unified rows (with form/route): {len(unified_df):,}")
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not build unified dataset: {e}")
        unified_df = pd.DataFrame()
    
    # Add rows without form/route for drugs that only have generic info
    try:
        generics_only = con.execute("""
            SELECT DISTINCT
                drugbank_id,
                atc_code,
                UPPER(TRIM(canonical_generic_name)) as generic_name,
                NULL as form,
                NULL as route,
                NULL as doses
            FROM drugbank_generics
            WHERE drugbank_id IS NOT NULL
              AND canonical_generic_name IS NOT NULL
              AND drugbank_id NOT IN (
                  SELECT DISTINCT drugbank_id FROM drugbank_products
                  WHERE dosage_form IS NOT NULL AND dosage_form != ''
              )
        """).fetchdf()
        
        if len(generics_only) > 0:
            unified_df = pd.concat([unified_df, generics_only], ignore_index=True)
            if verbose:
                print(f"  - Added generics without products: {len(generics_only):,}")
    except:
        pass
    
    # Add source tracking
    unified_df["sources"] = "drugbank"
    
    # Enrich with PNF data
    try:
        pnf_enrichment = con.execute("""
            SELECT DISTINCT
                NULL as drugbank_id,
                atc_code,
                UPPER(TRIM(generic_name)) as generic_name,
                UPPER(TRIM(form_token)) as form,
                UPPER(TRIM(route_allowed)) as route,
                NULL as doses
            FROM pnf
            WHERE generic_name IS NOT NULL AND generic_name != ''
        """).fetchdf()
        pnf_enrichment["sources"] = "pnf"
        unified_df = pd.concat([unified_df, pnf_enrichment], ignore_index=True)
        if verbose:
            print(f"  - Added PNF enrichment: {len(pnf_enrichment):,}")
    except:
        pass
    
    # Deduplicate
    unified_df = unified_df.drop_duplicates(
        subset=["drugbank_id", "atc_code", "generic_name", "form", "route"],
        keep="first"
    )
    
    if verbose:
        print(f"  - Final unified rows: {len(unified_df):,}")
    
    # Save unified reference
    unified_out = outputs_dir / "unified_drug_reference.parquet"
    unified_df.to_parquet(unified_out, index=False)
    unified_df.to_csv(outputs_dir / "unified_drug_reference.csv", index=False)
    if verbose:
        print(f"  - Saved: {unified_out}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("Unified Reference Dataset Complete")
        print("=" * 60)
        print(f"  - unified_drug_reference.parquet: {len(unified_df):,} rows")
        print(f"  - generics_lookup.parquet: {len(generics_agg):,} rows")
        if 'brands_agg' in dir():
            print(f"  - brands_lookup.parquet: {len(brands_agg):,} rows")
        if 'mixtures_df' in dir():
            print(f"  - mixtures_lookup.parquet: {len(mixtures_df):,} rows")
        if 'form_route_df' in dir():
            print(f"  - form_route_validity.parquet: {len(form_route_df):,} rows")
    
    con.close()
    
    return {
        "unified": unified_out,
        "generics": generics_out,
        "brands": outputs_dir / "brands_lookup.parquet",
        "mixtures": outputs_dir / "mixtures_lookup.parquet",
        "form_route": outputs_dir / "form_route_validity.parquet",
    }


if __name__ == "__main__":
    build_unified_reference()
