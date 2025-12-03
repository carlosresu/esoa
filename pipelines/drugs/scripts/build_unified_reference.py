#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build unified drug reference tables - LEAN version.

Tables produced (no explosions except valid form×route×dose combos):
    unified_generics.parquet   - drugbank_id → generic_name (lean, one per drug)
    unified_synonyms.parquet   - drugbank_id → synonyms (pipe-separated)
    unified_atc.parquet        - drugbank_id → atc_code (one row per valid combo)
    unified_products.parquet   - drugbank_id × form × route × dose (valid combos only)
    unified_brands.parquet     - brand_name → generic_name, drugbank_id
    unified_salts.parquet      - drugbank_id → salt forms
    unified_mixtures.parquet   - mixture components with component_key

Usage:
    python -m pipelines.drugs.scripts.build_unified_reference_v2
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def _save_table(df: pd.DataFrame, outputs_dir: Path, name: str, verbose: bool = True):
    """Save dataframe as parquet + csv."""
    parquet_path = outputs_dir / f"{name}.parquet"
    csv_path = outputs_dir / f"{name}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"  ✓ {name}: {len(df):,} rows")
    return parquet_path


def build_unified_reference(
    inputs_dir: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Build lean unified_* reference tables."""
    inputs_dir = Path(inputs_dir or INPUTS_DIR)
    outputs_dir = Path(outputs_dir or OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("Building LEAN unified_* reference tables")
        print("=" * 60)
    
    con = duckdb.connect(":memory:")
    
    # =========================================================================
    # Load source data (LEAN exports from drugbank_lean_export.R)
    # =========================================================================
    if verbose:
        print("\n[Step 1] Loading lean exports...")
    
    # Helper: prefer parquet over csv
    def load_table(table_name: str, basename: str):
        parquet_path = inputs_dir / f"{basename}.parquet"
        csv_path = inputs_dir / f"{basename}.csv"
        if parquet_path.exists():
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path}')")
            return parquet_path
        elif csv_path.exists():
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")
            return csv_path
        return None
    
    # Lean DrugBank tables (prefer parquet)
    lean_tables = [
        ("generics", "generics_lean"),
        ("synonyms", "synonyms_lean"),
        ("dosages", "dosages_lean"),
        ("atc", "atc_lean"),
        ("brands", "brands_lean"),
        ("salts", "salts_lean"),
        ("mixtures", "mixtures_lean"),
        ("products", "products_lean"),
    ]
    
    for table_name, basename in lean_tables:
        path = load_table(table_name, basename)
        if path and verbose:
            count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  - {table_name}: {count:,} rows")
    
    # Lookup tables (prefer parquet)
    lookup_tables = [
        ("lookup_salt_suffixes", "lookup_salt_suffixes"),
        ("lookup_pure_salts", "lookup_pure_salts"),
        ("lookup_form_canonical", "lookup_form_canonical"),
        ("lookup_route_canonical", "lookup_route_canonical"),
        ("lookup_form_to_route", "lookup_form_to_route"),
        ("lookup_per_unit", "lookup_per_unit"),
    ]
    
    for table_name, basename in lookup_tables:
        load_table(table_name, basename)
    
    # WHO ATC
    who_files = sorted(glob.glob(str(inputs_dir / "who_atc_*.parquet")))
    if who_files:
        who_path = who_files[-1]
        con.execute(f"CREATE TABLE who_atc AS SELECT * FROM read_parquet('{who_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM who_atc").fetchone()[0]
            print(f"  - who_atc: {count:,} rows")
    
    # FDA brands
    fda_files = sorted(glob.glob(str(inputs_dir / "fda_drug_*.parquet")))
    if fda_files:
        fda_path = fda_files[-1]
        con.execute(f"CREATE TABLE fda_brands AS SELECT * FROM read_parquet('{fda_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM fda_brands").fetchone()[0]
            print(f"  - fda_brands: {count:,} rows")

    # =========================================================================
    # TABLE 1: unified_generics - LEAN (one row per generic)
    # =========================================================================
    if verbose:
        print("\n[Step 2] Building unified_generics (lean)...")
    
    generics_df = con.execute("""
        SELECT DISTINCT
            drugbank_id,
            UPPER(TRIM(name)) as generic_name,
            name_key,
            'drugbank' as source
        FROM generics
        WHERE drugbank_id IS NOT NULL
          AND name IS NOT NULL AND name != ''
    """).fetchdf()
    
    # Add WHO-only entries
    try:
        existing = set(generics_df['generic_name'].str.upper().unique())
        who_df = con.execute("""
            SELECT DISTINCT
                NULL as drugbank_id,
                UPPER(TRIM(atc_name)) as generic_name,
                LOWER(REGEXP_REPLACE(atc_name, '[^a-zA-Z0-9 ]', '', 'g')) as name_key,
                'who' as source
            FROM who_atc
            WHERE atc_name IS NOT NULL AND atc_name != ''
        """).fetchdf()
        who_new = who_df[~who_df['generic_name'].isin(existing)]
        generics_df = pd.concat([generics_df, who_new], ignore_index=True)
    except Exception:
        pass
    
    generics_df = generics_df.drop_duplicates(subset=['generic_name'])
    
    # Add canonical drug name aliases for common combinations
    canonical_generics = [
        {"drugbank_id": "DB00766", "generic_name": "AMOXICILLIN + CLAVULANIC ACID", "name_key": "amoxicillin + clavulanic acid", "source": "canonical"},
        {"drugbank_id": None, "generic_name": "COTRIMOXAZOLE", "name_key": "cotrimoxazole", "source": "canonical"},
        {"drugbank_id": None, "generic_name": "SULFAMETHOXAZOLE + TRIMETHOPRIM", "name_key": "sulfamethoxazole + trimethoprim", "source": "canonical"},
    ]
    canonical_gen_df = pd.DataFrame(canonical_generics)
    generics_df = pd.concat([generics_df, canonical_gen_df], ignore_index=True).drop_duplicates(subset=['generic_name'])
    
    # =========================================================================
    # TABLE 2: unified_synonyms - drugbank_id → synonyms (aggregated)
    # =========================================================================
    if verbose:
        print("\n[Step 3] Building unified_synonyms...")
    
    synonyms_df = con.execute("""
        SELECT 
            s.drugbank_id,
            UPPER(TRIM(g.name)) as generic_name,
            STRING_AGG(DISTINCT UPPER(TRIM(s.synonym)), '|') as synonyms
        FROM synonyms s
        LEFT JOIN generics g ON s.drugbank_id = g.drugbank_id
        WHERE s.drugbank_id IS NOT NULL
          AND s.synonym IS NOT NULL AND s.synonym != ''
        GROUP BY s.drugbank_id, g.name
    """).fetchdf()
    
    # =========================================================================
    # TABLE 3: unified_atc_map - drugbank_id ↔ atc_code mapping
    # =========================================================================
    if verbose:
        print("\n[Step 4] Building unified_atc_map...")
    
    atc_map_df = con.execute("""
        SELECT DISTINCT
            a.drugbank_id,
            UPPER(TRIM(g.name)) as generic_name,
            TRIM(a.atc_code) as atc_code
        FROM atc a
        LEFT JOIN generics g ON a.drugbank_id = g.drugbank_id
        WHERE a.drugbank_id IS NOT NULL
          AND a.atc_code IS NOT NULL AND a.atc_code != ''
    """).fetchdf()
    
    # Add WHO ATC entries
    try:
        who_atc_df = con.execute("""
            SELECT DISTINCT
                NULL as drugbank_id,
                UPPER(TRIM(atc_name)) as generic_name,
                TRIM(atc_code) as atc_code
            FROM who_atc
            WHERE atc_code IS NOT NULL AND atc_code != ''
        """).fetchdf()
        atc_map_df = pd.concat([atc_map_df, who_atc_df], ignore_index=True)
    except Exception:
        pass
    
    atc_map_df = atc_map_df.drop_duplicates()
    
    # Add canonical drug name aliases for common combinations
    # These map user-friendly names to their ATC codes
    canonical_aliases = [
        # Amoxicillin + Clavulanic acid combinations -> J01CR02
        {"drugbank_id": "DB00766", "generic_name": "AMOXICILLIN + CLAVULANIC ACID", "atc_code": "J01CR02"},
        {"drugbank_id": "DB00766", "generic_name": "CO-AMOXICLAV", "atc_code": "J01CR02"},
        # Sulfamethoxazole + Trimethoprim -> J01EE01
        {"drugbank_id": None, "generic_name": "COTRIMOXAZOLE", "atc_code": "J01EE01"},
        {"drugbank_id": None, "generic_name": "SULFAMETHOXAZOLE + TRIMETHOPRIM", "atc_code": "J01EE01"},
    ]
    canonical_df = pd.DataFrame(canonical_aliases)
    atc_map_df = pd.concat([atc_map_df, canonical_df], ignore_index=True).drop_duplicates()
    
    # =========================================================================
    # TABLE 4: unified_dosages - drugbank_id × form × route × dose (VALID combos)
    # =========================================================================
    if verbose:
        print("\n[Step 5] Building unified_dosages (valid combos only)...")
    
    dosages_df = con.execute("""
        SELECT DISTINCT
            drugbank_id,
            UPPER(TRIM(form)) as form,
            UPPER(TRIM(route)) as route,
            UPPER(TRIM(strength)) as dose
        FROM dosages
        WHERE drugbank_id IS NOT NULL
    """).fetchdf()
    
    dosages_df = dosages_df.fillna('').drop_duplicates()
    if verbose:
        print(f"    - {len(dosages_df):,} valid combos")
    
    # =========================================================================
    # TABLE 5: unified_brands - brand → generic mapping
    # =========================================================================
    if verbose:
        print("\n[Step 6] Building unified_brands...")
    
    brands_list = []
    
    # FDA brands first
    try:
        fda_df = con.execute("""
            SELECT DISTINCT
                UPPER(TRIM(brand_name)) as brand_name,
                UPPER(TRIM(generic_name)) as generic_name,
                NULL as drugbank_id,
                'fda' as source
            FROM fda_brands
            WHERE brand_name IS NOT NULL AND brand_name != ''
        """).fetchdf()
        brands_list.append(fda_df)
    except Exception:
        pass
    
    # DrugBank brands (lean export)
    try:
        db_brands_df = con.execute("""
            SELECT DISTINCT
                UPPER(TRIM(b.brand)) as brand_name,
                UPPER(TRIM(g.name)) as generic_name,
                b.drugbank_id,
                'drugbank' as source
            FROM brands b
            LEFT JOIN generics g ON b.drugbank_id = g.drugbank_id
            WHERE b.brand IS NOT NULL AND b.brand != ''
        """).fetchdf()
        brands_list.append(db_brands_df)
    except Exception:
        pass
    
    brands_df = pd.concat(brands_list, ignore_index=True) if brands_list else pd.DataFrame()
    brands_df = brands_df.fillna('').drop_duplicates(subset=['brand_name'], keep='first')
    
    # =========================================================================
    # TABLE 6: unified_salts - drugbank_id → salt forms
    # =========================================================================
    if verbose:
        print("\n[Step 7] Building unified_salts...")
    
    salts_df = con.execute("""
        SELECT DISTINCT
            drugbank_id,
            UPPER(TRIM(name)) as salt_form,
            name_key as salt_key
        FROM salts
        WHERE drugbank_id IS NOT NULL
          AND name IS NOT NULL AND name != ''
    """).fetchdf()
    
    salts_df = salts_df.fillna('').drop_duplicates()
    
    # =========================================================================
    # TABLE 7: unified_mixtures - with component_key for fast lookup
    # =========================================================================
    if verbose:
        print("\n[Step 8] Building unified_mixtures...")
    
    # Lean export already has component_key_sorted and component_count
    mixtures_df = con.execute("""
        SELECT DISTINCT
            drugbank_id,
            UPPER(TRIM(mixture_name)) as mixture_name,
            component_generics,
            component_keys,
            component_key_sorted as component_key,
            component_count
        FROM mixtures
        WHERE drugbank_id IS NOT NULL
          AND component_generics IS NOT NULL
          AND component_generics != ''
    """).fetchdf()
    
    mixtures_df = mixtures_df.fillna('').drop_duplicates(subset=['component_key'], keep='first')
    
    # =========================================================================
    # Save all tables
    # =========================================================================
    if verbose:
        print("\n[Step 9] Saving tables...")
    
    output_paths = {}
    # Core lookup tables
    output_paths['unified_generics'] = _save_table(generics_df, outputs_dir, 'unified_generics', verbose)
    output_paths['unified_synonyms'] = _save_table(synonyms_df, outputs_dir, 'unified_synonyms', verbose)
    output_paths['unified_atc'] = _save_table(atc_map_df, outputs_dir, 'unified_atc', verbose)
    
    # Dosages table (VALID form × route × dose combos per drugbank_id)
    output_paths['unified_dosages'] = _save_table(dosages_df, outputs_dir, 'unified_dosages', verbose)
    
    # Other tables
    output_paths['unified_brands'] = _save_table(brands_df, outputs_dir, 'unified_brands', verbose)
    output_paths['unified_salts'] = _save_table(salts_df, outputs_dir, 'unified_salts', verbose)
    output_paths['unified_mixtures'] = _save_table(mixtures_df, outputs_dir, 'unified_mixtures', verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Summary (all tables are LEAN - valid combos only):")
        print(f"  unified_generics: {len(generics_df):,} (one per generic)")
        print(f"  unified_synonyms: {len(synonyms_df):,} (aggregated per drugbank_id)")
        print(f"  unified_atc: {len(atc_map_df):,} (drugbank_id ↔ atc, NO form/route/dose)")
        print(f"  unified_dosages: {len(dosages_df):,} (drugbank_id × form × route × dose)")
        print(f"  unified_brands: {len(brands_df):,} (one per brand)")
        print(f"  unified_salts: {len(salts_df):,}")
        print(f"  unified_mixtures: {len(mixtures_df):,} (deduplicated)")
        print("=" * 60)
    
    con.close()
    return output_paths


if __name__ == "__main__":
    build_unified_reference()
