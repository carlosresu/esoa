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
    # Load source data
    # =========================================================================
    if verbose:
        print("\n[Step 1] Loading source data...")
    
    # DrugBank generics (exploded - we'll extract what we need)
    db_generics_path = inputs_dir / "drugbank_generics_master.csv"
    if db_generics_path.exists():
        con.execute(f"CREATE TABLE drugbank_generics AS SELECT * FROM read_csv_auto('{db_generics_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(DISTINCT drugbank_id) FROM drugbank_generics").fetchone()[0]
            print(f"  - drugbank_generics: {count:,} unique drugs")
    
    # DrugBank products (valid form × route × dose combos)
    db_products_path = inputs_dir / "drugbank_products_export.csv"
    if db_products_path.exists():
        con.execute(f"CREATE TABLE drugbank_products AS SELECT * FROM read_csv_auto('{db_products_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM drugbank_products").fetchone()[0]
            print(f"  - drugbank_products: {count:,} rows")
    
    # DrugBank brands
    db_brands_path = inputs_dir / "drugbank_brands_master.csv"
    if db_brands_path.exists():
        con.execute(f"CREATE TABLE drugbank_brands AS SELECT * FROM read_csv_auto('{db_brands_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM drugbank_brands").fetchone()[0]
            print(f"  - drugbank_brands: {count:,} rows")
    
    # DrugBank mixtures
    db_mixtures_path = inputs_dir / "drugbank_mixtures_master.csv"
    if db_mixtures_path.exists():
        con.execute(f"CREATE TABLE drugbank_mixtures AS SELECT * FROM read_csv_auto('{db_mixtures_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM drugbank_mixtures").fetchone()[0]
            print(f"  - drugbank_mixtures: {count:,} rows")
    
    # DrugBank salts
    db_salts_path = inputs_dir / "drugbank_salts_master.csv"
    if db_salts_path.exists():
        con.execute(f"CREATE TABLE drugbank_salts AS SELECT * FROM read_csv_auto('{db_salts_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM drugbank_salts").fetchone()[0]
            print(f"  - drugbank_salts: {count:,} rows")
    
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
            UPPER(TRIM(canonical_generic_name)) as generic_name,
            'drugbank' as source
        FROM drugbank_generics
        WHERE drugbank_id IS NOT NULL
          AND canonical_generic_name IS NOT NULL
          AND canonical_generic_name != ''
    """).fetchdf()
    
    # Add WHO-only entries
    try:
        existing = set(generics_df['generic_name'].str.upper().unique())
        who_df = con.execute("""
            SELECT DISTINCT
                NULL as drugbank_id,
                UPPER(TRIM(atc_name)) as generic_name,
                'who' as source
            FROM who_atc
            WHERE atc_name IS NOT NULL AND atc_name != ''
        """).fetchdf()
        who_new = who_df[~who_df['generic_name'].isin(existing)]
        generics_df = pd.concat([generics_df, who_new], ignore_index=True)
    except Exception:
        pass
    
    generics_df = generics_df.drop_duplicates(subset=['generic_name'])
    
    # =========================================================================
    # TABLE 2: unified_synonyms - drugbank_id → synonyms (aggregated)
    # =========================================================================
    if verbose:
        print("\n[Step 3] Building unified_synonyms...")
    
    synonyms_df = con.execute("""
        SELECT 
            drugbank_id,
            UPPER(TRIM(canonical_generic_name)) as generic_name,
            STRING_AGG(DISTINCT UPPER(TRIM(lexeme)), '|') as synonyms
        FROM drugbank_generics
        WHERE drugbank_id IS NOT NULL
          AND lexeme IS NOT NULL AND lexeme != ''
        GROUP BY drugbank_id, canonical_generic_name
    """).fetchdf()
    
    # =========================================================================
    # TABLE 3: unified_atc_map - drugbank_id ↔ atc_code mapping
    # =========================================================================
    if verbose:
        print("\n[Step 4] Building unified_atc_map...")
    
    atc_map_df = con.execute("""
        SELECT DISTINCT
            drugbank_id,
            UPPER(TRIM(canonical_generic_name)) as generic_name,
            TRIM(atc_code) as atc_code
        FROM drugbank_generics
        WHERE drugbank_id IS NOT NULL
          AND atc_code IS NOT NULL AND atc_code != ''
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
    
    # =========================================================================
    # TABLE 4: unified_dosages - drugbank_id × form × route × dose (VALID combos)
    # NOTE: ATC codes have NO direct link to form/route/dose in raw DrugBank!
    # The dosages table has valid combos per drugbank_id only.
    # =========================================================================
    if verbose:
        print("\n[Step 5] Building unified_dosages (valid combos only)...")
    
    # Try to load lean dosages if available, otherwise use existing exports
    lean_dosages_path = inputs_dir.parent.parent / "dependencies" / "drugbank_generics" / "output" / "drugbank_dosages_lean.csv"
    
    if lean_dosages_path.exists():
        # Use lean export from raw dbdataset
        con.execute(f"CREATE TABLE drugbank_dosages AS SELECT * FROM read_csv_auto('{lean_dosages_path}')")
        dosages_df = con.execute("""
            SELECT DISTINCT
                drugbank_id,
                UPPER(TRIM(form)) as form,
                UPPER(TRIM(route)) as route,
                UPPER(TRIM(strength)) as dose
            FROM drugbank_dosages
            WHERE drugbank_id IS NOT NULL
        """).fetchdf()
        if verbose:
            print(f"    - Loaded from lean export: {len(dosages_df):,} valid combos")
    else:
        # Fallback to existing products export
        if verbose:
            print("    - Using drugbank_products as fallback")
        dosages_df = con.execute("""
            SELECT DISTINCT
                drugbank_id,
                UPPER(TRIM(dosage_form)) as form,
                UPPER(TRIM(route)) as route,
                UPPER(TRIM(strength)) as dose
            FROM drugbank_products
            WHERE drugbank_id IS NOT NULL
        """).fetchdf()
    
    dosages_df = dosages_df.fillna('').drop_duplicates()
    
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
    
    # DrugBank brands
    try:
        db_brands_df = con.execute("""
            SELECT DISTINCT
                UPPER(TRIM(brand_name)) as brand_name,
                UPPER(TRIM(canonical_generic_name)) as generic_name,
                drugbank_id,
                'drugbank' as source
            FROM drugbank_brands
            WHERE brand_name IS NOT NULL AND brand_name != ''
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
            parent_drugbank_id as drugbank_id,
            UPPER(TRIM(salt_name_normalized)) as salt_form
        FROM drugbank_salts
        WHERE parent_drugbank_id IS NOT NULL
          AND salt_name_normalized IS NOT NULL
    """).fetchdf()
    
    salts_df = salts_df.fillna('').drop_duplicates()
    
    # =========================================================================
    # TABLE 7: unified_mixtures - with component_key for fast lookup
    # =========================================================================
    if verbose:
        print("\n[Step 8] Building unified_mixtures...")
    
    mixtures_df = con.execute("""
        SELECT DISTINCT
            mixture_drugbank_id as drugbank_id,
            UPPER(TRIM(mixture_name)) as mixture_name,
            component_drugbank_ids,
            UPPER(TRIM(ingredient_components)) as component_generics
        FROM drugbank_mixtures
        WHERE mixture_drugbank_id IS NOT NULL
          AND ingredient_components IS NOT NULL
          AND ingredient_components != ''
    """).fetchdf()
    
    # Add component_key for fast lookup
    def make_component_key(components_str):
        if not components_str:
            return ""
        parts = [p.strip().upper() for p in components_str.split(';') if p.strip()]
        return '|'.join(sorted(parts))
    
    mixtures_df['component_key'] = mixtures_df['component_generics'].apply(make_component_key)
    mixtures_df['component_count'] = mixtures_df['component_generics'].apply(
        lambda x: len([p for p in x.split(';') if p.strip()]) if x else 0
    )
    mixtures_df = mixtures_df.drop_duplicates(subset=['component_key'], keep='first')
    
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
