#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build unified drug reference tables (Option B: Multiple Normalized Tables).

This produces MULTIPLE parquet files in outputs/drugs/:
    unified_generics.parquet  - Main reference: generic × atc × form × route × dose
    unified_brands.parquet    - Brand → generic mapping
    unified_synonyms.parquet  - Synonym → canonical generic mapping
    unified_salts.parquet     - Salt forms per generic
    unified_mixtures.parquet  - Mixture component mappings

Pipeline code should ONLY reference unified_* tables, never raw source files.

Sources loaded (internal only, not referenced by pipeline):
- DrugBank generics, products, mixtures, salts, brands
- WHO ATC, FDA brands, PNF lexicon

Usage:
    python -m pipelines.drugs.scripts.build_unified_reference
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# Paths
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
    """
    Build multiple normalized unified_* reference tables.
    
    Returns dict of table names to paths.
    """
    inputs_dir = Path(inputs_dir or INPUTS_DIR)
    outputs_dir = Path(outputs_dir or OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("Building unified_* reference tables")
        print("=" * 60)
    
    # Create in-memory DuckDB
    con = duckdb.connect(":memory:")
    
    # =========================================================================
    # Load source data into DuckDB
    # =========================================================================
    if verbose:
        print("\n[Step 1] Loading source data...")
    
    # DrugBank generics
    db_generics_path = inputs_dir / "drugbank_generics_master.csv"
    if db_generics_path.exists():
        con.execute(f"CREATE TABLE drugbank_generics AS SELECT * FROM read_csv_auto('{db_generics_path}')")
        count = con.execute("SELECT COUNT(*) FROM drugbank_generics").fetchone()[0]
        if verbose:
            print(f"  - drugbank_generics: {count:,} rows")
    
    # DrugBank products (for form/route)
    db_products_path = inputs_dir / "drugbank_products_export.csv"
    if db_products_path.exists():
        con.execute(f"CREATE TABLE drugbank_products AS SELECT * FROM read_csv_auto('{db_products_path}')")
        count = con.execute("SELECT COUNT(*) FROM drugbank_products").fetchone()[0]
        if verbose:
            print(f"  - drugbank_products: {count:,} rows")
    
    # DrugBank brands
    db_brands_path = inputs_dir / "drugbank_brands_master.csv"
    if db_brands_path.exists():
        con.execute(f"CREATE TABLE drugbank_brands AS SELECT * FROM read_csv_auto('{db_brands_path}')")
        count = con.execute("SELECT COUNT(*) FROM drugbank_brands").fetchone()[0]
        if verbose:
            print(f"  - drugbank_brands: {count:,} rows")
    
    # DrugBank mixtures
    db_mixtures_path = inputs_dir / "drugbank_mixtures_master.csv"
    if db_mixtures_path.exists():
        con.execute(f"CREATE TABLE drugbank_mixtures AS SELECT * FROM read_csv_auto('{db_mixtures_path}')")
        count = con.execute("SELECT COUNT(*) FROM drugbank_mixtures").fetchone()[0]
        if verbose:
            print(f"  - drugbank_mixtures: {count:,} rows")
    
    # DrugBank salts
    db_salts_path = inputs_dir / "drugbank_salts_master.csv"
    if db_salts_path.exists():
        con.execute(f"CREATE TABLE drugbank_salts AS SELECT * FROM read_csv_auto('{db_salts_path}')")
        count = con.execute("SELECT COUNT(*) FROM drugbank_salts").fetchone()[0]
        if verbose:
            print(f"  - drugbank_salts: {count:,} rows")
    
    # WHO ATC (latest)
    who_files = sorted(glob.glob(str(inputs_dir / "who_atc_*.parquet")))
    if who_files:
        who_path = who_files[-1]
        con.execute(f"CREATE TABLE who_atc AS SELECT * FROM read_parquet('{who_path}')")
        count = con.execute("SELECT COUNT(*) FROM who_atc").fetchone()[0]
        if verbose:
            print(f"  - who_atc: {count:,} rows")
    
    # FDA brands (latest)
    fda_files = sorted(glob.glob(str(inputs_dir / "fda_drug_*.parquet")))
    if fda_files:
        fda_path = fda_files[-1]
        con.execute(f"CREATE TABLE fda_brands AS SELECT * FROM read_parquet('{fda_path}')")
        count = con.execute("SELECT COUNT(*) FROM fda_brands").fetchone()[0]
        if verbose:
            print(f"  - fda_brands: {count:,} rows")
    
    # PNF lexicon
    pnf_path = inputs_dir / "pnf_lexicon.parquet"
    if pnf_path.exists():
        con.execute(f"CREATE TABLE pnf AS SELECT * FROM read_parquet('{pnf_path}')")
        count = con.execute("SELECT COUNT(*) FROM pnf").fetchone()[0]
        if verbose:
            print(f"  - pnf: {count:,} rows")
    
    # =========================================================================
    # TABLE 1: unified_generics - Main reference (generic × atc × form × route × dose)
    # =========================================================================
    if verbose:
        print("\n[Step 2] Building unified_generics...")
    
    # Step 2a: Get unique generics first (no explosion)
    base_generics = con.execute("""
        SELECT DISTINCT
            drugbank_id,
            COALESCE(atc_code, '') as atc_code,
            UPPER(TRIM(canonical_generic_name)) as generic_name
        FROM drugbank_generics
        WHERE drugbank_id IS NOT NULL
          AND canonical_generic_name IS NOT NULL
          AND canonical_generic_name != ''
    """).fetchdf()
    
    if verbose:
        print(f"    - Base generics: {len(base_generics):,}")
    
    # Step 2b: Get product variants separately (form/route/dose)
    product_variants = con.execute("""
        SELECT DISTINCT
            drugbank_id,
            UPPER(TRIM(dosage_form)) as form,
            UPPER(TRIM(route)) as route,
            UPPER(TRIM(strength)) as dose
        FROM drugbank_products
        WHERE drugbank_id IS NOT NULL
    """).fetchdf()
    
    if verbose:
        print(f"    - Product variants: {len(product_variants):,}")
    
    # Step 2c: Merge (memory efficient)
    generics_df = base_generics.merge(product_variants, on='drugbank_id', how='left')
    generics_df['source'] = 'drugbank'
    generics_df = generics_df.fillna('')
    generics_df = generics_df[['generic_name', 'drugbank_id', 'atc_code', 'form', 'route', 'dose', 'source']]
    generics_df = generics_df.drop_duplicates()
    
    # Add WHO-only entries
    try:
        existing_generics = set(generics_df['generic_name'].str.upper().unique())
        who_df = con.execute("""
            SELECT DISTINCT
                UPPER(TRIM(atc_name)) as generic_name,
                NULL as drugbank_id,
                atc_code,
                '' as form,
                '' as route,
                '' as dose,
                'who' as source
            FROM who_atc
            WHERE atc_name IS NOT NULL AND atc_name != ''
        """).fetchdf()
        who_new = who_df[~who_df['generic_name'].isin(existing_generics)]
        generics_df = pd.concat([generics_df, who_new], ignore_index=True)
    except Exception:
        pass
    
    generics_df = generics_df.fillna('').drop_duplicates()
    
    # Note: Mixture combinations are NOT added to unified_generics to keep it fast.
    # Instead, the tagger queries unified_mixtures on-demand when multiple generics detected.
    
    # =========================================================================
    # TABLE 2: unified_brands - Brand → generic mapping (normalized, one row per brand)
    # =========================================================================
    if verbose:
        print("\n[Step 3] Building unified_brands...")
    
    # FDA brands FIRST (Philippines context - FDA PH brands more relevant)
    brands_list = []
    
    try:
        fda_brands_df = con.execute("""
            SELECT DISTINCT
                UPPER(TRIM(brand_name)) as brand_name,
                UPPER(TRIM(generic_name)) as generic_name,
                NULL as drugbank_id,
                'fda' as source
            FROM fda_brands
            WHERE brand_name IS NOT NULL AND brand_name != ''
              AND generic_name IS NOT NULL AND generic_name != ''
        """).fetchdf()
        brands_list.append(fda_brands_df)
        if verbose:
            print(f"    - FDA brands: {len(fda_brands_df):,}")
    except Exception:
        pass
    
    # DrugBank brands second (simpler query to avoid swap)
    drugbank_brands_df = con.execute("""
        SELECT DISTINCT
            UPPER(TRIM(brand_name)) as brand_name,
            UPPER(TRIM(canonical_generic_name)) as generic_name,
            drugbank_id,
            'drugbank' as source
        FROM drugbank_brands
        WHERE brand_name IS NOT NULL AND brand_name != ''
          AND canonical_generic_name IS NOT NULL AND canonical_generic_name != ''
    """).fetchdf()
    brands_list.append(drugbank_brands_df)
    if verbose:
        print(f"    - DrugBank brands: {len(drugbank_brands_df):,}")
    
    brands_df = pd.concat(brands_list, ignore_index=True)
    
    # Keep first occurrence (FDA wins due to order)
    brands_df = brands_df.fillna('').drop_duplicates(subset=['brand_name'], keep='first')
    
    # =========================================================================
    # TABLE 3: unified_synonyms - Synonym → canonical generic mapping
    # =========================================================================
    if verbose:
        print("\n[Step 4] Building unified_synonyms...")
    
    synonyms_df = con.execute("""
        SELECT DISTINCT
            UPPER(TRIM(lexeme)) as synonym,
            UPPER(TRIM(canonical_generic_name)) as generic_name,
            drugbank_id
        FROM drugbank_generics
        WHERE lexeme IS NOT NULL AND lexeme != ''
          AND canonical_generic_name IS NOT NULL AND canonical_generic_name != ''
          AND UPPER(TRIM(lexeme)) != UPPER(TRIM(canonical_generic_name))
    """).fetchdf()
    
    synonyms_df = synonyms_df.fillna('').drop_duplicates(subset=['synonym'], keep='first')
    
    # =========================================================================
    # TABLE 4: unified_salts - Salt forms per generic
    # =========================================================================
    if verbose:
        print("\n[Step 5] Building unified_salts...")
    
    salts_df = con.execute("""
        SELECT DISTINCT
            UPPER(TRIM(s.salt_name_normalized)) as salt_form,
            s.parent_drugbank_id as drugbank_id,
            UPPER(TRIM(g.canonical_generic_name)) as generic_name
        FROM drugbank_salts s
        LEFT JOIN drugbank_generics g ON s.parent_drugbank_id = g.drugbank_id
        WHERE s.salt_name_normalized IS NOT NULL AND s.salt_name_normalized != ''
          AND s.parent_drugbank_id IS NOT NULL
    """).fetchdf()
    
    salts_df = salts_df.fillna('').drop_duplicates()
    
    # =========================================================================
    # TABLE 5: unified_mixtures - Mixture component mappings
    # =========================================================================
    if verbose:
        print("\n[Step 6] Building unified_mixtures...")
    
    mixtures_df = con.execute("""
        SELECT DISTINCT
            mixture_drugbank_id as drugbank_id,
            UPPER(TRIM(mixture_name)) as mixture_name,
            component_drugbank_ids,
            UPPER(TRIM(ingredient_components)) as component_generics
        FROM drugbank_mixtures
        WHERE mixture_drugbank_id IS NOT NULL
    """).fetchdf()
    
    mixtures_df = mixtures_df.fillna('').drop_duplicates()
    
    # =========================================================================
    # Save all tables
    # =========================================================================
    if verbose:
        print("\n[Step 7] Saving tables...")
    
    output_paths = {}
    output_paths['unified_generics'] = _save_table(generics_df, outputs_dir, 'unified_generics', verbose)
    output_paths['unified_brands'] = _save_table(brands_df, outputs_dir, 'unified_brands', verbose)
    output_paths['unified_synonyms'] = _save_table(synonyms_df, outputs_dir, 'unified_synonyms', verbose)
    output_paths['unified_salts'] = _save_table(salts_df, outputs_dir, 'unified_salts', verbose)
    output_paths['unified_mixtures'] = _save_table(mixtures_df, outputs_dir, 'unified_mixtures', verbose)
    
    # Also save legacy unified_drug_reference.parquet for backward compatibility
    # (this will be removed once tagger is updated)
    legacy_df = generics_df.copy()
    legacy_df.to_parquet(outputs_dir / "unified_drug_reference.parquet", index=False)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  unified_generics: {len(generics_df):,} rows ({generics_df['generic_name'].nunique():,} unique generics)")
        print(f"  unified_brands: {len(brands_df):,} brand→generic mappings")
        print(f"  unified_synonyms: {len(synonyms_df):,} synonym→generic mappings")
        print(f"  unified_salts: {len(salts_df):,} salt entries")
        print(f"  unified_mixtures: {len(mixtures_df):,} mixture entries")
        print("=" * 60)
    
    con.close()
    return output_paths


if __name__ == "__main__":
    build_unified_reference()
