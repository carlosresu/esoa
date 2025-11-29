#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build THE unified drug reference dataset.

This produces ONE parquet file that is THE source of truth for all drug matching:
    outputs/drugs/unified_drug_reference.parquet

Schema (EXPLODED by generic × drugbank_id × atc_code × form × route):
- generic_name: str - canonical name for matching
- drugbank_id: str - DrugBank identifier  
- atc_code: str - ATC code (single, not pipe-delimited)
- form: str - dosage form (TABLET, CAPSULE, etc.)
- route: str - administration route (ORAL, IV, etc.)
- synonyms: str - pipe-delimited alternate names
- brands: str - pipe-delimited brand names
- salt_forms: str - pipe-delimited salt suffixes
- doses: str - pipe-delimited known doses
- mixture_components: str - pipe-delimited component DrugBank IDs (for combos)
- sources: str - pipe-delimited data sources

Sources loaded:
- DrugBank generics (drugbank_generics_master.csv)
- DrugBank products (drugbank_products_export.csv)
- DrugBank mixtures (drugbank_mixtures_master.csv)
- DrugBank salts (drugbank_salts_master.csv)
- DrugBank brands (drugbank_brands_master.csv)
- WHO ATC (who_atc_*.parquet)
- FDA brands (fda_drug_*.parquet)
- PNF lexicon (pnf_lexicon.parquet)

Usage:
    python -m pipelines.drugs.scripts.build_unified_reference_v2
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


def build_unified_reference(
    inputs_dir: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Path:
    """
    Build THE unified drug reference dataset.
    
    Returns path to unified_drug_reference.parquet
    """
    inputs_dir = Path(inputs_dir or INPUTS_DIR)
    outputs_dir = Path(outputs_dir or OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("Building unified_drug_reference.parquet")
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
    # Build unified reference - EXPLODED by generic × drugbank_id × atc × form × route
    # =========================================================================
    if verbose:
        print("\n[Step 2] Building exploded reference...")
    
    # Base: DrugBank generics × products (form/route)
    unified_df = con.execute("""
        WITH base AS (
            SELECT DISTINCT
                g.drugbank_id,
                COALESCE(g.atc_code, '') as atc_code,
                UPPER(TRIM(g.canonical_generic_name)) as generic_name,
                UPPER(TRIM(p.dosage_form)) as form,
                UPPER(TRIM(p.route)) as route,
                p.strength as dose
            FROM drugbank_generics g
            LEFT JOIN drugbank_products p ON g.drugbank_id = p.drugbank_id
            WHERE g.drugbank_id IS NOT NULL
              AND g.canonical_generic_name IS NOT NULL
              AND g.canonical_generic_name != ''
        ),
        -- Aggregate doses per form/route combo
        with_doses AS (
            SELECT 
                drugbank_id,
                atc_code,
                generic_name,
                COALESCE(form, '') as form,
                COALESCE(route, '') as route,
                STRING_AGG(DISTINCT dose, '|') FILTER (WHERE dose IS NOT NULL AND dose != '') as doses
            FROM base
            GROUP BY drugbank_id, atc_code, generic_name, form, route
        )
        SELECT * FROM with_doses
    """).fetchdf()
    
    if verbose:
        print(f"  - Base rows (DrugBank): {len(unified_df):,}")
    
    # =========================================================================
    # Add synonyms from DrugBank lexemes
    # =========================================================================
    if verbose:
        print("\n[Step 3] Adding synonyms...")
    
    synonyms_df = con.execute("""
        SELECT 
            drugbank_id,
            STRING_AGG(DISTINCT UPPER(TRIM(lexeme)), '|') as synonyms
        FROM drugbank_generics
        WHERE drugbank_id IS NOT NULL AND lexeme IS NOT NULL AND lexeme != ''
        GROUP BY drugbank_id
    """).fetchdf()
    
    unified_df = unified_df.merge(synonyms_df, on='drugbank_id', how='left')
    unified_df['synonyms'] = unified_df['synonyms'].fillna('')
    
    if verbose:
        print(f"  - Added synonyms for {len(synonyms_df):,} generics")
    
    # =========================================================================
    # Note: Brands are NOT stored in exploded dataset to avoid bloat
    # They should be looked up separately by drugbank_id or generic_name
    # =========================================================================
    if verbose:
        print("\n[Step 4] Skipping brands in exploded dataset (lookup separately)...")
    
    unified_df['brands'] = ''  # Empty - brands looked up via separate query
    
    # =========================================================================
    # Add salt forms
    # =========================================================================
    if verbose:
        print("\n[Step 5] Adding salt forms...")
    
    try:
        salts_df = con.execute("""
            SELECT 
                drugbank_id,
                STRING_AGG(DISTINCT UPPER(TRIM(salt_names)), '|') as salt_forms
            FROM drugbank_generics
            WHERE drugbank_id IS NOT NULL AND salt_names IS NOT NULL AND salt_names != ''
            GROUP BY drugbank_id
        """).fetchdf()
        
        unified_df = unified_df.merge(salts_df, on='drugbank_id', how='left')
        unified_df['salt_forms'] = unified_df['salt_forms'].fillna('')
        
        if verbose:
            print(f"  - Salt forms for {len(salts_df):,} generics")
    except:
        unified_df['salt_forms'] = ''
    
    # =========================================================================
    # Note: Mixture components NOT stored in exploded dataset to avoid bloat
    # =========================================================================
    if verbose:
        print("\n[Step 6] Skipping mixture_components in exploded dataset...")
    
    unified_df['mixture_components'] = ''  # Empty - lookup separately if needed
    
    # =========================================================================
    # Add WHO ATC entries not in DrugBank
    # =========================================================================
    if verbose:
        print("\n[Step 7] Adding WHO-only entries...")
    
    try:
        existing_generics = set(unified_df['generic_name'].str.upper().unique())
        
        who_df = con.execute("""
            SELECT DISTINCT
                NULL as drugbank_id,
                atc_code,
                UPPER(TRIM(atc_name)) as generic_name,
                '' as form,
                '' as route,
                '' as doses,
                '' as synonyms,
                '' as brands,
                '' as salt_forms,
                '' as mixture_components
            FROM who_atc
            WHERE atc_name IS NOT NULL AND atc_name != ''
        """).fetchdf()
        
        who_new = who_df[~who_df['generic_name'].isin(existing_generics)]
        
        if len(who_new) > 0:
            unified_df = pd.concat([unified_df, who_new], ignore_index=True)
        
        if verbose:
            print(f"  - Added {len(who_new):,} WHO-only entries")
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not add WHO entries: {e}")
    
    # =========================================================================
    # Final cleanup
    # =========================================================================
    if verbose:
        print("\n[Step 8] Final cleanup...")
    
    # Add sources column
    def get_sources(row):
        sources = []
        if row.get('drugbank_id'):
            sources.append('drugbank')
        if not row.get('drugbank_id') and row.get('atc_code'):
            sources.append('who')
        return '|'.join(sources) if sources else 'drugbank'
    
    unified_df['sources'] = unified_df.apply(get_sources, axis=1)
    
    # Ensure all columns exist and are in order
    final_columns = [
        'generic_name', 'drugbank_id', 'atc_code', 'form', 'route',
        'synonyms', 'brands', 'salt_forms', 'doses', 'mixture_components', 'sources'
    ]
    
    for col in final_columns:
        if col not in unified_df.columns:
            unified_df[col] = ''
    
    unified_df = unified_df[final_columns]
    
    # Fill NaN with empty strings
    unified_df = unified_df.fillna('')
    
    # Remove completely empty rows
    unified_df = unified_df[unified_df['generic_name'] != '']
    
    # Deduplicate
    unified_df = unified_df.drop_duplicates()
    
    if verbose:
        print(f"  - Final rows: {len(unified_df):,}")
    
    # =========================================================================
    # Save
    # =========================================================================
    output_path = outputs_dir / "unified_drug_reference.parquet"
    unified_df.to_parquet(output_path, index=False)
    unified_df.to_csv(outputs_dir / "unified_drug_reference.csv", index=False)
    
    if verbose:
        print(f"\n✓ Saved: {output_path}")
        print(f"  Columns: {unified_df.columns.tolist()}")
        print(f"  Unique generics: {unified_df['generic_name'].nunique():,}")
        print(f"  With DrugBank ID: {(unified_df['drugbank_id'] != '').sum():,}")
        print(f"  With ATC code: {(unified_df['atc_code'] != '').sum():,}")
    
    con.close()
    return output_path


if __name__ == "__main__":
    build_unified_reference()
