#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the ONE source of truth: generics_master.parquet

This consolidates ALL drug reference data into a single dataset with one row per
generic drug. All related data (synonyms, brands, forms, etc.) are stored as
pipe-delimited columns.

Sources loaded:
- DrugBank generics (drugbank_generics_master.csv)
- DrugBank mixtures (drugbank_mixtures_master.csv)
- DrugBank salts (drugbank_salts_master.csv)
- DrugBank brands (drugbank_brands_master.csv)
- WHO ATC (who_atc_*.parquet)
- FDA brands (fda_drug_*.parquet)
- PNF lexicon (pnf_prepared.parquet)
- Regional synonyms (regional_synonyms.csv)

Output:
- generics_master.parquet - ONE file with ALL data, one row per generic

Schema:
- generic_name: str - canonical name for matching
- drugbank_id: str - DrugBank identifier (may be null for WHO-only entries)
- atc_codes: str - pipe-delimited ATC codes
- synonyms: str - pipe-delimited synonyms (all alternate names)
- brands: str - pipe-delimited brand names
- salt_forms: str - pipe-delimited salt suffixes
- doses: str - pipe-delimited known doses
- forms: str - pipe-delimited dosage forms
- routes: str - pipe-delimited administration routes
- mixture_of: str - pipe-delimited component DrugBank IDs (for combinations)
- sources: str - pipe-delimited data sources (drugbank|who|fda|pnf)
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def pipe_join(values) -> str:
    """Join non-null unique values with pipe."""
    if values is None:
        return ""
    unique = sorted(set(str(v).strip() for v in values if pd.notna(v) and str(v).strip()))
    return "|".join(unique)


def load_latest_file(pattern: str) -> Optional[pd.DataFrame]:
    """Load the latest file matching pattern."""
    files = sorted(glob.glob(str(INPUTS_DIR / pattern)))
    if not files:
        return None
    path = Path(files[-1])
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_generics_master(
    inputs_dir: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Path:
    """
    Build the consolidated generics_master.parquet.
    
    Returns path to output file.
    """
    inputs_dir = Path(inputs_dir or INPUTS_DIR)
    outputs_dir = Path(outputs_dir or OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("Building generics_master.parquet")
        print("=" * 60)
    
    # =========================================================================
    # Step 1: Load DrugBank generics as base
    # =========================================================================
    if verbose:
        print("\n[Step 1] Loading DrugBank generics...")
    
    drugbank_path = inputs_dir / "drugbank_generics_master.csv"
    if not drugbank_path.exists():
        raise FileNotFoundError(f"DrugBank generics not found: {drugbank_path}")
    
    db = pd.read_csv(drugbank_path)
    if verbose:
        print(f"  - Loaded: {len(db):,} rows")
    
    # Aggregate by drugbank_id
    generics = db.groupby('drugbank_id').agg({
        'canonical_generic_name': 'first',
        'lexeme': lambda x: pipe_join(x),
        'atc_code': lambda x: pipe_join(x.dropna()),
        'dose_norm': lambda x: pipe_join(x.dropna()),
        'form_norm': lambda x: pipe_join(x.dropna()),
        'route_norm': lambda x: pipe_join(x.dropna()),
        'salt_names': lambda x: pipe_join(x.dropna()),
    }).reset_index()
    
    generics = generics.rename(columns={
        'canonical_generic_name': 'generic_name',
        'lexeme': 'synonyms',
        'atc_code': 'atc_codes',
        'dose_norm': 'doses',
        'form_norm': 'forms',
        'route_norm': 'routes',
        'salt_names': 'salt_forms',
    })
    generics['sources'] = 'drugbank'
    
    if verbose:
        print(f"  - Aggregated to: {len(generics):,} unique generics")
    
    # =========================================================================
    # Step 2: Add regional synonyms
    # =========================================================================
    if verbose:
        print("\n[Step 2] Loading regional synonyms...")
    
    regional_path = inputs_dir / "regional_synonyms.csv"
    if regional_path.exists():
        regional = pd.read_csv(regional_path)
        
        # Build synonym mapping
        regional_map = {}
        for _, row in regional.iterrows():
            canonical = str(row['canonical_name']).upper().strip()
            variant = str(row['regional_variant']).upper().strip()
            if canonical not in regional_map:
                regional_map[canonical] = set()
            regional_map[canonical].add(variant)
            # Also add reverse
            if variant not in regional_map:
                regional_map[variant] = set()
            regional_map[variant].add(canonical)
        
        # Add to synonyms column
        def add_regional_synonyms(row):
            name = str(row['generic_name']).upper().strip()
            current = set(row['synonyms'].split('|')) if row['synonyms'] else set()
            regional_syns = regional_map.get(name, set())
            current.update(regional_syns)
            return pipe_join(current)
        
        generics['synonyms'] = generics.apply(add_regional_synonyms, axis=1)
        if verbose:
            print(f"  - Added {len(regional):,} regional synonyms")
    else:
        if verbose:
            print("  - No regional_synonyms.csv found")
    
    # =========================================================================
    # Step 3: Add DrugBank brands
    # =========================================================================
    if verbose:
        print("\n[Step 3] Loading DrugBank brands...")
    
    brands_path = inputs_dir / "drugbank_brands_master.csv"
    if brands_path.exists():
        brands_df = pd.read_csv(brands_path)
        
        # Aggregate brands by drugbank_id
        brand_agg = brands_df.groupby('drugbank_id').agg({
            'brand_name': lambda x: pipe_join(x),
        }).reset_index()
        brand_agg = brand_agg.rename(columns={'brand_name': 'brands'})
        
        # Merge
        generics = generics.merge(brand_agg, on='drugbank_id', how='left')
        generics['brands'] = generics['brands'].fillna('')
        
        if verbose:
            print(f"  - Added brands for {len(brand_agg):,} generics")
    else:
        generics['brands'] = ''
        if verbose:
            print("  - No drugbank_brands_master.csv found")
    
    # =========================================================================
    # Step 4: Add FDA brands
    # =========================================================================
    if verbose:
        print("\n[Step 4] Loading FDA brands...")
    
    fda_df = load_latest_file("fda_drug_*.parquet")
    if fda_df is None:
        fda_df = load_latest_file("fda_drug_*.csv")
    
    if fda_df is not None:
        # FDA has brand_name and generic_name columns
        fda_brands = fda_df.groupby('generic_name').agg({
            'brand_name': lambda x: pipe_join(x),
        }).reset_index()
        fda_brands['generic_name'] = fda_brands['generic_name'].str.upper().str.strip()
        fda_brands = fda_brands.rename(columns={'brand_name': 'fda_brands'})
        
        # Merge by generic name match
        generics['generic_name_upper'] = generics['generic_name'].str.upper().str.strip()
        generics = generics.merge(
            fda_brands, 
            left_on='generic_name_upper', 
            right_on='generic_name',
            how='left',
            suffixes=('', '_fda')
        )
        
        # Combine FDA brands with existing brands
        def combine_brands(row):
            existing = set(row['brands'].split('|')) if row['brands'] else set()
            fda = set(row.get('fda_brands', '').split('|')) if pd.notna(row.get('fda_brands')) else set()
            existing.update(fda)
            existing.discard('')
            return pipe_join(existing)
        
        generics['brands'] = generics.apply(combine_brands, axis=1)
        generics = generics.drop(columns=['generic_name_upper', 'generic_name_fda', 'fda_brands'], errors='ignore')
        
        if verbose:
            print(f"  - Added FDA brands from {len(fda_df):,} rows")
    else:
        if verbose:
            print("  - No FDA drug file found")
    
    # =========================================================================
    # Step 5: Add WHO ATC entries not in DrugBank
    # =========================================================================
    if verbose:
        print("\n[Step 5] Loading WHO ATC...")
    
    who_df = load_latest_file("who_atc_*.parquet")
    if who_df is None:
        who_df = load_latest_file("who_atc_*.csv")
    
    if who_df is not None:
        # Get WHO entries not already in generics
        existing_names = set(generics['generic_name'].str.upper().str.strip())
        
        who_names = who_df['atc_name'].str.upper().str.strip() if 'atc_name' in who_df.columns else who_df['generic_name'].str.upper().str.strip()
        who_new = who_df[~who_names.isin(existing_names)].copy()
        
        if len(who_new) > 0:
            who_new = who_new.groupby(who_names[~who_names.isin(existing_names)]).agg({
                'atc_code': lambda x: pipe_join(x) if 'atc_code' in who_df.columns else '',
            }).reset_index()
            who_new.columns = ['generic_name', 'atc_codes']
            who_new['drugbank_id'] = None
            who_new['synonyms'] = ''
            who_new['brands'] = ''
            who_new['salt_forms'] = ''
            who_new['doses'] = ''
            who_new['forms'] = ''
            who_new['routes'] = ''
            who_new['sources'] = 'who'
            
            generics = pd.concat([generics, who_new], ignore_index=True)
        
        if verbose:
            print(f"  - Added {len(who_new):,} WHO-only entries")
    else:
        if verbose:
            print("  - No WHO ATC file found")
    
    # =========================================================================
    # Step 6: Add mixture information
    # =========================================================================
    if verbose:
        print("\n[Step 6] Loading mixtures...")
    
    mixtures_path = inputs_dir / "drugbank_mixtures_master.csv"
    if mixtures_path.exists():
        mixtures_df = pd.read_csv(mixtures_path)
        
        # Aggregate mixture components by drugbank_id
        if 'drugbank_id' in mixtures_df.columns and 'ingredient_drugbank_ids' in mixtures_df.columns:
            mix_agg = mixtures_df.groupby('drugbank_id').agg({
                'ingredient_drugbank_ids': lambda x: pipe_join(x),
            }).reset_index()
            mix_agg = mix_agg.rename(columns={'ingredient_drugbank_ids': 'mixture_of'})
            
            generics = generics.merge(mix_agg, on='drugbank_id', how='left')
            generics['mixture_of'] = generics['mixture_of'].fillna('')
            
            if verbose:
                print(f"  - Added mixture info for {len(mix_agg):,} entries")
        else:
            generics['mixture_of'] = ''
    else:
        generics['mixture_of'] = ''
        if verbose:
            print("  - No drugbank_mixtures_master.csv found")
    
    # =========================================================================
    # Step 7: Final cleanup
    # =========================================================================
    if verbose:
        print("\n[Step 7] Final cleanup...")
    
    # Ensure all columns exist
    final_columns = [
        'generic_name', 'drugbank_id', 'atc_codes', 'synonyms', 'brands',
        'salt_forms', 'doses', 'forms', 'routes', 'mixture_of', 'sources'
    ]
    
    for col in final_columns:
        if col not in generics.columns:
            generics[col] = ''
    
    # Reorder columns
    generics = generics[final_columns]
    
    # Normalize generic_name
    generics['generic_name'] = generics['generic_name'].str.upper().str.strip()
    
    # Deduplicate by generic_name (keep first = DrugBank over WHO)
    generics = generics.drop_duplicates(subset=['generic_name'], keep='first')
    
    # Sort
    generics = generics.sort_values('generic_name').reset_index(drop=True)
    
    if verbose:
        print(f"  - Final dataset: {len(generics):,} generics")
    
    # =========================================================================
    # Save
    # =========================================================================
    output_path = outputs_dir / "generics_master.parquet"
    generics.to_parquet(output_path, index=False)
    generics.to_csv(outputs_dir / "generics_master.csv", index=False)
    
    if verbose:
        print(f"\n✓ Saved: {output_path}")
        print(f"  Columns: {generics.columns.tolist()}")
    
    return output_path


if __name__ == "__main__":
    build_generics_master()
