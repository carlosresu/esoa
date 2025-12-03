#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build unified drug reference tables - LEAN version.

Tables produced (no explosions except valid form×route×dose combos):
    unified_generics.csv     - drugbank_id → generic_name (lean, one per drug)
    unified_synonyms.csv     - drugbank_id → synonyms (pipe-separated)
    unified_atc.csv          - drugbank_id → atc_code (one row per valid combo)
    unified_dosages.csv      - drugbank_id × form × route × dose (valid combos only)
    unified_brands.csv       - brand_name → generic_name, drugbank_id
    unified_salts.csv        - drugbank_id → salt forms
    unified_mixtures.csv     - mixture components with component_key

Usage:
    python -m pipelines.drugs.scripts.build_unified_reference_v2
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from .unified_constants import CANONICAL_GENERICS, CANONICAL_ATC_MAPPINGS
from .tokenizer import extract_drug_details


ALL_DETAILS_COLS = [
    "salt_details", "brand_details", "indication_details", "alias_details",
    "type_details", "release_details", "form_details",
]

def _add_details_columns(df: pd.DataFrame, name_col: str = "generic_name") -> pd.DataFrame:
    """
    Add _details columns by parsing the specified name column.
    Columns: salt_details, brand_details, indication_details, alias_details,
             type_details, release_details, form_details
    """
    if name_col not in df.columns or df.empty:
        for col in ALL_DETAILS_COLS:
            df[col] = None
        return df
    
    details = df[name_col].fillna("").apply(extract_drug_details)
    for col in ALL_DETAILS_COLS:
        df[col] = details.apply(lambda d, c=col: d.get(c))
    return df

PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def _save_table(df: pd.DataFrame, outputs_dir: Path, name: str, verbose: bool = True):
    """Save dataframe as CSV (canonical format)."""
    csv_path = outputs_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"  ✓ {name}: {len(df):,} rows")
    return csv_path


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
    
    # Helper: load CSV files (canonical format)
    def load_table(table_name: str, basename: str):
        csv_path = inputs_dir / f"{basename}.csv"
        if csv_path.exists():
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_path}')")
            return csv_path
        return None
    
    # Lean DrugBank tables
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
    
    # Lookup tables
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
    who_files = sorted(glob.glob(str(inputs_dir / "who_atc_*.csv")))
    if who_files:
        who_path = who_files[-1]
        con.execute(f"CREATE TABLE who_atc AS SELECT * FROM read_csv_auto('{who_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM who_atc").fetchone()[0]
            print(f"  - who_atc: {count:,} rows")
    
    # FDA brands
    fda_files = sorted(glob.glob(str(inputs_dir / "fda_drug_*.csv")))
    if fda_files:
        fda_path = fda_files[-1]
        con.execute(f"CREATE TABLE fda_brands AS SELECT * FROM read_csv_auto('{fda_path}')")
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM fda_brands").fetchone()[0]
            print(f"  - fda_brands: {count:,} rows")
    
    # PNF (Philippine National Formulary) - prepared data
    pnf_path = inputs_dir / "pnf_prepared.csv"
    pnf_loaded = False
    if pnf_path.exists():
        con.execute(f"CREATE TABLE pnf AS SELECT * FROM read_csv_auto('{pnf_path}')")
        pnf_loaded = True
        if verbose:
            count = con.execute("SELECT COUNT(*) FROM pnf").fetchone()[0]
            print(f"  - pnf: {count:,} rows")

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
    
    # Add PNF generics (Philippine National Formulary)
    if pnf_loaded:
        try:
            existing = set(generics_df['generic_name'].str.upper().unique())
            pnf_df = con.execute("""
                SELECT DISTINCT
                    NULL as drugbank_id,
                    UPPER(TRIM(generic_normalized)) as generic_name,
                    LOWER(REGEXP_REPLACE(generic_normalized, '[^a-zA-Z0-9 ]', '', 'g')) as name_key,
                    'pnf' as source
                FROM pnf
                WHERE generic_normalized IS NOT NULL AND generic_normalized != ''
            """).fetchdf()
            pnf_new = pnf_df[~pnf_df['generic_name'].isin(existing)]
            generics_df = pd.concat([generics_df, pnf_new], ignore_index=True)
            if verbose:
                print(f"    + Added {len(pnf_new)} PNF-only generics")
        except Exception as e:
            if verbose:
                print(f"    ! PNF generics error: {e}")
    
    generics_df = generics_df.drop_duplicates(subset=['generic_name'])
    
    # Add canonical drug name aliases from unified_constants.py
    # These are combination drugs or special products that need explicit handling
    canonical_gen_df = pd.DataFrame(CANONICAL_GENERICS)
    # Add name_key column if not present
    if 'name_key' not in canonical_gen_df.columns:
        canonical_gen_df['name_key'] = canonical_gen_df['generic_name'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)
    
    # For canonicals with DrugBank IDs, UPDATE existing entries rather than just concat
    # This ensures we preserve DrugBank IDs even if PNF/WHO added the generic first
    canonical_with_db = canonical_gen_df[canonical_gen_df['drugbank_id'].notna()]
    canonical_no_db = canonical_gen_df[canonical_gen_df['drugbank_id'].isna()]
    
    # Update existing entries with DrugBank IDs from canonicals
    for _, row in canonical_with_db.iterrows():
        mask = generics_df['generic_name'] == row['generic_name']
        if mask.any():
            generics_df.loc[mask, 'drugbank_id'] = row['drugbank_id']
            generics_df.loc[mask, 'source'] = 'canonical'
        else:
            # Add new entry
            generics_df = pd.concat([generics_df, pd.DataFrame([row])], ignore_index=True)
    
    # Add canonicals without DrugBank ID only if not already present
    existing = set(generics_df['generic_name'].str.upper().unique())
    canonical_no_db_new = canonical_no_db[~canonical_no_db['generic_name'].str.upper().isin(existing)]
    generics_df = pd.concat([generics_df, canonical_no_db_new], ignore_index=True)
    
    generics_df = generics_df.drop_duplicates(subset=['generic_name'])
    
    # Also add raw PNF molecule names (not just normalized) for better matching
    if pnf_loaded:
        try:
            existing = set(generics_df['generic_name'].str.upper().unique())
            pnf_raw_df = con.execute("""
                SELECT DISTINCT
                    NULL as drugbank_id,
                    UPPER(TRIM(raw_molecule)) as generic_name,
                    LOWER(REGEXP_REPLACE(raw_molecule, '[^a-zA-Z0-9 ]', '', 'g')) as name_key,
                    'pnf_raw' as source
                FROM pnf
                WHERE raw_molecule IS NOT NULL AND raw_molecule != ''
            """).fetchdf()
            pnf_raw_new = pnf_raw_df[~pnf_raw_df['generic_name'].isin(existing)]
            generics_df = pd.concat([generics_df, pnf_raw_new], ignore_index=True)
            if verbose:
                print(f"    + Added {len(pnf_raw_new)} PNF raw molecule names")
        except Exception as e:
            if verbose:
                print(f"    ! PNF raw error: {e}")
    
    # Add _details columns by parsing generic names
    # This preserves information like salt forms, brand names, indications, and aliases
    generics_df = _add_details_columns(generics_df, "generic_name")
    
    # Also try to pull PNF details if available (more precise since already parsed)
    if pnf_loaded:
        try:
            # Build column list for SQL query
            details_cols_sql = ", ".join(ALL_DETAILS_COLS)
            pnf_details = con.execute(f"""
                SELECT DISTINCT
                    UPPER(TRIM(generic_normalized)) as generic_name,
                    {details_cols_sql}
                FROM pnf
                WHERE generic_normalized IS NOT NULL
            """).fetchdf()
            # Merge PNF details into generics_df (prefer PNF details when available)
            for _, prow in pnf_details.iterrows():
                mask = generics_df['generic_name'] == prow['generic_name']
                if mask.any():
                    for col in ALL_DETAILS_COLS:
                        if col in prow.index and pd.notna(prow[col]) and prow[col]:
                            generics_df.loc[mask, col] = prow[col]
        except Exception:
            pass  # PNF may not have details columns yet
    
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
    
    # Add PNF ATC entries (Philippine National Formulary)
    if pnf_loaded:
        try:
            pnf_atc_df = con.execute("""
                SELECT DISTINCT
                    NULL as drugbank_id,
                    UPPER(TRIM(generic_normalized)) as generic_name,
                    TRIM(atc_code) as atc_code
                FROM pnf
                WHERE atc_code IS NOT NULL AND atc_code != ''
                  AND generic_normalized IS NOT NULL AND generic_normalized != ''
            """).fetchdf()
            before_count = len(atc_map_df)
            atc_map_df = pd.concat([atc_map_df, pnf_atc_df], ignore_index=True)
            if verbose:
                print(f"    + Added {len(pnf_atc_df)} PNF ATC mappings")
        except Exception as e:
            if verbose:
                print(f"    ! PNF ATC error: {e}")
    
    atc_map_df = atc_map_df.drop_duplicates()
    
    # Add canonical ATC mappings from unified_constants.py
    # These map combination names to their ATC codes
    canonical_df = pd.DataFrame(CANONICAL_ATC_MAPPINGS)
    atc_map_df = pd.concat([atc_map_df, canonical_df], ignore_index=True).drop_duplicates()
    
    # =========================================================================
    # TABLE 4: unified_dosages - drugbank_id × form × route × dose (VALID combos)
    # =========================================================================
    if verbose:
        print("\n[Step 5] Building unified_dosages (valid combos only)...")
    
    # DrugBank dosages (canonical)
    dosages_df = con.execute("""
        SELECT DISTINCT
            d.drugbank_id,
            UPPER(TRIM(g.name)) as generic_name,
            UPPER(TRIM(d.form)) as form,
            UPPER(TRIM(d.route)) as route,
            UPPER(TRIM(d.strength)) as dose,
            'drugbank' as source
        FROM dosages d
        LEFT JOIN generics g ON d.drugbank_id = g.drugbank_id
        WHERE d.drugbank_id IS NOT NULL
    """).fetchdf()
    
    if verbose:
        print(f"    - DrugBank: {len(dosages_df):,} combos")
    
    # Add PNF form/route/dose combos (Philippine-context valid combinations)
    # These are critical for matching Philippine hospital data (ESOA) to Philippine formulary (Annex F)
    if pnf_loaded:
        try:
            pnf_dosages_df = con.execute("""
                SELECT DISTINCT
                    NULL as drugbank_id,
                    UPPER(TRIM(generic_normalized)) as generic_name,
                    UPPER(TRIM(form_token)) as form,
                    UPPER(TRIM(route_allowed)) as route,
                    CASE 
                        WHEN strength_mg IS NOT NULL THEN CAST(CAST(strength_mg AS INTEGER) AS VARCHAR) || ' MG'
                        WHEN strength IS NOT NULL AND unit IS NOT NULL THEN CAST(CAST(strength AS INTEGER) AS VARCHAR) || ' ' || UPPER(unit)
                        ELSE NULL
                    END as dose,
                    'pnf' as source
                FROM pnf
                WHERE generic_normalized IS NOT NULL AND generic_normalized != ''
            """).fetchdf()
            dosages_df = pd.concat([dosages_df, pnf_dosages_df], ignore_index=True)
            if verbose:
                print(f"    + PNF: {len(pnf_dosages_df):,} Philippine-context combos")
        except Exception as e:
            if verbose:
                print(f"    ! PNF dosages error: {e}")
    
    dosages_df = dosages_df.fillna('').drop_duplicates()
    if verbose:
        print(f"    - Total: {len(dosages_df):,} valid combos")
    
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
