"""
Pipeline runner functions for drug tagging.

These functions are called by the run_* scripts in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .io_utils import reorder_columns_after, write_csv_and_parquet
from .spinner import run_with_spinner
from .tagger import UnifiedTagger
from .unified_constants import (
    GARBAGE_TOKENS,
    ALL_DRUG_SYNONYMS,
    DRUGBANK_COMPONENT_SYNONYMS,
)


# Default paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_DIR / "raw" / "drugs"
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"

PIPELINE_RAW_DIR = Path(os.environ.get("PIPELINE_RAW_DIR", RAW_DIR))
PIPELINE_INPUTS_DIR = Path(os.environ.get("PIPELINE_INPUTS_DIR", INPUTS_DIR))
PIPELINE_OUTPUTS_DIR = Path(os.environ.get("PIPELINE_OUTPUTS_DIR", OUTPUTS_DIR))


def run_annex_f_tagging(
    annex_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Run Annex F tagging (Part 2).
    
    Returns dict with results summary.
    """
    if annex_path is None:
        annex_path = PIPELINE_RAW_DIR / "annex_f.csv"
    if output_path is None:
        output_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.csv"
    
    # Load Annex F
    if not annex_path.exists():
        raise FileNotFoundError(f"Annex F not found: {annex_path}")
    
    annex_df = pd.read_csv(annex_path)
    
    # Initialize and load tagger
    tagger = UnifiedTagger(
        outputs_dir=PIPELINE_OUTPUTS_DIR,
        inputs_dir=PIPELINE_INPUTS_DIR,
        verbose=False,
    )
    tagger.load()
    
    # Tag descriptions
    results_df = run_with_spinner(
        f"Tag {len(annex_df):,} Annex F entries",
        lambda: tagger.tag_descriptions(
            annex_df,
            text_column="Drug Description",
            id_column="Drug Code",
        )
    )
    
    # Merge results
    annex_df["row_idx"] = range(len(annex_df))
    merge_cols = ["row_idx", "atc_code", "drugbank_id", "generic_name", "reference_text", 
                  "match_score", "match_reason", "sources",
                  "dose", "form", "route", "type_details", "release_details", "form_details",
                  "salt_details", "brand_details", "indication_details", "alias_details"]
    # Only include columns that exist in results_df
    merge_cols = [c for c in merge_cols if c in results_df.columns]
    merged = annex_df.merge(
        results_df[merge_cols],
        on="row_idx",
        how="left",
    ).drop(columns=["row_idx"])
    
    # Rename columns
    merged = merged.rename(columns={
        "generic_name": "matched_generic_name",
        "reference_text": "matched_reference_text",
        "sources": "matched_source",
    })
    
    # Reorder columns
    merged = reorder_columns_after(merged, "Drug Description", "matched_reference_text")
    
    # Write outputs
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_with_spinner("Write outputs", lambda: write_csv_and_parquet(merged, output_path))
    
    tagger.close()
    
    # Summary
    total = len(merged)
    matched_atc = merged["atc_code"].notna().sum()
    matched_drugbank = merged["drugbank_id"].notna().sum()
    reason_counts = {str(reason): int(count) for reason, count in merged["match_reason"].value_counts().items() if pd.notna(reason)}

    results = {
        "total": total,
        "matched_atc": matched_atc,
        "matched_atc_pct": 100 * matched_atc / total if total else 0,
        "matched_drugbank": matched_drugbank,
        "matched_drugbank_pct": 100 * matched_drugbank / total if total else 0,
        "output_path": output_path,
        "reason_counts": reason_counts,
    }
    
    # Log metrics
    log_metrics("annex_f", {
        "total": total,
        "matched_atc": matched_atc,
        "matched_atc_pct": round(results["matched_atc_pct"], 2),
        "matched_drugbank": matched_drugbank,
        "matched_drugbank_pct": round(results["matched_drugbank_pct"], 2),
    })
    
    return results


def run_esoa_tagging(
    esoa_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict:
    """
    Run ESOA tagging (Part 3).
    
    Returns dict with results summary.
    """
    if esoa_path is None:
        esoa_path = PIPELINE_INPUTS_DIR / "esoa_combined.csv"
        if not esoa_path.exists():
            esoa_path = PIPELINE_INPUTS_DIR / "esoa_prepared.csv"
    if output_path is None:
        output_path = PIPELINE_OUTPUTS_DIR / "esoa_with_atc.csv"
    
    # Load ESOA
    if not esoa_path.exists():
        raise FileNotFoundError(f"ESOA not found: {esoa_path}")
    
    esoa_df = pd.read_csv(esoa_path)
    
    # Determine text column
    text_column = None
    for col in ["raw_text", "ITEM_DESCRIPTION", "DESCRIPTION", "Drug Description", "description"]:
        if col in esoa_df.columns:
            text_column = col
            break
    
    if not text_column:
        raise ValueError(f"No text column found. Columns: {list(esoa_df.columns)}")
    
    # Initialize and load tagger
    tagger = UnifiedTagger(
        outputs_dir=PIPELINE_OUTPUTS_DIR,
        inputs_dir=PIPELINE_INPUTS_DIR,
        verbose=False,
    )
    tagger.load()
    
    # Use tag_batch with deduplication for performance
    total = len(esoa_df)
    results_df = tagger.tag_batch(
        esoa_df,
        text_column=text_column,
        chunk_size=10000,
        show_progress=show_progress,
        deduplicate=True,
    )
    
    # Map results back to original rows by text
    # results_df has 'input_text' column with the original text
    results_df = results_df.rename(columns={"input_text": "_tag_text"})
    esoa_df["_tag_text"] = esoa_df[text_column].fillna("").astype(str)
    
    merge_cols = ["_tag_text", "atc_code", "drugbank_id", "generic_name", "reference_text", 
                  "match_score", "match_reason", "sources",
                  "dose", "form", "route", "type_details", "release_details", "form_details",
                  "salt_details", "brand_details", "indication_details", "alias_details"]
    # Only include columns that exist in results_df
    merge_cols = [c for c in merge_cols if c in results_df.columns]
    merged = esoa_df.merge(
        results_df[merge_cols],
        on="_tag_text",
        how="left",
    ).drop(columns=["_tag_text"])
    
    # Rename columns
    merged = merged.rename(columns={
        "atc_code": "atc_code_final",
        "drugbank_id": "drugbank_id_final",
        "generic_name": "generic_final",
        "reference_text": "matched_reference_text",
        "sources": "reference_source",
    })
    
    # Reorder columns
    merged = reorder_columns_after(merged, text_column, "matched_reference_text")
    
    # Write outputs
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_with_spinner("Write outputs", lambda: write_csv_and_parquet(merged, output_path))
    
    tagger.close()
    
    # Summary
    matched_atc = merged["atc_code_final"].notna() & (merged["atc_code_final"] != "")
    matched_atc_count = matched_atc.sum()
    matched_drugbank = merged["drugbank_id_final"].notna() & (merged["drugbank_id_final"] != "")
    matched_drugbank_count = matched_drugbank.sum()
    
    results = {
        "total": total,
        "matched_atc": matched_atc_count,
        "matched_atc_pct": 100 * matched_atc_count / total if total else 0,
        "matched_drugbank": matched_drugbank_count,
        "matched_drugbank_pct": 100 * matched_drugbank_count / total if total else 0,
        "output_path": output_path,
    }
    
    reason_counts = {str(reason): int(count) for reason, count in merged["match_reason"].value_counts().items() if pd.notna(reason)}

    if verbose:
        print(f"\nESOA tagging complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Has ATC: {matched_atc_count:,} ({results['matched_atc_pct']:.1f}%)")
        print(f"  Has DrugBank ID: {matched_drugbank_count:,} ({results['matched_drugbank_pct']:.1f}%)")
        print("\nMatch reasons:")
        for reason, count in list(reason_counts.items())[:10]:
            pct = 100 * count / total if total else 0
            print(f"  {reason}: {count:,} ({pct:.1f}%)")
    
    # Log metrics
    log_metrics("esoa", {
        "total": total,
        "matched_atc": matched_atc_count,
        "matched_atc_pct": round(results["matched_atc_pct"], 2),
        "matched_drugbank": matched_drugbank_count,
        "matched_drugbank_pct": round(results["matched_drugbank_pct"], 2),
    })
    
    return results


def run_esoa_to_drug_code(
    esoa_path: Optional[Path] = None,
    annex_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Run ESOA to Drug Code matching (Part 4).
    
    Matches ESOA items to Annex F drug codes using EXACT matching:
    - Generic name must match exactly
    - ATC code must match (drug_code is unique per ATC)
    
    Returns dict with results summary.
    """
    if esoa_path is None:
        esoa_path = PIPELINE_OUTPUTS_DIR / "esoa_with_atc.csv"
        if not esoa_path.exists():
            esoa_path = PIPELINE_OUTPUTS_DIR / "esoa_with_atc.parquet"
    if annex_path is None:
        annex_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.csv"
        if not annex_path.exists():
            annex_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.parquet"
    if output_path is None:
        output_path = PIPELINE_OUTPUTS_DIR / "esoa_with_drug_code.csv"
    
    # Load data
    if not esoa_path.exists():
        raise FileNotFoundError(f"ESOA with ATC not found: {esoa_path}")
    if not annex_path.exists():
        raise FileNotFoundError(f"Annex F with ATC not found: {annex_path}")
    
    if str(esoa_path).endswith('.parquet'):
        esoa_df = run_with_spinner("Load ESOA", lambda: pd.read_parquet(esoa_path))
    else:
        esoa_df = run_with_spinner("Load ESOA", lambda: pd.read_csv(esoa_path))
    
    if str(annex_path).endswith('.parquet'):
        annex_df = run_with_spinner("Load Annex F", lambda: pd.read_parquet(annex_path))
    else:
        annex_df = run_with_spinner("Load Annex F", lambda: pd.read_csv(annex_path))
    
    if verbose:
        print(f"  ESOA rows: {len(esoa_df):,}")
        print(f"  Annex F rows: {len(annex_df):,}")
    
    # Build Annex F lookup index by generic name
    def normalize_for_match(s):
        if pd.isna(s):
            return ""
        return str(s).upper().strip()
    
    # Build synonym mappings from generics_master and merge with static constants
    generics_master_path = PIPELINE_OUTPUTS_DIR / "generics_master.csv"
    if not generics_master_path.exists():
        generics_master_path = PIPELINE_OUTPUTS_DIR / "generics_master.parquet"
    all_synonyms = dict(ALL_DRUG_SYNONYMS)  # Start with static synonyms from unified_constants
    
    if generics_master_path.exists():
        if str(generics_master_path).endswith('.parquet'):
            gm = pd.read_parquet(generics_master_path)
        else:
            gm = pd.read_csv(generics_master_path)
        for _, row in gm.iterrows():
            generic = str(row['generic_name']).upper().strip()
            synonyms_str = row.get('synonyms', '')
            if synonyms_str:
                for syn in str(synonyms_str).split('|'):
                    syn = syn.upper().strip()
                    if syn and syn != generic:
                        # Bidirectional mapping
                        all_synonyms[syn] = generic
                        all_synonyms[generic] = syn
        if verbose:
            print(f"  Loaded synonyms: {len(ALL_DRUG_SYNONYMS)} static + {len(all_synonyms) - len(ALL_DRUG_SYNONYMS)} from generics_master")
    
    def get_all_name_variants(name):
        """Get all possible name variants for matching."""
        variants = {name}
        if name in all_synonyms:
            variants.add(all_synonyms[name])
        # Also check if name appears as a value (reverse lookup)
        for syn, canonical in all_synonyms.items():
            if canonical == name:
                variants.add(syn)
        return variants
    
    # Use constants from unified_constants.py (imported at module level)
    # GARBAGE_TOKENS, ALL_DRUG_SYNONYMS, DRUGBANK_COMPONENT_SYNONYMS
    
    import re
    
    def extract_dose_mg(text):
        """Extract dose in mg from text. Returns None if not found.
        
        For percentage-based drugs (e.g., 5% DEXTROSE, 0.9% SODIUM CHLORIDE),
        returns the percentage as a string (e.g., "5%", "0.9%") for exact matching.
        """
        if not text:
            return None
        text = str(text).upper()
        
        # First check for percentage-based doses (IV solutions, etc.)
        # These are matched as strings, not converted to mg
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if pct_match:
            # For percentage drugs, return the percentage as a string
            return f"{pct_match.group(1)}%"
        
        # Match patterns like "600MG", "600 MG", "100MG/5ML", "200 MG/ML"
        # For concentrations like 100MG/5ML, extract 100
        patterns = [
            r'(\d+(?:\.\d+)?)\s*MG(?:/\d+\s*ML)?(?:\s|$|[^A-Z])',  # 600MG, 100MG/5ML
            r'(\d+(?:\.\d+)?)\s*MCG',  # 500MCG -> convert to mg
            r'(\d+(?:\.\d+)?)\s*G(?:\s|$|[^A-Z])',  # 1G -> convert to mg
        ]
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if match:
                val = float(match.group(1))
                if i == 1:  # MCG -> MG
                    val = val / 1000
                elif i == 2:  # G -> MG
                    val = val * 1000
                return val
        return None
    
    annex_lookup = {}  # generic_name -> list of candidates
    drugbank_lookup = {}  # drugbank_id -> list of candidates
    for _, row in annex_df.iterrows():
        drug_code = row.get("Drug Code")
        if pd.isna(drug_code):
            continue
        
        generic_raw = row.get("matched_generic_name") or row.get("generic_name") or ""
        drug_desc = row.get("Drug Description") or ""
        
        # Extract clean generics from pipe-separated string (Annex F also has garbage)
        annex_generics = []
        for part in str(generic_raw).split('|'):
            part = part.strip().upper()
            if not part or part in GARBAGE_TOKENS or len(part) <= 2:
                continue
            # Skip pure dose patterns (e.g., "500MG", "100ML")
            # But allow drug names with numbers (e.g., "GENTAMICIN C2", "VITAMIN B12")
            if re.match(r'^\d+(\.\d+)?\s*(MG|ML|MCG|G|IU|%|CC|L)$', part, re.IGNORECASE):
                continue
            if part.replace('.', '').isdigit():
                continue
            annex_generics.append(part)
        
        if not annex_generics:
            continue
        
        atc = normalize_for_match(row.get("atc_code"))
        drugbank_id = row.get("drugbank_id")
        if pd.notna(drugbank_id):
            drugbank_id = str(drugbank_id).strip()
        else:
            drugbank_id = None
        
        # Extract dose from Drug Description
        annex_dose = extract_dose_mg(drug_desc)
        
        # Extract form and route from Annex F
        annex_form = normalize_for_match(row.get("form"))
        annex_route = normalize_for_match(row.get("route"))
        
        candidate = {
            "drug_code": drug_code,
            "atc_code": atc,
            "drugbank_id": drugbank_id,
            "generic_name": annex_generics[0],  # Primary generic
            "dose_mg": annex_dose,
            "form": annex_form,
            "route": annex_route,
            "description": drug_desc,
        }
        
        # Index by each generic component and its synonyms
        for generic in annex_generics:
            if generic not in annex_lookup:
                annex_lookup[generic] = []
            annex_lookup[generic].append(candidate)
            
            # Also index by base name without parentheticals (e.g., "ASCORBIC ACID (VITAMIN C)" -> "ASCORBIC ACID")
            base_generic = re.sub(r'\s*\([^)]*\)', '', generic).strip()
            if base_generic and base_generic != generic:
                if base_generic not in annex_lookup:
                    annex_lookup[base_generic] = []
                annex_lookup[base_generic].append(candidate)
            
            # Also add synonym mappings (from unified_constants)
            if generic in ALL_DRUG_SYNONYMS:
                syn = ALL_DRUG_SYNONYMS[generic]
                if syn not in annex_lookup:
                    annex_lookup[syn] = []
                annex_lookup[syn].append(candidate)
            # Check base generic for synonyms too
            if base_generic and base_generic in ALL_DRUG_SYNONYMS:
                syn = ALL_DRUG_SYNONYMS[base_generic]
                if syn not in annex_lookup:
                    annex_lookup[syn] = []
                annex_lookup[syn].append(candidate)
        
        # Index by drugbank_id
        if drugbank_id:
            if drugbank_id not in drugbank_lookup:
                drugbank_lookup[drugbank_id] = []
            drugbank_lookup[drugbank_id].append(candidate)
    
    if verbose:
        print(f"  Annex F lookup: {len(annex_lookup):,} unique generics")
        print(f"  DrugBank lookup: {len(drugbank_lookup):,} unique drugbank_ids")
    
    def extract_clean_generics(generic_str):
        """Extract clean generic names from pipe-separated string."""
        if not generic_str:
            return []
        parts = [p.strip().upper() for p in str(generic_str).split('|')]
        # Filter out garbage and deduplicate while preserving order
        seen = set()
        clean = []
        for p in parts:
            if not p or p in GARBAGE_TOKENS or p in seen or len(p) <= 2:
                continue
            # Skip if looks like a pure dose (e.g., "500MG", "100ML", "10%")
            # But allow vitamin names like "B1", "B12", "B6"
            import re
            if re.match(r'^\d+(\.\d+)?\s*(MG|ML|MCG|G|IU|%|CC|L)$', p, re.IGNORECASE):
                continue
            # Skip pure numbers
            if p.replace('.', '').isdigit():
                continue
            seen.add(p)
            clean.append(p)
        return clean
    
    def extract_generics_from_description(desc):
        """Fallback: extract generic names from DESCRIPTION when generic_final is empty."""
        if not desc:
            return []
        desc = str(desc).upper()
        generics = []
        
        # Split on common separators
        # Handle "ALUMINUM+MAGNESIUM", "IBUPROFEN + PARACETAMOL", etc.
        parts = re.split(r'[+/]|\s+AND\s+|\s+\+\s+', desc)
        
        for part in parts:
            # Extract the first word(s) before dose info
            # e.g., "ALUMINUM 200MG" -> "ALUMINUM"
            match = re.match(r'^([A-Z][A-Z\s\-]+?)(?:\s*\d|\s*\(|$)', part.strip())
            if match:
                generic = match.group(1).strip()
                # Clean up
                generic = re.sub(r'\s+', ' ', generic)
                if generic and len(generic) > 2 and generic not in GARBAGE_TOKENS:
                    generics.append(generic)
        
        return generics
    
    # Match ESOA to Annex F - STRICT MATCHING
    # Only matches when: generic + dose + form + route all match (salt can vary)
    def match_to_drug_code(row):
        generic_raw = row.get("generic_final") or row.get("generic_name") or ""
        
        # Fix known wrong synonyms (from unified_constants)
        for wrong, correct in DRUGBANK_COMPONENT_SYNONYMS.items():
            if wrong in str(generic_raw).upper():
                generic_raw = str(generic_raw).upper().replace(wrong, correct)
        
        generics = extract_clean_generics(generic_raw)
        
        # Fallback: if no generics from generic_final, try extracting from DESCRIPTION
        esoa_desc = row.get("DESCRIPTION") or ""
        if not generics:
            generics = extract_generics_from_description(esoa_desc)
        
        if not generics:
            return None, "no_generic"
        
        # Extract ESOA attributes for matching
        esoa_dose = extract_dose_mg(esoa_desc)
        esoa_form = normalize_for_match(row.get("form"))
        esoa_route = normalize_for_match(row.get("route"))
        
        # Try each generic component against the lookup
        candidates = []
        for generic in generics:
            # Try all name variants (original + synonyms)
            for variant in get_all_name_variants(generic):
                candidates.extend(annex_lookup.get(variant, []))
        
        if not candidates:
            return None, "generic_not_in_annex"
        
        # Deduplicate candidates by drug_code
        seen_codes = set()
        unique_candidates = []
        for c in candidates:
            if c["drug_code"] not in seen_codes:
                seen_codes.add(c["drug_code"])
                unique_candidates.append(c)
        candidates = unique_candidates
        
        # Helper to check if form/route match (None matches anything)
        def form_matches(cand_form, esoa_form):
            if not esoa_form or not cand_form:
                return True  # Missing form in either = compatible
            return cand_form == esoa_form
        
        def route_matches(cand_route, esoa_route):
            if not esoa_route or not cand_route:
                return True  # Missing route in either = compatible
            return cand_route == esoa_route
        
        # STRICT MATCHING: Require generic + dose + form + route
        # Only return a match if dose matches (and form/route if available)
        if esoa_dose:
            perfect_matches = [
                c for c in candidates 
                if c.get("dose_mg") == esoa_dose 
                and form_matches(c.get("form"), esoa_form)
                and route_matches(c.get("route"), esoa_route)
            ]
            if perfect_matches:
                return perfect_matches[0]["drug_code"], "matched_perfect"
        
        # No perfect match found
        return None, "no_perfect_match"
    
    if verbose:
        print("\nMatching ESOA to Drug Codes...")
    
    results = esoa_df.apply(match_to_drug_code, axis=1, result_type="expand")
    esoa_df["drug_code"] = results[0]
    esoa_df["drug_code_match_reason"] = results[1]
    
    # Write outputs
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_with_spinner("Write outputs", lambda: write_csv_and_parquet(esoa_df, output_path))
    
    # Summary
    total = len(esoa_df)
    matched = esoa_df["drug_code"].notna().sum()
    
    reason_counts = {str(reason): int(count) for reason, count in esoa_df["drug_code_match_reason"].value_counts().items() if pd.notna(reason)}
    result_summary = {
        "total": total,
        "matched": matched,
        "matched_pct": 100 * matched / total if total else 0,
        "output_path": output_path,
        "reason_counts": reason_counts,
    }
    
    if verbose:
        print(f"\nPart 4 complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Matched: {matched:,} ({result_summary['matched_pct']:.1f}%)")
        print("\nMatch reasons:")
        for reason, count in reason_counts.items():
            pct = 100 * count / total if total else 0
            print(f"  {reason}: {count:,} ({pct:.1f}%)")
    
    # Log metrics
    log_metrics("esoa_to_drug_code", {
        "total": total,
        "matched": matched,
        "matched_pct": round(result_summary["matched_pct"], 2),
    })
    
    return result_summary


def load_fda_food_lookup(inputs_dir: Path = None) -> dict:
    """
    Load FDA food data for fallback matching.
    
    Returns dict mapping normalized product names to registration info.
    """
    import glob
    
    if inputs_dir is None:
        inputs_dir = PIPELINE_INPUTS_DIR
    
    # Find latest FDA food file
    food_files = sorted(glob.glob(str(inputs_dir / "fda_food_*.csv")))
    if not food_files:
        food_files = sorted(glob.glob(str(inputs_dir / "fda_food_*.parquet")))
    
    if not food_files:
        return {}
    
    food_path = Path(food_files[-1])
    
    if str(food_path).endswith('.parquet'):
        food_df = pd.read_parquet(food_path)
    else:
        food_df = pd.read_csv(food_path)
    
    # Build lookup by brand_name and product_name
    lookup = {}
    for _, row in food_df.iterrows():
        brand = str(row.get("brand_name", "")).upper().strip()
        product = str(row.get("product_name", "")).upper().strip()
        reg_num = row.get("registration_number", "")
        
        if brand and brand != "-":
            lookup[brand] = {"type": "fda_food_brand", "registration": reg_num}
        if product and product != "-":
            lookup[product] = {"type": "fda_food_product", "registration": reg_num}
    
    return lookup


def check_fda_food_fallback(
    text: str,
    food_lookup: dict,
) -> tuple:
    """
    Check if text matches FDA food database.
    
    Returns (match_type, registration_number) or (None, None).
    """
    if not text or not food_lookup:
        return None, None
    
    text_upper = text.upper().strip()
    
    # Direct match
    if text_upper in food_lookup:
        info = food_lookup[text_upper]
        return info["type"], info.get("registration", "")
    
    # Token-based match (check if any token matches)
    tokens = text_upper.split()
    for token in tokens:
        if len(token) >= 4 and token in food_lookup:
            info = food_lookup[token]
            return f"{info['type']}_partial", info.get("registration", "")
    
    return None, None


def log_metrics(
    run_type: str,
    metrics: dict,
    metrics_path: Optional[Path] = None,
) -> None:
    """
    Log pipeline run metrics to history file.
    
    Args:
        run_type: Type of run (annex_f, esoa, esoa_to_drug_code)
        metrics: Dict with metric values
        metrics_path: Path to metrics history file
    """
    from datetime import datetime
    
    if metrics_path is None:
        metrics_path = PIPELINE_OUTPUTS_DIR / "metrics_history.csv"
    
    # Build row
    row = {
        "timestamp": datetime.now().isoformat(),
        "run_type": run_type,
        **metrics,
    }
    
    # Append to CSV
    file_exists = metrics_path.exists()
    
    metrics_df = pd.DataFrame([row])
    if file_exists:
        metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)


def get_metrics_summary(metrics_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Get metrics history summary.
    
    Returns DataFrame with all historical metrics.
    """
    if metrics_path is None:
        metrics_path = PIPELINE_OUTPUTS_DIR / "metrics_history.csv"
    
    if not metrics_path.exists():
        return pd.DataFrame()
    
    return pd.read_csv(metrics_path)


def print_metrics_comparison(verbose: bool = True) -> None:
    """Print comparison of latest metrics vs previous runs."""
    df = get_metrics_summary()
    
    if df.empty:
        if verbose:
            print("No metrics history found.")
        return
    
    if verbose:
        print("\n" + "=" * 60)
        print("METRICS HISTORY")
        print("=" * 60)
        
        # Group by run_type and show latest
        for run_type in df["run_type"].unique():
            subset = df[df["run_type"] == run_type].tail(5)
            print(f"\n{run_type.upper()}:")
            print(subset.to_string(index=False))
