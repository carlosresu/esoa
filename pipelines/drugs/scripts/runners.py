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
    
    if verbose:
        print("=" * 60)
        print("Part 2: Match Annex F with ATC/DrugBank IDs")
        print("=" * 60)
    
    # Load Annex F
    if not annex_path.exists():
        raise FileNotFoundError(f"Annex F not found: {annex_path}")
    
    annex_df = run_with_spinner("Load Annex F", lambda: pd.read_csv(annex_path))
    
    # Initialize and load tagger
    tagger = run_with_spinner(
        "Initialize tagger",
        lambda: UnifiedTagger(
            outputs_dir=PIPELINE_OUTPUTS_DIR,
            inputs_dir=PIPELINE_INPUTS_DIR,
            verbose=False,
        )
    )
    run_with_spinner("Load reference data", lambda: tagger.load())
    
    # Tag descriptions
    if verbose:
        print(f"\nTagging {len(annex_df):,} Annex F entries...")
    
    results_df = run_with_spinner(
        "Tag descriptions",
        lambda: tagger.tag_descriptions(
            annex_df,
            text_column="Drug Description",
            id_column="Drug Code",
        )
    )
    
    # Merge results
    annex_df["row_idx"] = range(len(annex_df))
    merged = annex_df.merge(
        results_df[["row_idx", "atc_code", "drugbank_id", "generic_name", "reference_text", "match_score", "match_reason", "sources"]],
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
    
    results = {
        "total": total,
        "matched_atc": matched_atc,
        "matched_atc_pct": 100 * matched_atc / total if total else 0,
        "matched_drugbank": matched_drugbank,
        "matched_drugbank_pct": 100 * matched_drugbank / total if total else 0,
        "output_path": output_path,
    }
    
    if verbose:
        print(f"\nAnnex F tagging complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Has ATC: {matched_atc:,} ({results['matched_atc_pct']:.1f}%)")
        print(f"  Has DrugBank ID: {matched_drugbank:,} ({results['matched_drugbank_pct']:.1f}%)")
        
        print("\nMatch reasons:")
        for reason, count in merged["match_reason"].value_counts().items():
            print(f"  {reason}: {count:,} ({100*count/total:.1f}%)")
    
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
    
    if verbose:
        print("=" * 60)
        print("Part 3: Match ESOA with ATC/DrugBank IDs")
        print("=" * 60)
    
    # Load ESOA
    if not esoa_path.exists():
        raise FileNotFoundError(f"ESOA not found: {esoa_path}")
    
    esoa_df = run_with_spinner("Load ESOA", lambda: pd.read_csv(esoa_path))
    
    # Determine text column
    text_column = None
    for col in ["raw_text", "ITEM_DESCRIPTION", "DESCRIPTION", "Drug Description", "description"]:
        if col in esoa_df.columns:
            text_column = col
            break
    
    if not text_column:
        raise ValueError(f"No text column found. Columns: {list(esoa_df.columns)}")
    
    if verbose:
        print(f"  Using text column: {text_column}")
    
    # Initialize and load tagger
    tagger = run_with_spinner(
        "Initialize tagger",
        lambda: UnifiedTagger(
            outputs_dir=PIPELINE_OUTPUTS_DIR,
            inputs_dir=PIPELINE_INPUTS_DIR,
            verbose=verbose,  # Enable for progress output
        )
    )
    run_with_spinner("Load reference data", lambda: tagger.load())
    
    # Tag with deduplication (146K unique vs 258K total)
    total = len(esoa_df)
    if verbose:
        print(f"\nTagging {total:,} ESOA entries (with deduplication)...")
    
    # Use tag_batch with deduplication for performance
    results_df = tagger.tag_batch(
        esoa_df,
        text_column=text_column,
        chunk_size=10000,
        show_progress=verbose,
        deduplicate=True,
    )
    
    # Map results back to original rows by text
    # results_df has 'input_text' column with the original text
    results_df = results_df.rename(columns={"input_text": "_tag_text"})
    esoa_df["_tag_text"] = esoa_df[text_column].fillna("").astype(str)
    
    merged = esoa_df.merge(
        results_df[["_tag_text", "atc_code", "drugbank_id", "generic_name", "reference_text", "match_score", "match_reason", "sources"]],
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
    
    if verbose:
        print(f"\nESOA tagging complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Has ATC: {matched_atc_count:,} ({results['matched_atc_pct']:.1f}%)")
        print(f"  Has DrugBank ID: {matched_drugbank_count:,} ({results['matched_drugbank_pct']:.1f}%)")
        
        print("\nMatch reasons:")
        for reason, count in merged["match_reason"].value_counts().head(10).items():
            print(f"  {reason}: {count:,} ({100*count/total:.1f}%)")
    
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
        esoa_path = PIPELINE_OUTPUTS_DIR / "esoa_with_atc.parquet"
        if not esoa_path.exists():
            esoa_path = PIPELINE_OUTPUTS_DIR / "esoa_with_atc.csv"
    if annex_path is None:
        annex_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.parquet"
        if not annex_path.exists():
            annex_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.csv"
    if output_path is None:
        output_path = PIPELINE_OUTPUTS_DIR / "esoa_with_drug_code.csv"
    
    if verbose:
        print("=" * 60)
        print("Part 4: Match ESOA to Annex F Drug Codes")
        print("=" * 60)
    
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
    
    # Build synonym mappings from generics_master (the ONE source of truth)
    generics_master_path = PIPELINE_OUTPUTS_DIR / "generics_master.parquet"
    all_synonyms = {}
    
    if generics_master_path.exists():
        gm = pd.read_parquet(generics_master_path)
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
            print(f"  Loaded synonyms from generics_master: {len(all_synonyms):,} mappings")
    
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
    
    annex_lookup = {}
    for _, row in annex_df.iterrows():
        drug_code = row.get("Drug Code")
        if pd.isna(drug_code):
            continue
        
        generic = normalize_for_match(row.get("matched_generic_name") or row.get("generic_name"))
        if not generic:
            continue
        
        atc = normalize_for_match(row.get("atc_code"))
        
        if generic not in annex_lookup:
            annex_lookup[generic] = []
        annex_lookup[generic].append({
            "drug_code": drug_code,
            "atc_code": atc,
            "generic_name": generic,
        })
    
    if verbose:
        print(f"  Annex F lookup: {len(annex_lookup):,} unique generics")
    
    # Match ESOA to Annex F
    def match_to_drug_code(row):
        generic = normalize_for_match(row.get("generic_final") or row.get("generic_name"))
        atc = normalize_for_match(row.get("atc_code_final") or row.get("atc_code"))
        
        if not generic:
            return None, "no_generic"
        
        # Try all name variants (original + synonyms)
        candidates = []
        for variant in get_all_name_variants(generic):
            candidates.extend(annex_lookup.get(variant, []))
        
        if not candidates:
            return None, "generic_not_in_annex"
        
        # Filter by ATC if available
        if atc:
            atc_matches = [c for c in candidates if c["atc_code"] == atc]
            if atc_matches:
                return atc_matches[0]["drug_code"], "matched_generic_atc"
        
        # Fall back to generic-only match
        return candidates[0]["drug_code"], "matched_generic_only"
    
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
    
    result_summary = {
        "total": total,
        "matched": matched,
        "matched_pct": 100 * matched / total if total else 0,
        "output_path": output_path,
    }
    
    if verbose:
        print(f"\nPart 4 complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Matched: {matched:,} ({result_summary['matched_pct']:.1f}%)")
        
        print("\nMatch reasons:")
        for reason, count in esoa_df["drug_code_match_reason"].value_counts().items():
            print(f"  {reason}: {count:,} ({100*count/total:.1f}%)")
    
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
    food_files = sorted(glob.glob(str(inputs_dir / "fda_food_*.parquet")))
    if not food_files:
        food_files = sorted(glob.glob(str(inputs_dir / "fda_food_*.csv")))
    
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
