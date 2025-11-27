#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3 (v2): Match ESOA rows to ATC codes and DrugBank IDs.

This version uses the UNIFIED TAGGER (same algorithm as Part 2) to ensure
consistent tagging between Annex F and ESOA.

Prerequisites:
- Run Part 1 (prepare_dependencies) first to ensure reference data is fresh
- Run build_unified_reference.py to create the unified reference dataset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from pipelines.drugs.scripts.spinner import run_with_spinner

PROJECT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"

# Use environment variable for inputs if set (for GitIgnored data)
PIPELINE_INPUTS_DIR = Path(os.environ.get("PIPELINE_INPUTS_DIR", INPUTS_DIR))
PIPELINE_OUTPUTS_DIR = Path(os.environ.get("PIPELINE_OUTPUTS_DIR", OUTPUTS_DIR))


def _write_csv_and_parquet(df: pd.DataFrame, csv_path: Path) -> None:
    """Write DataFrame to both CSV and Parquet."""
    df.to_csv(csv_path, index=False)
    parquet_path = csv_path.with_suffix(".parquet")
    try:
        # Convert object columns to string for parquet compatibility
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("").astype(str)
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"Warning: Parquet write failed: {e}", file=sys.stderr)


def run_part_3_unified(
    esoa_path: Optional[str] = None,
    out_filename: str = "esoa_with_atc.csv",
    standalone: bool = True,
) -> dict:
    """
    Run Part 3 using the unified tagger.
    
    Returns dict with results summary.
    """
    from pipelines.drugs.scripts.unified_tagger import UnifiedTagger
    
    if standalone:
        print("=" * 60)
        print("Part 3 (v2): Match ESOA with ATC/DrugBank IDs")
        print("Using Unified Tagger")
        print("=" * 60)
    
    # Resolve ESOA path
    if esoa_path:
        resolved_esoa = Path(esoa_path)
        if not resolved_esoa.is_absolute():
            resolved_esoa = PROJECT_DIR / resolved_esoa
    else:
        resolved_esoa = PIPELINE_INPUTS_DIR / "esoa_combined.csv"
        if not resolved_esoa.exists():
            resolved_esoa = PIPELINE_INPUTS_DIR / "esoa_prepared.csv"
    
    if not resolved_esoa.exists():
        raise FileNotFoundError(f"ESOA file not found: {resolved_esoa}")
    
    # Load ESOA
    esoa_df = run_with_spinner("Load ESOA", lambda: pd.read_csv(resolved_esoa))
    
    # Determine text column
    text_column = None
    for col in ["raw_text", "ITEM_DESCRIPTION", "DESCRIPTION", "Drug Description", "description"]:
        if col in esoa_df.columns:
            text_column = col
            break
    
    if not text_column:
        raise ValueError(f"No text column found in ESOA. Columns: {list(esoa_df.columns)}")
    
    if standalone:
        print(f"  Using text column: {text_column}")
    
    # Initialize tagger
    tagger = run_with_spinner(
        "Initialize unified tagger",
        lambda: UnifiedTagger(
            outputs_dir=PIPELINE_OUTPUTS_DIR,
            inputs_dir=PIPELINE_INPUTS_DIR,
            verbose=False,
        )
    )
    
    # Load reference data
    run_with_spinner("Load reference data", lambda: tagger.load())
    
    # Tag all descriptions
    total = len(esoa_df)
    if standalone:
        print(f"\nTagging {total:,} ESOA entries...")
    
    # Process in batches for progress reporting
    batch_size = 5000
    results_list = []
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_df = esoa_df.iloc[start:end].copy()
        batch_df["_batch_idx"] = range(start, end)
        
        batch_results = tagger.tag_descriptions(
            batch_df,
            text_column=text_column,
            id_column="_batch_idx",
        )
        batch_results = batch_results.rename(columns={"id": "_batch_idx"})
        results_list.append(batch_results)
        
        if standalone:
            pct = 100 * end / total
            print(f"  Processed {end:,}/{total:,} ({pct:.1f}%)")
    
    results_df = pd.concat(results_list, ignore_index=True)
    
    # Merge results back with original ESOA
    esoa_df["_batch_idx"] = range(len(esoa_df))
    merged = esoa_df.merge(
        results_df[["_batch_idx", "atc_code", "drugbank_id", "generic_name", "match_score", "match_reason", "sources"]],
        on="_batch_idx",
        how="left",
    )
    merged = merged.drop(columns=["_batch_idx"])
    
    # Rename columns for compatibility with Part 4
    merged = merged.rename(columns={
        "atc_code": "atc_code_final",
        "drugbank_id": "drugbank_id_final",
        "generic_name": "generic_final",
        "sources": "reference_source",
    })
    
    # Write outputs
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PIPELINE_OUTPUTS_DIR / out_filename
    
    run_with_spinner("Write outputs", lambda: _write_csv_and_parquet(merged, output_path))
    
    # Close tagger
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
    
    if standalone:
        print(f"\nESOA tagging complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Has ATC: {matched_atc_count:,} ({results['matched_atc_pct']:.1f}%)")
        print(f"  Has DrugBank ID: {matched_drugbank_count:,} ({results['matched_drugbank_pct']:.1f}%)")
        
        # Show match reasons breakdown
        print("\nMatch reasons:")
        for reason, count in merged["match_reason"].value_counts().head(10).items():
            print(f"  {reason}: {count:,} ({100*count/total:.1f}%)")
    
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 3 (v2): Match ESOA rows to ATC codes using unified tagger."
    )
    parser.add_argument(
        "--esoa",
        metavar="PATH",
        help="Path to ESOA CSV (default: inputs/drugs/esoa_combined.csv).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default="esoa_with_atc.csv",
        help="Output filename (default: esoa_with_atc.csv).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    
    run_part_3_unified(
        esoa_path=args.esoa,
        out_filename=args.out,
        standalone=True,
    )
    
    print("\nNext: Run Part 4 to bridge ESOA to Annex F Drug Codes")


if __name__ == "__main__":
    main(sys.argv[1:])
