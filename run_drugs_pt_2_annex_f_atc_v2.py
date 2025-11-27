#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2 (v2): Match Annex F entries to ATC codes and DrugBank IDs.

This version uses the UNIFIED TAGGER (same algorithm as Part 3) to ensure
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


def run_part_2_unified(standalone: bool = True) -> dict:
    """
    Run Part 2 using the unified tagger.
    
    Returns dict with results summary.
    """
    from pipelines.drugs.scripts.unified_tagger import UnifiedTagger
    
    if standalone:
        print("=" * 60)
        print("Part 2 (v2): Match Annex F with ATC/DrugBank IDs")
        print("Using Unified Tagger")
        print("=" * 60)
    
    # Load Annex F
    annex_path = PIPELINE_INPUTS_DIR / "annex_f.csv"
    if not annex_path.exists():
        raise FileNotFoundError(f"Annex F not found: {annex_path}")
    
    annex_df = run_with_spinner("Load Annex F", lambda: pd.read_csv(annex_path))
    
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
    if standalone:
        print(f"\nTagging {len(annex_df):,} Annex F entries...")
    
    results_df = run_with_spinner(
        "Tag Annex F descriptions",
        lambda: tagger.tag_descriptions(
            annex_df,
            text_column="Drug Description",
            id_column="Drug Code",
        )
    )
    
    # Merge results back with original Annex F
    annex_df["row_idx"] = range(len(annex_df))
    merged = annex_df.merge(
        results_df[["row_idx", "atc_code", "drugbank_id", "generic_name", "match_score", "match_reason", "sources"]],
        on="row_idx",
        how="left",
    )
    merged = merged.drop(columns=["row_idx"])
    
    # Rename columns for compatibility with Part 4
    merged = merged.rename(columns={
        "generic_name": "matched_generic_name",
        "sources": "matched_source",
    })
    
    # Write outputs
    PIPELINE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PIPELINE_OUTPUTS_DIR / "annex_f_with_atc.csv"
    
    run_with_spinner("Write outputs", lambda: _write_csv_and_parquet(merged, output_path))
    
    # Close tagger
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
    
    if standalone:
        print(f"\nAnnex F tagging complete: {output_path}")
        print(f"  Total: {total:,}")
        print(f"  Has ATC: {matched_atc:,} ({results['matched_atc_pct']:.1f}%)")
        print(f"  Has DrugBank ID: {matched_drugbank:,} ({results['matched_drugbank_pct']:.1f}%)")
        
        # Show match reasons breakdown
        print("\nMatch reasons:")
        for reason, count in merged["match_reason"].value_counts().items():
            print(f"  {reason}: {count:,} ({100*count/total:.1f}%)")
    
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 2 (v2): Match Annex F entries to ATC codes using unified tagger."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    
    run_part_2_unified(standalone=True)
    
    print("\nNext: Run Part 3 to match ESOA rows to ATC/DrugBank IDs")


if __name__ == "__main__":
    main(sys.argv[1:])
