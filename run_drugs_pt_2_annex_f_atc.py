#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2: Match Annex F entries to ATC codes and DrugBank IDs.

This script:
- Loads Annex F, PNF lexicon, and DrugBank generics/mixtures
- Matches each Annex F Drug Description to reference data
- Assigns ATC codes and DrugBank IDs
- Outputs annex_f_with_atc.csv for use in Part 4

Prerequisites:
- Run Part 1 (prepare_dependencies) first to ensure reference data is fresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from pipelines.drugs.scripts.spinner import run_with_spinner

PROJECT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def run_part_2(workers: int = 8, use_threads: bool = False, standalone: bool = True) -> dict:
    """
    Run Part 2: Match Annex F with ATC/DrugBank IDs.
    
    Returns dict with results summary.
    """
    import pandas as pd
    
    # Import the matching module
    from pipelines.drugs.scripts.match_annex_f_with_atc import (
        _read_table,
        _normalize_annex_df,
        _build_generic_phrases,
        _build_aho_automaton,
        _build_generic_to_atc_map,
        _build_mixture_lookup,
        build_reference_rows,
        _build_reference_index,
        _group_drugbank_refs_by_id,
        _build_generic_to_drugbank_id,
        _precompute_annex_records,
        match_annex_with_atc,
        _write_csv_and_parquet,
        DRUGS_DIR,
        OUTPUTS_DRUGS_DIR,
    )
    
    if standalone:
        print("=" * 60)
        print("Part 2: Match Annex F with ATC/DrugBank IDs")
        print("=" * 60)
    
    # Load data
    annex_path = DRUGS_DIR / "annex_f.csv"
    pnf_path = DRUGS_DIR / "pnf_lexicon.csv"
    drugbank_path = DRUGS_DIR / "drugbank_generics_master.csv"
    mixture_path = DRUGS_DIR / "drugbank_mixtures_master.csv"
    
    annex_df = run_with_spinner("Load Annex F source", lambda: _read_table(annex_path, required=True))
    annex_df = run_with_spinner("Normalize Annex F descriptions", lambda: _normalize_annex_df(annex_df))
    mixture_df = run_with_spinner("Load DrugBank mixtures", lambda: _read_table(mixture_path, required=False))
    pnf_df = run_with_spinner("Load PNF lexicon", lambda: _read_table(pnf_path, required=False))
    drugbank_df = run_with_spinner("Load DrugBank generics", lambda: _read_table(drugbank_path, required=True))
    
    # Build indexes
    generic_phrases = run_with_spinner("Build generic phrase list", lambda: _build_generic_phrases(drugbank_df))
    generic_automaton = run_with_spinner(
        "Build generic automaton", lambda: _build_aho_automaton(generic_phrases) if generic_phrases else None
    )
    generic_atc_map = run_with_spinner(
        "Index generic phrases to ATC codes", lambda: _build_generic_to_atc_map(drugbank_df)
    )
    mixture_lookup = run_with_spinner(
        "Index mixture components", lambda: _build_mixture_lookup(mixture_df) if not mixture_df.empty else {}
    )
    reference_rows = run_with_spinner(
        "Assemble reference rows", lambda: build_reference_rows(pnf_df, drugbank_df)
    )
    reference_index = run_with_spinner("Build reference token index", lambda: _build_reference_index(reference_rows))
    drugbank_refs_by_id = run_with_spinner(
        "Index DrugBank references by id", lambda: _group_drugbank_refs_by_id(reference_rows)
    )
    generic_to_drugbank = run_with_spinner(
        "Build generic to DrugBank ID lookup", lambda: _build_generic_to_drugbank_id(drugbank_df)
    )
    
    # Prepare Annex F records
    brand_patterns = []  # Not used in current implementation
    annex_records = run_with_spinner(
        "Prepare Annex F records",
        lambda: _precompute_annex_records(annex_df, brand_patterns, generic_phrases, generic_atc_map),
    )
    
    # Run matching
    match_df, tie_df, unresolved_df = run_with_spinner(
        "Match Annex F against references",
        lambda: match_annex_with_atc(
            annex_records,
            reference_rows,
            reference_index,
            brand_patterns,
            None if use_threads else generic_automaton,
            generic_phrases,
            generic_atc_map,
            mixture_lookup,
            drugbank_refs_by_id,
            generic_to_drugbank,
            max_workers=workers,
            use_threads=use_threads,
        ),
    )
    
    # Write outputs
    OUTPUTS_DRUGS_DIR.mkdir(parents=True, exist_ok=True)
    match_path = OUTPUTS_DRUGS_DIR / "annex_f_with_atc.csv"
    ties_path = OUTPUTS_DRUGS_DIR / "annex_f_atc_ties.csv"
    unresolved_path = OUTPUTS_DRUGS_DIR / "annex_f_atc_unresolved.csv"
    
    def reorder(df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "Drug Code", "Drug Description", "fuzzy_basis", "matched_reference_raw",
            "matched_source", "matched_generic_name", "matched_lexicon", "match_count",
            "matched_secondary_lexicon", "secondary_match_count", "atc_code", "drugbank_id",
            "primary_matching_tokens", "secondary_matching_tokens",
        ]
        existing = [c for c in cols if c in df.columns]
        remaining = [c for c in df.columns if c not in existing]
        return df.loc[:, existing + remaining]
    
    def _write_outputs() -> None:
        _write_csv_and_parquet(reorder(match_df), match_path)
        _write_csv_and_parquet(reorder(tie_df), ties_path)
        _write_csv_and_parquet(reorder(unresolved_df), unresolved_path)
    
    run_with_spinner("Write Annex F outputs", _write_outputs)
    
    # Summary
    total = len(match_df)
    matched = match_df["atc_code"].notna().sum()
    has_drugbank = match_df["drugbank_id"].notna().sum()
    
    results = {
        "total": total,
        "matched_atc": matched,
        "matched_atc_pct": 100 * matched / total if total else 0,
        "has_drugbank": has_drugbank,
        "has_drugbank_pct": 100 * has_drugbank / total if total else 0,
        "output_path": match_path,
    }
    
    if standalone:
        print(f"\nAnnex F ATC matches saved to {match_path}")
        print(f"  Total: {total}")
        print(f"  Matched with ATC: {matched} ({results['matched_atc_pct']:.1f}%)")
        print(f"  Has DrugBank ID: {has_drugbank} ({results['has_drugbank_pct']:.1f}%)")
    
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 2: Match Annex F entries to ATC codes and DrugBank IDs."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--use-threads",
        action="store_true",
        help="Use thread pool instead of process pool.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_part_2(workers=args.workers, use_threads=args.use_threads, standalone=True)
    
    print("\nNext: Run Part 3 to match ESOA rows to ATC/DrugBank IDs")


if __name__ == "__main__":
    main(sys.argv[1:])
