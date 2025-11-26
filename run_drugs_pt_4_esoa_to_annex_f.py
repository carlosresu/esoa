#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 4: Bridge ESOA rows to Annex F Drug Codes via ATC/DrugBank ID.

This script:
- Loads ESOA rows with ATC/DrugBank IDs (from Part 3)
- Loads Annex F rows with ATC/DrugBank IDs (from Part 2)
- For each ESOA row, finds Annex F candidates with matching ATC/DrugBank ID
- Scores candidates by dose, form, route similarity
- Selects the best Drug Code for each ESOA row

This is the final step that produces the ESOA → Drug Code mapping.

Prerequisites:
- Run Part 2 (annex_f_atc) to have Annex F tagged with ATC/DrugBank IDs
- Run Part 3 (esoa_atc) to have ESOA tagged with ATC/DrugBank IDs
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from pipelines.drugs.scripts.spinner import run_with_spinner

PROJECT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def build_annex_f_index(annex_atc_df: pd.DataFrame) -> tuple[dict, dict]:
    """Build ATC → Annex F and DrugBank ID → Annex F lookup indexes."""
    atc_to_annex = defaultdict(list)
    drugbank_to_annex = defaultdict(list)

    for _, row in annex_atc_df.iterrows():
        row_dict = row.to_dict()
        atc = str(row.get("atc_code", "")).strip()
        db_id = str(row.get("drugbank_id", "")).strip()

        if atc and atc != "nan":
            # Handle pipe-separated ATC codes
            for code in atc.split("|"):
                code = code.strip()
                if code:
                    atc_to_annex[code].append(row_dict)

        if db_id and db_id != "nan":
            drugbank_to_annex[db_id].append(row_dict)

    return dict(atc_to_annex), dict(drugbank_to_annex)


def find_annex_f_candidates(
    esoa_row: dict,
    atc_to_annex: dict,
    drugbank_to_annex: dict,
) -> list[dict]:
    """Find Annex F candidates for an ESOA row based on ATC/DrugBank ID."""
    candidates = []
    seen_drug_codes = set()

    # Try ATC code first
    atc = str(esoa_row.get("probable_atc", "") or esoa_row.get("who_atc_codes", "")).strip()
    if atc and atc != "nan":
        for code in atc.split("|"):
            code = code.strip()
            for annex_row in atc_to_annex.get(code, []):
                drug_code = annex_row.get("Drug Code")
                if drug_code and drug_code not in seen_drug_codes:
                    seen_drug_codes.add(drug_code)
                    candidates.append(annex_row)

    # Also try DrugBank ID
    # Note: ESOA matching may store this differently - check column names
    db_id = str(esoa_row.get("drugbank_id", "")).strip()
    if db_id and db_id != "nan":
        for annex_row in drugbank_to_annex.get(db_id, []):
            drug_code = annex_row.get("Drug Code")
            if drug_code and drug_code not in seen_drug_codes:
                seen_drug_codes.add(drug_code)
                candidates.append(annex_row)

    return candidates


def score_annex_f_candidate(esoa_row: dict, annex_row: dict) -> float:
    """Score how well an Annex F row matches an ESOA row."""
    score = 0.0

    # Base score for having a match
    score += 10.0

    # TODO: Add dose, form, route scoring
    # This will be implemented based on the parsed fields from both sides

    return score


def match_esoa_to_annex_f(
    esoa_df: pd.DataFrame,
    atc_to_annex: dict,
    drugbank_to_annex: dict,
) -> pd.DataFrame:
    """Match each ESOA row to the best Annex F Drug Code."""
    results = []

    for _, esoa_row in esoa_df.iterrows():
        esoa_dict = esoa_row.to_dict()

        # Find candidates
        candidates = find_annex_f_candidates(esoa_dict, atc_to_annex, drugbank_to_annex)

        # Score and select best
        best_score = -1
        best_annex = None

        for annex_row in candidates:
            score = score_annex_f_candidate(esoa_dict, annex_row)
            if score > best_score:
                best_score = score
                best_annex = annex_row

        # Build result row
        result = {
            "esoa_idx": esoa_dict.get("esoa_idx"),
            "raw_text": esoa_dict.get("raw_text"),
            "esoa_atc": esoa_dict.get("probable_atc") or esoa_dict.get("who_atc_codes"),
            "candidate_count": len(candidates),
            "matched_drug_code": best_annex.get("Drug Code") if best_annex else None,
            "matched_drug_description": best_annex.get("Drug Description") if best_annex else None,
            "match_score": best_score if best_annex else None,
        }
        results.append(result)

    return pd.DataFrame(results)


def run_part_4(
    esoa_atc_filename: str = "esoa_with_atc.csv",
    annex_atc_filename: str = "annex_f_with_atc.csv",
    out_filename: str = "esoa_matched_drug_codes.csv",
    standalone: bool = True,
) -> dict:
    """
    Run Part 4: Bridge ESOA to Annex F Drug Codes.
    
    Returns dict with results summary.
    """
    if standalone:
        print("=" * 60)
        print("Part 4: Bridge ESOA to Annex F Drug Codes")
        print("=" * 60)

    # Resolve paths
    esoa_atc_path = OUTPUTS_DIR / esoa_atc_filename
    annex_atc_path = OUTPUTS_DIR / annex_atc_filename
    out_path = OUTPUTS_DIR / out_filename

    if not esoa_atc_path.exists():
        raise FileNotFoundError(f"ESOA with ATC not found: {esoa_atc_path}. Run Part 3 first.")

    if not annex_atc_path.exists():
        raise FileNotFoundError(f"Annex F with ATC not found: {annex_atc_path}. Run Part 2 first.")

    # Load data
    esoa_df = run_with_spinner("Load ESOA with ATC", lambda: pd.read_csv(esoa_atc_path))
    annex_df = run_with_spinner("Load Annex F with ATC", lambda: pd.read_csv(annex_atc_path))

    # Build index
    atc_to_annex, drugbank_to_annex = run_with_spinner(
        "Build Annex F index", lambda: build_annex_f_index(annex_df)
    )

    # Match
    matched_df = run_with_spinner(
        "Match ESOA to Annex F Drug Codes",
        lambda: match_esoa_to_annex_f(esoa_df, atc_to_annex, drugbank_to_annex),
    )

    # Save output
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_with_spinner("Write output CSV", lambda: matched_df.to_csv(out_path, index=False))

    # Summary
    total = len(matched_df)
    matched = matched_df["matched_drug_code"].notna().sum()

    results = {
        "total": total,
        "matched": matched,
        "matched_pct": 100 * matched / total if total else 0,
        "unmatched": total - matched,
        "atc_codes_indexed": len(atc_to_annex),
        "drugbank_ids_indexed": len(drugbank_to_annex),
        "output_path": out_path,
    }

    if standalone:
        print(f"\n  ATC codes indexed: {len(atc_to_annex)}")
        print(f"  DrugBank IDs indexed: {len(drugbank_to_annex)}")
        print(f"  Total ESOA rows: {total}")
        print(f"  Matched to Drug Code: {matched} ({results['matched_pct']:.1f}%)")
        print(f"  Unmatched: {total - matched}")
        print(f"  Output: {out_path}")

    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 4: Bridge ESOA rows to Annex F Drug Codes via ATC/DrugBank ID."
    )
    parser.add_argument(
        "--esoa-atc",
        metavar="PATH",
        default="esoa_with_atc.csv",
        help="ESOA with ATC/DrugBank IDs (default: outputs/drugs/esoa_with_atc.csv).",
    )
    parser.add_argument(
        "--annex-atc",
        metavar="PATH",
        default="annex_f_with_atc.csv",
        help="Annex F with ATC/DrugBank IDs (default: outputs/drugs/annex_f_with_atc.csv).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default="esoa_matched_drug_codes.csv",
        help="Output filename (default: esoa_matched_drug_codes.csv).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_part_4(
        esoa_atc_filename=args.esoa_atc,
        annex_atc_filename=args.annex_atc,
        out_filename=args.out,
        standalone=True,
    )
    
    print("\nPipeline complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
