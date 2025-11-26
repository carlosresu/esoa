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
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def _run_with_spinner(label: str, func):
    """Run func() while showing a lightweight CLI spinner."""
    import threading

    done = threading.Event()
    result = []
    err = []

    def worker():
        try:
            result.append(func())
        except BaseException as exc:
            err.append(exc)
        finally:
            done.set()

    start = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    frames = "|/-\\"
    idx = 0
    while not done.wait(0.1):
        elapsed = time.perf_counter() - start
        sys.stdout.write(f"\r{frames[idx % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        idx += 1
    thread.join()
    elapsed = time.perf_counter() - start
    status = "done" if not err else "error"
    sys.stdout.write(f"\r[{status}] {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return result[0] if result else None


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

    print("=" * 60)
    print("Part 4: Bridge ESOA to Annex F Drug Codes")
    print("=" * 60)

    # Resolve paths
    esoa_atc_path = OUTPUTS_DIR / args.esoa_atc
    annex_atc_path = OUTPUTS_DIR / args.annex_atc
    out_path = OUTPUTS_DIR / args.out

    if not esoa_atc_path.exists():
        print(f"[error] ESOA with ATC not found: {esoa_atc_path}")
        print("  Run Part 3 first: python run_drugs_pt_3_esoa_atc.py")
        sys.exit(1)

    if not annex_atc_path.exists():
        print(f"[error] Annex F with ATC not found: {annex_atc_path}")
        print("  Run Part 2 first: python run_drugs_pt_2_annex_f_atc.py")
        sys.exit(1)

    # Load data
    print(f"[info] Loading ESOA with ATC: {esoa_atc_path}")
    esoa_df = _run_with_spinner("Load ESOA with ATC", lambda: pd.read_csv(esoa_atc_path))

    print(f"[info] Loading Annex F with ATC: {annex_atc_path}")
    annex_df = _run_with_spinner("Load Annex F with ATC", lambda: pd.read_csv(annex_atc_path))

    # Build index
    atc_to_annex, drugbank_to_annex = _run_with_spinner(
        "Build Annex F index", lambda: build_annex_f_index(annex_df)
    )
    print(f"  - ATC codes indexed: {len(atc_to_annex)}")
    print(f"  - DrugBank IDs indexed: {len(drugbank_to_annex)}")

    # Match
    matched_df = _run_with_spinner(
        "Match ESOA to Annex F Drug Codes",
        lambda: match_esoa_to_annex_f(esoa_df, atc_to_annex, drugbank_to_annex),
    )

    # Save output
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(out_path, index=False)

    # Summary
    total = len(matched_df)
    matched = matched_df["matched_drug_code"].notna().sum()

    print("\n" + "=" * 60)
    print("Part 4 Complete: ESOA → Annex F Drug Code Mapping")
    print("=" * 60)
    print(f"  Output: {out_path}")
    print(f"  Total ESOA rows: {total}")
    print(f"  Matched to Drug Code: {matched} ({100*matched/total:.1f}%)")
    print(f"  Unmatched: {total - matched}")


if __name__ == "__main__":
    main(sys.argv[1:])
