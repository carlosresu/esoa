#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3: Match ESOA rows to ATC codes and DrugBank IDs.

This script:
- Loads prepared ESOA data
- Matches each ESOA description to reference data (PNF, WHO, FDA, DrugBank)
- Assigns ATC codes and DrugBank IDs to ESOA rows
- Outputs esoa_with_atc.csv for use in Part 4

This uses the existing drugs pipeline matching logic from:
- pipelines/drugs/scripts/match_features_drugs.py (molecule detection)
- pipelines/drugs/scripts/match_scoring_drugs.py (scoring and classification)

Prerequisites:
- Run Part 1 (prepare_dependencies) first
- Run Part 2 (annex_f_atc) to have Annex F tagged
"""

from __future__ import annotations

import argparse
import sys
import time
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 3: Match ESOA rows to ATC codes and DrugBank IDs."
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
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Skip Excel output generation.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("=" * 60)
    print("Part 3: Match ESOA Rows with ATC/DrugBank IDs")
    print("=" * 60)

    # Resolve ESOA path
    if args.esoa:
        esoa_path = Path(args.esoa)
        if not esoa_path.is_absolute():
            esoa_path = PROJECT_DIR / esoa_path
    else:
        esoa_path = INPUTS_DIR / "esoa_combined.csv"
        if not esoa_path.exists():
            esoa_path = INPUTS_DIR / "esoa_prepared.csv"

    if not esoa_path.exists():
        print(f"[error] ESOA file not found: {esoa_path}")
        sys.exit(1)

    print(f"[info] Using ESOA: {esoa_path}")

    # Import the existing pipeline matching logic
    from pipelines.drugs.scripts.match_drugs import match
    from pipelines.drugs.scripts.prepare_drugs import prepare

    # Prepare inputs if needed
    pnf_prepared = INPUTS_DIR / "pnf_prepared.csv"
    esoa_prepared = INPUTS_DIR / "esoa_prepared.csv"
    annex_prepared = INPUTS_DIR / "annex_f.csv"

    if not pnf_prepared.exists() or not esoa_prepared.exists():
        print("[info] Running preparation step...")
        pnf_csv = INPUTS_DIR / "pnf.csv"
        if not pnf_csv.exists():
            print(f"[error] PNF source not found: {pnf_csv}")
            sys.exit(1)
        prepare(str(pnf_csv), str(esoa_path), str(INPUTS_DIR))

    # Run matching
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / args.out

    def _timing_hook(label: str, elapsed: float) -> None:
        print(f"  [{elapsed:7.2f}s] {label}")

    print("\n[info] Running ESOA matching pipeline...")
    match(
        str(annex_prepared),
        str(pnf_prepared),
        str(esoa_prepared),
        str(out_path),
        timing_hook=_timing_hook,
        skip_excel=args.skip_excel,
    )

    print("\n" + "=" * 60)
    print("Part 3 Complete: ESOA Rows Tagged with ATC/DrugBank IDs")
    print("=" * 60)
    print(f"  Output: {out_path}")
    print("\nNext: Run Part 4 (esoa_to_annex_f) to bridge ESOA to Annex F Drug Codes")


if __name__ == "__main__":
    main(sys.argv[1:])
