#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience runner that launches the Drugs script using a pre-normalized Annex F."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_drugs_all_parts as run_dm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the DrugsAndMedicine pipeline with a prepared Annex F dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annex", default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "annex_f.csv"), help="Path to the prepared Annex F CSV")
    parser.add_argument("--pnf", default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "pnf.csv"), help="Path to PNF CSV")
    parser.add_argument(
        "--esoa",
        default=None,
        help="Path to eSOA CSV (defaults to concatenated inputs/drugs/esoa_pt_*.csv)",
    )
    parser.add_argument("--out", default="esoa_matched_drugs.csv", help="Output CSV filename (stored under ./outputs/drugs)")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R preprocessing scripts")
    parser.add_argument("--skip-brandmap", action="store_true", help="Skip building FDA brand map CSV")
    parser.add_argument("--skip-excel", action="store_true", help="Skip writing XLSX output (CSV and summaries still produced)")
    parser.add_argument("--skip-unknowns", action="store_true", help="Skip resolve_unknowns enrichment after matching")
    args = parser.parse_args(argv)

    annex_path = run_dm._resolve_repo_path(args.annex)
    if not annex_path.is_file():
        raise FileNotFoundError(
            f"Annex F CSV not found at {annex_path}. Provide a normalized Annex F dataset before running the pipeline."
        )

    forwarded: list[str] = [
        "run_drugs_all_parts.py",
        "--annex",
        str(annex_path),
        "--pnf",
        args.pnf,
        "--out",
        args.out,
    ]
    if args.esoa is not None:
        forwarded.extend(["--esoa", args.esoa])
    if args.skip_r:
        forwarded.append("--skip-r")
    if args.skip_brandmap:
        forwarded.append("--skip-brandmap")
    if args.skip_excel:
        forwarded.append("--skip-excel")
    if args.skip_unknowns:
        forwarded.append("--skip-unknowns")

    original_argv = sys.argv
    try:
        sys.argv = forwarded
        run_dm.main_entry()
    finally:
        sys.argv = original_argv

    run_dm._prune_dated_exports(Path(run_dm.THIS_DIR) / "dependencies" / "atcd" / "output")
    run_dm._prune_dated_exports(run_dm._ensure_inputs_dir())


if __name__ == "__main__":
    main(sys.argv[1:])
