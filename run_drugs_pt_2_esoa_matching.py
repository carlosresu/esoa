#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 runner: execute the full Drugs pipeline assuming Annex F is already prepared."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_drugs_all_parts as run_dm


def _default_annex_prepared() -> str:
    return str(run_dm.PIPELINE_INPUTS_SUBDIR / "annex_f_prepared.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the DrugsAndMedicine pipeline using a pre-prepared Annex F.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annex", default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "annex_f.csv"), help="Path to Annex F CSV")
    parser.add_argument("--annex-prepared", default=_default_annex_prepared(), help="Path to annex_f_prepared.csv")
    parser.add_argument("--pnf", default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "pnf.csv"), help="Path to PNF CSV")
    parser.add_argument(
        "--esoa",
        default=None,
        help="Path to eSOA CSV (defaults to concatenated inputs/drugs/esoa_pt_*.csv when omitted)",
    )
    parser.add_argument("--out", default="esoa_matched_drugs.csv", help="Output CSV filename (stored under ./outputs/drugs)")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R preprocessing scripts")
    parser.add_argument("--skip-brandmap", action="store_true", help="Skip building FDA brand map CSV")
    parser.add_argument("--skip-drugbank", action="store_true", help="Skip running the DrugBank aggregation helper")
    parser.add_argument("--skip-excel", action="store_true", help="Skip writing XLSX output (CSV and summaries still produced)")
    parser.add_argument("--skip-unknowns", action="store_true", help="Skip resolve_unknowns enrichment after matching")
    args = parser.parse_args(argv)

    annex_prepared = run_dm._resolve_repo_path(args.annex_prepared)
    if not annex_prepared.is_file():
        raise FileNotFoundError(
            f"Prepared Annex F not found at {annex_prepared}. Run run_drugs_pt_1_annex_f.py first or adjust --annex-prepared."
        )

    forwarded: list[str] = [
        "run_drugs_all_parts.py",
        "--skip-annex-stage",
        "--annex",
        args.annex,
        "--pnf",
        args.pnf,
        "--out",
        args.out,
        "--annex-prepared",
        str(annex_prepared),
    ]
    if args.esoa is not None:
        forwarded.extend(["--esoa", args.esoa])
    if args.skip_r:
        forwarded.append("--skip-r")
    if args.skip_brandmap:
        forwarded.append("--skip-brandmap")
    if args.skip_drugbank:
        forwarded.append("--skip-drugbank")
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
