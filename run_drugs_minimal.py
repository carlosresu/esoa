#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal DrugsAndMedicine pipeline runner skipping R/brand map rebuild and Excel export."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import run_drugs as run_dm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal DrugsAndMedicine pipeline (skips R/brand map rebuild, no Excel output)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pnf", default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "pnf.csv"), help="Path to PNF CSV")
    parser.add_argument("--esoa", default=None, help="Path to eSOA CSV (defaults to concatenated esoa_pt_*.csv)")
    parser.add_argument("--out", default="esoa_matched_drugs.csv", help="Output CSV filename (stored under ./outputs/drugs)")
    args = parser.parse_args(argv)

    minimal_args = [
        "run_drugs.py",
        "--pnf", args.pnf,
        "--out", args.out,
        "--skip-r",
        "--skip-brandmap",
        "--skip-excel",
    ]
    if args.esoa is not None:
        minimal_args.extend(["--esoa", args.esoa])

    original_argv = sys.argv
    try:
        sys.argv = minimal_args
        run_dm.main_entry()
    finally:
        sys.argv = original_argv

    run_dm._prune_dated_exports(Path(run_dm.THIS_DIR) / "dependencies" / "atcd" / "output")
    run_dm._prune_dated_exports(run_dm._ensure_inputs_dir())


if __name__ == "__main__":
    main(sys.argv[1:])
