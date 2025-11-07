#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2 minimal runner: skip R/brandmap/Excel while using a prepared Annex F."""
from __future__ import annotations

import argparse
import sys

import run_drugs_pt_2_esoa_matching as stage2
import run_drugs_all_parts as run_dm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the minimal Drugs pipeline (no R/brand map/Excel) using a prepared Annex F.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annex", default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "annex_f.csv"), help="Path to Annex F CSV")
    parser.add_argument("--annex-prepared", default=None, help="Path to annex_f_prepared.csv (defaults to inputs/drugs/annex_f_prepared.csv)")
    parser.add_argument("--pnf", default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "pnf.csv"), help="Path to PNF CSV")
    parser.add_argument("--esoa", default=None, help="Path to eSOA CSV (defaults to concatenated esoa_pt_*.csv)")
    parser.add_argument("--out", default="esoa_matched_drugs.csv", help="Output CSV filename (stored under ./outputs/drugs)")
    args = parser.parse_args(argv)

    forwarded = [
        "--annex",
        args.annex,
        "--pnf",
        args.pnf,
        "--out",
        args.out,
        "--skip-r",
        "--skip-brandmap",
        "--skip-excel",
    ]
    if args.annex_prepared:
        forwarded.extend(["--annex-prepared", args.annex_prepared])
    if args.esoa is not None:
        forwarded.extend(["--esoa", args.esoa])

    stage2.main(forwarded)


if __name__ == "__main__":
    main(sys.argv[1:])
