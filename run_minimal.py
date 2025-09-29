#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal ESOA pipeline runner skipping installs/R/brand map rebuild and Excel export."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal ESOA pipeline (no installs/R/brand map rebuild, no Excel output)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pnf", default=f"{run.DEFAULT_INPUTS_DIR}/pnf.csv", help="Path to PNF CSV")
    parser.add_argument("--esoa", default=f"{run.DEFAULT_INPUTS_DIR}/esoa.csv", help="Path to eSOA CSV")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV filename (stored under ./outputs)")
    args = parser.parse_args(argv)

    minimal_args = [
        "run.py",
        "--pnf", args.pnf,
        "--esoa", args.esoa,
        "--out", args.out,
        "--skip-install",
        "--skip-r",
        "--skip-brandmap",
    ]

    original_argv = sys.argv
    try:
        sys.argv = minimal_args
        run.main_entry()
    finally:
        sys.argv = original_argv

    outputs_dir = Path(run.THIS_DIR) / run.OUTPUTS_DIR
    excel_path = outputs_dir / (Path(args.out).stem + ".xlsx")
    if excel_path.exists():
        excel_path.unlink()
        print(f"Removed Excel export: {excel_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
