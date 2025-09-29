#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal ESOA pipeline runner skipping installs/R/brand map rebuild and Excel export."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from datetime import datetime

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
        "--skip-excel",
    ]

    original_argv = sys.argv
    try:
        sys.argv = minimal_args
        run.main_entry()
    finally:
        sys.argv = original_argv

    _prune_dated_exports(Path(run.THIS_DIR) / "dependencies" / "atcd" / "output", "who_atc_", "csv")
    _prune_dated_exports(Path(run.THIS_DIR) / "inputs", "", "csv")


def _prune_dated_exports(directory: Path, prefix: str, extension: str) -> None:
    """Delete older dated files (YYYY-MM-DD) when newer ones exist."""
    if not directory.is_dir():
        return

    dated_files: dict[str, Path] = {}
    for path in directory.glob(f"{prefix}*{extension}"):
        stem = path.stem
        if prefix and not stem.startswith(prefix):
            continue
        date_part = stem[len(prefix):]
        date_part = date_part.split("_", 1)[0]
        try:
            parsed = datetime.strptime(date_part, "%Y-%m-%d").date()
        except ValueError:
            continue
        key = parsed.isoformat()
        dated_files[key] = path if (key not in dated_files or path.stat().st_mtime > dated_files[key].stat().st_mtime) else dated_files[key]

    if not dated_files:
        return

    newest_date = max(dated_files.keys())
    for key, latest_path in dated_files.items():
        if key < newest_date and latest_path.exists():
            latest_path.unlink()


if __name__ == "__main__":
    main(sys.argv[1:])
