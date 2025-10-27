# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py — Dose/form-aware preparation + dose-aware matching.

Exports:
- prepare(pnf_csv, esoa_csv, outdir) -> (pnf_prepared_csv, esoa_prepared_csv)
- match(annex_prepared_csv, pnf_prepared_csv, esoa_prepared_csv, out_csv) -> out_csv
- run_all(annex_csv, pnf_csv, esoa_csv, outdir, out_csv) -> out_csv
"""

import argparse
import os
from pathlib import Path

from scripts.prepare import prepare
from scripts.match import match
from scripts.prepare_annex_f import prepare_annex_f


def run_all(
    annex_csv: str,
    pnf_csv: str,
    esoa_csv: str,
    outdir: str = ".",
    out_csv: str = "esoa_matched.csv",
) -> str:
    """Execute preparation then matching so callers get a single convenience entry point."""
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    annex_prepared = prepare_annex_f(annex_csv, str(outdir_path / "annex_f_prepared.csv"))
    # Generate prepared inputs for both datasets and capture their output paths.
    pnf_prepared, esoa_prepared = prepare(pnf_csv, esoa_csv, outdir)
    # Chain directly into the matcher using the freshly prepared files.
    return match(annex_prepared, pnf_prepared, esoa_prepared, out_csv)


def _cli():
    """Parse CLI arguments and run the full pipeline with user-provided paths."""
    # Collect all supported command-line options.
    ap = argparse.ArgumentParser(description="Dose-aware drug matching pipeline")
    ap.add_argument("--annex", required=False, default="annex_f.csv")
    ap.add_argument("--pnf", required=False, default="pnf.csv")
    ap.add_argument("--esoa", required=False, default="esoa.csv")
    ap.add_argument("--outdir", required=False, default=".")
    ap.add_argument("--out", required=False, default="esoa_matched.csv")
    args = ap.parse_args()
    # Execute the end-to-end pipeline with the parsed arguments.
    annex_path = args.annex
    pnf_path = args.pnf
    esoa_path = args.esoa
    if not os.path.isabs(annex_path):
        annex_path = os.path.join(args.outdir, annex_path)
    run_all(annex_path, pnf_path, esoa_path, args.outdir, args.out)


if __name__ == "__main__":
    _cli()
