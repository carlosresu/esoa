# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py — Dose/form-aware preparation + dose-aware matching.

Exports:
- prepare(pnf_csv, esoa_csv, outdir) -> (pnf_prepared.csv, esoa_prepared.csv)
- match(pnf_prepared, esoa_prepared, out_path) -> out_path
- run_all(pnf_csv, esoa_csv, outdir, out_path) -> out_path
"""

import argparse

from scripts.prepare import prepare
from scripts.match import match


def run_all(
    pnf_csv: str,
    esoa_csv: str,
    outdir: str = ".",
    out_path: str = "esoa_matched.parquet",
    *,
    export_csv: bool = False,
    export_excel: bool = False,
) -> str:
    """Execute preparation then matching so callers get a single convenience entry point."""
    # Generate prepared inputs for both datasets and capture their output paths.
    pnf_prepared, esoa_prepared = prepare(pnf_csv, esoa_csv, outdir)
    # Chain directly into the matcher using the freshly prepared files.
    return match(
        pnf_prepared,
        esoa_prepared,
        out_path,
        export_csv=export_csv,
        export_excel=export_excel,
    )


def _cli():
    """Parse CLI arguments and run the full pipeline with user-provided paths."""
    # Collect all supported command-line options.
    ap = argparse.ArgumentParser(description="Dose-aware drug matching pipeline")
    ap.add_argument("--pnf", required=False, default="pnf.csv")
    ap.add_argument("--esoa", required=False, default="esoa.csv")
    ap.add_argument("--outdir", required=False, default=".")
    ap.add_argument("--out", required=False, default="esoa_matched.parquet")
    ap.add_argument("--export-csv", action="store_true", help="Also emit CSV copies of the results")
    ap.add_argument("--export-excel", action="store_true", help="Also emit XLSX copy of the matched dataset")
    args = ap.parse_args()
    # Execute the end-to-end pipeline with the parsed arguments.
    run_all(
        args.pnf,
        args.esoa,
        args.outdir,
        args.out,
        export_csv=args.export_csv,
        export_excel=args.export_excel,
    )


if __name__ == "__main__":
    _cli()
