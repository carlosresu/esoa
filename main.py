# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py — Dose/form-aware preparation + dose-aware matching.

Exports:
- prepare(pnf_csv, esoa_csv, outdir) -> (pnf_prepared_csv, esoa_prepared_csv)
- match(pnf_prepared_csv, esoa_prepared_csv, out_csv) -> out_csv
- run_all(pnf_csv, esoa_csv, outdir, out_csv) -> out_csv
"""

import argparse

from scripts.prepare import prepare
from scripts.match import match


def run_all(pnf_csv: str, esoa_csv: str, outdir: str = ".", out_csv: str = "esoa_matched.csv") -> str:
    pnf_prepared, esoa_prepared = prepare(pnf_csv, esoa_csv, outdir)
    return match(pnf_prepared, esoa_prepared, out_csv)


def _cli():
    ap = argparse.ArgumentParser(description="Dose-aware drug matching pipeline")
    ap.add_argument("--pnf", required=False, default="pnf.csv")
    ap.add_argument("--esoa", required=False, default="esoa.csv")
    ap.add_argument("--outdir", required=False, default=".")
    ap.add_argument("--out", required=False, default="esoa_matched.csv")
    args = ap.parse_args()
    run_all(args.pnf, args.esoa, args.outdir, args.out)


if __name__ == "__main__":
    _cli()
