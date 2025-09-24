#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py — Imports main.py and runs the whole pipeline end-to-end.

Usage:
  python run.py --pnf pnf.csv --esoa esoa.csv --outdir . --out esoa_matched.csv
"""

import argparse
import main  # <- this imports the functions from main.py

def main_entry():
    parser = argparse.ArgumentParser(description="Run full ESOA pipeline (prepare → match)")
    parser.add_argument("--pnf", default="pnf.csv", help="Path to raw PNF CSV")
    parser.add_argument("--esoa", default="esoa.csv", help="Path to raw eSOA CSV")
    parser.add_argument("--outdir", default=".", help="Folder to write prepared files")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV path for matched results")
    args = parser.parse_args()

    main.run_all(args.pnf, args.esoa, args.outdir, args.out)
    print("\n>>> Pipeline complete. Final output:", args.out)

if __name__ == "__main__":
    main_entry()
