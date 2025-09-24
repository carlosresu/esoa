#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py — Install requirements, then import main.py and run the pipeline.

Usage:
  python run.py --pnf pnf.csv --esoa esoa.csv --outdir . --out esoa_matched.csv
  # Optional:
  # python run.py --requirements path/to/requirements.txt --skip-install
"""
import argparse
import os
import subprocess
import sys

def install_requirements(req_path: str):
    """Install requirements with the same Python interpreter running this script."""
    if not req_path:
        return
    if not os.path.isfile(req_path):
        print(f">>> requirements file not found at {req_path}; skipping install")
        return
    print(f">>> Installing dependencies from {req_path} using {sys.executable} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", req_path],
        check=True,
    )
    print(">>> Dependencies installed.")

def main_entry():
    parser = argparse.ArgumentParser(description="Run full ESOA pipeline (prepare → match)")
    parser.add_argument("--pnf", default="pnf.csv", help="Path to raw PNF CSV")
    parser.add_argument("--esoa", default="esoa.csv", help="Path to raw eSOA CSV")
    parser.add_argument("--outdir", default=".", help="Folder to write prepared files")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV path for matched results")
    parser.add_argument("--requirements", default="requirements.txt", help="Path to requirements file")
    parser.add_argument("--skip-install", action="store_true", help="Skip installing requirements")
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements(args.requirements)

    # Import AFTER installing deps so pandas/pyahocorasick are available
    import main  # noqa: E402

    main.run_all(args.pnf, args.esoa, args.outdir, args.out)
    print("\n>>> Pipeline complete. Final output:", args.out)

if __name__ == "__main__":
    main_entry()
