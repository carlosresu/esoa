# ===============================
# File: run.py (top-level)
# ===============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py — Install requirements, run ATC preprocessing (R scripts from ./dependencies/atcd),
then import main.py and run the pipeline.

Usage:
  python run.py --pnf pnf.csv --esoa esoa.csv --outdir . --out esoa_matched.csv
  # Optional:
  # python run.py --requirements path/to/requirements.txt --skip-install --skip-r
"""

import argparse
import os
import shutil
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


def run_r_scripts():
    """
    Run the ATC R preprocessing scripts with cwd=./dependencies/atcd
    so relative outputs are written to ./dependencies/atcd/output/*

    NOTE: scripts.match looks for WHO molecules under ./dependencies/atcd/output,
    so we ensure that directory exists.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    atcd_dir = os.path.join(here, "dependencies", "atcd")

    if not os.path.isdir(atcd_dir):
        raise FileNotFoundError(f"ATC directory not found: {atcd_dir}")

    out_dir = os.path.join(atcd_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript not found in PATH. Please install R and ensure 'Rscript' is available.")

    scripts = ["atcd.R", "export.R", "filter.R"]

    for script in scripts:
        script_path = os.path.join(atcd_dir, script)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Required R script not found: {script_path}")

    for script in scripts:
        print(f">>> Running R script (cwd={atcd_dir}): {script}")
        subprocess.run([rscript, script], check=True, cwd=atcd_dir)

    print(">>> All R scripts completed successfully.")


def main_entry():
    parser = argparse.ArgumentParser(description="Run full ESOA pipeline (ATC → prepare → match)")
    parser.add_argument("--pnf", default="pnf.csv", help="Path to raw PNF CSV")
    parser.add_argument("--esoa", default="esoa.csv", help="Path to raw eSOA CSV")
    parser.add_argument("--outdir", default=".", help="Folder to write prepared files")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV path for matched results")
    parser.add_argument("--requirements", default="requirements.txt", help="Path to requirements file")
    parser.add_argument("--skip-install", action="store_true", help="Skip installing requirements")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R scripts")
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements(args.requirements)

    if not args.skip_r:
        run_r_scripts()

    import main  # noqa: E402

    main.run_all(args.pnf, args.esoa, args.outdir, args.out)
    print("\n>>> Pipeline complete. Final output:", args.out)


if __name__ == "__main__":
    main_entry()
