#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py — End-to-end execution of the drug matching pipeline.

Steps:
1. Create .venv with Python 3.11 if available (fallback to current Python)
2. Install dependencies from requirements.txt
3. Run prepare.py (dose/form-aware preparation)
4. Run main.py (dose-aware matcher)

Outputs:
- pnf_prepared.csv
- esoa_prepared.csv
- esoa_matched.csv
"""

import os
import sys
import subprocess
import platform
import shutil

def run(cmd, shell=False):
    print(f"\n>>> Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, shell=shell, check=True)

def version_str(exe):
    try:
        out = subprocess.check_output([exe, "--version"], text=True).strip()
        return out
    except Exception:
        return "Unknown"

def main():
    venv_dir = ".venv"

    # Prefer python3.11 for venv creation if available
    preferred_python = shutil.which("python3.11")
    if preferred_python:
        base_python = preferred_python
    else:
        base_python = sys.executable
        print(">>> WARNING: python3.11 not found on PATH. Using current Python "
              f"({version_str(base_python)}). Some packages are better tested on 3.11.")

    # Step 1. Create venv if missing
    if not os.path.exists(venv_dir):
        print(f">>> Creating venv with: {base_python} ({version_str(base_python)})")
        run([base_python, "-m", "venv", venv_dir])
    else:
        print(">>> venv already exists, skipping creation")

    # Step 2. Paths inside venv
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")

    print(f">>> Venv python: {python_path} ({version_str(python_path)})")

    # Step 3. Install dependencies
    run([pip_path, "install", "-r", "requirements.txt"])

    # Step 4. Run data engineering
    run([python_path, "prepare.py",
         "--pnf", "pnf.csv",
         "--esoa", "esoa.csv",
         "--outdir", "."])

    # Step 5. Run matcher
    run([python_path, "main.py",
         "--pnf", "pnf_prepared.csv",
         "--esoa", "esoa_prepared.csv",
         "--out", "esoa_matched.csv"])

    print("\n>>> Pipeline complete. Final output: esoa_matched.csv")

if __name__ == "__main__":
    main()
