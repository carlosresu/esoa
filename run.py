#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — Install requirements, optionally run ATC (R) preprocessing,
then run the ESOA pipeline (prepare → match).

Usage:
  python run.py --pnf inputs/pnf.csv --esoa inputs/esoa.csv --out esoa_matched.csv
  # Optional flags:
  #   --requirements requirements.txt
  #   --skip-install
  #   --skip-r
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------
THIS_DIR: Path = Path(__file__).resolve().parent
DEFAULT_INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
ATCD_SUBDIR = Path("dependencies") / "atcd"
ATCD_SCRIPTS: tuple[str, ...] = ("atcd.R", "export.R", "filter.R")


# Ensure local imports (e.g., main.py) work when called from other CWDs
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def install_requirements(req_path: str | os.PathLike[str]) -> None:
    """
    Install dependencies listed in the given requirements file using
    the same Python interpreter running this script.

    Quietly installs (stdout/stderr suppressed). No-op if file missing.
    """
    req = Path(req_path)
    if not req or not req.is_file():
        return

    with open(os.devnull, "w") as devnull:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(req),
            ],
            cwd=str(THIS_DIR),
            stdout=devnull,
            stderr=devnull,
        )


def _resolve_input_path(p: str | os.PathLike[str], default_subdir: str = DEFAULT_INPUTS_DIR) -> Path:
    """
    Resolve a user-provided path to an existing file. If the provided
    path is not a file, also try ./<default_subdir>/<basename> relative
    to THIS_DIR.

    Raises FileNotFoundError if nothing exists.
    """
    if not p:
        raise FileNotFoundError("No input path provided.")

    pth = Path(p)
    if pth.is_file():
        return pth

    candidate = THIS_DIR / default_subdir / pth.name
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Input file not found: {pth!s}. "
        f"Tried: {pth.resolve()!s} and {candidate!s}. "
        f"Place the file under ./{default_subdir}/ or pass --pnf/--esoa with a correct path."
    )


def _ensure_outputs_dir() -> Path:
    """
    Ensure the outputs directory exists under THIS_DIR and return it.
    """
    outdir = THIS_DIR / OUTPUTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _assert_all_exist(root: Path, files: Iterable[str | os.PathLike[str]]) -> None:
    for f in files:
        fp = root / f
        if not fp.is_file():
            raise FileNotFoundError(f"Required file not found: {fp}")


# ---------------------------------------------------------------------
# ATC (R) Scripts
# ---------------------------------------------------------------------
def run_r_scripts() -> None:
    """
    Execute the ATC preprocessing R scripts located in ./dependencies/atcd.
    Requires Rscript to be available on PATH.
    """
    atcd_dir = THIS_DIR / ATCD_SUBDIR
    if not atcd_dir.is_dir():
        raise FileNotFoundError(f"ATC directory not found: {atcd_dir}")

    out_dir = atcd_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript not found in PATH. Please install R and ensure 'Rscript' is available.")

    _assert_all_exist(atcd_dir, ATCD_SCRIPTS)

    # Run each script quietly; raise on non-zero exit.
    with open(os.devnull, "w") as devnull:
        for script in ATCD_SCRIPTS:
            subprocess.run(
                [rscript, script],
                check=True,
                cwd=str(atcd_dir),
                stdout=devnull,
                stderr=devnull,
            )


# ---------------------------------------------------------------------
# Debug: create a single master.py concatenating key files for inspection
# ---------------------------------------------------------------------
def create_master_file(root_dir: Path) -> None:
    """
    Concatenate key repository files into ./debug/master.py for quick inspection.
    Missing files are skipped silently.
    """
    debug_dir = root_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = debug_dir / "master.py"

    files_to_concatenate = [
        root_dir / "scripts" / "aho.py",
        root_dir / "scripts" / "combos.py",
        root_dir / "scripts" / "dose.py",
        root_dir / "scripts" / "match_features.py",
        root_dir / "scripts" / "match_scoring.py",
        root_dir / "scripts" / "match_outputs.py",
        root_dir / "scripts" / "match.py",
        root_dir / "scripts" / "prepare.py",
        root_dir / "scripts" / "routes_forms.py",
        root_dir / "scripts" / "text_utils.py",
        root_dir / "scripts" / "who_molecules.py",
        root_dir / "main.py",
        root_dir / "run.py",
    ]

    header_text = """\
# INSTRUCTIONS:
# 1. With every query, I will provide the file contents of my Python repository.
# 2. The files are marked with their paths as comments (e.g., # <file_path>).
# 3. Your task is to analyze the provided code, then respond to my queries by providing the complete/corrected/expanded code for any file that needs to be changed or created, as a downloadable file(s).
# 4. For each file you modify or create, use the following format:
# - change 1: replace <file_path> by downloading the below
#   <download file here>
# - change 2: replace <file_path> by downloading the below
#   <download file here>
# - <and so on...>
# 5. DO NOT SEND ANY CODE IN CHAT. Only give me drop-in files I replace my files with.

# START OF REPO FILES
"""
    footer_text = """\
# END OF REPO FILES
"""

    with output_file_path.open("w", encoding="utf-8") as outfile:
        outfile.write(header_text)
        for file_path in files_to_concatenate:
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(root_dir).as_posix()
            outfile.write(f"\n# <{relative_path}>\n")
            with file_path.open("r", encoding="utf-8") as infile:
                outfile.write(infile.read())
            outfile.write("\n")
        outfile.write(footer_text)


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
def main_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Run full ESOA pipeline (ATC → prepare → match)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pnf", default=f"{DEFAULT_INPUTS_DIR}/pnf.csv", help="Path to PNF CSV")
    parser.add_argument("--esoa", default=f"{DEFAULT_INPUTS_DIR}/esoa.csv", help="Path to eSOA CSV")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV filename (saved under ./outputs)")
    parser.add_argument("--requirements", default="requirements.txt", help="Requirements file to install")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install of requirements")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R preprocessing scripts")
    args = parser.parse_args()

    # Build debug/master.py for inspection
    create_master_file(THIS_DIR)

    # Install Python deps (quiet)
    if not args.skip_install and args.requirements:
        install_requirements(args.requirements)

    # Run ATC preprocessing (R)
    if not args.skip_r:
        run_r_scripts()

    # Import main (after potential dependency install)
    import main  # noqa: WPS433  (allow runtime import for pipeline)

    pnf_path = _resolve_input_path(args.pnf)
    esoa_path = _resolve_input_path(args.esoa)

    outdir = _ensure_outputs_dir()
    out_path = outdir / Path(args.out).name

    main.run_all(str(pnf_path), str(esoa_path), str(outdir), str(out_path))


if __name__ == "__main__":
    try:
        main_entry()
    except Exception as e:  # Re-raise after printing for CI visibility
        print(f"ERROR: {e}", file=sys.stderr)
        raise
