#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — Install requirements, optionally run ATC (R) preprocessing,
then build the FDA brand map (CSV export), then run the ESOA pipeline (prepare → match).

Scraper module path: scripts/fda_ph_drug_scraper.py (executed as a module)
Brand map outputs to: ./inputs/fda_brand_map_YYYY-MM-DD.csv
Prepared files (pnf_prepared.csv, esoa_prepared.csv) are written to: ./inputs
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

THIS_DIR: Path = Path(__file__).resolve().parent
DEFAULT_INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
ATCD_SUBDIR = Path("dependencies") / "atcd"
ATCD_SCRIPTS: tuple[str, ...] = ("atcd.R", "export.R", "filter.R")

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

def install_requirements(req_path: str | os.PathLike[str]) -> None:
    req = Path(req_path) if req_path else None
    if not req or not req.is_file():
        return
    with open(os.devnull, "w") as devnull:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)],
            cwd=str(THIS_DIR),
            stdout=devnull,
            stderr=devnull,
        )

def _resolve_input_path(p: str | os.PathLike[str], default_subdir: str = DEFAULT_INPUTS_DIR) -> Path:
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
    outdir = THIS_DIR / OUTPUTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _ensure_inputs_dir() -> Path:
    inp = THIS_DIR / DEFAULT_INPUTS_DIR
    inp.mkdir(parents=True, exist_ok=True)
    return inp

def _assert_all_exist(root: Path, files: Iterable[str | os.PathLike[str]]) -> None:
    for f in files:
        fp = root / f
        if not fp.is_file():
            raise FileNotFoundError(f"Required file not found: {fp}")

def run_r_scripts() -> None:
    atcd_dir = THIS_DIR / ATCD_SUBDIR
    if not atcd_dir.is_dir():
        print(f">>> ATC directory not found: {atcd_dir}; skipping R preprocessing.")
        return
    out_dir = atcd_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    rscript = shutil.which("Rscript")
    if not rscript:
        print(">>> Rscript not found in PATH; skipping ATC R preprocessing.")
        return
    try:
        _assert_all_exist(atcd_dir, ATCD_SCRIPTS)
    except FileNotFoundError as e:
        print(f">>> {e}; skipping ATC R preprocessing.")
        return
    with open(os.devnull, "w") as devnull:
        for script in ATCD_SCRIPTS:
            subprocess.run([rscript, script], check=True, cwd=str(atcd_dir), stdout=devnull, stderr=devnull)

def create_master_file(root_dir: Path) -> None:
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
        root_dir / "scripts" / "fda_ph_drug_scraper.py",
        root_dir / "scripts" / "brand_map.py",
        root_dir / "main.py",
        root_dir / "run.py",
    ]
    header_text = "# START OF REPO FILES"
    footer_text = "# END OF REPO FILES"
    with output_file_path.open("w", encoding="utf-8") as outfile:
        outfile.write(header_text)
        for file_path in files_to_concatenate:
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(root_dir).as_posix()
            outfile.write(f"\\n# <{relative_path}>\\n")
            with file_path.open("r", encoding="utf-8") as infile:
                outfile.write(infile.read())
            outfile.write("\\n")
        outfile.write(footer_text)

def build_brand_map(inputs_dir: Path, outfile: Path | None) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_csv = outfile or (inputs_dir / f"fda_brand_map_{date_str}.csv")
    with open(os.devnull, "w") as devnull:
        subprocess.run(
            [sys.executable, "-m", "scripts.fda_ph_drug_scraper", "--outdir", str(inputs_dir), "--outfile", str(out_csv)],
            check=True,
            cwd=str(THIS_DIR),
            stdout=devnull,
            stderr=devnull,
        )
    return out_csv

def main_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Run full ESOA pipeline (ATC → build brand map → prepare → match)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pnf", default=f"{DEFAULT_INPUTS_DIR}/pnf.csv", help="Path to PNF CSV")
    parser.add_argument("--esoa", default=f"{DEFAULT_INPUTS_DIR}/esoa.csv", help="Path to eSOA CSV")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV filename (saved under ./outputs)")
    parser.add_argument("--requirements", default="requirements.txt", help="Requirements file to install")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install of requirements")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R preprocessing scripts")
    parser.add_argument("--skip-brandmap", action="store_true", help="Skip building FDA brand map CSV")
    args = parser.parse_args()

    create_master_file(THIS_DIR)

    if not args.skip_install and args.requirements:
        install_requirements(args.requirements)

    if not args.skip_r:
        run_r_scripts()

    outdir = _ensure_outputs_dir()
    inputs_dir = _ensure_inputs_dir()

    if not args.skip_brandmap:
        try:
            bm = build_brand_map(inputs_dir, outfile=None)
            print(f">>> Built FDA brand map: {bm}")
        except Exception as e:
            print(f">>> FDA brand map build failed: {e}")

    import main  # noqa: WPS433

    pnf_path = _resolve_input_path(args.pnf)
    esoa_path = _resolve_input_path(args.esoa)

    out_path = outdir / Path(args.out).name
    # IMPORTANT: write prepared CSVs into ./inputs (not ./outputs)
    main.run_all(str(pnf_path), str(esoa_path), str(inputs_dir), str(out_path))

    # NOTE: We intentionally removed the extra summary print to avoid duplicate summaries.
    # The detailed summary already comes from scripts.match_outputs.write_outputs().

if __name__ == "__main__":
    try:
        main_entry()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
