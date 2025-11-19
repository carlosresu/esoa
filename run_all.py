#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refresh core drug reference datasets (PNF, WHO, FDA, DrugBank) in sequence."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from pipelines.drugs.constants import PIPELINE_INPUTS_DIR, PROJECT_ROOT
from pipelines.drugs.pipeline import DrugsAndMedicinePipeline
from pipelines.drugs.scripts.prepare_drugs import prepare
from pipelines.drugs.scripts.run_drugbank_drugs import main as run_drugbank_generics
from pipelines.drugs.scripts.scrape_fda_food_products_drugs import main as run_fda_food_scraper

PROJECT_DIR = PROJECT_ROOT
DRUGS_INPUTS_DIR = PIPELINE_INPUTS_DIR


def _ensure_inputs_dir() -> Path:
    DRUGS_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return DRUGS_INPUTS_DIR


def _natural_esoa_part_order(path: Path) -> tuple[int, str]:
    """Sort helper that orders esoa_pt_* files by their numeric suffix."""
    for token in path.stem.split("_"):
        if token.isdigit():
            return int(token), path.name
    return sys.maxsize, path.name


def _concatenate_csv(parts: Sequence[Path], dest: Path) -> Path:
    """Concatenate multiple CSV files (identical headers) into dest."""
    header: List[str] | None = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as out_handle:
        writer: csv.writer | None = None
        for part in parts:
            if not part.is_file():
                continue
            with part.open("r", newline="", encoding="utf-8-sig") as in_handle:
                reader = csv.reader(in_handle)
                try:
                    file_header = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = file_header
                    writer = csv.writer(out_handle)
                    writer.writerow(header)
                elif header != file_header:
                    raise ValueError(
                        f"Header mismatch while concatenating {part.name}; expected {header} but found {file_header}."
                    )
                assert writer is not None
                writer.writerows(reader)
    return dest


def _resolve_esoa_source(inputs_dir: Path, esoa_hint: Optional[str]) -> Path:
    """Resolve the eSOA CSV path, concatenating esoa_pt_* files when present."""
    search_dirs: List[Path] = []
    if esoa_hint:
        hint = Path(esoa_hint)
        if not hint.is_absolute():
            hint = (PROJECT_DIR / hint).resolve()
        if hint.is_dir():
            search_dirs.append(hint)
        elif hint.is_file():
            return hint
        else:
            candidate = inputs_dir / hint.name
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"Unable to resolve eSOA input at {hint}")
    search_dirs.append(inputs_dir)
    seen: set[Path] = set()
    for directory in search_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        part_files = sorted(directory.glob("esoa_pt_*.csv"), key=_natural_esoa_part_order)
        if part_files:
            return _concatenate_csv(part_files, directory / "esoa_combined.csv")
        for name in ("esoa_combined.csv", "esoa.csv", "esoa_prepared.csv"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"No eSOA CSV found. Provide esoa_pt_*.csv files or esoa.csv under {inputs_dir} (or use --esoa)."
    )


def _ensure_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} missing at {path}")
    return path


def _find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def refresh_pnf(esoa_hint: Optional[str]) -> Path:
    """Run pipelines.drugs.scripts.prepare_drugs against the current PNF + eSOA inputs."""
    inputs_dir = _ensure_inputs_dir()
    pnf_csv = _ensure_file(inputs_dir / "pnf.csv", "PNF source CSV")
    esoa_csv = _resolve_esoa_source(inputs_dir, esoa_hint)
    print(f"[pnf] Preparing PNF dataset from {pnf_csv} and {esoa_csv}")
    pnf_out, _ = prepare(str(pnf_csv), str(esoa_csv), str(inputs_dir))
    out_path = Path(pnf_out).resolve()
    print(f"[pnf] Wrote normalized dataset to {out_path}")
    return out_path


def refresh_who(inputs_dir: Path) -> Path:
    """Trigger the WHO ATC R scripts and return the freshest molecules export."""
    print("[who] Running dependencies/atcd R scripts...")
    DrugsAndMedicinePipeline._run_r_scripts(PROJECT_DIR, inputs_dir)
    latest = _find_latest_file(inputs_dir, "who_atc_*_molecules.csv")
    if latest is None:
        raise FileNotFoundError("WHO ATC export not found after running the R scripts.")
    print(f"[who] Latest molecules export: {latest}")
    return latest


def refresh_fda_brand_map(inputs_dir: Path) -> Path:
    print("[fda_drug] Building FDA brand map export...")
    path = DrugsAndMedicinePipeline._build_brand_map(inputs_dir)
    print(f"[fda_drug] Brand map available at {path}")
    return path


def refresh_fda_food(inputs_dir: Path, quiet: bool = True) -> Path:
    print("[fda_food] Scraping FDA PH food catalog...")
    argv: List[str] = ["--outdir", str(inputs_dir)]
    if quiet:
        argv.append("--quiet")
    code = run_fda_food_scraper(argv)
    if code not in (0, None):
        raise RuntimeError(f"FDA food scraper exited with status {code}")
    out_path = inputs_dir / "fda_food_products.csv"
    if not out_path.is_file():
        raise FileNotFoundError(f"Expected FDA food output at {out_path} but it was not created.")
    print(f"[fda_food] Catalog refreshed at {out_path}")
    return out_path


def refresh_drugbank_generics_exports() -> tuple[Optional[Path], Optional[Path]]:
    """Invoke the DrugBank R helper to regenerate generics + brand exports."""
    print("[drugbank_generics] Launching dependencies/drugbank_generics/drugbank.R...")
    run_drugbank_generics()
    inputs_generics = DRUGS_INPUTS_DIR / "drugbank_generics.csv"
    inputs_brands = DRUGS_INPUTS_DIR / "drugbank_brands.csv"
    if not inputs_generics.exists():
        print(f"[drugbank_generics] Warning: {inputs_generics} not found after refresh.")
    if not inputs_brands.exists():
        print(f"[drugbank_generics] Warning: {inputs_brands} not found after refresh.")
    return (
        inputs_generics if inputs_generics.is_file() else None,
        inputs_brands if inputs_brands.is_file() else None,
    )


def run_drugbank_mixtures() -> Path:
    script_path = PROJECT_DIR / "dependencies" / "drugbank_generics" / "drugbank_mixtures.R"
    if not script_path.is_file():
        raise FileNotFoundError(f"DrugBank mixtures R script not found at {script_path}")
    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript executable not found on PATH.")
    print(f"[drugbank_mixtures] Running {script_path}...")
    subprocess.run([rscript, str(script_path)], check=True, cwd=str(script_path.parent))
    output_path = DRUGS_INPUTS_DIR / "drugbank_mixtures_master.csv"
    if not output_path.is_file():
        raise FileNotFoundError(
            f"DrugBank mixtures output not found at {output_path}. Check the R script logs for details."
        )
    print(f"[drugbank_mixtures] Output written to {output_path}")
    return output_path


def _maybe_run_drugbank_brands_script(include_flag: bool) -> None:
    if not include_flag:
        return
    script_path = PROJECT_DIR / "dependencies" / "drugbank_generics" / "drugbank_brands.R"
    if not script_path.is_file():
        print(f"[drugbank_brands] Placeholder script not found at {script_path}; skipping.")
        return
    rscript = shutil.which("Rscript")
    if not rscript:
        print("[drugbank_brands] Rscript executable not found; cannot run placeholder.")
        return
    print(f"[drugbank_brands] Executing placeholder R script {script_path}...")
    subprocess.run([rscript, str(script_path)], check=True, cwd=str(script_path.parent))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the reference datasets required by the DrugsAndMedicine pipeline."
    )
    parser.add_argument(
        "--esoa",
        metavar="PATH",
        help="Optional path or directory containing the eSOA CSV. Defaults to inputs/drugs.",
    )
    parser.add_argument(
        "--include-fda-food",
        action="store_true",
        help="Scrape the FDA PH food catalog in addition to the default steps.",
    )
    parser.add_argument(
        "--include-drugbank-brands",
        action="store_true",
        help="Attempt to run the (future) DrugBank brands pipeline.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    inputs_dir = _ensure_inputs_dir()
    artifacts: dict[str, Path] = {}

    artifacts["pnf_prepared"] = refresh_pnf(args.esoa)
    artifacts["who_molecules"] = refresh_who(inputs_dir)
    if args.include_fda_food:
        artifacts["fda_food_catalog"] = refresh_fda_food(inputs_dir)
    artifacts["fda_brand_map"] = refresh_fda_brand_map(inputs_dir)
    generics_path, brands_path = refresh_drugbank_generics_exports()
    if generics_path:
        artifacts["drugbank_generics"] = generics_path
    if brands_path:
        artifacts["drugbank_brands_csv"] = brands_path
    artifacts["drugbank_mixtures"] = run_drugbank_mixtures()
    _maybe_run_drugbank_brands_script(args.include_drugbank_brands)

    print("\nArtifacts updated:")
    for label, path in artifacts.items():
        print(f"  - {label}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
