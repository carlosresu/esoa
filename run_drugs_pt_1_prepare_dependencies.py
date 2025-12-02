#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 1: Prepare all reference dependencies for the drugs pipeline.

This script refreshes:
- WHO ATC (via R scripts in dependencies/atcd)
- DrugBank generics/mixtures (via R scripts in dependencies/drugbank_generics)
- FDA brand map (from FDA drug exports)
- FDA food catalog (from FDA food exports)
- PNF lexicon (normalize and parse dose/route/form)
- Annex F (ensure normalized)

Run this before the matching steps to ensure all reference data is fresh.
"""
from __future__ import annotations

# === Auto-activate virtual environment ===
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_VENV_PYTHON = _SCRIPT_DIR / ".venv" / "bin" / "python"

if _VENV_PYTHON.exists() and sys.executable != str(_VENV_PYTHON):
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), __file__] + sys.argv[1:])
# === End auto-activate ===

import argparse
from typing import Optional, Sequence

# Sync shared scripts to submodules before running
from pipelines.drugs.scripts.sync_to_submodules import sync_all
sync_all()

from pipelines.drugs.scripts.spinner import run_with_spinner


def run_part_1(
    esoa_path: Optional[str] = None,
    skip_who: bool = False,
    skip_drugbank: bool = False,
    skip_fda_brand: bool = False,
    skip_fda_food: bool = False,
    skip_pnf: bool = False,
    allow_fda_food_scrape: bool = False,
    standalone: bool = True,
) -> dict[str, Path]:
    """
    Run Part 1: Prepare all dependencies.
    
    Returns dict of artifact paths.
    """
    # Import here to avoid circular imports
    from run_drugs_all import (
        PROJECT_ROOT,
        _ensure_inputs_dir,
        _ensure_parquet_sibling,
        refresh_pnf,
        refresh_who,
        refresh_fda_brand_map,
        refresh_fda_food,
        refresh_drugbank_generics_exports,
        ensure_drugbank_mixtures_output,
    )
    
    if standalone:
        print("=" * 60)
        print("Part 1: Prepare Dependencies")
        print("=" * 60)
    
    project_root = PROJECT_ROOT
    inputs_dir = _ensure_inputs_dir()
    artifacts: dict[str, Path] = {}

    # 1. WHO ATC
    if not skip_who:
        artifacts["who_molecules"] = run_with_spinner(
            "Refresh WHO ATC exports", lambda: refresh_who(inputs_dir, verbose=False)
        )
    elif standalone:
        print("[skip] WHO ATC exports")

    # 2. DrugBank (runs each R script via native shell with live spinner/timer)
    if not skip_drugbank:
        # Each script: os.system() in thread + live timer, cores-1 workers
        generics_path, brands_path = refresh_drugbank_generics_exports(verbose=False)
        if generics_path:
            artifacts["drugbank_generics"] = generics_path
        # Quick check, no spinner needed
        mixtures_path = ensure_drugbank_mixtures_output(verbose=False)
        if mixtures_path:
            artifacts["drugbank_mixtures"] = mixtures_path
    elif standalone:
        print("[skip] DrugBank generics/mixtures")

    # 3. FDA Brand Map
    if not skip_fda_brand:
        artifacts["fda_brand_map"] = run_with_spinner(
            "Build FDA brand map", lambda: refresh_fda_brand_map(inputs_dir, verbose=False)
        )
    elif standalone:
        print("[skip] FDA brand map")

    # 4. FDA Food
    if not skip_fda_food:
        artifacts["fda_food_catalog"] = run_with_spinner(
            "Refresh FDA food catalog",
            lambda: refresh_fda_food(
                inputs_dir,
                allow_scrape=allow_fda_food_scrape,
                verbose=False,
            ),
        )
    elif standalone:
        print("[skip] FDA food catalog")

    # 5. PNF
    if not skip_pnf:
        artifacts["pnf_prepared"] = run_with_spinner(
            "Prepare PNF dataset", lambda: refresh_pnf(esoa_path, verbose=False)
        )
    elif standalone:
        print("[skip] PNF preparation")

    # 6. Annex F (just verify it exists in raw/)
    def _verify_annex_f() -> Path:
        raw_dir = project_root / "raw" / "drugs"
        annex_path = raw_dir / "annex_f.csv"
        if not annex_path.is_file():
            raise FileNotFoundError(
                f"Annex F CSV not found at {annex_path}. "
                "Please provide a normalized annex_f.csv in raw/drugs/."
            )
        _ensure_parquet_sibling(annex_path, verbose=False)
        return annex_path
    
    artifacts["annex_f"] = run_with_spinner("Verify Annex F", _verify_annex_f)

    # Ensure Parquet siblings
    for path in artifacts.values():
        if path and path.suffix.lower() == ".csv":
            _ensure_parquet_sibling(path, verbose=False)

    if standalone:
        print("\nPart 1 artifacts:")
        for label, path in artifacts.items():
            print(f"  - {label}: {path}")
    
    return artifacts


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 1: Prepare all reference dependencies for the drugs pipeline."
    )
    parser.add_argument(
        "--esoa",
        metavar="PATH",
        help="Optional path to eSOA CSV (for PNF preparation).",
    )
    parser.add_argument(
        "--skip-who",
        action="store_true",
        help="Skip WHO ATC R preprocessing.",
    )
    parser.add_argument(
        "--skip-drugbank",
        action="store_true",
        help="Skip DrugBank generics/mixtures refresh.",
    )
    parser.add_argument(
        "--skip-fda-brand",
        action="store_true",
        help="Skip FDA brand map generation.",
    )
    parser.add_argument(
        "--skip-fda-food",
        action="store_true",
        help="Skip FDA food catalog refresh.",
    )
    parser.add_argument(
        "--skip-pnf",
        action="store_true",
        help="Skip PNF preparation.",
    )
    parser.add_argument(
        "--allow-fda-food-scrape",
        action="store_true",
        default=False,
        help="Enable HTML scraping fallback for FDA food.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_part_1(
        esoa_path=args.esoa,
        skip_who=args.skip_who,
        skip_drugbank=args.skip_drugbank,
        skip_fda_brand=args.skip_fda_brand,
        skip_fda_food=args.skip_fda_food,
        skip_pnf=args.skip_pnf,
        allow_fda_food_scrape=args.allow_fda_food_scrape,
        standalone=True,
    )
    
    print("\nNext: Run Part 2 to tag Annex F with ATC/DrugBank IDs")


if __name__ == "__main__":
    main(sys.argv[1:])
