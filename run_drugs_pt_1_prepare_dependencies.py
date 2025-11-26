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

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from run_all import (
    _ensure_inputs_dir,
    _run_with_spinner,
    refresh_pnf,
    refresh_who,
    refresh_fda_brand_map,
    refresh_fda_food,
    refresh_drugbank_generics_exports,
    ensure_drugbank_mixtures_output,
    _ensure_parquet_sibling,
    _ensure_file,
    PROJECT_DIR,
    DRUGS_INPUTS_DIR,
)


def ensure_annex_f(inputs_dir: Path, *, verbose: bool = True) -> Path:
    """Ensure Annex F CSV exists and has a Parquet sibling."""
    annex_path = inputs_dir / "annex_f.csv"
    if not annex_path.is_file():
        raise FileNotFoundError(
            f"Annex F CSV not found at {annex_path}. "
            "Please provide a normalized annex_f.csv in inputs/drugs/."
        )
    _ensure_parquet_sibling(annex_path, verbose=verbose)
    if verbose:
        print(f"[annex_f] Verified at {annex_path}")
    return annex_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 1: Prepare all reference dependencies for the drugs pipeline."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress messages.",
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

    inputs_dir = _ensure_inputs_dir()
    artifacts: dict[str, Path] = {}
    verbose = args.verbose

    print("=" * 60)
    print("Part 1: Prepare Dependencies")
    print("=" * 60)

    # 1. WHO ATC
    if not args.skip_who:
        artifacts["who_molecules"] = _run_with_spinner(
            "Refresh WHO ATC exports", lambda: refresh_who(inputs_dir, verbose=verbose)
        )
    else:
        print("[skip] WHO ATC exports")

    # 2. DrugBank
    if not args.skip_drugbank:
        generics_path, brands_path = _run_with_spinner(
            "Refresh DrugBank generics exports",
            lambda: refresh_drugbank_generics_exports(verbose=verbose),
        )
        if generics_path:
            artifacts["drugbank_generics"] = generics_path
        mixtures_path = _run_with_spinner(
            "Check DrugBank mixtures output",
            lambda: ensure_drugbank_mixtures_output(verbose=verbose),
        )
        if mixtures_path:
            artifacts["drugbank_mixtures"] = mixtures_path
    else:
        print("[skip] DrugBank generics/mixtures")

    # 3. FDA Brand Map
    if not args.skip_fda_brand:
        artifacts["fda_brand_map"] = _run_with_spinner(
            "Build FDA brand map", lambda: refresh_fda_brand_map(inputs_dir, verbose=verbose)
        )
    else:
        print("[skip] FDA brand map")

    # 4. FDA Food
    if not args.skip_fda_food:
        artifacts["fda_food_catalog"] = _run_with_spinner(
            "Refresh FDA food catalog",
            lambda: refresh_fda_food(
                inputs_dir,
                allow_scrape=args.allow_fda_food_scrape,
                verbose=verbose,
            ),
        )
    else:
        print("[skip] FDA food catalog")

    # 5. PNF
    if not args.skip_pnf:
        artifacts["pnf_prepared"] = _run_with_spinner(
            "Prepare PNF dataset", lambda: refresh_pnf(args.esoa, verbose=verbose)
        )
    else:
        print("[skip] PNF preparation")

    # 6. Annex F (just verify it exists)
    artifacts["annex_f"] = _run_with_spinner(
        "Verify Annex F", lambda: ensure_annex_f(inputs_dir, verbose=verbose)
    )

    # Ensure Parquet siblings
    for path in artifacts.values():
        if path and path.suffix.lower() == ".csv":
            _ensure_parquet_sibling(path, verbose=verbose)

    print("\n" + "=" * 60)
    print("Part 1 Complete: Dependencies Prepared")
    print("=" * 60)
    for label, path in artifacts.items():
        print(f"  - {label}: {path}")
    print("\nNext: Run Part 2 (match_annex_f_with_atc) to tag Annex F with ATC/DrugBank IDs")


if __name__ == "__main__":
    main(sys.argv[1:])
