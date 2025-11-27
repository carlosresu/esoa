#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3: Match ESOA rows to ATC codes and DrugBank IDs.

This script:
- Loads prepared ESOA data
- Matches each ESOA description to reference data (PNF, WHO, FDA, DrugBank)
- Assigns ATC codes and DrugBank IDs to ESOA rows
- Outputs esoa_with_atc.csv for use in Part 4

This uses the existing drugs pipeline matching logic from:
- pipelines/drugs/scripts/match_features_drugs.py (molecule detection)
- pipelines/drugs/scripts/match_scoring_drugs.py (scoring and classification)

Prerequisites:
- Run Part 1 (prepare_dependencies) first
- Run Part 2 (annex_f_atc) to have Annex F tagged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from pipelines.drugs.scripts.spinner import run_with_spinner

PROJECT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def run_part_3(
    esoa_path: Optional[str] = None,
    out_filename: str = "esoa_with_atc.csv",
    skip_excel: bool = False,
    standalone: bool = True,
) -> dict:
    """
    Run Part 3: Match ESOA with ATC/DrugBank IDs.
    
    Returns dict with results summary.
    """
    import pandas as pd
    from pipelines.drugs.scripts.match_drugs import match
    from pipelines.drugs.scripts.prepare_drugs import prepare
    
    if standalone:
        print("=" * 60)
        print("Part 3: Match ESOA Rows with ATC/DrugBank IDs")
        print("=" * 60)

    # Resolve ESOA path
    if esoa_path:
        resolved_esoa = Path(esoa_path)
        if not resolved_esoa.is_absolute():
            resolved_esoa = PROJECT_DIR / resolved_esoa
    else:
        resolved_esoa = INPUTS_DIR / "esoa_combined.csv"
        if not resolved_esoa.exists():
            resolved_esoa = INPUTS_DIR / "esoa_prepared.csv"

    if not resolved_esoa.exists():
        raise FileNotFoundError(f"ESOA file not found: {resolved_esoa}")

    # Prepare inputs if needed
    pnf_prepared = INPUTS_DIR / "pnf_prepared.csv"
    esoa_prepared = INPUTS_DIR / "esoa_prepared.csv"
    # Use Part 2 output (annex_f_with_atc.csv) which has ATC codes and parsed info
    annex_prepared = OUTPUTS_DIR / "annex_f_with_atc.csv"
    if not annex_prepared.exists():
        # Fallback to raw annex_f if Part 2 hasn't been run
        annex_prepared = INPUTS_DIR / "annex_f.csv"

    if not pnf_prepared.exists() or not esoa_prepared.exists():
        pnf_csv = INPUTS_DIR / "pnf.csv"
        if not pnf_csv.exists():
            raise FileNotFoundError(f"PNF source not found: {pnf_csv}")
        run_with_spinner(
            "Prepare PNF and ESOA",
            lambda: prepare(str(pnf_csv), str(resolved_esoa), str(INPUTS_DIR))
        )

    # Load reference catalogues
    def _load_catalogues():
        from pipelines.drugs.scripts.match_drugs import _assemble_reference_catalogue
        annex_df = pd.read_csv(annex_prepared)
        pnf_df = pd.read_csv(pnf_prepared)
        
        # Map Part 2 output columns to expected column names if needed
        if "Drug Description" in annex_df.columns and "raw_description" not in annex_df.columns:
            annex_df["raw_description"] = annex_df["Drug Description"]
            annex_df["normalized_description"] = annex_df.get("fuzzy_basis", annex_df["Drug Description"])
            annex_df["generic_name"] = annex_df.get("matched_generic_name", annex_df.get("parsed_molecules", ""))
            annex_df["drug_code"] = annex_df["Drug Code"]
            # Add missing columns with defaults
            for col in ["route_allowed", "form_token", "dose_kind", "strength", "unit", 
                       "per_val", "per_unit", "pct", "strength_mg", "ratio_mg_per_ml", "route_evidence"]:
                if col not in annex_df.columns:
                    annex_df[col] = ""
        
        return _assemble_reference_catalogue(annex_df, pnf_df)
    
    reference_df = run_with_spinner("Load reference catalogues", _load_catalogues)
    esoa_df = run_with_spinner("Load ESOA prepared CSV", lambda: pd.read_csv(esoa_prepared))

    # Build features
    from pipelines.drugs.scripts.match_features_drugs import build_features
    
    # The build_features function has its own spinners, so we call it directly
    features_df = build_features(reference_df, esoa_df, timing_hook=None)

    # Score & classify
    from pipelines.drugs.scripts.match_scoring_drugs import score_and_classify
    
    out_df = run_with_spinner(
        "Score & classify matches",
        lambda: score_and_classify(features_df, reference_df)
    )

    # Write outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / out_filename

    from pipelines.drugs.scripts.match_outputs_drugs import write_outputs
    
    # write_outputs has its own spinners
    write_outputs(out_df, str(out_path), timing_hook=None, skip_excel=skip_excel)

    # Summary
    total = len(out_df)
    results = {
        "total": total,
        "output_path": out_path,
    }
    
    if standalone:
        print(f"\nESO matching complete: {out_path}")
        print(f"  Total rows: {total}")
    
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 3: Match ESOA rows to ATC codes and DrugBank IDs."
    )
    parser.add_argument(
        "--esoa",
        metavar="PATH",
        help="Path to ESOA CSV (default: inputs/drugs/esoa_combined.csv).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default="esoa_with_atc.csv",
        help="Output filename (default: esoa_with_atc.csv).",
    )
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Skip Excel output generation.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_part_3(
        esoa_path=args.esoa,
        out_filename=args.out,
        skip_excel=args.skip_excel,
        standalone=True,
    )
    
    print("\nNext: Run Part 4 to bridge ESOA to Annex F Drug Codes")


if __name__ == "__main__":
    main(sys.argv[1:])
