#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequentially invoke registered ITEM_REF_CODE pipelines."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List

from pipelines import list_pipelines


def _default_pipeline_order() -> List[str]:
    """Return a deterministic ordering of ITEM_REF_CODE values."""
    return [pipeline.item_ref_code for pipeline in list_pipelines()]


def _run_drugs_and_medicine() -> None:
    import run_drugs_and_medicine

    original = sys.argv
    try:
        sys.argv = ["run_drugs_and_medicine.py"]
        run_drugs_and_medicine.main_entry()
    finally:
        sys.argv = original


def _run_laboratory_and_diagnostic() -> None:
    import run_laboratory_and_diagnostic as lab_runner

    lab_runner.main([])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple ITEM_REF_CODE pipelines sequentially (defaults to all registered).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        help="Subset of ITEM_REF_CODE values to run (defaults to all registered pipelines).",
    )
    parser.add_argument(
        "--include-stubs",
        action="store_true",
        help="(Reserved) Attempt to run any pipelines still marked as stubs.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    selected = args.pipelines or _default_pipeline_order()
    for code in selected:
        print(f"=== Running pipeline: {code} ===")
        if code == "DrugsAndMedicine":
            _run_drugs_and_medicine()
            continue
        if code == "LaboratoryAndDiagnostic":
            _run_laboratory_and_diagnostic()
            continue
        print(f"Unknown ITEM_REF_CODE '{code}'. Skipping.")


if __name__ == "__main__":
    main(sys.argv[1:])
