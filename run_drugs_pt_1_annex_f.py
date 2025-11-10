#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 runner: prepare and preview Annex F prior to the full Drugs pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_drugs_all_parts as run_dm


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare annex_f_prepared.csv and display a quick preview.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annex",
        default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "annex_f.csv"),
        help="Path to the raw Annex F CSV.",
    )
    parser.add_argument(
        "--out",
        default=str(run_dm.PIPELINE_INPUTS_SUBDIR / "annex_f_prepared.csv"),
        help="Destination CSV for the prepared Annex F output.",
    )
    args = parser.parse_args(argv)

    inputs_dir = run_dm._ensure_inputs_dir()
    annex_path = run_dm._resolve_input_path(args.annex)
    out_path = run_dm._resolve_repo_path(args.out)
    run_dm._prepare_annex_core(annex_path, inputs_dir, output_path=out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
