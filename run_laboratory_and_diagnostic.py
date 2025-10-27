#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for the LaboratoryAndDiagnostic pipeline (Laboratory & Diagnostic normalization)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipelines import PipelineContext, PipelineOptions, PipelineRunParams, get_pipeline, slugify_item_ref_code

ITEM_REF_CODE = "LaboratoryAndDiagnostic"
PIPELINE_SLUG = slugify_item_ref_code(ITEM_REF_CODE)


def _resolve_esoa_path(esoa_arg: str, inputs_dir: Path) -> Path:
    """Resolve a user-provided Laboratory & Diagnostic CSV path."""
    candidate = Path(esoa_arg)
    if not candidate.is_absolute():
        candidate_cwd = (Path.cwd() / candidate).resolve()
        if candidate_cwd.is_file():
            candidate = candidate_cwd
        else:
            candidate = (inputs_dir / candidate).resolve()

    if not candidate.is_file():
        raise FileNotFoundError(
            f"Laboratory & Diagnostic eSOA source not found at {candidate}. "
            f"Place the CSV under {inputs_dir} or pass --esoa with a valid path."
        )
    return candidate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the LaboratoryAndDiagnostic pipeline (Lab & DX matching scaffold).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--esoa", default=None, help="Optional additional Laboratory & Diagnostic CSV to merge with the raw eSOA sources")
    parser.add_argument("--outdir", default=None, help="Destination directory for outputs (defaults to ./outputs/laboratory_and_diagnostic)")
    parser.add_argument("--out", default="esoa_matched_labs.csv", help="Matched CSV filename")
    parser.add_argument("--skip-excel", action="store_true", help="Skip XLSX export when the pipeline adds support")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent
    inputs_dir = (project_root / "inputs" / PIPELINE_SLUG).resolve()
    outputs_dir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (project_root / "outputs" / PIPELINE_SLUG).resolve()
    )
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    esoa_path = _resolve_esoa_path(args.esoa, inputs_dir) if args.esoa else None
    out_csv = outputs_dir / Path(args.out).name

    pipeline = get_pipeline(ITEM_REF_CODE)
    context = PipelineContext(project_root=project_root, inputs_dir=inputs_dir, outputs_dir=outputs_dir)
    params = PipelineRunParams(annex_csv=None, pnf_csv=None, esoa_csv=esoa_path, out_csv=out_csv)
    options = PipelineOptions(skip_excel=args.skip_excel, extra={"out_csv": out_csv})

    pipeline.run(context, params, options)


if __name__ == "__main__":
    main()
