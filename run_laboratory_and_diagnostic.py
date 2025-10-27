#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for the LaboratoryAndDiagnostic pipeline (currently stubbed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipelines import PipelineContext, PipelineOptions, PipelineRunParams, get_pipeline, slugify_item_ref_code

ITEM_REF_CODE = "LaboratoryAndDiagnostic"
PIPELINE_SLUG = slugify_item_ref_code(ITEM_REF_CODE)


def _resolve_esoa_path(esoa_arg: str | None, inputs_dir: Path) -> Path:
    """Determine the Laboratory & Diagnostic eSOA source CSV."""
    if esoa_arg:
        candidate = Path(esoa_arg)
    else:
        candidate = inputs_dir / "LabAndDx.csv"

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
        description="Run the LaboratoryAndDiagnostic pipeline (experimental stub).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--esoa", default=None, help="Path to Laboratory & Diagnostic CSV (defaults to inputs/laboratory_and_diagnostic/LabAndDx.csv)")
    parser.add_argument("--outdir", default=None, help="Destination directory for outputs (defaults to ./outputs/laboratory_and_diagnostic)")
    parser.add_argument("--out", default="laboratory_and_diagnostic_matched.csv", help="Matched CSV filename")
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

    esoa_path = _resolve_esoa_path(args.esoa, inputs_dir)
    out_csv = outputs_dir / Path(args.out).name

    pipeline = get_pipeline(ITEM_REF_CODE)
    context = PipelineContext(project_root=project_root, inputs_dir=inputs_dir, outputs_dir=outputs_dir)
    params = PipelineRunParams(annex_csv=None, pnf_csv=None, esoa_csv=esoa_path, out_csv=out_csv)
    options = PipelineOptions(skip_excel=args.skip_excel, extra={"out_csv": out_csv})

    try:
        pipeline.run(context, params, options)
    except NotImplementedError as exc:
        print(
            f"{ITEM_REF_CODE} pipeline is not yet implemented. {exc}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
