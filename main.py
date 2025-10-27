# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py — Category-aware pipeline loader around the eSOA matching workflows.

Exports:
- prepare, match, prepare_annex_f (compatibility re-export from scripts/)
- run_all(annex_csv, pnf_csv, esoa_csv, outdir, out_csv, item_ref_code) -> out_csv
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from pipelines import (
    PipelineContext,
    PipelineOptions,
    PipelineRunParams,
    get_pipeline,
    slugify_item_ref_code,
)
from pipelines.drugs.scripts.prepare_drugs import prepare  # re-exported for backward compatibility
from pipelines.drugs.scripts.match_drugs import match  # re-exported for backward compatibility
from pipelines.drugs.scripts.prepare_annex_f_drugs import prepare_annex_f  # re-exported for backward compatibility


def _resolve_path(
    path: str | os.PathLike[str] | None,
    *,
    base_dir: Path,
    fallback_dir: Optional[Path] = None,
    default_name: Optional[str] = None,
) -> Optional[Path]:
    if path is None:
        if default_name is None:
            return None
        candidate = base_dir / default_name
        if candidate.exists():
            return candidate.resolve()
        return candidate

    candidate = Path(path)
    search_paths: list[Path] = []

    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.append(base_dir / candidate)
        if fallback_dir is not None:
            search_paths.append(fallback_dir / candidate)
        search_paths.append(Path.cwd() / candidate)
        search_paths.append(candidate)

    for option in search_paths:
        if option.exists():
            return option.resolve()

    # Fall back to the first search path even if it does not yet exist; callers can handle creation.
    return search_paths[0].resolve()


def run_all(
    annex_csv: str | os.PathLike[str] | None,
    pnf_csv: str | os.PathLike[str] | None,
    esoa_csv: str | os.PathLike[str],
    outdir: str | os.PathLike[str] | None = None,
    out_csv: str | os.PathLike[str] = "esoa_matched.csv",
    *,
    item_ref_code: str = "DrugsAndMedicine",
    skip_excel: bool = False,
) -> str:
    """Execute the registered pipeline for the requested ITEM_REF_CODE."""
    project_root = Path(__file__).resolve().parent
    slug = slugify_item_ref_code(item_ref_code)

    inputs_dir = (project_root / "inputs" / slug).resolve()
    inputs_dir.mkdir(parents=True, exist_ok=True)

    if outdir is None:
        output_dir = (project_root / "outputs" / slug).resolve()
    else:
        output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = get_pipeline(item_ref_code)
    fallback_inputs = project_root / "inputs"

    annex_default = "annex_f.csv" if item_ref_code == "DrugsAndMedicine" else None
    pnf_default = "pnf.csv" if item_ref_code == "DrugsAndMedicine" else None
    esoa_default = "esoa_combined.csv" if item_ref_code == "DrugsAndMedicine" else None

    params = PipelineRunParams(
        annex_csv=_resolve_path(annex_csv, base_dir=inputs_dir, fallback_dir=fallback_inputs, default_name=annex_default),
        pnf_csv=_resolve_path(pnf_csv, base_dir=inputs_dir, fallback_dir=fallback_inputs, default_name=pnf_default),
        esoa_csv=_resolve_path(esoa_csv, base_dir=inputs_dir, fallback_dir=fallback_inputs, default_name=esoa_default),
        out_csv=(output_dir / Path(out_csv).name),
    )
    context = PipelineContext(
        project_root=project_root,
        inputs_dir=inputs_dir,
        outputs_dir=output_dir,
    )
    options = PipelineOptions(
        skip_excel=skip_excel,
        extra={"out_csv": params.out_csv},
    )
    result = pipeline.run(context, params, options)
    return str(result.matched_csv)


def _cli() -> None:
    """Parse CLI arguments and run the requested pipeline."""
    parser = argparse.ArgumentParser(description="Modular eSOA matching pipeline runner")
    parser.add_argument("--annex", required=False)
    parser.add_argument("--pnf", required=False)
    parser.add_argument("--esoa", required=False)
    parser.add_argument("--outdir", required=False)
    parser.add_argument("--out", required=False, default="esoa_matched.csv")
    parser.add_argument(
        "--item-ref-code",
        required=False,
        default="DrugsAndMedicine",
        help="ITEM_REF_CODE category to process (registered pipelines only)",
    )
    parser.add_argument("--skip-excel", action="store_true", help="Skip XLSX export when supported by the pipeline")
    args = parser.parse_args()

    run_all(
        args.annex,
        args.pnf,
        args.esoa,
        args.outdir,
        args.out,
        item_ref_code=args.item_ref_code,
        skip_excel=args.skip_excel,
    )


if __name__ == "__main__":
    _cli()
