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
)
from scripts.prepare import prepare  # re-exported for backward compatibility
from scripts.match import match  # re-exported for backward compatibility
from scripts.prepare_annex_f import prepare_annex_f  # re-exported for backward compatibility


def _resolve_path(
    path: str | os.PathLike[str] | None,
    *,
    base_dir: Path,
    fallback_dir: Optional[Path] = None,
) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        base_candidate = (base_dir / candidate)
        if base_candidate.exists():
            candidate = base_candidate.resolve()
        elif fallback_dir is not None:
            fb_candidate = (fallback_dir / candidate)
            if fb_candidate.exists():
                candidate = fb_candidate.resolve()
            else:
                candidate = base_candidate.resolve()
        else:
            candidate = base_candidate.resolve()
    return candidate


def run_all(
    annex_csv: str | os.PathLike[str] | None,
    pnf_csv: str | os.PathLike[str] | None,
    esoa_csv: str | os.PathLike[str],
    outdir: str | os.PathLike[str] = ".",
    out_csv: str | os.PathLike[str] = "esoa_matched.csv",
    *,
    item_ref_code: str = "DrugsAndMedicine",
    skip_excel: bool = False,
) -> str:
    """Execute the registered pipeline for the requested ITEM_REF_CODE."""
    project_root = Path(__file__).resolve().parent
    output_dir = Path(outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = get_pipeline(item_ref_code)
    params = PipelineRunParams(
        annex_csv=_resolve_path(annex_csv, base_dir=Path.cwd(), fallback_dir=output_dir),
        pnf_csv=_resolve_path(pnf_csv, base_dir=Path.cwd(), fallback_dir=output_dir),
        esoa_csv=_resolve_path(esoa_csv, base_dir=Path.cwd(), fallback_dir=output_dir),
        out_csv=(output_dir / Path(out_csv).name),
    )
    context = PipelineContext(
        project_root=project_root,
        inputs_dir=output_dir,
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
    parser.add_argument("--annex", required=False, default="annex_f.csv")
    parser.add_argument("--pnf", required=False, default="pnf.csv")
    parser.add_argument("--esoa", required=False, default="esoa.csv")
    parser.add_argument("--outdir", required=False, default=".")
    parser.add_argument("--out", required=False, default="esoa_matched.csv")
    parser.add_argument(
        "--item-ref-code",
        required=False,
        default="DrugsAndMedicine",
        help="ITEM_REF_CODE category to process (registered pipelines only)",
    )
    parser.add_argument("--skip-excel", action="store_true", help="Skip XLSX export when supported by the pipeline")
    args = parser.parse_args()

    annex_path = Path(args.annex)
    if not annex_path.is_absolute():
        annex_path = Path(args.outdir) / annex_path
    run_all(
        annex_path,
        args.pnf,
        args.esoa,
        args.outdir,
        args.out,
        item_ref_code=args.item_ref_code,
        skip_excel=args.skip_excel,
    )


if __name__ == "__main__":
    _cli()
