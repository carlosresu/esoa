#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline implementation for ITEM_REF_CODE == 'DrugsAndMedicine'."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from ..base import (
    BasePipeline,
    PipelineContext,
    PipelineOptions,
    PipelinePreparedInputs,
    PipelineResult,
    PipelineRunParams,
    TimingHook,
)
from ..registry import register_pipeline
from .scripts.prepare_drugs import prepare
from .scripts.match_drugs import match

THIS_DIR = Path(__file__).resolve().parents[2]
ATCD_SUBDIR = Path("dependencies") / "atcd"
ATCD_SCRIPTS: tuple[str, ...] = ("atcd.R", "export.R", "filter.R")


def _require_parquet(path: Path, *, label: str) -> Path:
    """
    Prefer a parquet path, but allow CSV when no parquet sibling exists to preserve legacy inputs.

    If a CSV is provided and a parquet sibling exists, the parquet file is used. Otherwise the
    original path is returned so callers can continue with CSV inputs.
    """
    if path.suffix.lower() == ".parquet":
        return path
    parquet_peer = path.with_suffix(".parquet")
    if parquet_peer.is_file():
        return parquet_peer
    return path


@register_pipeline
class DrugsAndMedicinePipeline(BasePipeline):
    """Concrete pipeline for the existing drug matching workflow."""

    item_ref_code = "DrugsAndMedicine"
    display_name = "Drugs and Medicine"
    description = "Matches drug brands and molecules against Annex F + PNF references."

    def pre_run(
        self,
        context: PipelineContext,
        params: PipelineRunParams,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> Mapping[str, Path]:
        artifacts: Dict[str, Path] = {}
        spinner = self._spinner(options)

        if not options.flag("skip_r"):
            elapsed = self._run_stage(
                spinner,
                "ATC R preprocessing",
                lambda: self._run_r_scripts(context.project_root, context.inputs_dir),
            )
            if timing_hook:
                timing_hook("ATC R preprocessing", elapsed)

        if not options.flag("skip_brandmap"):
            elapsed, map_path = self._run_stage_with_result(
                spinner,
                "Build FDA brand map",
                lambda: self._build_brand_map(context.inputs_dir),
            )
            if timing_hook:
                timing_hook("Build FDA brand map", elapsed)
            if map_path:
                artifacts["fda_brand_map"] = map_path

        return artifacts

    def prepare_inputs(
        self,
        context: PipelineContext,
        params: PipelineRunParams,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelinePreparedInputs:
        if params.annex_csv is None or params.pnf_csv is None:
            raise ValueError("DrugsAndMedicine pipeline requires Annex F and PNF CSV inputs.")

        spinner = self._spinner(options)
        prepared_paths: Dict[str, Path] = {}

        def _prepare() -> None:
            annex_path = _require_parquet(Path(params.annex_csv), label="Annex F prepared input")
            if not annex_path.is_file():
                raise FileNotFoundError(
                    f"Annex input not found: {annex_path} (expected a prepared Annex F dataset)."
                )
            if annex_path.suffix.lower() != ".parquet":
                try:
                    import polars as pl  # Local import to avoid hard dependency at import time.

                    annex_parquet = annex_path.with_suffix(".parquet")
                    pl.read_csv(annex_path).write_parquet(annex_parquet)
                    annex_path = annex_parquet
                except Exception:
                    pass
            prepared_paths["annex"] = annex_path.resolve()
            pnf_prep, esoa_prep = prepare(
                str(_require_parquet(Path(params.pnf_csv), label="PNF input")),
                str(_require_parquet(Path(params.esoa_csv), label="eSOA input")),
                str(context.inputs_dir),
            )
            prepared_paths["pnf"] = Path(_require_parquet(Path(pnf_prep), label="PNF prepared output")).resolve()
            prepared_paths["esoa"] = Path(_require_parquet(Path(esoa_prep), label="eSOA prepared output")).resolve()

        elapsed = self._run_stage(spinner, "Prepare inputs", _prepare)
        if timing_hook:
            timing_hook("Prepare inputs", elapsed)

        return PipelinePreparedInputs(
            esoa_csv=prepared_paths["esoa"],
            annex_csv=prepared_paths["annex"],
            pnf_csv=prepared_paths["pnf"],
        )

    def match(
        self,
        context: PipelineContext,
        prepared: PipelinePreparedInputs,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelineResult:
        out_csv = options.extra.get("out_csv")
        if out_csv is None:
            raise ValueError("PipelineOptions.extra must provide 'out_csv' Path for match output.")
        out_path = Path(out_csv)
        matched_path = match(
            str(prepared.annex_csv),
            str(prepared.pnf_csv),
            str(prepared.esoa_csv),
            str(out_path),
            timing_hook=timing_hook,
            skip_excel=options.skip_excel,
        )
        return PipelineResult(
            matched_csv=Path(matched_path),
            prepared=prepared,
        )

    def post_run(
        self,
        context: PipelineContext,
        result: PipelineResult,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> None:
        if options.flag("skip_unknowns"):
            return
        spinner = self._spinner(options)
        elapsed = self._run_stage(
            spinner,
            "Resolve unknowns",
            lambda: self._run_resolve_unknowns(context.project_root),
        )
        if timing_hook:
            timing_hook("Resolve unknowns", elapsed)

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def _spinner(options: PipelineOptions) -> Optional[Callable[[str, Callable[[], None]], float]]:
        spinner = options.extra.get("spinner")
        if callable(spinner):
            return spinner
        return None

    @staticmethod
    def _run_stage(
        spinner: Optional[Callable[[str, Callable[[], None]], float]],
        label: str,
        func: Callable[[], None],
    ) -> float:
        if spinner:
            return spinner(label, func)
        t0 = time.perf_counter()
        func()
        return time.perf_counter() - t0

    @staticmethod
    def _run_stage_with_result(
        spinner: Optional[Callable[[str, Callable[[], None]], float]],
        label: str,
        func: Callable[[], Path],
    ) -> tuple[float, Optional[Path]]:
        result: Dict[str, Path] = {}

        def _wrapper() -> None:
            result["value"] = func()

        elapsed = DrugsAndMedicinePipeline._run_stage(spinner, label, _wrapper)
        return elapsed, result.get("value")

    @staticmethod
    def _run_r_scripts(project_root: Path, dest_inputs_dir: Path) -> None:
        atcd_dir = project_root / ATCD_SUBDIR
        if not atcd_dir.is_dir():
            return
        out_dir = atcd_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        rscript = shutil.which("Rscript")
        if not rscript:
            return
        scripts = [atcd_dir / script for script in ATCD_SCRIPTS]
        if not all(script.is_file() for script in scripts):
            return
        with open(os.devnull, "w") as devnull:
            for script in ATCD_SCRIPTS:
                subprocess.run(
                    [rscript, script],
                    check=True,
                    cwd=str(atcd_dir),
                    stdout=devnull,
                    stderr=devnull,
                )
        # Sync generated WHO ATC exports into the pipeline inputs directory and
        # remove legacy copies left at the repository root.
        dest_inputs_dir.mkdir(parents=True, exist_ok=True)
        root_inputs_dir = project_root / "inputs"

        def _hydrate(dest: Path, source: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Overwrite in-place so the freshest export is used for this run.
            if dest.exists():
                dest.unlink()
            shutil.copy2(source, dest)
            parquet_src = source.with_suffix(".parquet")
            if parquet_src.is_file():
                shutil.copy2(parquet_src, dest.with_suffix(".parquet"))
            elif dest.suffix.lower() == ".csv":
                try:
                    import polars as pl  # Local import to avoid polluting module import time.

                    pl.read_csv(dest).write_parquet(dest.with_suffix(".parquet"))
                except Exception:
                    pass

        patterns = ("who_atc_*_molecules.csv", "who_atc_*_molecules.parquet")
        for pattern in patterns:
            for csv_path in out_dir.glob(pattern):
                dest_name = csv_path.name.replace("_molecules", "")
                _hydrate(dest_inputs_dir / dest_name, csv_path)

            # Move any remnants from the legacy ./inputs/ directory into the
            # pipeline-scoped inputs folder so future runs only see the new layout.
            for legacy_path in root_inputs_dir.glob(pattern):
                target = dest_inputs_dir / legacy_path.name.replace("_molecules", "")
                if target.exists():
                    target.unlink()
                shutil.move(str(legacy_path), str(target))

    @staticmethod
    def _build_brand_map(inputs_dir: Path) -> Path:
        def _pick_latest(paths: list[Path]) -> Optional[Path]:
            if not paths:
                return None
            dated: list[tuple[str, Path]] = []
            for p in paths:
                match = re.search(r"fda_(?:brand_map|drug)_(\\d{4}-\\d{2}-\\d{2})\\.csv$", p.name)
                if match:
                    dated.append((match.group(1), p))
            if dated:
                dated.sort(key=lambda tup: tup[0])
                return dated[-1][1]
            return max(paths, key=lambda p: p.stat().st_mtime)

        existing_maps = sorted(
            list(inputs_dir.glob("fda_drug_*.parquet"))
            + list(inputs_dir.glob("fda_brand_map_*.parquet"))
            + list(inputs_dir.glob("fda_drug_*.csv"))
            + list(inputs_dir.glob("fda_brand_map_*.csv"))
        )
        module_output_dir = THIS_DIR / "dependencies" / "fda_ph_scraper" / "output"
        module_output_dir.mkdir(parents=True, exist_ok=True)
        with open(os.devnull, "w") as devnull:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "dependencies.fda_ph_scraper.drug_scraper",
                        "--outdir",
                        str(module_output_dir),
                    ],
                    check=True,
                    cwd=str(THIS_DIR),
                    stdout=devnull,
                    stderr=devnull,
                )
            except subprocess.CalledProcessError as exc:
                latest_existing = _pick_latest(existing_maps)
                if latest_existing:
                    return latest_existing
                raise RuntimeError(
                    "Building FDA brand map failed and no prior map is available. "
                    "Re-run with --skip-brandmap if the FDA site is unreachable."
                ) from exc
        brand_files = sorted(
            list(module_output_dir.glob("fda_drug_*.parquet"))
            + list(module_output_dir.glob("fda_brand_map_*.parquet"))
            + list(module_output_dir.glob("fda_drug_*.csv"))
            + list(module_output_dir.glob("fda_brand_map_*.csv"))
        )
        selected = _pick_latest(brand_files)
        if not selected:
            latest_existing = _pick_latest(existing_maps)
            if latest_existing:
                return latest_existing
            raise RuntimeError(
                "No FDA brand map was created; rerun with --skip-brandmap once the site is reachable."
            )
        dest = inputs_dir / selected.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected, dest)
        parquet_src = selected.with_suffix(".parquet")
        dest_parquet = dest.with_suffix(".parquet")
        if parquet_src.is_file():
            shutil.copy2(parquet_src, dest_parquet)
        elif dest.suffix.lower() == ".csv":
            try:
                import polars as pl  # Local import to avoid imposing a hard dependency at module import time.

                pl.read_csv(dest).write_parquet(dest_parquet)
            except Exception:
                pass
        return dest_parquet if dest_parquet.is_file() else dest

    @staticmethod
    def _run_resolve_unknowns(project_root: Path) -> None:
        pipeline_path = project_root / "pipelines" / "drugs" / "scripts" / "resolve_unknowns_drugs.py"
        legacy_path = project_root / "resolve_unknowns.py"
        if pipeline_path.is_file():
            mod_name = "pipelines.drugs.scripts.resolve_unknowns_drugs"
        elif legacy_path.is_file():
            mod_name = "resolve_unknowns"
        else:
            return
        with open(os.devnull, "w") as devnull:
            subprocess.run(
                [sys.executable, "-m", mod_name],
                check=True,
                cwd=str(project_root),
                stdout=devnull,
                stderr=devnull,
            )
