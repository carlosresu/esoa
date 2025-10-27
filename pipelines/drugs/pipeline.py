#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline implementation for ITEM_REF_CODE == 'DrugsAndMedicine'."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
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
from scripts.prepare import prepare
from scripts.prepare_annex_f import prepare_annex_f
from scripts.match import match

THIS_DIR = Path(__file__).resolve().parents[2]
ATCD_SUBDIR = Path("dependencies") / "atcd"
ATCD_SCRIPTS: tuple[str, ...] = ("atcd.R", "export.R", "filter.R")


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
                lambda: self._run_r_scripts(context.project_root),
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
            prepared_paths["annex"] = Path(
                prepare_annex_f(str(params.annex_csv), str(context.inputs_dir / "annex_f_prepared.csv"))
            ).resolve()
            pnf_prep, esoa_prep = prepare(str(params.pnf_csv), str(params.esoa_csv), str(context.inputs_dir))
            prepared_paths["pnf"] = Path(pnf_prep).resolve()
            prepared_paths["esoa"] = Path(esoa_prep).resolve()

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
    def _run_r_scripts(project_root: Path) -> None:
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

    @staticmethod
    def _build_brand_map(inputs_dir: Path) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_csv = inputs_dir / f"fda_brand_map_{date_str}.csv"
        if out_csv.exists():
            return out_csv
        existing_maps = sorted(inputs_dir.glob("fda_brand_map_*.csv"), reverse=True)
        with open(os.devnull, "w") as devnull:
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.fda_ph_drug_scraper",
                        "--outdir",
                        str(inputs_dir),
                        "--outfile",
                        str(out_csv),
                    ],
                    check=True,
                    cwd=str(THIS_DIR),
                    stdout=devnull,
                    stderr=devnull,
                )
            except subprocess.CalledProcessError as exc:
                if existing_maps:
                    return existing_maps[0]
                raise RuntimeError(
                    "Building FDA brand map failed and no prior map is available. "
                    "Re-run with --skip-brandmap if the FDA site is unreachable."
                ) from exc
        return out_csv

    @staticmethod
    def _run_resolve_unknowns(project_root: Path) -> None:
        scripts_path = project_root / "scripts" / "resolve_unknowns.py"
        root_path = project_root / "resolve_unknowns.py"
        if scripts_path.is_file():
            mod_name = "scripts.resolve_unknowns"
        elif root_path.is_file():
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
