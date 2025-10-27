#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline implementation for ITEM_REF_CODE == 'LaboratoryAndDiagnostic'."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
from .constants import PIPELINE_RAW_DIR
from .scripts import match_labs_records, prepare_labs_inputs


@register_pipeline
class LaboratoryAndDiagnosticPipeline(BasePipeline):
    """Normalize and match LaboratoryAndDiagnostic eSOA entries."""

    item_ref_code = "LaboratoryAndDiagnostic"
    display_name = "Laboratory & Diagnostic"
    description = (
        "Standardizes LaboratoryAndDiagnostic eSOA descriptions by matching against the "
        "official Labs catalog first, then falling back to Diagnostics.xlsx descriptions."
    )

    def prepare_inputs(
        self,
        context: PipelineContext,
        params: PipelineRunParams,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelinePreparedInputs:
        raw_dir = PIPELINE_RAW_DIR
        csv_source = raw_dir / "03 ESOA_ITEM_LIB.csv"
        tsv_source = raw_dir / "03 ESOA_ITEM_LIB.tsv"
        dest_csv = context.inputs_dir / "esoa_prepared_labs.csv"
        prepared_path = prepare_labs_inputs(csv_source, tsv_source, None, dest_csv)
        return PipelinePreparedInputs(esoa_csv=prepared_path)

    def match(
        self,
        context: PipelineContext,
        prepared: PipelinePreparedInputs,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelineResult:
        if prepared.esoa_csv is None:
            raise ValueError("LaboratoryAndDiagnostic pipeline requires a prepared eSOA CSV.")

        out_csv = options.extra.get("out_csv") if options.extra else None
        output_path = Path(out_csv) if out_csv else context.outputs_dir / "esoa_matched_labs.csv"

        master_csv = context.inputs_dir / "labs.csv"
        if not master_csv.is_file():
            raise FileNotFoundError(
                f"Master Laboratory & Diagnostic catalog not found at {master_csv}."
            )
        diagnostics_xlsx = PIPELINE_RAW_DIR / "Diagnostics.xlsx"

        matched_path = match_labs_records(
            Path(prepared.esoa_csv),
            master_csv,
            diagnostics_xlsx,
            output_path,
            skip_excel=options.skip_excel,
        )
        return PipelineResult(matched_csv=matched_path, prepared=prepared)
