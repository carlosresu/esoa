#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Placeholder pipeline for ITEM_REF_CODE == 'LaboratoryAndDiagnostic'."""

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


@register_pipeline
class LaboratoryAndDiagnosticPipeline(BasePipeline):
    """Scaffolding that future lab/diagnostic matching logic can build upon."""

    item_ref_code = "LaboratoryAndDiagnostic"
    display_name = "Laboratory & Diagnostic"
    description = (
        "Stub pipeline that keeps the registry aware of laboratory and diagnostic items. "
        "Implement category-specific normalization and matching logic here."
    )

    def prepare_inputs(
        self,
        context: PipelineContext,
        params: PipelineRunParams,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelinePreparedInputs:
        return PipelinePreparedInputs(esoa_csv=params.esoa_csv)

    def match(
        self,
        context: PipelineContext,
        prepared: PipelinePreparedInputs,
        options: PipelineOptions,
        *,
        timing_hook: Optional[TimingHook] = None,
    ) -> PipelineResult:
        raise NotImplementedError(
            "LaboratoryAndDiagnostic pipeline does not yet implement matching. "
            "Add algorithm-specific logic under pipelines/lab_and_dx/."
        )
