#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level pipeline registry that exposes category-specific matching workflows."""

from .base import (
    BasePipeline,
    PipelineContext,
    PipelineOptions,
    PipelinePreparedInputs,
    PipelineResult,
    PipelineRunParams,
)
from .registry import PIPELINE_REGISTRY, get_pipeline, list_pipelines
from .utils import slugify_item_ref_code

__all__ = [
    "BasePipeline",
    "PipelineContext",
    "PipelineOptions",
    "PipelinePreparedInputs",
    "PipelineResult",
    "PipelineRunParams",
    "PIPELINE_REGISTRY",
    "get_pipeline",
    "list_pipelines",
    "slugify_item_ref_code",
]
