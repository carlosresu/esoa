#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants for the LaboratoryAndDiagnostic pipeline."""

from __future__ import annotations

from pathlib import Path

from ..utils import slugify_item_ref_code

ITEM_REF_CODE: str = "LaboratoryAndDiagnostic"
PIPELINE_SLUG: str = slugify_item_ref_code(ITEM_REF_CODE)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PIPELINE_INPUTS_DIR: Path = PROJECT_ROOT / "inputs" / PIPELINE_SLUG
PIPELINE_OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs" / PIPELINE_SLUG
PIPELINE_RAW_DIR: Path = PROJECT_ROOT / "raw"

__all__ = [
    "ITEM_REF_CODE",
    "PIPELINE_SLUG",
    "PIPELINE_INPUTS_DIR",
    "PIPELINE_OUTPUTS_DIR",
    "PIPELINE_RAW_DIR",
    "PROJECT_ROOT",
]
