#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants for the DrugsAndMedicine pipeline."""

from __future__ import annotations

from pathlib import Path

from ..utils import slugify_item_ref_code

ITEM_REF_CODE: str = "DrugsAndMedicine"
PIPELINE_SLUG: str = slugify_item_ref_code(ITEM_REF_CODE)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
PIPELINE_INPUTS_DIR: Path = PROJECT_ROOT / "inputs" / "drugs"
PIPELINE_OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs" / "drugs"
PIPELINE_RAW_DIR: Path = PROJECT_ROOT / "raw"
PIPELINE_WHO_ATC_DIR: Path = PIPELINE_INPUTS_DIR
PIPELINE_DRUGBANK_BRANDS_PATH: Path = PIPELINE_INPUTS_DIR / "drugbank_brands.csv"
# Parquet-first storage expectation for all drug pipeline intermediates/outputs.
PIPELINE_PRIMARY_STORAGE_FORMAT: str = "parquet"

__all__ = [
    "ITEM_REF_CODE",
    "PIPELINE_SLUG",
    "PIPELINE_INPUTS_DIR",
    "PIPELINE_OUTPUTS_DIR",
    "PIPELINE_RAW_DIR",
    "PIPELINE_WHO_ATC_DIR",
    "PIPELINE_DRUGBANK_BRANDS_PATH",
    "PIPELINE_PRIMARY_STORAGE_FORMAT",
    "PROJECT_ROOT",
]
