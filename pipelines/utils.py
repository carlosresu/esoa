#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers shared across pipeline runner entry points."""

from __future__ import annotations

import re


def slugify_item_ref_code(item_ref_code: str) -> str:
    """Convert an ITEM_REF_CODE like 'DrugsAndMedicine' into a slug ('drugs_and_medicine')."""
    if not item_ref_code:
        raise ValueError("item_ref_code must be a non-empty string.")
    slug = re.sub(r"(?<!^)(?=[A-Z])", "_", item_ref_code).lower()
    return slug


__all__ = ["slugify_item_ref_code"]
