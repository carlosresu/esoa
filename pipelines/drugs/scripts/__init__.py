#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exports convenience aliases for DrugsAndMedicine pipeline modules."""

from .prepare_drugs import prepare
from .tagging import UnifiedTagger, tag_descriptions, tag_single
from .runners import run_annex_f_tagging, run_esoa_tagging

__all__ = [
    "prepare",
    "UnifiedTagger",
    "tag_descriptions",
    "tag_single",
    "run_annex_f_tagging",
    "run_esoa_tagging",
]

# Backward compatibility alias
match = run_annex_f_tagging
