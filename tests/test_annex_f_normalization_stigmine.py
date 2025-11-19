#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for Annex F normalization issues involving 'STIGMINE' molecules."""

import pytest

from pipelines.drugs.scripts.text_utils_drugs import normalize_text, _normalize_text_basic
from pipelines.drugs.scripts.generic_normalization import normalize_generic

STIGMINE_NAMES = [
    "NEOSTIGMINE",
    "PHYSOSTIGMINE",
    "PYRIDOSTIGMINE",
    "RIVASTIGMINE",
]


@pytest.mark.parametrize("name", STIGMINE_NAMES)
def test_stigmine_names_not_corrupted(name: str) -> None:
    norm = normalize_text(name).upper()
    basic = _normalize_text_basic(name).upper()
    generic = normalize_generic(name).upper()

    assert "STIGMINE" in norm
    assert "STIGMINE" in basic
    assert "STIGMINE" in generic

    assert "STIGINE" not in norm
    assert "STIGINE" not in basic
    assert "STIGINE" not in generic


def test_gm_unit_normalization_only_in_unit_context() -> None:
    assert " 1 G " in f" {normalize_text('1 gm')} ".upper()
    assert " 2 G " in f" {normalize_text('2 gms')} ".upper()
    assert normalize_text("NEOSTIGMINE").upper() == "NEOSTIGMINE"
    assert "NEOSTIGMINE" in normalize_text("NEOSTIGMINE 1 gm").upper()
