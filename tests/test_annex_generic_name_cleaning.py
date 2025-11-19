#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Annex F generic candidate cleaning and selection heuristics."""

from run_drugs_pt_1_parse_annex_f import (
    _build_candidate_from_raw_description,
    _clean_generic_candidates,
    _select_generic,
)
from pipelines.drugs.scripts.generic_normalization import normalize_generic


def test_remove_form_and_packaging_from_generic_candidates() -> None:
    candidates = [
        "VITAMINS INTRAVENOUS, FAT-SOLUBLE SOLUTION 10 ML AMPULE",
        "VITAMINS FAT-SOLUBLE",
        "VITAMINS",
    ]
    cleaned = _clean_generic_candidates(candidates)
    joined = _select_generic(cleaned)

    assert "SOLUTION" not in joined
    assert "ML" not in joined
    assert "AMPULE" not in joined
    assert not any(ch.isdigit() for ch in joined)
    assert "VITAMIN" in joined


def test_avoid_single_trivial_acid_generic() -> None:
    raw = "AMINO ACID SOLUTIONS FOR RENAL CONDITIONS 3.50% 500 mL BOTTLE"
    norm = normalize_generic(raw).upper()
    assert norm != "ACID"
    assert "AMINO" in norm
    assert "RENAL" in norm


def test_keep_all_components_in_combo() -> None:
    candidates = ["ALUMINUM HYDROXIDE", "MAGNESIUM HYDROXIDE"]
    cleaned = _clean_generic_candidates(candidates)
    joined = _select_generic(cleaned)
    assert "ALUMINUM HYDROXIDE" in joined
    assert "MAGNESIUM HYDROXIDE" in joined
    assert " + " in joined


def test_deduplicate_overlapping_combo_candidates() -> None:
    candidates = [
        "ALENDRONATE + CHOLECALCIFEROL (VIT. D3) ( AS SODIUM SALT) 70 MG + 2800 IU TABLET",
        "ALENDRONATE + CHOLECALCIFEROL VIT",
        "ALENDRONATE + CHOLECALCIFEROL",
    ]
    cleaned = _clean_generic_candidates(candidates)
    joined = _select_generic(cleaned)
    assert joined == "ALENDRONATE + CHOLECALCIFEROL"


def test_build_candidate_from_raw_description_combination() -> None:
    raw = "ALUMINUM HYDROXIDE + MAGNESIUM HYDROXIDE 225 mg + 200 mg/5 mL SUSPENSION 250 mL BOTTLE"
    candidate = _build_candidate_from_raw_description(raw)
    assert candidate == "ALUMINUM HYDROXIDE + MAGNESIUM HYDROXIDE"


def test_build_candidate_from_raw_description_strips_forms() -> None:
    raw = "ALENDRONATE + CHOLECALCIFEROL (VIT. D3) ( as SODIUM SALT) 70 mg + 2800 IU TABLET"
    candidate = _build_candidate_from_raw_description(raw)
    assert candidate == "ALENDRONATE + CHOLECALCIFEROL"
