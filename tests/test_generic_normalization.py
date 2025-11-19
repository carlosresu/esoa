#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the explicit drug normalization rule set."""

from __future__ import annotations

import unittest

from pipelines.drugs.scripts.generic_normalization import normalize_generic


class GenericNormalizationTests(unittest.TestCase):
    def assertNormalized(self, raw: str, expected: str) -> None:
        self.assertEqual(normalize_generic(raw), expected)

    def test_vitamins_collapsing(self) -> None:
        self.assertNormalized(
            "VITAMINS WATER-SOLUBLE INTRAVENOUS SOLUTION",
            "VITAMINS",
        )

    def test_dextrose_water(self) -> None:
        self.assertNormalized(
            "DEXTROSE IN WATER 5% SOLUTION",
            "DEXTROSE",
        )

    def test_dextrose_saline(self) -> None:
        self.assertNormalized(
            "DEXTROSE IN SODIUM CHLORIDE SOLUTION",
            "DEXTROSE + SODIUM CHLORIDE",
        )

    def test_saline_only_salts(self) -> None:
        self.assertNormalized(
            "SODIUM CHLORIDE INJECTION USP",
            "SODIUM CHLORIDE",
        )

    def test_multi_ingredient_combos(self) -> None:
        self.assertNormalized(
            "DEXTROSE IN LACTATED RINGERS WITH POTASSIUM CHLORIDE AND MAGNESIUM SULFATE",
            "DEXTROSE + LACTATED RINGERS + POTASSIUM CHLORIDE + MAGNESIUM SULFATE",
        )

    def test_prevent_truncation(self) -> None:
        self.assertNormalized(
            "SODIUM + CHLORIDE INJECTION",
            "SODIUM CHLORIDE",
        )


if __name__ == "__main__":
    unittest.main()
