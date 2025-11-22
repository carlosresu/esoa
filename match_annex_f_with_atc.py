#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper entrypoint to match Annex F entries against ATC catalogs (Polars-first)."""

from __future__ import annotations

from pipelines.drugs.scripts.annex_atc_match import run_annex_atc_match


def main() -> None:
    run_annex_atc_match()


if __name__ == "__main__":
    main()
