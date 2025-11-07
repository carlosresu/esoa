#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible alias for the stage 2 minimal runner."""
from __future__ import annotations

import sys

import run_drugs_pt_2_esoa_matching_minimal as stage2_min


def main(argv: list[str] | None = None) -> None:
    stage2_min.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
