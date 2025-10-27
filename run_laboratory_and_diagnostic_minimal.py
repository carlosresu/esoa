#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal launcher for the LaboratoryAndDiagnostic pipeline (Lab & DX matching scaffold)."""

from __future__ import annotations

import argparse
import sys

import run_laboratory_and_diagnostic as lab_runner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the LaboratoryAndDiagnostic pipeline without optional extras (Lab & DX matching scaffold).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--esoa", default=None, help="Optional Laboratory & Diagnostic CSV to merge with the raw eSOA sources")
    parser.add_argument("--out", default="laboratory_and_diagnostic_matched.csv", help="Matched CSV filename")
    args = parser.parse_args(argv)

    cli_args: list[str] = ["--out", args.out, "--skip-excel"]
    if args.esoa is not None:
        cli_args = ["--esoa", args.esoa, "--out", args.out, "--skip-excel"]

    lab_runner.main(cli_args)


if __name__ == "__main__":
    main(sys.argv[1:])
