#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal launcher for the LaboratoryAndDiagnostic pipeline."""

from __future__ import annotations

import argparse
import sys

import run_labs as labs_runner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the LaboratoryAndDiagnostic pipeline without optional extras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--esoa", default=None, help="Optional LaboratoryAndDiagnostic CSV to merge with the raw eSOA sources")
    parser.add_argument("--out", default="esoa_matched_labs.csv", help="Matched CSV filename")
    args = parser.parse_args(argv)

    cli_args: list[str] = ["--out", args.out, "--skip-excel"]
    if args.esoa is not None:
        cli_args = ["--esoa", args.esoa, "--out", args.out, "--skip-excel"]

    labs_runner.main(cli_args)


if __name__ == "__main__":
    main(sys.argv[1:])
