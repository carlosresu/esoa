#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 0 runner: refresh the DrugBank exports once before running other Drugs pipeline stages."""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List

import run_drugs_all_parts as run_dm


def _run_drugbank_export(extra_args: List[str] | None = None) -> None:
    """Execute the DrugBank aggregation helper so CSVs land under dependencies/ and inputs/drugs."""
    cmd = [sys.executable, "-m", "pipelines.drugs.scripts.run_drugbank_drugs"]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.check_call(cmd, cwd=str(run_dm.THIS_DIR))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DrugBank aggregation helper as a standalone stage. "
            "Invoke this whenever the upstream DrugBank dataset is refreshed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-spinner",
        action="store_true",
        help="Disable the spinner/timing wrapper (the underlying Python module still streams its output).",
    )
    args, extra = parser.parse_known_args(argv)

    if args.no_spinner:
        _run_drugbank_export(extra)
        return

    def _task() -> None:
        _run_drugbank_export(extra)

    run_dm.run_with_spinner("DrugBank aggregation", _task)


if __name__ == "__main__":
    main(sys.argv[1:])
