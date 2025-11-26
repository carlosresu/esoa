#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2: Match Annex F entries to ATC codes and DrugBank IDs.

This script:
- Loads Annex F, PNF lexicon, and DrugBank generics/mixtures
- Matches each Annex F Drug Description to reference data
- Assigns ATC codes and DrugBank IDs
- Outputs annex_f_with_atc.csv for use in Part 4

Prerequisites:
- Run Part 1 (prepare_dependencies) first to ensure reference data is fresh
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_DIR = Path(__file__).resolve().parent


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 2: Match Annex F entries to ATC codes and DrugBank IDs."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--use-threads",
        action="store_true",
        help="Use thread pool instead of process pool.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print("=" * 60)
    print("Part 2: Match Annex F with ATC/DrugBank IDs")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "pipelines.drugs.scripts.match_annex_f_with_atc",
        "--workers",
        str(args.workers),
    ]
    if args.use_threads:
        cmd.append("--use-threads")

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_DIR))
    except subprocess.CalledProcessError as exc:
        print(f"[error] Annex F matching failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)

    print("\n" + "=" * 60)
    print("Part 2 Complete: Annex F Tagged with ATC/DrugBank IDs")
    print("=" * 60)
    print("  Output: outputs/drugs/annex_f_with_atc.csv")
    print("\nNext: Run Part 3 (match_esoa_with_atc) to match ESOA rows to ATC/DrugBank IDs")


if __name__ == "__main__":
    main(sys.argv[1:])
