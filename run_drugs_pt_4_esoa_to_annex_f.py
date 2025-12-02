#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 4: Bridge ESOA rows to Annex F Drug Codes via ATC/DrugBank ID.

This script uses the unified runner which:
- Loads ESOA rows with ATC/DrugBank IDs (from Part 3)
- Loads Annex F rows with ATC/DrugBank IDs (from Part 2)
- Matches by generic name and ATC code
- Outputs esoa_with_drug_code.csv/parquet

Prerequisites:
- Run Part 2 (annex_f_atc) to have Annex F tagged with ATC/DrugBank IDs
- Run Part 3 (esoa_atc) to have ESOA tagged with ATC/DrugBank IDs
"""

# === Auto-activate virtual environment ===
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_VENV_PYTHON = _SCRIPT_DIR / ".venv" / "bin" / "python"

if _VENV_PYTHON.exists() and sys.executable != str(_VENV_PYTHON):
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), __file__] + sys.argv[1:])
# === End auto-activate ===

# Sync shared scripts to submodules before running
from pipelines.drugs.scripts.sync_to_submodules import sync_all
sync_all()

from pipelines.drugs.scripts.runners import run_esoa_to_drug_code

if __name__ == "__main__":
    run_esoa_to_drug_code()
    print("\nPipeline complete!")
