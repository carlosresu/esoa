#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3: Match ESOA rows to ATC codes and DrugBank IDs.

Prerequisites:
- Run Part 1 (prepare_dependencies) first
- Run build_unified_reference.py to create the unified reference dataset
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

from pipelines.drugs.scripts.runners import run_esoa_tagging

if __name__ == "__main__":
    run_esoa_tagging()
    print("\nNext: Run Part 4 to bridge ESOA to Annex F Drug Codes")
