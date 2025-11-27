#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2: Match Annex F entries to ATC codes and DrugBank IDs.

Prerequisites:
- Run Part 1 (prepare_dependencies) first
- Run build_unified_reference.py to create the unified reference dataset
"""

import sys

from pipelines.drugs.scripts.runners import run_annex_f_tagging

if __name__ == "__main__":
    run_annex_f_tagging()
    print("\nNext: Run Part 3 to match ESOA rows to ATC/DrugBank IDs")
