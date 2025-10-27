#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile run_drugs.main_entry with pyinstrument and write reports under ./outputs.

Usage mirrors run_drugs; pass the same CLI flags. Produces HTML and text
profiling artifacts timestamped inside the DrugsAndMedicine outputs directory so
teams can inspect the hottest sections of the pipeline end-to-end.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys

try:
    from pyinstrument import Profiler
except ModuleNotFoundError:  # pragma: no cover - runtime dependency bootstrapping
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstrument>=4.4"], stdout=sys.stdout)
    from pyinstrument import Profiler

import run_drugs as run_dm


def _profiled_main() -> None:
    """Execute run.main_entry under pyinstrument, emitting HTML and text reports."""
    profiler = Profiler()
    profiler.start()
    try:
        run_dm.main_entry()
    finally:
        profiler.stop()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        outputs_dir = run_dm._ensure_outputs_dir()
        html_path = outputs_dir / f"pyinstrument_profile_{timestamp}.html"
        text_path = outputs_dir / f"pyinstrument_profile_{timestamp}.txt"
        html_path.write_text(profiler.output_html(), encoding="utf-8")
        text_path.write_text(
            profiler.output_text(unicode=True, color=False),
            encoding="utf-8",
        )
        print(f"pyinstrument profile written to {html_path} and {text_path}")


def main() -> None:
    """Entry point to keep CLI parity with run_drugs while enabling profiling."""
    _profiled_main()


if __name__ == "__main__":
    main()
