#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, time
import os, pandas as pd
from typing import Callable

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    """Wrap a callable with a lightweight spinner to show progress inside module-level scripts."""
    import threading
    done = threading.Event()
    err = []

    def worker():
        try:
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"; i = 0
    while not done.is_set():
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r{frames[i % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        time.sleep(0.1); i += 1
    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return elapsed

from .match_features import build_features
from .match_scoring import score_and_classify
from .match_outputs import write_outputs

def match(
    pnf_prepared_csv: str,
    esoa_prepared_csv: str,
    out_csv: str = "esoa_matched.csv",
    *,
    timing_hook: Callable[[str, float], None] | None = None,
    skip_excel: bool = False,
) -> str:
    """Run the feature build, scoring, and output-writing stages on prepared inputs exactly as outlined in pipeline.md steps 6–15."""
    def _timed(label: str, func: Callable[[], None]) -> float:
        elapsed = _run_with_spinner(label, func)
        if timing_hook:
            timing_hook(label, elapsed)
        return elapsed

    # Load inputs
    # Use small mutable containers so closures can assign to outer scope by reference.
    pnf_df = [None]
    esoa_df = [None]
    _timed("Load PNF prepared CSV", lambda: pnf_df.__setitem__(0, pd.read_csv(pnf_prepared_csv)))
    _timed("Load eSOA prepared CSV", lambda: esoa_df.__setitem__(0, pd.read_csv(esoa_prepared_csv)))

    # Build features — inner function prints its own sub-spinners; do not show outer spinner
    # Feed the matcher-specific feature engineering step.
    features_df = build_features(pnf_df[0], esoa_df[0], timing_hook=timing_hook)

    # Score & classify
    # Prepare container for the scored DataFrame so the closure can mutate it.
    out_df = [None]
    _timed("Score & classify", lambda: out_df.__setitem__(0, score_and_classify(features_df, pnf_df[0])))

    # Write outputs — inner module prints its own sub-spinners; do not show outer spinner
    out_path = os.path.abspath(out_csv)
    # Persist the outputs (CSV, XLSX, summaries) while capturing timing metrics.
    write_outputs(out_df[0], out_path, timing_hook=timing_hook, skip_excel=skip_excel)

    return out_path
