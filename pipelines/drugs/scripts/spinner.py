#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared spinner utility for pipeline scripts."""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


LabelType = str | Callable[[float], str]


def run_with_spinner(
    label: LabelType,
    func: Callable[[], T],
    completion_label: Optional[Callable[[float], str]] = None,
) -> T:
    """Run func() while showing a lightweight CLI spinner with elapsed time.
    
    Args:
        label: Text to show during execution. Can be a string or a
               callback(elapsed_seconds) -> str for dynamic updates (e.g., ETA).
        func: Function to execute
        completion_label: Optional callback(elapsed_seconds) -> str for completion line.
                         If None, uses the label (evaluated at completion time if callable).
    
    Output format: ⠋ XXXX.XXs label (during) / ⣿ XXXX.XXs label (done)
    """
    done = threading.Event()
    result: list[T] = []
    err: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(func())
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)
        finally:
            done.set()

    def get_label(elapsed: float) -> str:
        return label(elapsed) if callable(label) else label

    start = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0
    while not done.wait(0.1):
        elapsed = time.perf_counter() - start
        current_label = get_label(elapsed)
        sys.stdout.write(f"\r{frames[idx % len(frames)]} {elapsed:7.2f}s {current_label}    ")
        sys.stdout.flush()
        idx += 1
    thread.join()
    elapsed = time.perf_counter() - start
    complete = "⣿" if not err else "✗"
    final_label = completion_label(elapsed) if completion_label else get_label(elapsed)
    sys.stdout.write(f"\r{complete} {elapsed:7.2f}s {final_label}    \n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return result[0] if result else None  # type: ignore[return-value]
