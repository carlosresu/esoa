#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared spinner utility for pipeline scripts."""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_spinner(label: str, func: Callable[[], T]) -> T:
    """Run func() while showing a lightweight CLI spinner with elapsed time."""
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

    start = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    frames = "|/-\\"
    idx = 0
    while not done.wait(0.1):
        elapsed = time.perf_counter() - start
        sys.stdout.write(f"\r{frames[idx % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        idx += 1
    thread.join()
    elapsed = time.perf_counter() - start
    status = "done" if not err else "error"
    sys.stdout.write(f"\r[{status}] {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return result[0] if result else None  # type: ignore[return-value]
