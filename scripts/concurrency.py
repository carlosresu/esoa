#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal concurrency helpers for cpu-bound row-wise transforms."""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar, List

T = TypeVar("T")
R = TypeVar("R")


def _available_cpus() -> int:
    """Best-effort logical cpu count respecting scheduler affinity."""
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        count = os.cpu_count()
        return count if isinstance(count, int) and count > 0 else 1


def _env_requested_workers() -> int | None:
    """Parse ESOA_MAX_WORKERS to allow opt-in tuning or disabling."""
    raw = os.getenv("ESOA_MAX_WORKERS")
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0:
        return 0
    return value


def resolve_worker_count(explicit: int | None = None, task_size: int | None = None) -> int:
    """Resolve a safe worker count given env overrides and task size."""
    requested = _env_requested_workers()
    if requested is not None:
        if requested <= 0:
            return 1
        return max(1, min(requested, _available_cpus()))

    if explicit is not None:
        return max(1, min(explicit, _available_cpus()))

    cpu_cap = _available_cpus()
    if cpu_cap <= 1:
        return 1

    if not task_size or task_size < 2000:
        return 1

    # Scale workers by workload size but avoid oversubscription.
    target = min(cpu_cap, max(2, math.ceil(task_size / 20000)))
    return max(1, min(target, cpu_cap))


def maybe_parallel_map(
    seq: Sequence[T] | Iterable[T],
    func: Callable[[T], R],
    *,
    max_workers: int | None = None,
    parallel_threshold: int = 2000,
) -> List[R]:
    """Apply func across seq, using a process pool when the workload is large enough.

    The ESOA_MAX_WORKERS env var can pin the worker count (set to 1 to disable
    parallelism). On small workloads the function falls back to serial execution.
    """
    values = list(seq)
    count = len(values)
    if count == 0:
        return []

    workers = resolve_worker_count(explicit=max_workers, task_size=count)
    if workers <= 1 or count < parallel_threshold:
        return [func(item) for item in values]

    # Balance chunks so each worker gets at least a few hundred items.
    chunksize = max(1, min(1000, count // (workers * 4)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, values, chunksize=chunksize))


__all__ = ["maybe_parallel_map", "resolve_worker_count"]
