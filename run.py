#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — Full ESOA pipeline with on-demand spinner and timing.

Console behavior:
  • Only the spinner/timer lines and the final timing summary are printed.
  • The heavy “Match & write outputs” step uses a tqdm progress bar that starts immediately.

File outputs:
  • scripts/match_outputs.py writes ./outputs/summary.txt (overwritten each run).
  • Finally runs resolve_unknowns.py to analyze unmatched terms and write its outputs under ./outputs.
"""
from __future__ import annotations

import ensurepip
import os
import subprocess
import sys
import time
from pathlib import Path


THIS_DIR: Path = Path(__file__).resolve().parent


def _bootstrap_requirements(req_file: Path | None = None) -> None:
    req_path = req_file if req_file is not None else THIS_DIR / "requirements.txt"
    if not req_path.is_file():
        return

    def _run(cmd: list[str], *, devnull: object, suppress: bool = True) -> None:
        stdout = devnull if suppress else None
        stderr = devnull if suppress else None
        subprocess.check_call(cmd, cwd=str(THIS_DIR), stdout=stdout, stderr=stderr)

    with open(os.devnull, "w") as devnull:
        try:
            _run([sys.executable, "-m", "pip", "--version"], devnull=devnull)
        except subprocess.CalledProcessError:
            try:
                ensurepip.bootstrap()
            except Exception:
                pass
            try:
                _run([sys.executable, "-m", "pip", "--version"], devnull=devnull)
            except subprocess.CalledProcessError:
                try:
                    _run([sys.executable, "-m", "ensurepip", "--default-pip"], devnull=devnull)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError("pip is unavailable and could not be bootstrapped.") from exc

        try:
            _run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                devnull=devnull,
            )
        except subprocess.CalledProcessError:
            pass

        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(req_path),
        ]
        try:
            _run(install_cmd, devnull=devnull)
        except subprocess.CalledProcessError:
            time.sleep(3)
            print(
                "! Initial dependency install failed; retrying with verbose output...",
                file=sys.stderr,
            )
            _run(install_cmd, devnull=devnull, suppress=False)


_bootstrap_requirements()

import argparse
import csv
import re
import threading
from datetime import datetime
from typing import Callable, Iterable

from pipelines import PipelineContext, PipelineOptions, PipelineRunParams, get_pipeline

DEFAULT_INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"

# Ensure local imports when called from other CWDs
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# ----------------------------
# Utilities
# ----------------------------
def _resolve_input_path(p: str | os.PathLike[str], default_subdir: str = DEFAULT_INPUTS_DIR) -> Path:
    """Resolve user-provided paths, falling back to ./inputs/{filename} when relative."""
    # Fail fast when no argument is supplied.
    if not p:
        raise FileNotFoundError("No input path provided.")
    pth = Path(p)
    # Accept absolute or pre-resolved file references as-is.
    if pth.is_file():
        return pth
    candidate = THIS_DIR / default_subdir / pth.name
    # Look in the project-relative inputs directory for convenience.
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Input file not found: {pth!s}. "
        f"Tried: {pth.resolve()!s} and {candidate!s}. "
        f"Place the file under ./{default_subdir}/ or pass --pnf/--esoa with a correct path."
    )

def _natural_esoa_part_order(path: Path) -> tuple[int, str]:
    """Sort helper to order esoa_pt_*.csv using the first numeric suffix when available."""
    match = re.search(r"(\d+)", path.stem)
    index = int(match.group(1)) if match else sys.maxsize
    return index, path.name

def _concatenate_csv(parts: list[Path], dest: Path) -> None:
    """Row-bind multiple CSV files (assuming identical headers) into dest."""
    header: list[str] | None = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as out_handle:
        writer: csv.writer | None = None
        for part in parts:
            with part.open("r", newline="", encoding="utf-8-sig") as in_handle:
                reader = csv.reader(in_handle)
                try:
                    file_header = next(reader)
                except StopIteration:
                    continue  # Skip empty files gracefully.
                if header is None:
                    header = file_header
                    writer = csv.writer(out_handle)
                    writer.writerow(header)
                elif file_header != header:
                    raise ValueError(
                        f"Header mismatch when concatenating {part.name}; expected {header} but found {file_header}."
                    )
                assert writer is not None
                for row in reader:
                    writer.writerow(row)

def _resolve_esoa_path(esoa_arg: str) -> Path:
    """Resolve the eSOA input, concatenating esoa_pt_*.csv files when present."""
    raw_arg = Path(esoa_arg)
    search_dir = raw_arg.parent if raw_arg.parent else Path(DEFAULT_INPUTS_DIR)
    if not search_dir.is_absolute():
        search_dir = (THIS_DIR / search_dir).resolve()
    if not search_dir.exists():
        search_dir = (_ensure_inputs_dir()).resolve()

    part_files = sorted(search_dir.glob("esoa_pt_*.csv"), key=_natural_esoa_part_order)
    if part_files:
        combined = search_dir / "esoa_combined.csv"
        _concatenate_csv(part_files, combined)
        return combined

    return _resolve_input_path(esoa_arg)

def _ensure_outputs_dir() -> Path:
    """Make sure the outputs directory exists and return its filesystem path."""
    outdir = THIS_DIR / OUTPUTS_DIR
    # Create the directory tree so downstream writes succeed.
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _ensure_inputs_dir() -> Path:
    """Create the inputs directory when missing so upstream steps can drop files."""
    inp = THIS_DIR / DEFAULT_INPUTS_DIR
    # Mirror the outputs helper to ensure inputs/ exists during bootstrapping.
    inp.mkdir(parents=True, exist_ok=True)
    return inp

def _assert_all_exist(root: Path, files: Iterable[str | os.PathLike[str]]) -> None:
    """Validate that every filename under root exists before shelling out to R scripts."""
    for f in files:
        fp = root / f
        # Surface a clear error when an expected script is missing.
        if not fp.is_file():
            raise FileNotFoundError(f"Required file not found: {fp}")

def _prune_dated_exports(directory: Path) -> None:
    """Keep only the newest YYYY-MM-DD CSV per prefix/suffix family under directory."""
    if not directory.is_dir():
        return

    date_rx = re.compile(r"\d{4}-\d{2}-\d{2}")
    grouped: dict[tuple[str, str], list[tuple[str, Path]]] = {}

    for path in directory.glob("*.csv"):
        stem = path.stem
        match = date_rx.search(stem)
        if not match:
            continue

        prefix = stem[: match.start()]
        suffix = stem[match.end():]
        date_token = stem[match.start() : match.end()]

        try:
            datetime.strptime(date_token, "%Y-%m-%d")
        except ValueError:
            continue

        key = (prefix, suffix)
        grouped.setdefault(key, []).append((date_token, path))

    for key, entries in grouped.items():
        latest_date = max(date for date, _ in entries)
        for date, path in entries:
            if date != latest_date and path.exists():
                path.unlink()

# ----------------------------
# Spinner + timing
# ----------------------------
def run_with_spinner(label: str, func: Callable[[], None], start_delay: float = 0.0) -> float:
    """Run func() in a worker thread with a live spinner immediately (no delay).
    Returns elapsed seconds.
    """
    done = threading.Event()
    err: list[BaseException] = []

    def worker():
        try:
            # Execute the workload in a background thread while the spinner animates.
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            # Signal the spinner loop to exit once the task finishes.
            done.set()

    t0 = time.perf_counter()
    # Spawn the worker thread immediately so the spinner reflects real time.
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0

    while not done.is_set():
        elapsed = time.perf_counter() - t0
        frame = spinner_frames[idx % len(spinner_frames)]
        # Continuously update the console line with elapsed time and spinner state.
        sys.stdout.write(f"\r{frame} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

    # Ensure the worker concluded before finalizing output.
    th.join()
    elapsed = time.perf_counter() - t0
    # Emit a final success line with the total runtime.
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()

    if err:
        raise err[0]
    return elapsed

# ----------------------------
# Timing aggregation
# ----------------------------
GROUP_DEFINITIONS: list[tuple[str, tuple[str, ...]]] = [
    (
        "Setup & Prerequisites",
        (
            "ATC R preprocessing",
            "Build FDA brand map",
            "Prepare inputs",
        ),
    ),
    (
        "Data Loading & Validation",
        (
            "Load PNF prepared CSV",
            "Load eSOA prepared CSV",
            "Validate inputs",
        ),
    ),
    (
        "Reference Indexes",
        (
            "Index PNF names",
            "Load WHO molecules",
            "Load FDA brand map",
            "Build brand automata",
            "Index FDA generics",
            "Build PNF automata",
            "Build PNF partial index",
        ),
    ),
    (
        "Text Normalization",
        (
            "Normalize ESOA text",
            "Parse dose/route/form (raw)",
            "Apply brand→generic swaps",
            "Parse dose/route/form (basis)",
        ),
    ),
    (
        "Matching & Detection",
        (
            "Scan PNF hits",
            "Partial PNF fallback",
            "Detect WHO molecules",
            "Fuzzy reference fallback",
        ),
    ),
    (
        "Feature Enrichment",
        (
            "Compute combo features",
            "Extract unknown tokens",
            "Compute presence flags",
        ),
    ),
    (
        "Scoring & Outputs",
        (
            "Score & classify",
            "Write matched CSV",
            "Write Excel",
            "Write unknown words CSV",
            "Write summary.txt",
        ),
    ),
    (
        "Post-processing",
        (
            "Resolve unknowns",
        ),
    ),
]

STEP_TO_GROUP: dict[str, str] = {}
GROUP_ORDER: list[str] = []
for group_name, step_names in GROUP_DEFINITIONS:
    GROUP_ORDER.append(group_name)
    for step in step_names:
        STEP_TO_GROUP[step] = group_name

DEFAULT_GROUP = "Other"

class TimingCollector:
    """Accumulate per-step timings and expose grouped rollups for summary output."""
    def __init__(self) -> None:
        # Store (step label, duration) pairs in insertion order.
        self._entries: list[tuple[str, float]] = []

    def add(self, label: str, seconds: float) -> None:
        """Track the elapsed time for a named pipeline step."""
        self._entries.append((label, seconds))

    @property
    def entries(self) -> list[tuple[str, float]]:
        # Provide an immutable snapshot to avoid external mutation.
        return list(self._entries)

    def grouped_totals(self) -> dict[str, float]:
        """Roll up timings by high-level group, preserving the predefined order."""
        totals: dict[str, float] = {group: 0.0 for group in GROUP_ORDER}
        other_total = 0.0
        for label, seconds in self._entries:
            group = STEP_TO_GROUP.get(label)
            if group:
                # Accumulate durations within the matched group.
                totals[group] += seconds
            else:
                # Collect timings that do not map to a known group.
                other_total += seconds
        if other_total > 0.0:
            totals[DEFAULT_GROUP] = totals.get(DEFAULT_GROUP, 0.0) + other_total
        return totals

    def total(self) -> float:
        """Return the aggregate runtime of every recorded step."""
        # Sum directly over the recorded durations for quick reuse.
        return sum(seconds for _, seconds in self._entries)

def _print_grouped_summary(timings: TimingCollector) -> None:
    """Render grouped timing totals to stdout in a compact report."""
    totals = timings.grouped_totals()
    non_zero = [(group, secs) for group, secs in totals.items() if secs > 0.0]
    if not non_zero:
        return
    label_width = max(len(group) for group, _ in non_zero)
    # Print grouped totals followed by the aggregate wall clock time.
    print("\n=== Timing Summary ===")
    total = 0.0
    for group, secs in non_zero:
        print(f"• {group:<{label_width}} {secs:9.2f}s")
        total += secs
    dash_count = label_width + 16
    print("-" * dash_count)
    padding = max(label_width - len("Total"), 0)
    print(f"• Total{'':<{padding}} {total:9.2f}s")

# ----------------------------
# Main entry
# ----------------------------
def main_entry() -> None:
    """CLI front-end that mirrors README flow with pluggable ITEM_REF_CODE pipelines."""
    parser = argparse.ArgumentParser(
        description="Run eSOA pipeline with spinner+timing using the registered ITEM_REF_CODE pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Flags align with README guidance: allow skipping R preprocessing or FDA brand rebuilds when rerunning.
    parser.add_argument("--annex", default=f"{DEFAULT_INPUTS_DIR}/annex_f.csv", help="Path to Annex F CSV")
    parser.add_argument("--pnf", default=f"{DEFAULT_INPUTS_DIR}/pnf.csv", help="Path to PNF CSV")
    parser.add_argument("--esoa", default=f"{DEFAULT_INPUTS_DIR}/esoa.csv", help="Path to eSOA CSV")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV filename (saved under ./outputs)")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R preprocessing scripts")
    parser.add_argument("--skip-brandmap", action="store_true", help="Skip building FDA brand map CSV")
    parser.add_argument("--skip-excel", action="store_true", help="Skip writing XLSX output (CSV and summaries still produced)")
    parser.add_argument(
        "--item-ref-code",
        default="DrugsAndMedicine",
        help="ITEM_REF_CODE category to process (must have a registered pipeline)",
    )
    parser.add_argument(
        "--skip-unknowns",
        action="store_true",
        help="Skip running resolve_unknowns enrichment after matching when supported by the pipeline",
    )
    args = parser.parse_args()

    pipeline = get_pipeline(args.item_ref_code)

    outdir = _ensure_outputs_dir()
    inputs_dir = _ensure_inputs_dir()
    # Resolve the requested raw inputs, enforcing the fallback search behavior.
    def _optional(path: str | os.PathLike[str]) -> Path | None:
        try:
            return _resolve_input_path(path)
        except FileNotFoundError:
            print(
                f"! Optional input {path!s} not found; passing None to pipeline {pipeline.item_ref_code}.",
                file=sys.stderr,
            )
            return None

    annex_path = _optional(args.annex)
    pnf_path = _optional(args.pnf)
    esoa_path = _resolve_esoa_path(args.esoa)
    out_path = outdir / Path(args.out).name

    timings = TimingCollector()
    context = PipelineContext(
        project_root=THIS_DIR,
        inputs_dir=inputs_dir,
        outputs_dir=outdir,
    )
    params = PipelineRunParams(
        annex_csv=annex_path,
        pnf_csv=pnf_path,
        esoa_csv=esoa_path,
        out_csv=out_path,
    )
    options = PipelineOptions(
        skip_excel=args.skip_excel,
        extra={
            "spinner": run_with_spinner,
            "skip_r": args.skip_r,
            "skip_brandmap": args.skip_brandmap,
            "skip_unknowns": args.skip_unknowns,
            "out_csv": out_path,
        },
    )

    pipeline.run(context, params, options, timing_hook=timings.add)

    # Final timing summary (console only)
    _print_grouped_summary(timings)

    _prune_dated_exports(THIS_DIR / "dependencies" / "atcd" / "output")
    _prune_dated_exports(THIS_DIR / DEFAULT_INPUTS_DIR)

if __name__ == "__main__":
    try:
        main_entry()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
