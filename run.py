#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — Full ESOA pipeline with on-demand spinner and timing.

Console behavior:
  • Only the spinner/timer lines and the final timing summary are printed.
  • All other steps run silently.
File outputs:
  • scripts/match_outputs.py writes ./outputs/summary.txt (overwritten each run).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

THIS_DIR: Path = Path(__file__).resolve().parent
DEFAULT_INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
ATCD_SUBDIR = Path("dependencies") / "atcd"
ATCD_SCRIPTS: tuple[str, ...] = ("atcd.R", "export.R", "filter.R")

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


# ----------------------------
# Utilities
# ----------------------------
def _resolve_input_path(p: str | os.PathLike[str], default_subdir: str = DEFAULT_INPUTS_DIR) -> Path:
    if not p:
        raise FileNotFoundError("No input path provided.")
    pth = Path(p)
    if pth.is_file():
        return pth
    candidate = THIS_DIR / default_subdir / pth.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Input file not found: {pth!s}. "
        f"Tried: {pth.resolve()!s} and {candidate!s}. "
        f"Place the file under ./{default_subdir}/ or pass --pnf/--esoa with a correct path."
    )


def _ensure_outputs_dir() -> Path:
    outdir = THIS_DIR / OUTPUTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _ensure_inputs_dir() -> Path:
    inp = THIS_DIR / DEFAULT_INPUTS_DIR
    inp.mkdir(parents=True, exist_ok=True)
    return inp


def _assert_all_exist(root: Path, files: Iterable[str | os.PathLike[str]]) -> None:
    for f in files:
        fp = root / f
        if not fp.is_file():
            raise FileNotFoundError(f"Required file not found: {fp}")


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
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0

    # Show spinner immediately (per request)
    while not done.is_set():
        elapsed = time.perf_counter() - t0
        frame = spinner_frames[idx % len(spinner_frames)]
        sys.stdout.write(f"\r{frame} {label} … {elapsed:0.1f}s")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {label} — done in {elapsed:0.2f}s\n")
    sys.stdout.flush()

    if err:
        raise err[0]
    return elapsed


# ----------------------------
# Steps (silent)
# ----------------------------
def install_requirements(req_path: str | os.PathLike[str]) -> None:
    req = Path(req_path) if req_path else None
    if not req or not req.is_file():
        return
    with open(os.devnull, "w") as devnull:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)],
            cwd=str(THIS_DIR),
            stdout=devnull,
            stderr=devnull,
        )


def run_r_scripts() -> None:
    atcd_dir = THIS_DIR / ATCD_SUBDIR
    if not atcd_dir.is_dir():
        return
    out_dir = atcd_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    rscript = shutil.which("Rscript")
    if not rscript:
        return
    try:
        _assert_all_exist(atcd_dir, ATCD_SCRIPTS)
    except FileNotFoundError:
        return
    with open(os.devnull, "w") as devnull:
        for script in ATCD_SCRIPTS:
            subprocess.run([rscript, script], check=True, cwd=str(atcd_dir), stdout=devnull, stderr=devnull)


def create_master_file(root_dir: Path) -> None:
    """Silent; best-effort concatenation for debugging (no console output)."""
    debug_dir = root_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = debug_dir / "master.py"

    files_to_concatenate = [
        root_dir / "scripts" / "aho.py",
        root_dir / "scripts" / "combos.py",
        root_dir / "scripts" / "dose.py",
        root_dir / "scripts" / "match_features.py",
        root_dir / "scripts" / "match_scoring.py",
        root_dir / "scripts" / "match_outputs.py",
        root_dir / "scripts" / "match.py",
        root_dir / "scripts" / "prepare.py",
        root_dir / "scripts" / "routes_forms.py",
        root_dir / "scripts" / "text_utils.py",
        root_dir / "scripts" / "who_molecules.py",
        root_dir / "scripts" / "fda_ph_drug_scraper.py",
        root_dir / "scripts" / "brand_map.py",
        root_dir / "main.py",
        root_dir / "run.py",
    ]

    header_text = "# START OF REPO FILES"
    footer_text = "# END OF REPO FILES"

    try:
        with output_file_path.open("w", encoding="utf-8", newline="\n") as outfile:
            outfile.write(header_text + "\n")
            for file_path in files_to_concatenate:
                if not file_path.is_file():
                    continue
                relative_path = file_path.relative_to(root_dir).as_posix()
                outfile.write(f"# <{relative_path}>\n")
                outfile.write(file_path.read_text(encoding="utf-8", errors="ignore"))
                outfile.write("\n")
            outfile.write(footer_text + "\n")
    except Exception:
        # Silent best-effort
        pass


def build_brand_map(inputs_dir: Path, outfile: Path | None) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_csv = outfile or (inputs_dir / f"fda_brand_map_{date_str}.csv")
    with open(os.devnull, "w") as devnull:
        subprocess.run(
            [sys.executable, "-m", "scripts.fda_ph_drug_scraper", "--outdir", str(inputs_dir), "--outfile", str(out_csv)],
            check=True,
            cwd=str(THIS_DIR),
            stdout=devnull,
            stderr=devnull,
        )
    return out_csv


# ----------------------------
# Main entry
# ----------------------------
def main_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Run full ESOA pipeline (ATC → brand map → prepare → match) with spinner+timing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pnf", default=f"{DEFAULT_INPUTS_DIR}/pnf.csv", help="Path to PNF CSV")
    parser.add_argument("--esoa", default=f"{DEFAULT_INPUTS_DIR}/esoa.csv", help="Path to eSOA CSV")
    parser.add_argument("--out", default="esoa_matched.csv", help="Output CSV filename (saved under ./outputs)")
    parser.add_argument("--requirements", default="requirements.txt", help="Requirements file to install")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install of requirements")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R preprocessing scripts")
    parser.add_argument("--skip-brandmap", action="store_true", help="Skip building FDA brand map CSV")
    args = parser.parse_args()

    # Silent helper
    create_master_file(THIS_DIR)

    outdir = _ensure_outputs_dir()
    inputs_dir = _ensure_inputs_dir()
    pnf_path = _resolve_input_path(args.pnf)
    esoa_path = _resolve_input_path(args.esoa)
    out_path = outdir / Path(args.out).name

    from scripts.prepare import prepare as _prepare
    from scripts.match import match as _match

    timings: list[tuple[str, float]] = []

    if not args.skip_install and args.requirements:
        t = run_with_spinner("Install requirements", lambda: install_requirements(args.requirements))
        timings.append(("Install requirements", t))

    if not args.skip_r:
        t = run_with_spinner("ATC R preprocessing", run_r_scripts)
        timings.append(("ATC R preprocessing", t))

    if not args.skip_brandmap:
        t = run_with_spinner("Build FDA brand map", lambda: build_brand_map(inputs_dir, outfile=None))
        timings.append(("Build FDA brand map", t))

    t = run_with_spinner("Prepare inputs", lambda: _prepare(str(pnf_path), str(esoa_path), str(inputs_dir)))
    timings.append(("Prepare inputs", t))

    t = run_with_spinner(
        "Match & write outputs",
        lambda: _match(str(inputs_dir / "pnf_prepared.csv"), str(inputs_dir / "esoa_prepared.csv"), str(out_path)),
    )
    timings.append(("Match & write outputs", t))

    # Final timing summary (console only)
    print("\n=== Timing Summary ===")
    total = 0.0
    for name, secs in timings:
        print(f"• {name:<24} {secs:8.2f}s")
        total += secs
    print(f"{'-'*38}\n• Total{'':<21} {total:8.2f}s")


if __name__ == "__main__":
    try:
        main_entry()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
