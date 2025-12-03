#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the complete 4-part drugs pipeline:

Part 1: Prepare dependencies (WHO ATC, DrugBank, FDA, PNF, Annex F)
Part 2: Match Annex F with ATC/DrugBank IDs
Part 3: Match ESOA with ATC/DrugBank IDs
Part 4: Bridge ESOA to Annex F Drug Codes via ATC/DrugBank ID
"""
from __future__ import annotations

# === Auto-activate virtual environment ===
# This allows the script to work with Code Runner or direct execution
import os
import sys
from pathlib import Path
from datetime import datetime

_SCRIPT_DIR = Path(__file__).resolve().parent
_VENV_PYTHON = _SCRIPT_DIR / ".venv" / "bin" / "python"

# Re-execute with venv Python if not already in venv
if _VENV_PYTHON.exists() and sys.executable != str(_VENV_PYTHON):
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), __file__] + sys.argv[1:])
# === End auto-activate ===

import argparse
import csv
import re
import shutil
import subprocess
from typing import Callable, List, Optional, Sequence, TypeVar, Mapping

import pandas as pd

# Sync shared scripts to submodules before running
from pipelines.drugs.scripts.sync_to_submodules import sync_all
sync_all()

from pipelines.drugs.constants import PIPELINE_INPUTS_DIR, PROJECT_ROOT
from pipelines.drugs.pipeline import DrugsAndMedicinePipeline
from pipelines.drugs.scripts.prepare import prepare

PROJECT_DIR = PROJECT_ROOT
DRUGS_INPUTS_DIR = PIPELINE_INPUTS_DIR
RUN_SUMMARY_PATH = PROJECT_ROOT / "run_summary.md"
RUN_SUMMARY_SECTIONS: dict[str, list[str]] = {}
T = TypeVar("T")


def add_run_summary(section: str, lines: str | Sequence[str]) -> None:
    entries = RUN_SUMMARY_SECTIONS.setdefault(section, [])
    if isinstance(lines, str):
        entries.append(lines)
    else:
        entries.extend([line for line in lines if line])


def write_run_summary() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry_lines = [f"## Run completed {timestamp}", ""]
    sections_order = [
        "Code State",
        "Part 1: Prepare Dependencies",
        "Part 2: Match Annex F with ATC/DrugBank IDs",
        "Part 3: Match ESOA with ATC/DrugBank IDs",
        "Part 4: Bridge ESOA to Annex F Drug Codes",
        "Overall",
    ]
    for section in sections_order:
        entries = RUN_SUMMARY_SECTIONS.get(section)
        if not entries:
            continue
        entry_lines.append(f"### {section}")
        entry_lines.extend(entries)
        entry_lines.append("")
    entry_text = "\n".join(entry_lines).strip() + "\n\n"
    if RUN_SUMMARY_PATH.exists():
        existing = RUN_SUMMARY_PATH.read_text().rstrip() + "\n\n"
    else:
        existing = "# Pipeline Run History\n\n"
    RUN_SUMMARY_PATH.write_text(existing + entry_text)
    RUN_SUMMARY_SECTIONS.clear()


def capture_code_state() -> None:
    lines: list[str] = []
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_DIR).decode().strip()
    except Exception:
        branch = "unknown"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR).decode().strip()[:7]
    except Exception:
        commit = "unknown"
    lines.append(f"- Branch: {branch}")
    lines.append(f"- Commit: {commit}")
    try:
        status = subprocess.check_output(["git", "status", "-sb"], cwd=PROJECT_DIR).decode().splitlines()
        dirty_lines = [line.strip() for line in status[1:]]
        clean = not dirty_lines
        lines.append(f"- Working tree: {'clean' if clean else 'dirty'}")
        for line in dirty_lines[:5]:
            lines.append(f"  - {line}")
    except Exception:
        lines.append("- Working tree: unknown")
    add_run_summary("Code State", lines)


def _format_reason_lines(reason_counts: Mapping[str, int], total: int, prefix: str = "- Match reasons:") -> list[str]:
    if not reason_counts or not total:
        return []
    lines = [prefix]
    for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True):
        pct = 100 * count / total if total else 0
        lines.append(f"  - {reason}: {count:,} ({pct:.1f}%)")
    return lines

# Regex to match dated files: name_YYYY-MM-DD.ext or name_YYYY-MM-DD_*.ext
DATED_FILE_PATTERN = re.compile(r"^(.+?)_(\d{4}-\d{2}-\d{2})(?:_.*)?(\.\w+)$")


def purge_old_dated_files(directory: Path, quiet: bool = False) -> int:
    """
    Remove all but the latest version of dated files in a directory.
    
    Files matching pattern: name_YYYY-MM-DD.ext or name_YYYY-MM-DD_suffix.ext
    Keeps the most recent date for each (name, ext) group.
    
    Returns count of deleted files.
    """
    if not directory.exists():
        return 0
    
    # Group files by (base_name, extension)
    groups: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = DATED_FILE_PATTERN.match(path.name)
        if match:
            base_name, date_str, ext = match.groups()
            key = (base_name, ext)
            if key not in groups:
                groups[key] = []
            groups[key].append((date_str, path))
    
    deleted = 0
    for (base_name, ext), files in groups.items():
        if len(files) <= 1:
            continue
        # Sort by date descending, keep the latest
        files.sort(key=lambda x: x[0], reverse=True)
        latest_date, latest_path = files[0]
        for date_str, path in files[1:]:
            try:
                path.unlink()
                deleted += 1
                if not quiet:
                    print(f"[purge] Removed old file: {path.name} (keeping {latest_path.name})")
            except Exception:
                pass
    
    return deleted


def _ensure_inputs_dir() -> Path:
    DRUGS_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return DRUGS_INPUTS_DIR


def _run_python_module(module: str, argv: Sequence[str], *, cwd: Path | None = None, verbose: bool = True) -> None:
    command = [sys.executable, "-m", module, *argv]
    workdir = cwd or PROJECT_DIR
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    try:
        subprocess.run(command, check=True, cwd=str(workdir), stdout=stdout, stderr=stderr)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Module {module} exited with status {exc.returncode}") from exc


def _find_rscript() -> Optional[Path]:
    """Resolve the Rscript executable across common Windows/Linux locations."""
    env_override = os.environ.get("RSCRIPT_PATH")
    if env_override:
        candidate = Path(env_override)
        if candidate.is_file():
            return candidate
    which = shutil.which("Rscript")
    if which:
        return Path(which)
    r_home = os.environ.get("R_HOME")
    if r_home:
        for rel in ("bin/Rscript", "bin/Rscript.exe", "bin/x64/Rscript.exe", "bin/x64/Rscript"):
            candidate = Path(r_home) / rel
            if candidate.is_file():
                return candidate
    for base in (Path("C:/Program Files/R"), Path("C:/Program Files (x86)/R")):
        versions = [p for p in base.glob("R-*") if p.is_dir()]
        versions.sort(key=lambda p: p.name)
        for root in reversed(versions):
            for rel in ("bin/x64/Rscript.exe", "bin/Rscript.exe", "bin/Rscript"):
                candidate = root / rel
                if candidate.is_file():
                    return candidate
    return None


def _run_with_spinner(label: str, func: Callable[[], T]) -> T:
    """Run func() while showing a braille spinner before the elapsed time."""
    import threading
    import time

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
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    idx = 0
    while not done.wait(0.1):
        elapsed = time.perf_counter() - start
        sys.stdout.write(f"\r{frames[idx % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        idx += 1
    thread.join()
    elapsed = time.perf_counter() - start
    complete = "⣿" if not err else "✗"
    sys.stdout.write(f"\r{complete} {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return result[0] if result else None  # type: ignore[return-value]


def _run_r_script(
    script_path: Path,
    *,
    verbose: bool = True,
    quiet_mode: bool = True,
    workers: Optional[int] = None,
) -> None:
    """Run an R script via Rscript.
    
    Args:
        script_path: Path to the .R script
        verbose: If True, show R output in terminal; if False, redirect to log file
        quiet_mode: If True, set ESOA_DRUGBANK_QUIET=1 to suppress R messages
        workers: Number of parallel workers (None = let R auto-detect like RStudio)
    """
    rscript = _find_rscript()
    if not rscript:
        raise FileNotFoundError(
            "Rscript executable not found. Install R or set RSCRIPT_PATH/R_HOME or add Rscript to PATH."
        )
    env = os.environ.copy()
    env.setdefault("RSCRIPT_PATH", str(rscript))
    
    # Only set quiet mode if explicitly requested (RStudio doesn't set this)
    if quiet_mode:
        env["ESOA_DRUGBANK_QUIET"] = "1"
    elif "ESOA_DRUGBANK_QUIET" in env:
        del env["ESOA_DRUGBANK_QUIET"]
    
    # Only set workers if explicitly provided (otherwise let R auto-detect like RStudio)
    if workers is not None:
        env["ESOA_DRUGBANK_WORKERS"] = str(workers)
    # If not set, don't override - let R use detectCores() like RStudio does
    
    # Avoid file redirection which causes buffering delays
    # Use DEVNULL to discard output without buffering overhead
    if not verbose:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
    else:
        stdout = None
        stderr = None
    try:
        subprocess.run(
            [str(rscript), str(script_path)],
            check=True,
            cwd=str(script_path.parent),
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{script_path.name} exited with status {exc.returncode}") from exc


def _mirror_module_output(source: Path, dest: Path) -> None:
    """Mirror a module export into the Drugs pipeline inputs directory."""
    if not source.is_file():
        raise FileNotFoundError(f"Expected output from module at {source}, but it was not created.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _sort_esoa_parts(path: Path) -> tuple[int, str]:
    """Sort helper that orders esoa_pt_* files by their numeric suffix."""
    for token in path.stem.split("_"):
        if token.isdigit():
            return int(token), path.name
    return sys.maxsize, path.name


def _concatenate_csv(parts: Sequence[Path], dest: Path) -> Path:
    """Concatenate multiple CSV files (identical headers) into dest, deduplicating rows."""
    import pandas as pd
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    for part in parts:
        if not part.is_file():
            continue
        try:
            df = pd.read_csv(part, encoding="utf-8-sig")
            dfs.append(df)
        except Exception:
            continue
    
    if not dfs:
        raise ValueError("No valid CSV files to concatenate")
    
    # Concatenate and deduplicate
    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    
    if before != after:
        add_run_summary(
            "Part 1: Prepare Dependencies",
            f"- [esoa] Deduplicated: {before:,} → {after:,} rows (removed {before - after:,} duplicates)",
        )
    
    # Save CSV only
    combined.to_csv(dest, index=False)
    return dest


def _resolve_esoa_source(inputs_dir: Path, esoa_hint: Optional[str]) -> Path:
    """Resolve the eSOA CSV path, concatenating esoa_pt_* files from raw/ when present."""
    raw_dir = PROJECT_ROOT / "raw" / "drugs"
    
    # If explicit hint provided, use it
    if esoa_hint:
        hint = Path(esoa_hint)
        if not hint.is_absolute():
            hint = (PROJECT_DIR / hint).resolve()
        if hint.is_file():
            return hint
        raise FileNotFoundError(f"Unable to resolve eSOA input at {hint}")
    
    # Check for esoa_pt_*.csv in raw/drugs/ and combine them
    part_files = sorted(raw_dir.glob("esoa_pt_*.csv"), key=_sort_esoa_parts)
    if part_files:
        return _concatenate_csv(part_files, inputs_dir / "esoa_combined.csv")
    
    # Fall back to existing combined file in inputs/
    for name in ("esoa_combined.csv", "esoa.csv", "esoa_prepared.csv"):
        candidate = inputs_dir / name
        if candidate.is_file():
            return candidate
    
    raise FileNotFoundError(
        f"No eSOA CSV found. Provide esoa_pt_*.csv files in raw/drugs/ or esoa_combined.csv in {inputs_dir}."
    )


def _ensure_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} missing at {path}")
    return path


def _find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def refresh_pnf(esoa_hint: Optional[str], *, verbose: bool = True) -> Path:
    """Run pipelines.drugs.scripts.prepare_drugs against the current PNF + eSOA inputs."""
    inputs_dir = _ensure_inputs_dir()
    raw_dir = PROJECT_ROOT / "raw" / "drugs"
    pnf_csv = _ensure_file(raw_dir / "pnf.csv", "PNF source CSV (in raw/drugs/)")
    esoa_csv = _resolve_esoa_source(inputs_dir, esoa_hint)
    if verbose:
        print(f"[pnf] Preparing PNF dataset from {pnf_csv} and {esoa_csv}")
    pnf_out, _ = prepare(str(pnf_csv), str(esoa_csv), str(inputs_dir))
    out_path = Path(pnf_out).resolve()
    if verbose:
        print(f"[pnf] Wrote normalized dataset to {out_path}")
    return out_path


def refresh_who(inputs_dir: Path, *, verbose: bool = True) -> Path:
    """Trigger the WHO ATC R scripts and return the freshest molecules export."""
    if verbose:
        print("[who] Running dependencies/atcd R scripts...")
    DrugsAndMedicinePipeline._run_r_scripts(PROJECT_DIR, inputs_dir)
    candidates = list(inputs_dir.glob("who_atc_*_molecules.csv"))
    candidates.extend(
        p
        for p in inputs_dir.glob("who_atc_*.csv")
        if re.fullmatch(r"who_atc_\d{4}-\d{2}-\d{2}\.csv", p.name)
    )
    latest = sorted(candidates)[-1] if candidates else None
    if latest is None:
        raise FileNotFoundError(
            "WHO ATC export not found after running the R scripts (expected who_atc_<date>.csv)."
        )
    if verbose:
        print(f"[who] Latest molecules export: {latest}")
    return latest


def refresh_fda_brand_map(inputs_dir: Path, *, verbose: bool = True) -> Path:
    if verbose:
        print("[fda_drug] Building FDA brand map export...")
    path = DrugsAndMedicinePipeline._build_brand_map(inputs_dir)
    if verbose:
        print(f"[fda_drug] Brand map available at {path}")
    return path


def refresh_fda_food(
    inputs_dir: Path, quiet: bool = True, *, allow_scrape: bool = False, verbose: bool = True
) -> Path:
    if verbose:
        print("[fda_food] Scraping FDA PH food catalog...")
    module_name = "dependencies.fda_ph_scraper.food_scraper"
    module_output_dir = PROJECT_DIR / "dependencies" / "fda_ph_scraper" / "output"
    module_output_dir.mkdir(parents=True, exist_ok=True)
    argv: List[str] = ["--outdir", str(module_output_dir)]
    if quiet:
        argv.append("--quiet")
    if allow_scrape:
        argv.append("--allow-scrape")
    _run_python_module(module_name, argv, verbose=verbose)
    food_outputs = [
        path
        for path in module_output_dir.glob("fda_food_*.csv")
        if "export" not in path.name and "products" not in path.name
    ]
    latest = sorted(food_outputs)[-1] if food_outputs else None
    if latest is None:
        raise FileNotFoundError("FDA food catalog not produced.")
    dest_path = inputs_dir / latest.name
    _mirror_module_output(latest, dest_path)
    # Clean up legacy copies if present
    for pattern in ("fda_food_products*.csv", "fda_food_export_*.csv"):
        for legacy in inputs_dir.glob(pattern):
            legacy.unlink(missing_ok=True)
    if verbose:
        print(f"[fda_food] Catalog refreshed at {dest_path}")
    return dest_path


def refresh_drugbank_generics_exports(*, verbose: bool = True) -> tuple[Optional[Path], Optional[Path]]:
    """Run DrugBank R scripts with minimal Python overhead using native shell."""
    import time
    import shutil
    
    drugbank_dir = PROJECT_DIR / "dependencies" / "drugbank_generics"
    
    # Find Rscript
    rscript = shutil.which("Rscript")
    if not rscript:
        raise FileNotFoundError("Rscript not found on PATH")
    
    # Default to 8 workers or fewer on smaller systems (AGENTS.md #6)
    import multiprocessing
    worker_count = min(8, multiprocessing.cpu_count())
    
    # Use lean export script (replaces all old R scripts)
    scripts = [
        "drugbank_lean_export.R",
    ]
    
    # Set env vars for R scripts
    os.environ["ESOA_DRUGBANK_WORKERS"] = str(worker_count)
    os.environ["ESOA_DRUGBANK_QUIET"] = "1"
    
    import threading
    
    for script_name in scripts:
        script_path = drugbank_dir / script_name
        if not script_path.is_file():
            print(f"[skip] {script_name} (not found)")
            continue
        
        # Build shell command - suppress ALL output (stdout AND stderr)
        if sys.platform == "win32":
            cmd = f'cd /d "{drugbank_dir}" && "{rscript}" "{script_path}" >nul 2>&1'
        else:
            cmd = f'cd "{drugbank_dir}" && "{rscript}" "{script_path}" >/dev/null 2>&1'
        
        # Run with live timer update
        start = time.perf_counter()
        done_event = threading.Event()
        exit_code_holder = [0]
        
        def run_r(shell_cmd=cmd):
            exit_code_holder[0] = os.system(shell_cmd)
            done_event.set()
        
        # Start R in background thread
        r_thread = threading.Thread(target=run_r, daemon=True)
        r_thread.start()
        
        # Update timer while R runs
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        idx = 0
        while not done_event.wait(0.1):
            elapsed = time.perf_counter() - start
            sys.stdout.write(f"\r{frames[idx % len(frames)]} {elapsed:7.2f}s {script_name}")
            sys.stdout.flush()
            idx += 1
        
        r_thread.join()
        elapsed = time.perf_counter() - start
        
        if exit_code_holder[0] == 0:
            sys.stdout.write(f"\r⣿ {elapsed:7.2f}s {script_name}\n")
        else:
            sys.stdout.write(f"\r✗ {elapsed:7.2f}s {script_name} (exit {exit_code_holder[0]})\n")
        sys.stdout.flush()
    
    module_output = drugbank_dir / "output"
    
    # Copy lean exports (CSV-only per updated AGENTS.md policy)
    lean_basenames = [
        "generics_lean",
        "synonyms_lean", 
        "dosages_lean",
        "atc_lean",
        "brands_lean",
        "salts_lean",
        "mixtures_lean",
        "products_lean",
        # Lookup tables
        "lookup_salt_suffixes",
        "lookup_pure_salts",
        "lookup_form_canonical",
        "lookup_route_canonical",
        "lookup_form_to_route",
        "lookup_per_unit",
    ]
    
    for basename in lean_basenames:
        for ext in (".csv",):
            source = module_output / f"{basename}{ext}"
            if source.is_file():
                _mirror_module_output(source, DRUGS_INPUTS_DIR / f"{basename}{ext}")
    
    inputs_generics = DRUGS_INPUTS_DIR / "generics_lean.csv"
    if not inputs_generics.exists():
        if verbose:
            print(f"[drugbank] Warning: {inputs_generics} not found after refresh.")
    
    return (
        inputs_generics if inputs_generics.is_file() else None,
        DRUGS_INPUTS_DIR / "brands_lean.csv" if (DRUGS_INPUTS_DIR / "brands_lean.csv").is_file() else None,
    )


def ensure_drugbank_mixtures_output(*, verbose: bool = True) -> Optional[Path]:
    output_path = DRUGS_INPUTS_DIR / "mixtures_lean.csv"
    if output_path.is_file():
        return output_path
    if verbose:
        print(f"[drugbank] Warning: mixtures_lean.csv not found at {output_path}")
    return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete 4-part drugs pipeline."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress messages.",
    )
    parser.add_argument(
        "--esoa",
        metavar="PATH",
        help="Path to eSOA CSV. Defaults to inputs/drugs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for Part 2 (default: 8).",
    )
    parser.add_argument(
        "--use-threads",
        action="store_true",
        help="Use thread pool instead of process pool for Part 2.",
    )
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Skip Excel output generation in Part 3.",
    )
    # Part selection
    parser.add_argument(
        "--only",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only the specified part (1-4).",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Start from the specified part (default: 1).",
    )
    # Skip flags for Part 1
    parser.add_argument(
        "--skip-who",
        action="store_true",
        help="Skip WHO ATC refresh in Part 1.",
    )
    parser.add_argument(
        "--skip-drugbank",
        action="store_true",
        help="Skip DrugBank refresh in Part 1.",
    )
    parser.add_argument(
        "--skip-fda-brand",
        action="store_true",
        help="Skip FDA brand map in Part 1.",
    )
    parser.add_argument(
        "--skip-fda-food",
        action="store_false",
        dest="include_fda_food",
        help="Skip FDA food catalog in Part 1.",
    )
    parser.add_argument(
        "--include-fda-food",
        action="store_true",
        default=True,
        dest="include_fda_food",
        help="Include FDA food catalog in Part 1 (default).",
    )
    parser.add_argument(
        "--skip-pnf",
        action="store_true",
        help="Skip PNF preparation in Part 1.",
    )
    parser.add_argument(
        "--allow-fda-food-scrape",
        action="store_true",
        default=False,
        help="Enable HTML scraping fallback for FDA food.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _ensure_inputs_dir()
    
    # Auto-purge old dated files in inputs/drugs (silent)
    purge_old_dated_files(DRUGS_INPUTS_DIR, quiet=True)

    print("=" * 60)
    print("ESOA DRUGS PIPELINE")
    print("=" * 60)
    capture_code_state()

    # Determine which parts to run
    if args.only:
        parts_to_run = [args.only]
    else:
        parts_to_run = list(range(args.start_from, 5))

    part2_stats: dict | None = None
    part3_stats: dict | None = None
    part4_stats: dict | None = None

    # Import part functions
    from run_drugs_pt_1_prepare_dependencies import run_part_1
    from pipelines.drugs.scripts.runners import run_annex_f_tagging, run_esoa_tagging, run_esoa_to_drug_code

    # Run selected parts
    if 1 in parts_to_run:
        print("PART 1: Prepare Dependencies")
        print("=" * 60)
        artifacts = run_part_1(
            esoa_path=args.esoa,
            skip_who=args.skip_who,
            skip_drugbank=args.skip_drugbank,
            skip_fda_brand=args.skip_fda_brand,
            skip_fda_food=not args.include_fda_food,
            skip_pnf=args.skip_pnf,
            allow_fda_food_scrape=args.allow_fda_food_scrape,
            standalone=False,
        )
        add_run_summary(
            "Part 1: Prepare Dependencies",
            [
                "- WHO ATC refreshed",
                "- DrugBank lean export refreshed",
                "- FDA brand map rebuilt",
                "- FDA food catalog refreshed",
                "- PNF prepared",
                "- Annex F verified",
            ],
        )

    if 2 in parts_to_run:
        print("\nPART 2: Match Annex F with ATC/DrugBank IDs")
        print("=" * 60)
        part2_stats = run_annex_f_tagging(verbose=False)
        lines = [
            f"- Total rows: {part2_stats['total']:,}",
            f"- Matched ATC: {part2_stats['matched_atc']:,} ({part2_stats['matched_atc_pct']:.1f}%)",
            f"- Matched DrugBank ID: {part2_stats['matched_drugbank']:,} ({part2_stats['matched_drugbank_pct']:.1f}%)",
            f"- Output: {part2_stats['output_path']}",
        ]
        lines.extend(_format_reason_lines(part2_stats.get("reason_counts", {}), part2_stats["total"]))
        add_run_summary("Part 2: Match Annex F with ATC/DrugBank IDs", lines)

    if 3 in parts_to_run:
        print("\nPART 3: Match ESOA with ATC/DrugBank IDs")
        print("=" * 60)
        from pathlib import Path
        esoa_path = Path(args.esoa) if args.esoa else None
        part3_stats = run_esoa_tagging(esoa_path=esoa_path, verbose=False, show_progress=True)
        lines = [
            f"- Total rows: {part3_stats['total']:,}",
            f"- Matched ATC: {part3_stats['matched_atc']:,} ({part3_stats['matched_atc_pct']:.1f}%)",
            f"- Matched DrugBank ID: {part3_stats['matched_drugbank']:,} ({part3_stats['matched_drugbank_pct']:.1f}%)",
            f"- Output: {part3_stats['output_path']}",
        ]
        lines.extend(_format_reason_lines(part3_stats.get("reason_counts", {}), part3_stats["total"]))
        add_run_summary("Part 3: Match ESOA with ATC/DrugBank IDs", lines)

    if 4 in parts_to_run:
        print("\nPART 4: Bridge ESOA to Annex F Drug Codes")
        print("=" * 60)
        part4_stats = run_esoa_to_drug_code(verbose=False)
        lines = [
            f"- Total rows: {part4_stats['total']:,}",
            f"- Matched drug codes: {part4_stats['matched']:,} ({part4_stats['matched_pct']:.1f}%)",
            f"- Output: {part4_stats['output_path']}",
        ]
        lines.extend(_format_reason_lines(part4_stats.get("reason_counts", {}), part4_stats["total"]))
        add_run_summary("Part 4: Bridge ESOA to Annex F Drug Codes", lines)

    overall_lines: list[str] = []
    if part3_stats:
        overall_lines.append(
            f"- ESOA ATC coverage: {part3_stats['matched_atc']:,}/{part3_stats['total']:,} ({part3_stats['matched_atc_pct']:.1f}%)"
        )
        overall_lines.append(
            f"- ESOA DrugBank coverage: {part3_stats['matched_drugbank']:,}/{part3_stats['total']:,} ({part3_stats['matched_drugbank_pct']:.1f}%)"
        )
    if part4_stats:
        overall_lines.append(
            f"- ESOA → Drug code coverage: {part4_stats['matched']:,}/{part4_stats['total']:,} ({part4_stats['matched_pct']:.1f}%)"
        )
        overall_lines.append(f"- Final output: {part4_stats['output_path']}")
    if overall_lines:
        add_run_summary("Overall", overall_lines)

    write_run_summary()

    print("\nPIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
