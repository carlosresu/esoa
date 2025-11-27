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

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, TypeVar

import pandas as pd

from pipelines.drugs.constants import PIPELINE_INPUTS_DIR, PROJECT_ROOT
from pipelines.drugs.pipeline import DrugsAndMedicinePipeline
from pipelines.drugs.scripts.prepare_drugs import prepare

PROJECT_DIR = PROJECT_ROOT
DRUGS_INPUTS_DIR = PIPELINE_INPUTS_DIR
T = TypeVar("T")


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
    """Run func() while showing a lightweight CLI spinner with elapsed time."""
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


def _run_r_script(script_path: Path, *, verbose: bool = True) -> None:
    rscript = _find_rscript()
    if not rscript:
        raise FileNotFoundError(
            "Rscript executable not found. Install R or set RSCRIPT_PATH/R_HOME or add Rscript to PATH."
        )
    env = os.environ.copy()
    env.setdefault("ESOA_DRUGBANK_QUIET", "1")
    env.setdefault("RSCRIPT_PATH", str(rscript))
    # Allow R helpers to use all local cores by default (tunable via env).
    if "ESOA_DRUGBANK_WORKERS" not in env:
        cpu_count = os.cpu_count() or 13
        env["ESOA_DRUGBANK_WORKERS"] = str(max(1, cpu_count - 1))
    log_path: Path | None = None
    if not verbose:
        log_path = script_path.with_suffix(".log")
        stdout = log_path.open("w", encoding="utf-8")
        stderr = subprocess.STDOUT
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
        if log_path and log_path.is_file():
            print(f"[rscript] Failure log: {log_path}")
        raise RuntimeError(f"{script_path.name} exited with status {exc.returncode}") from exc
    finally:
        if stdout not in (None, sys.stdout, sys.stderr):
            try:
                stdout.close()  # type: ignore[arg-type]
            except Exception:
                pass


def _copy_to_pipeline_inputs(source: Path, dest: Path) -> None:
    """Mirror a module export into the Drugs pipeline inputs directory."""
    if not source.is_file():
        raise FileNotFoundError(f"Expected output from module at {source}, but it was not created.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    parquet_src = source.with_suffix(".parquet")
    if parquet_src.is_file():
        shutil.copy2(parquet_src, dest.with_suffix(".parquet"))


def _ensure_parquet_sibling(csv_path: Path, *, verbose: bool = True) -> Optional[Path]:
    """Create a Parquet sibling for a CSV if missing."""
    if csv_path.suffix.lower() != ".csv":
        return None
    parquet_path = csv_path.with_suffix(".parquet")
    if parquet_path.exists():
        return parquet_path
    try:
        df = pd.read_csv(csv_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        if verbose:
            print(f"[parquet] Warning: could not create {parquet_path} from {csv_path}: {exc}")
        return None


def _natural_esoa_part_order(path: Path) -> tuple[int, str]:
    """Sort helper that orders esoa_pt_* files by their numeric suffix."""
    for token in path.stem.split("_"):
        if token.isdigit():
            return int(token), path.name
    return sys.maxsize, path.name


def _concatenate_csv(parts: Sequence[Path], dest: Path) -> Path:
    """Concatenate multiple CSV files (identical headers) into dest."""
    header: List[str] | None = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as out_handle:
        writer: csv.writer | None = None
        for part in parts:
            if not part.is_file():
                continue
            with part.open("r", newline="", encoding="utf-8-sig") as in_handle:
                reader = csv.reader(in_handle)
                try:
                    file_header = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = file_header
                    writer = csv.writer(out_handle)
                    writer.writerow(header)
                elif header != file_header:
                    raise ValueError(
                        f"Header mismatch while concatenating {part.name}; expected {header} but found {file_header}."
                    )
                assert writer is not None
                writer.writerows(reader)
    _ensure_parquet_sibling(dest)
    return dest


def _resolve_esoa_source(inputs_dir: Path, esoa_hint: Optional[str]) -> Path:
    """Resolve the eSOA CSV path, concatenating esoa_pt_* files when present."""
    search_dirs: List[Path] = []
    if esoa_hint:
        hint = Path(esoa_hint)
        if not hint.is_absolute():
            hint = (PROJECT_DIR / hint).resolve()
        if hint.is_dir():
            search_dirs.append(hint)
        elif hint.is_file():
            return hint
        else:
            candidate = inputs_dir / hint.name
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"Unable to resolve eSOA input at {hint}")
    search_dirs.append(inputs_dir)
    seen: set[Path] = set()
    for directory in search_dirs:
        directory = directory.resolve()
        if directory in seen:
            continue
        seen.add(directory)
        part_files = sorted(directory.glob("esoa_pt_*.csv"), key=_natural_esoa_part_order)
        if part_files:
            return _concatenate_csv(part_files, directory / "esoa_combined.csv")
        for name in ("esoa_combined.csv", "esoa.csv", "esoa_prepared.csv"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"No eSOA CSV found. Provide esoa_pt_*.csv files or esoa.csv under {inputs_dir} (or use --esoa)."
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
    pnf_csv = _ensure_file(inputs_dir / "pnf.csv", "PNF source CSV")
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
    _copy_to_pipeline_inputs(latest, dest_path)
    # Clean up legacy copies if present
    for pattern in ("fda_food_products*.csv", "fda_food_products*.parquet", "fda_food_export_*.csv"):
        for legacy in inputs_dir.glob(pattern):
            legacy.unlink(missing_ok=True)
    if verbose:
        print(f"[fda_food] Catalog refreshed at {dest_path}")
    return dest_path


def refresh_drugbank_generics_exports(*, verbose: bool = True) -> tuple[Optional[Path], Optional[Path]]:
    """Invoke the DrugBank R helper to regenerate generics + brand exports."""
    if verbose:
        print("[drugbank_generics] Launching dependencies/drugbank_generics/drugbank_all.R...")
    script_path = PROJECT_DIR / "dependencies" / "drugbank_generics" / "drugbank_all.R"
    if not script_path.is_file():
        raise FileNotFoundError(f"DrugBank helper not found at {script_path}")
    _run_r_script(script_path, verbose=verbose)
    module_output = script_path.parent / "output"
    for filename in (
        "drugbank_generics_master.csv",
        "drugbank_mixtures_master.csv",
        "drugbank_brands_master.csv",
        "drugbank_products_export.csv",
        "drugbank_salts_master.csv",
        "drugbank_pure_salts.csv",
        "drugbank_salt_suffixes.csv",
    ):
        source = module_output / filename
        if source.is_file():
            _copy_to_pipeline_inputs(source, DRUGS_INPUTS_DIR / filename)
    inputs_generics_master = DRUGS_INPUTS_DIR / "drugbank_generics_master.csv"
    inputs_brands = DRUGS_INPUTS_DIR / "drugbank_brands.csv"
    if not inputs_generics_master.exists():
        if verbose:
            print(f"[drugbank_generics] Warning: {inputs_generics_master} not found after refresh.")
    if not inputs_brands.exists():
        if verbose:
            print(
                "[drugbank_generics] Note: drugbank_brands.csv was not produced. "
                "Run with --include-drugbank-brands when the placeholder script is implemented."
            )
    return (
        inputs_generics_master if inputs_generics_master.is_file() else None,
        inputs_brands if inputs_brands.is_file() else None,
    )


def ensure_drugbank_mixtures_output(*, verbose: bool = True) -> Optional[Path]:
    output_path = DRUGS_INPUTS_DIR / "drugbank_mixtures_master.csv"
    if output_path.is_file():
        return output_path
    if verbose:
        print(f"[drugbank_mixtures] Warning: expected output not found at {output_path}")
    return None


def _maybe_run_drugbank_brands_script(include_flag: bool, *, verbose: bool = True) -> None:
    if not include_flag:
        return
    script_path = PROJECT_DIR / "dependencies" / "drugbank_generics" / "drugbank_brands.R"
    if not script_path.is_file():
        if verbose:
            print(f"[drugbank_brands] Placeholder script not found at {script_path}; skipping.")
        return
    rscript = shutil.which("Rscript")
    if not rscript:
        if verbose:
            print("[drugbank_brands] Rscript executable not found; cannot run placeholder.")
        return
    if verbose:
        print(f"[drugbank_brands] Executing placeholder R script {script_path}...")
    env = os.environ.copy()
    env.setdefault("ESOA_DRUGBANK_QUIET", "1")
    subprocess.run([rscript, str(script_path)], check=True, cwd=str(script_path.parent), env=env)




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

    print("=" * 60)
    print("ESOA DRUGS PIPELINE")
    print("=" * 60)

    # Determine which parts to run
    if args.only:
        parts_to_run = [args.only]
    else:
        parts_to_run = list(range(args.start_from, 5))

    # Import part functions
    from run_drugs_pt_1_prepare_dependencies import run_part_1
    from pipelines.drugs.scripts.runners import run_annex_f_tagging, run_esoa_tagging
    from run_drugs_pt_4_esoa_to_annex_f import run_part_4

    # Run selected parts
    if 1 in parts_to_run:
        print("\n" + "=" * 60)
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
        print("\nPart 1 artifacts:")
        for label, path in artifacts.items():
            print(f"  - {label}: {path}")

    if 2 in parts_to_run:
        print("\n" + "=" * 60)
        print("PART 2: Match Annex F with ATC/DrugBank IDs")
        print("=" * 60)
        results = run_annex_f_tagging(verbose=False)
        print(f"\nPart 2 results:")
        print(f"  - Matched with ATC: {results['matched_atc']} ({results['matched_atc_pct']:.1f}%)")
        print(f"  - Has DrugBank ID: {results['matched_drugbank']} ({results['matched_drugbank_pct']:.1f}%)")

    if 3 in parts_to_run:
        print("\n" + "=" * 60)
        print("PART 3: Match ESOA with ATC/DrugBank IDs")
        print("=" * 60)
        from pathlib import Path
        esoa_path = Path(args.esoa) if args.esoa else None
        results = run_esoa_tagging(esoa_path=esoa_path, verbose=False)
        print(f"\nPart 3 results:")
        print(f"  - Total rows: {results['total']}")
        print(f"  - Matched with ATC: {results['matched_atc']} ({results['matched_atc_pct']:.1f}%)")
        print(f"  - Output: {results['output_path']}")

    if 4 in parts_to_run:
        print("\n" + "=" * 60)
        print("PART 4: Bridge ESOA to Annex F Drug Codes")
        print("=" * 60)
        results = run_part_4(
            esoa_atc_filename="esoa_with_atc.csv",
            annex_atc_filename="annex_f_with_atc.csv",
            out_filename="esoa_matched_drug_codes.csv",
            standalone=False,
        )
        print(f"\nPart 4 results:")
        print(f"  - Matched to Drug Code: {results['matched']} ({results['matched_pct']:.1f}%)")
        print(f"  - Unmatched: {results['unmatched']}")
        print(f"  - Output: {results['output_path']}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
