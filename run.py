#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

# --- Ensure repo root is importable (robust when invoked from other CWDs) ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


def install_requirements(req_path: str):
    """Install requirements with the same Python interpreter running this script."""
    if not req_path:
        return
    if not os.path.isfile(req_path):
        print(f">>> Skipping install: requirements file not found: {req_path}")
        return
    print(f">>> Installing dependencies from: {req_path}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", req_path],
        cwd=THIS_DIR,
    )
    print(">>> Dependencies installed.")


def _resolve_input_path(p: str, default_subdir: str = "inputs") -> str:
    """
    Resolve an input path with smart fallbacks:
      1) If 'p' exists as-is -> return it
      2) If basename(p) exists under ./inputs/ -> return that
      3) Otherwise -> raise FileNotFoundError with a helpful message
    """
    if not p:
        raise FileNotFoundError("No input path provided")

    # Absolute or relative path provided and exists
    if os.path.isfile(p):
        return p

    # Try ./inputs/<basename>
    base = os.path.basename(p)
    candidate = os.path.join(THIS_DIR, default_subdir, base)
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(
        f"Input file not found: {p!r}. "
        f"Tried: {os.path.abspath(p)} and {candidate!r}. "
        "Place the file under ./inputs/ or pass --pnf/--esoa with a correct path."
    )


def _outputs_dir() -> str:
    """
    Always use ./outputs relative to repo root for intermediate + final outputs.
    Creates the directory if missing.
    """
    d = os.path.join(THIS_DIR, "outputs")
    os.makedirs(d, exist_ok=True)
    return d


def main_entry():
    parser = argparse.ArgumentParser(description="Run full ESOA pipeline (prepare → match)")
    parser.add_argument(
        "--pnf",
        default="inputs/pnf.csv",
        help="Path to PNF CSV (default: inputs/pnf.csv)",
    )
    parser.add_argument(
        "--esoa",
        default="inputs/esoa.csv",
        help="Path to ESOA CSV (default: inputs/esoa.csv)",
    )
    parser.add_argument(
        "--out",
        default="esoa_matched.csv",
        help="Final matched CSV filename (always written to ./outputs/)",
    )
    parser.add_argument("--requirements", default="", help="Optional requirements.txt to install")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install step")

    args = parser.parse_args()

    if args.requirements and not args.skip_install:
        install_requirements(args.requirements)

    # Defer heavy imports until after optional install
    import main  # noqa: E402

    # Resolve inputs and outputs
    pnf_path = _resolve_input_path(args.pnf)
    esoa_path = _resolve_input_path(args.esoa)

    # Force all outputs to ./outputs/
    outdir = _outputs_dir()
    out_path = os.path.join(outdir, os.path.basename(args.out))

    print(">>> Resolved paths:")
    print(f"    PNF   : {pnf_path}")
    print(f"    ESOA  : {esoa_path}")
    print(f"    OUTDIR: {outdir}")
    print(f"    OUT   : {out_path}")

    # Run the full pipeline (always writing under ./outputs)
    main.run_all(pnf_path, esoa_path, outdir, out_path)


if __name__ == "__main__":
    try:
        main_entry()
    except Exception as e:
        # Bubble up with a helpful message; keep original traceback for debugging
        print(f"ERROR: {e}", file=sys.stderr)
        raise
