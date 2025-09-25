# ===============================
# File: run.py (top-level)
# ===============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py — Install requirements, (optionally) run ATC preprocessing (R scripts from ./dependencies/atcd),
then import main.py and run the pipeline.

Usage examples:
  python run.py
  python run.py --pnf inputs/pnf.csv --esoa inputs/esoa.csv --out esoa_matched.csv
  python run.py --requirements requirements.txt
  python run.py --skip-r
  python run.py --skip-install
"""

import argparse
import os
import shutil
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
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-r", req_path],
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


def run_r_scripts():
    """
    Run the ATC R preprocessing scripts with cwd=./dependencies/atcd
    so relative outputs are written to ./dependencies/atcd/output/*

    NOTE: scripts.match looks for WHO molecules under ./dependencies/atcd/output,
    so we ensure that directory exists.
    """
    atcd_dir = os.path.join(THIS_DIR, "dependencies", "atcd")

    if not os.path.isdir(atcd_dir):
        raise FileNotFoundError(f"ATC directory not found: {atcd_dir}")

    out_dir = os.path.join(atcd_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError("Rscript not found in PATH. Please install R and ensure 'Rscript' is available.")

    scripts = ["atcd.R", "export.R", "filter.R"]

    for script in scripts:
        script_path = os.path.join(atcd_dir, script)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Required R script not found: {script_path}")

    for script in scripts:
        print(f">>> Running R script (cwd={atcd_dir}): {script}")
        subprocess.run([rscript, script], check=True, cwd=atcd_dir)

    print(">>> All R scripts completed successfully.")

def create_master_file(root_dir: str):
    """
    Concatenates all specified Python files into a single master.py file.
    """
    debug_dir = os.path.join(root_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    output_file_path = os.path.join(debug_dir, "master.py")
    
    # Files to be included in the master file
    files_to_concatenate = [
        os.path.join(root_dir, "scripts", "aho.py"),
        os.path.join(root_dir, "scripts", "combos.py"),
        os.path.join(root_dir, "scripts", "dose.py"),
        os.path.join(root_dir, "scripts", "match.py"),
        os.path.join(root_dir, "scripts", "prepare.py"),
        os.path.join(root_dir, "scripts", "routes_forms.py"),
        os.path.join(root_dir, "scripts", "text_utils.py"),
        os.path.join(root_dir, "scripts", "who_molecules.py"),
        os.path.join(root_dir, "main.py"),
        os.path.join(root_dir, "run.py"),
    ]

    header_text = """
# INSTRUCTIONS:
# 1. With every query, I will provide the file contents of my Python repository.
# 2. The files are marked with their paths as comments (e.g., # <file_path>).
# 3. Your task is to analyze the provided code, then respond to my queries by providing the complete/corrected/expanded code for any file that needs to be changed or created, as a downloadable file(s).
# 4. For each file you modify or create, use the following format:
# - change 1: edit <file_path> by pasting the below
#   <entire_file_content_here>
# - change 2: create <file_path> by pasting the below
#   <entire_file_content_here>
# - <and so on...>
# 5. DO NOT SEND ANY CODE IN CHAT. Only give me drop-in files I replace my files with.

# START OF REPO FILES
"""

    footer_text = """
# END OF REPO FILES
"""

    print(">>> Creating master.py...")
    
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        outfile.write(header_text)
        
        for file_path in files_to_concatenate:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found, skipping: {file_path}")
                continue
            
            relative_path = os.path.relpath(file_path, root_dir)
            outfile.write(f"\n# <{relative_path}>\n")
            
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
            
            outfile.write("\n")
        
        outfile.write(footer_text)
        
    print(f">>> master.py created successfully at: {output_file_path}")


def main_entry():
    parser = argparse.ArgumentParser(description="Run full ESOA pipeline (ATC → prepare → match)")
    parser.add_argument(
        "--pnf",
        default="inputs/pnf.csv",
        help="Path to raw PNF CSV (default: inputs/pnf.csv)",
    )
    parser.add_argument(
        "--esoa",
        default="inputs/esoa.csv",
        help="Path to raw eSOA CSV (default: inputs/esoa.csv)",
    )
    parser.add_argument(
        "--out",
        default="esoa_matched.csv",
        help="Final matched CSV filename (always written to ./outputs/)",
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Path to requirements file (default: requirements.txt)",
    )
    parser.add_argument("--skip-install", action="store_true", help="Skip installing requirements")
    parser.add_argument("--skip-r", action="store_true", help="Skip running ATC R scripts")
    args = parser.parse_args()

    # Create the master.py file first
    create_master_file(THIS_DIR)

    if not args.skip_install and args.requirements:
        install_requirements(args.requirements)

    if not args.skip_r:
        run_r_scripts()

    # Defer heavy imports until after optional install & R preprocessing
    import main  # noqa: E402

    # Resolve inputs and outputs
    pnf_path = _resolve_input_path(args.pnf)
    esoa_path = _resolve_input_path(args.esoa)

    # Force all pipeline outputs to ./outputs/
    outdir = _outputs_dir()
    out_path = os.path.join(outdir, os.path.basename(args.out))

    print(">>> Resolved paths:")
    print(f"    PNF   : {pnf_path}")
    print(f"    ESOA  : {esoa_path}")
    print(f"    OUTDIR: {outdir}")
    print(f"    OUT   : {out_path}")

    # Run the full pipeline (always writing under ./outputs)
    main.run_all(pnf_path, esoa_path, outdir, out_path)

    print("\n>>> Pipeline complete. Final output:", out_path)


if __name__ == "__main__":
    try:
        main_entry()
    except Exception as e:
        # Bubble up with a helpful message; keep original traceback for debugging
        print(f"ERROR: {e}", file=sys.stderr)
        raise