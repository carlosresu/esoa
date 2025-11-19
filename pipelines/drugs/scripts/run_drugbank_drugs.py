#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..constants import (
    PIPELINE_DRUGBANK_BRANDS_PATH,
    PIPELINE_DRUGBANK_GENERICS_PATH,
    PIPELINE_INPUTS_DIR,
)


def stream_r_script(executable: str, script_path: Path, *, env: dict[str, str] | None = None) -> int:
    """Run the R script while streaming its combined stdout/stderr."""
    process = subprocess.Popen(
        [executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None  # stdout is redirected above
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    finally:
        process.stdout.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    r_script = project_root / "dependencies" / "drugbank_generics" / "drugbank_all.R"

    if not r_script.exists():
        sys.stderr.write(f"R script not found at {r_script}\n")
        sys.exit(1)

    print(f"Launching DrugBank aggregation via {r_script}")

    env = os.environ.copy()
    env.setdefault("ESOA_DRUGBANK_QUIET", "1")

    try:
        return_code = stream_r_script("Rscript", r_script, env=env)
    except FileNotFoundError:
        sys.stderr.write("Rscript executable not found. Please install R or adjust PATH.\n")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted by user.\n")
        sys.exit(130)

    if return_code != 0:
        sys.stderr.write(
            f"R script exited with status {return_code}. See output above for details.\n"
        )
        sys.exit(return_code)

    output_dir = project_root / "dependencies" / "drugbank_generics" / "output"
    copies: dict[str, list[Path]] = {
        "drugbank_generics_master.csv": [
            output_dir / "drugbank_generics.csv",
            PIPELINE_DRUGBANK_GENERICS_PATH,
            PIPELINE_INPUTS_DIR / "drugbank_generics_master.csv",
            PIPELINE_INPUTS_DIR / "generics.csv",  # legacy fallback
        ],
        "drugbank_mixtures_master.csv": [
            PIPELINE_INPUTS_DIR / "drugbank_mixtures_master.csv",
        ],
    }

    brand_source = output_dir / "drugbank_brands.csv"
    if brand_source.is_file():
        copies["drugbank_brands.csv"] = [PIPELINE_DRUGBANK_BRANDS_PATH]

    def _resolve_source(path_candidates: list[Path]) -> Path | None:
        for candidate in path_candidates:
            if candidate.is_file():
                return candidate
        return None

    def _same_path(left: Path, right: Path) -> bool:
        return left.resolve(strict=False) == right.resolve(strict=False)

    for filename, targets in copies.items():
        source_candidates = [
            PIPELINE_INPUTS_DIR / filename,
            output_dir / filename,
        ]
        source = _resolve_source(source_candidates)
        if source is None:
            sys.stderr.write(f"Warning: expected {filename} not found in output paths; nothing copied.\n")
            continue
        for target in targets:
            if _same_path(source, target):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"Copied {source.name} to {target}")


if __name__ == "__main__":
    main()
