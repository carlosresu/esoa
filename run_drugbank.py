#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    r_script = project_root / "dependencies" / "drugbank" / "drugbank.R"

    if not r_script.exists():
        sys.stderr.write(f"R script not found at {r_script}\n")
        sys.exit(1)

    try:
        subprocess.run(["Rscript", str(r_script)], check=True)
    except FileNotFoundError:
        sys.stderr.write("Rscript executable not found. Please install R or adjust PATH.\n")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"R script exited with status {exc.returncode}. See output above for details.\n"
        )
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
