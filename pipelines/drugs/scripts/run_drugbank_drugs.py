#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


def stream_r_script(executable: str, script_path: Path) -> int:
    """Run the R script while streaming its combined stdout/stderr."""
    process = subprocess.Popen(
        [executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
    r_script = project_root / "dependencies" / "drugbank_generics" / "drugbank.R"

    if not r_script.exists():
        sys.stderr.write(f"R script not found at {r_script}\n")
        sys.exit(1)

    print(f"Launching DrugBank aggregation via {r_script}")

    try:
        return_code = stream_r_script("Rscript", r_script)
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


if __name__ == "__main__":
    main()
