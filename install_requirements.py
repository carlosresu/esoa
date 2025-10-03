#!/usr/bin/env python3
"""Install project dependencies defined in requirements.txt."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    if not requirements_path.exists():
        print("requirements.txt not found.", file=sys.stderr)
        sys.exit(1)

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    print("Running:", " ".join(cmd))

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"pip install failed with exit code {exc.returncode}.", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
