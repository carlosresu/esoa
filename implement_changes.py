#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implement_changes.py
--------------------
Open and run directly (no arguments needed). Designed to be run from VS Code's
"Run Python File" without using the terminal for options.

Behavior:
- Looks for ./debug/master.py (fallback: ./master.py) containing a region:
    # START OF REPO FILES
      ... blocks ...
    # END OF REPO FILES
- Each file block starts with a header:
    "# <path/to/file.py>"
  or "# File: path/to/file.py"
- Writes each block to its corresponding path (root *.py or scripts/*.py),
  OVERWRITING existing files.

A short summary is printed at the end.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Config (no CLI required) ---
DEFAULT_MASTER_CANDIDATES = [Path("./debug/master.py"), Path("./master.py")]
REPO_ROOT = Path(".")

RE_START = re.compile(r'^\s*#\s*START OF REPO FILES\s*$', re.IGNORECASE)
RE_END   = re.compile(r'^\s*#\s*END OF REPO FILES\s*$', re.IGNORECASE)

RE_HEADER_ANGLE = re.compile(r'^\s*#\s*<\s*(?P<path>[^>]+?)\s*>\s*$')   # "# <scripts/aho.py>"
RE_HEADER_FILE  = re.compile(r'^\s*#\s*File:\s*(?P<path>.+?)\s*$')      # "# File: scripts/aho.py"

def _is_allowed_target(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    parts = path.parts
    if len(parts) == 1:
        return True  # root-level .py like run.py
    if parts[0] == "scripts" and len(parts) >= 2:
        return True
    return False

def _find_master_file() -> Path:
    for p in DEFAULT_MASTER_CANDIDATES:
        if p.exists():
            return p
    # If nothing found, point to the first candidate for the error message
    return DEFAULT_MASTER_CANDIDATES[0]

def parse_master(master_path: Path) -> Dict[Path, str]:
    text = master_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    try:
        start_idx = next(i for i, ln in enumerate(lines) if RE_START.match(ln))
        end_idx   = next(i for i, ln in enumerate(lines[start_idx+1:], start=start_idx+1) if RE_END.match(ln))
    except StopIteration:
        raise RuntimeError("Could not find '# START OF REPO FILES' / '# END OF REPO FILES' region in master file.")

    region = lines[start_idx+1:end_idx]

    files: Dict[Path, List[str]] = {}
    current_path: Optional[Path] = None
    current_buf: List[str] = []

    def flush_current():
        nonlocal current_path, current_buf, files
        if current_path is not None:
            files[current_path] = current_buf[:]
        current_path = None
        current_buf = []

    for ln in region:
        m_angle = RE_HEADER_ANGLE.match(ln)
        m_file  = RE_HEADER_FILE.match(ln)
        if m_angle or m_file:
            flush_current()
            hdr_path = (m_angle or m_file).group("path").strip()
            hdr_path = hdr_path.split()[0]  # take first token only
            target = Path(Path(hdr_path).as_posix())
            current_path = target if _is_allowed_target(target) else None
            continue

        if current_path is not None:
            current_buf.append(ln)

    flush_current()

    # Normalize to single trailing newline
    return {p: ("\n".join(buf).rstrip("\n") + "\n") for p, buf in files.items()}

def write_outputs(mapping: Dict[Path, str], repo_root: Path) -> List[Tuple[Path, int]]:
    written: List[Tuple[Path, int]] = []
    for rel, content in mapping.items():
        target = (repo_root / rel).resolve()
        if not str(target).startswith(str(repo_root.resolve())):
            raise RuntimeError(f"Unsafe path resolved outside repo: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        written.append((target, len(content.encode("utf-8"))))
    return written

def main() -> int:
    master_path = _find_master_file()
    if not master_path.exists():
        print(f"[ERROR] Master file not found at expected location: {master_path}", file=sys.stderr)
        print("Tip: Place your concatenated file at ./debug/master.py", file=sys.stderr)
        return 2

    try:
        mapping = parse_master(master_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse master file: {e}", file=sys.stderr)
        return 3

    if not mapping:
        print("[WARN] No eligible .py files found to write. Ensure headers like '# <scripts/xxx.py>' or '# File: scripts/xxx.py' are within the START/END region.")
        return 0

    written = write_outputs(mapping, REPO_ROOT)

    print("\nApplied changes from:", master_path.as_posix())
    print("Overwritten files:")
    for path, nbytes in written:
        try:
            rel = path.relative_to(REPO_ROOT.resolve())
        except ValueError:
            rel = path
        print(f"  • {rel}  ({nbytes} bytes)")

    print("\nDone.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
