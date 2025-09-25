#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resolve_unknowns.py
-------------------
Reads ./outputs/unknown_words.csv and looks up whether each unknown token
appears (whole or partial) in any generic name listed in:
  • PNF: ./inputs/pnf_prepared.csv   (column: generic_name)
  • FDA brand map: newest of ./inputs/fda_brand_map_*.csv   (column: generic_name)
  • WHO ATC molecules: newest of ./dependencies/atcd/output/who_atc_*_molecules.csv (column: atc_name)

Outputs ./outputs/missed_generics.csv with columns:
  unknown_word, source, reference_name, match_kind, reference_path
"""

import csv
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
WHO_DIR = ROOT / "dependencies" / "atcd" / "output"

def _read_csv_firstcol(path: Path) -> List[str]:
    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            _header = next(reader)
        except StopIteration:
            return out
        for row in reader:
            if not row:
                continue
            out.append((row[0] or "").strip())
    return [x for x in out if x]

def _read_col(path: Path, colname: str) -> List[str]:
    if not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or colname not in reader.fieldnames:
            return out
        for r in reader:
            v = (r.get(colname) or "").strip()
            if v:
                out.append(v)
    return out

def _pick_newest(pattern: Path) -> Optional[Path]:
    files = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)
    return Path(files[0]) if files else None

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def _variants(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    v = {_norm(s)}
    v.add(_norm(s.replace("_", " ")))
    v.add(_norm(s.replace("-", " ")))
    return list(v)

def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _norm(s))

def _match_kind(unknown: str, ref: str) -> Tuple[bool, str]:
    """Return (is_match, kind) where kind is 'whole' or 'partial'."""
    u = _norm(unknown)
    if not u:
        return (False, "")
    ref_norm = _norm(ref)
    if not ref_norm:
        return (False, "")
    if u in ref_norm:
        tokens = _tokenize(ref_norm)
        if u in tokens:
            return (True, "whole")
        return (True, "partial")
    for uv in _variants(u):
        if uv and uv in ref_norm:
            tokens = _tokenize(ref_norm)
            if uv in tokens:
                return (True, "whole")
            return (True, "partial")
    return (False, "")

def main():
    unknowns_path = OUTPUTS / "unknown_words.csv"
    if not unknowns_path.is_file():
        print(f"ERROR: {unknowns_path} not found.", file=sys.stderr)
        sys.exit(1)

    unknown_words = _read_csv_firstcol(unknowns_path)
    if not unknown_words:
        print("No unknown words found (empty file)." )
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        outpath = OUTPUTS / "missed_generics.csv"
        with outpath.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["unknown_word","source","reference_name","match_kind","reference_path"])
        print(str(outpath))
        return

    pnf_path = INPUTS / "pnf_prepared.csv"
    pnf_names = _read_col(pnf_path, "generic_name")

    fda_path = _pick_newest(INPUTS / "fda_brand_map_*.csv")
    fda_names = _read_col(fda_path, "generic_name") if fda_path else []

    if not fda_names:
        legacy_fda = _pick_newest(INPUTS / "brand_map_*.csv")
        if legacy_fda:
            fda_path = legacy_fda
            fda_names = _read_col(legacy_fda, "generic_name")

    who_path = _pick_newest(ROOT / "dependencies" / "atcd" / "output" / "who_atc_*_molecules.csv")
    who_names = _read_col(who_path, "atc_name") if who_path else []

    sources = [
        ("PNF", pnf_names, pnf_path if pnf_path.is_file() else None),
        ("FDA", fda_names, fda_path if fda_path else None),
        ("WHO", who_names, who_path if who_path else None),
    ]

    results = []
    for uw in unknown_words:
        if not uw:
            continue
        for src_name, names, src_path in sources:
            if not names:
                continue
            for ref in names:
                ok, kind = _match_kind(uw, ref)
                if ok:
                    results.append([uw, src_name, ref, kind, str(src_path) if src_path else ""])

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUTS / "missed_generics.csv"
    with outpath.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unknown_word","source","reference_name","match_kind","reference_path"])
        writer.writerows(results)

    print(str(outpath))

if __name__ == "__main__":
    main()
