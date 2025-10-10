#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resolve_unknowns.py (fast n-gram token-boundary matches with source priority)
-----------------------------------------------------------------------------
Ultra-fast matching by pre-indexing UNKNOWN phrases as token n-grams.
For each reference string, we tokenize and scan all contiguous n-grams whose
lengths exist in the unknowns index, doing O(tokens) hash lookups.

Rules:
- Tokens are sequences of [a-z0-9], splitting on spaces/underscores/hyphens and other non-alphanumerics.
- NO within-token substring matches.
- "whole": unknown tokens equal the entire reference tokens.
- "partial": unknown tokens are a proper contiguous subset of the reference tokens
             (e.g., "tranexamic" matches "tranexamic acid", "tranexamic_acid", "tranexamic-acid").

Input:
  ./outputs/unknown_words.csv  (columns: word,count)

Search lists:
  • PNF: ./inputs/pnf_prepared.csv (generic_name)
  • FDA brand map: newest of ./inputs/fda_brand_map_*.csv OR ./inputs/brand_map_*.csv (generic_name)
  • WHO ATC: newest of ./dependencies/atcd/output/who_atc_*_molecules.csv (atc_name)

Output:
  ./outputs/missed_generics.csv with columns:
    unknown_word, unknown_count, source, reference_name, match_kind, reference_path

Priority:
  PNF over WHO over FDA (i.e., if an unknown matches in PNF, ignore WHO/FDA; if not, use WHO; else FDA).
"""
import csv
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

from scripts.reference_data import load_drugbank_generics, load_ignore_words

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
WHO_DIR = ROOT / "dependencies" / "atcd" / "output"

# Common nouns that should be ignored (already handled in match_outputs but kept for safety)
COMMON_UNKNOWN_STOPWORDS = {
    "bottle",
    "bottles",
    "box",
    "boxes",
    "syringe",
    "syringes",
    "softgel",
    "softgels",
    "mc",
    "none",
    "content",
}

COMMON_UNKNOWN_STOPWORDS |= set(load_ignore_words())
_, _drugbank_tokens, _ = load_drugbank_generics()
COMMON_UNKNOWN_STOPWORDS |= set(_drugbank_tokens)

# -----------------------------
# Helpers
# -----------------------------
_token_re = re.compile(r"[a-z0-9]+")

def _norm(s: str) -> str:
    """Lowercase a string and collapse internal whitespace for stable tokenization."""
    return re.sub(r"\s+", " ", s.lower().strip())

def _tokens(s: str) -> List[str]:
    """Split normalized text into alphanumeric tokens used by the n-gram index."""
    return _token_re.findall(_norm(s))

def _read_unknowns_with_counts(path: Path) -> Dict[str, int]:
    """Read unknown_words.csv with header [word,count] -> dict word->count."""
    out: Dict[str, int] = {}
    # Bail out when the source file is absent.
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "word" not in reader.fieldnames:
            return out
        for r in reader:
            # Normalize the unknown token and associated count field.
            w = (r.get("word") or "").strip()
            c_raw = (r.get("count") or "").strip()
            if not w:
                continue
            try:
                c = int(c_raw) if c_raw != "" else 0
            except Exception:
                c = 0
            # Persist the observed count (defaults to 0 on parsing failure).
            out[w] = c
    return out

def _read_col(path: Path, colname: str) -> List[str]:
    """Collect non-empty values from a named CSV column, preserving original order."""
    if not path or not path.is_file():
        return []
    out = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or colname not in reader.fieldnames:
            return out
        for r in reader:
            v = (r.get(colname) or "").strip()
            if v:
                # Preserve meaningful entries exactly as they appear in the file.
                out.append(v)
    return out

def _pick_newest(pattern: Path) -> Optional[Path]:
    """Select the newest file matching the glob pattern, if any exist."""
    # Sort matches by mtime so the freshest export is preferred.
    files = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)
    return Path(files[0]) if files else None

# -----------------------------
# Index unknowns as token n-grams
# -----------------------------
def _build_unknown_index(unknowns: Iterable[str]) -> Tuple[Dict[int, Dict[Tuple[str, ...], List[str]]], set]:
    """
    Returns:
      - index_by_len: { L -> { token_tuple -> [original_unknowns...] } }
      - lengths: set of L values present
    """
    index_by_len: Dict[int, Dict[Tuple[str, ...], List[str]]] = {}
    lengths: set = set()
    for u in unknowns:
        if not u:
            continue
        toks = tuple(_tokens(u))
        if not toks:
            continue
        L = len(toks)
        lengths.add(L)
        # Group unknowns first by token length then by the exact token tuple.
        bucket = index_by_len.setdefault(L, {})
        bucket.setdefault(toks, []).append(u)
    return index_by_len, lengths

# -----------------------------
# Scan one source using the index
# -----------------------------
def _scan_source(
    source_name: str,
    names: List[str],
    src_path: Optional[Path],
    index_by_len: Dict[int, Dict[Tuple[str, ...], List[str]]],
    lengths: set,
) -> List[List[str]]:
    """Return match rows describing how each reference string lines up with unknown tokens."""
    results: List[List[str]] = []
    refpath_str = str(src_path) if src_path else ""
    for ref in names:
        ref_tokens = _tokens(ref)
        if not ref_tokens:
            continue
        n = len(ref_tokens)
        # Whole match: when ref tokens match an unknown exactly
        if n in index_by_len:
            lst = index_by_len[n].get(tuple(ref_tokens))
            if lst:
                for u in lst:
                    # Capture full-length alignments for the report.
                    results.append([u, source_name, ref, "whole", refpath_str])
        # Partial matches: contiguous subsequences shorter than full length
        for L in lengths:
            if L >= n:
                continue
            bucket = index_by_len.get(L)
            if not bucket:
                continue
            for i in range(0, n - L + 1):
                window = tuple(ref_tokens[i:i+L])
                lst = bucket.get(window)
                if lst:
                    for u in lst:
                        # Record partial overlaps that may explain shorter unknown spans.
                        results.append([u, source_name, ref, "partial", refpath_str])
    return results

def main():
    """CLI entry point orchestrating unknown-word enrichment across reference vocabularies, producing the `missed_generics.csv` cues highlighted in README's unknown-handling section."""
    unknowns_path = OUTPUTS / "unknown_words.csv"
    if not unknowns_path.is_file():
        print(f"ERROR: {unknowns_path} not found.", file=sys.stderr)
        sys.exit(1)

    unknown_counts = _read_unknowns_with_counts(unknowns_path)
    for key in list(unknown_counts.keys()):
        norm_key = key.strip()
        # Drop empty strings, single letters, or known non-generic jargon tokens.
        if not norm_key or len(norm_key) == 1 or norm_key.lower() in COMMON_UNKNOWN_STOPWORDS:
            unknown_counts.pop(key)
    if not unknown_counts:
        print("No unknown words found (empty file).")
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        outpath = OUTPUTS / "missed_generics.csv"
        with outpath.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["unknown_word","unknown_count","source","reference_name","match_kind","reference_path"])
        print(str(outpath))
        return

    # Preserve the file order to maintain deterministic reporting.
    unknown_words = list(unknown_counts.keys())

    # Build fast index of unknown token n-grams
    index_by_len, lengths = _build_unknown_index(unknown_words)

    # Load sources
    pnf_path = INPUTS / "pnf_prepared.csv"
    pnf_names = _read_col(pnf_path, "generic_name")

    fda_path = _pick_newest(INPUTS / "fda_brand_map_*.csv")
    fda_names = _read_col(fda_path, "generic_name") if fda_path else []

    if not fda_names:
        legacy_fda = _pick_newest(INPUTS / "brand_map_*.csv")
        if legacy_fda:
            fda_path = legacy_fda
            fda_names = _read_col(legacy_fda, "generic_name")

    who_path = _pick_newest(WHO_DIR / "who_atc_*_molecules.csv")
    who_names = _read_col(who_path, "atc_name") if who_path else []

    # Scan all sources independently
    all_results: List[List[str]] = []
    if pnf_names:
        all_results.extend(_scan_source("PNF", pnf_names, pnf_path if pnf_path.is_file() else None, index_by_len, lengths))
    if who_names:
        all_results.extend(_scan_source("WHO", who_names, who_path if who_path else None, index_by_len, lengths))
    if fda_names:
        all_results.extend(_scan_source("FDA", fda_names, fda_path if fda_path else None, index_by_len, lengths))

    # Deduplicate rows
    seen_rows = set()
    deduped: List[List[str]] = []
    for row in all_results:
        t = tuple(row)
        if t not in seen_rows:
            seen_rows.add(t)
            # Keep only the first observation of each (unknown, source, ref, kind) tuple.
            deduped.append(row)

    # Prioritize per unknown: PNF > WHO > FDA
    priority = {"PNF": 0, "WHO": 1, "FDA": 2}
    by_unknown: Dict[str, Dict[str, List[List[str]]]] = {}
    for u, src, ref, kind, rpath in deduped:
        by_unknown.setdefault(u, {}).setdefault(src, []).append([u, src, ref, kind, rpath])

    filtered_rows: List[List[str]] = []
    for u, src_map in by_unknown.items():
        # pick the best available source for this unknown
        if not src_map:
            continue
        best_src = min(src_map.keys(), key=lambda s: priority.get(s, 99))
        filtered_rows.extend(src_map[best_src])

    # Add counts column; keep deterministic order: by unknown, then ref
    # Sorting ensures reproducible output and easier diffing of subsequent runs.
    filtered_rows.sort(
        key=lambda r: (
            -unknown_counts.get(r[0], 0),
            r[0],
            r[1],
            r[2],
            r[3],
        )
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUTS / "missed_generics.csv"
    with outpath.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unknown_word","unknown_count","source","reference_name","match_kind","reference_path"])
        for u, src, ref, kind, rpath in filtered_rows:
            writer.writerow([u, unknown_counts.get(u, 0), src, ref, kind, rpath])

    # Print the output path for automation hooks and quick discovery.
    print(str(outpath))

if __name__ == "__main__":
    main()
