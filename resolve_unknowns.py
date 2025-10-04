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
  ./outputs/unknown_words.parquet  (columns: word,count)

Search lists:
  • PNF: ./inputs/pnf_prepared.parquet (generic_name)
  • FDA brand map: newest of ./inputs/fda_brand_map_*.csv OR ./inputs/brand_map_*.csv (generic_name)
  • WHO ATC: newest of ./dependencies/atcd/output/who_atc_*_molecules.csv (atc_name)

Output:
  ./outputs/missed_generics.parquet with columns:
    unknown_word, unknown_count, source, reference_name, match_kind, reference_path

Priority:
  PNF over WHO over FDA (i.e., if an unknown matches in PNF, ignore WHO/FDA; if not, use WHO; else FDA).
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from scripts.io_utils import read_dataframe, write_parquet

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
    """Load unknown word counts from a parquet/csv table with columns [word,count]."""
    if not path.is_file():
        return {}
    try:
        df = read_dataframe(path)
    except Exception:
        return {}

    if "word" not in df.columns:
        return {}

    words = df["word"].fillna("").astype(str).str.strip()
    counts_series = df.get("count")
    if counts_series is None:
        counts_iter = [0] * len(words)
    else:
        counts_iter = counts_series.fillna(0)

    out: Dict[str, int] = {}
    for word, count in zip(words, counts_iter):
        if not word:
            continue
        try:
            out[word] = int(count)
        except Exception:
            try:
                out[word] = int(float(count))
            except Exception:
                out[word] = 0
    return out

def _read_col(path: Path, colname: str) -> List[str]:
    """Collect non-empty values from a named column, preserving original order."""
    if not path or not path.is_file():
        return []
    try:
        df = read_dataframe(path)
    except Exception:
        return []
    if colname not in df.columns:
        return []
    series = df[colname].fillna("").astype(str).str.strip()
    return [v for v in series if v]

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

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point orchestrating unknown-word enrichment across reference vocabularies, producing the `missed_generics` cues highlighted in README's unknown-handling section."""

    parser = argparse.ArgumentParser(
        description="Resolve unknown eSOA tokens against PNF/WHO/FDA references",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also emit CSV copies alongside the Parquet outputs.",
    )
    args = parser.parse_args(argv)

    unknowns_path = OUTPUTS / "unknown_words.parquet"
    if not unknowns_path.is_file():
        fallback_unknowns = OUTPUTS / "unknown_words.csv"
        if fallback_unknowns.is_file():
            unknowns_path = fallback_unknowns
        else:
            print(
                "ERROR: unknown_words.{parquet,csv} not found under ./outputs. Run the matcher first.",
                file=sys.stderr,
            )
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
        outpath = OUTPUTS / "missed_generics.parquet"
        empty_df = pd.DataFrame(
            columns=[
                "unknown_word",
                "unknown_count",
                "source",
                "reference_name",
                "match_kind",
                "reference_path",
            ]
        )
        write_parquet(empty_df, outpath)
        if args.export_csv:
            empty_df.to_csv(outpath.with_suffix(".csv"), index=False, encoding="utf-8")
        print(str(outpath))
        return

    # Preserve the file order to maintain deterministic reporting.
    unknown_words = list(unknown_counts.keys())

    # Build fast index of unknown token n-grams
    index_by_len, lengths = _build_unknown_index(unknown_words)

    # Load sources
    pnf_path = INPUTS / "pnf_prepared.parquet"
    if not pnf_path.is_file():
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
    outpath = OUTPUTS / "missed_generics.parquet"
    rows = [
        {
            "unknown_word": u,
            "unknown_count": int(unknown_counts.get(u, 0)),
            "source": src,
            "reference_name": ref,
            "match_kind": kind,
            "reference_path": rpath,
        }
        for u, src, ref, kind, rpath in filtered_rows
    ]
    result_df = pd.DataFrame(rows)
    write_parquet(result_df, outpath)
    if args.export_csv:
        result_df.to_csv(outpath.with_suffix(".csv"), index=False, encoding="utf-8")

    # Print the output path for automation hooks and quick discovery.
    print(str(outpath))


if __name__ == "__main__":
    main()
