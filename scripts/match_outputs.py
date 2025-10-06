#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Output writers for matched eSOA records, summaries, and review artifacts."""

from __future__ import annotations
import sys, time
import glob
import json
import os, pandas as pd
import re
from pathlib import Path
from typing import Callable, List, Optional, Set

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    """Provide a local progress spinner so downstream modules stay dependency-free."""
    import threading
    done = threading.Event()
    err = []

    def worker():
        try:
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"; i = 0
    while not done.is_set():
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r{frames[i % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        time.sleep(0.1); i += 1
    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return elapsed

OUTPUT_COLUMNS = [
    "esoa_idx","raw_text","parentheticals",
    "normalized","norm_compact","match_basis","match_basis_norm_basic",
    "probable_brands","did_brand_swap","brand_swap_added_generic","fda_dose_corroborated","fda_generics_list",
    "molecules_recognized","molecules_recognized_list","molecules_recognized_count",
    "dose_recognized","dosage_parsed_raw","dosage_parsed",
    "route_raw","form_raw","route_evidence_raw",
    "route","route_source","route_text","form","form_source","form_text",
    "form_ok","route_ok","route_form_imputations","route_evidence",
    "present_in_pnf","present_in_who","present_in_fda_generic","probable_atc",
    "generic_id","generic_final","molecule_token","pnf_hits_gids","pnf_hits_count","pnf_hits_tokens",
    "who_molecules_list","who_molecules","who_atc_codes_list","who_atc_codes","who_atc_count","who_atc_has_ddd","who_atc_adm_r","who_route_tokens","who_form_tokens",
    "selected_form","selected_route_allowed","selected_variant",
    "selected_dose_kind","selected_strength","selected_unit","selected_strength_mg",
    "selected_per_val","selected_per_unit","selected_ratio_mg_per_ml","selected_pct",
    "dose_sim","looks_combo_final","combo_reason","combo_known_generics_count",
    "non_therapeutic_summary","non_therapeutic_detail","non_therapeutic_tokens","non_therapeutic_hits","non_therapeutic_best",
    "unknown_kind","unknown_words_list","unknown_words",
    "qty_pnf","qty_who","qty_fda_drug","qty_fda_food","qty_unknown",
    "atc_code_final","confidence",
    "match_molecule(s)","match_quality","detail_final",
    "bucket_final","why_final","reason_final",
]

# Common non-clinical nouns that should not be escalated as unknowns
COMMON_UNKNOWN_STOPWORDS = {
    "bottle",
    "bottles",
    "box",
    "boxes",
    "syringe",
    "syringes",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_KNOWN_TOKENS_CACHE: Optional[Set[str]] = None


def _extract_tokens(values: Iterable[object]) -> Set[str]:
    tokens: Set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        for match in _TOKEN_RE.findall(value.lower()):
            tokens.add(match)
    return tokens


def _known_tokens() -> Set[str]:
    global _KNOWN_TOKENS_CACHE
    if _KNOWN_TOKENS_CACHE is not None:
        return _KNOWN_TOKENS_CACHE

    tokens: Set[str] = set()
    project_root = Path(__file__).resolve().parent.parent
    inputs_dir = project_root / "inputs"

    # PNF prepared
    pnf_prepared = inputs_dir / "pnf_prepared.csv"
    if pnf_prepared.is_file():
        try:
            df_pnf = pd.read_csv(pnf_prepared, usecols=["generic_name", "synonyms"], dtype=str)
            tokens.update(_extract_tokens(df_pnf.get("generic_name", [])))
            tokens.update(_extract_tokens(df_pnf.get("synonyms", [])))
        except Exception:
            pass

    # FDA brand map (latest)
    brand_maps = sorted(glob.glob(str(inputs_dir / "fda_brand_map_*.csv")))
    if brand_maps:
        try:
            df_brand = pd.read_csv(brand_maps[-1], usecols=["brand_name", "generic_name"], dtype=str)
            tokens.update(_extract_tokens(df_brand.get("brand_name", [])))
            tokens.update(_extract_tokens(df_brand.get("generic_name", [])))
        except Exception:
            pass

    # FDA food catalog (optional)
    food_catalog = inputs_dir / "fda_food_products.csv"
    if food_catalog.is_file():
        try:
            df_food = pd.read_csv(
                food_catalog,
                usecols=["brand_name", "product_name", "company_name"],
                dtype=str,
            )
            tokens.update(_extract_tokens(df_food.get("brand_name", [])))
            tokens.update(_extract_tokens(df_food.get("product_name", [])))
            tokens.update(_extract_tokens(df_food.get("company_name", [])))
        except Exception:
            pass

    # WHO molecules (latest)
    who_dir = project_root / "dependencies" / "atcd" / "output"
    who_files = sorted(glob.glob(str(who_dir / "who_atc_*_molecules.csv")))
    if who_files:
        try:
            df_who = pd.read_csv(who_files[-1], usecols=["atc_name"], dtype=str)
            tokens.update(_extract_tokens(df_who.get("atc_name", [])))
        except Exception:
            pass

    _KNOWN_TOKENS_CACHE = tokens
    return _KNOWN_TOKENS_CACHE

def _generate_summary_lines(out_small: pd.DataFrame, mode: str) -> List[str]:
    """Produce human-readable distribution summaries for review files."""
    total = len(out_small)
    bucket_order = ["Auto-Accept", "Candidates", "Needs review", "Unknown"]

    if mode == "default":
        lines: List[str] = []
        seen: Set[str] = set()

        def _normalize(series: pd.Series, fallback: str) -> pd.Series:
            normalized = (
                series.fillna("")
                .astype(str)
                .str.strip()
                .replace({"": fallback})
            )
            return normalized

        def _emit_bucket(bucket: str, bucket_rows: pd.DataFrame) -> None:
            if bucket_rows.empty:
                return
            count = int(len(bucket_rows))
            pct_bucket = round(count / float(total) * 100, 2) if total else 0.0
            lines.append(f"{bucket}: {count:,} ({pct_bucket}%)")
            match_molecule = _normalize(
                bucket_rows.get("match_molecule(s)", pd.Series(["N/A"] * len(bucket_rows), index=bucket_rows.index)),
                "N/A",
            )
            match_quality = _normalize(
                bucket_rows.get("match_quality", pd.Series(["N/A"] * len(bucket_rows), index=bucket_rows.index)),
                "N/A",
            )
            grouped = (
                pd.DataFrame(
                    {
                        "match_molecule": match_molecule,
                        "match_quality": match_quality,
                    }
                )
                .groupby(["match_molecule", "match_quality"], dropna=False)
                .size()
                .reset_index(name="n")
                .sort_values(by=["n", "match_molecule", "match_quality"], ascending=[False, True, True])
            )
            top_rows = grouped.head(5)
            shown = int(top_rows["n"].sum())
            for _, row in top_rows.iterrows():
                pct_total = round(row["n"] / float(total) * 100, 2) if total else 0.0
                pct_bucket = round(row["n"] / float(count) * 100, 2) if count else 0.0
                lines.append(
                    f"  {row['match_molecule']}: {row['match_quality']}: {int(row['n']):,} "
                    f"({pct_total}% of total, {pct_bucket}% of bucket)"
                )
            remainder = count - shown
            if remainder > 0:
                pct_total_remainder = round(remainder / float(total) * 100, 2) if total else 0.0
                pct_bucket_remainder = round(remainder / float(count) * 100, 2) if count else 0.0
                lines.append(
                    f"  Other combinations: {remainder:,} "
                    f"({pct_total_remainder}% of total, {pct_bucket_remainder}% of bucket)"
                )

        for bucket in bucket_order:
            bucket_rows = out_small.loc[out_small["bucket_final"].eq(bucket)].copy()
            _emit_bucket(bucket, bucket_rows)
            seen.add(bucket)

        extra_buckets = [
            bucket
            for bucket in out_small.get("bucket_final", pd.Series(dtype=str)).dropna().unique()
            if bucket not in seen
        ]
        for bucket in sorted(extra_buckets):
            bucket_rows = out_small.loc[out_small["bucket_final"].eq(bucket)].copy()
            _emit_bucket(bucket, bucket_rows)

        return lines

    lines = ["Distribution Summary"]
    qty_columns = ["qty_pnf", "qty_who", "qty_fda_drug", "qty_fda_food", "qty_unknown"]

    def _append_top_values(bucket_df: pd.DataFrame, column: str, label: str, limit: int = 5) -> None:
        if column not in bucket_df.columns:
            return
        series = (
            bucket_df[column]
            .fillna("N/A")
            .astype(str)
            .str.strip()
            .replace({"": "N/A"})
        )
        top = series.value_counts().head(limit)
        for value, count in top.items():
            pct_total = round(count / float(total) * 100, 2) if total else 0.0
            lines.append(f"    {label}: {value}: {count:,} ({pct_total}%)")

    for bucket in bucket_order:
        bucket_rows = out_small.loc[out_small["bucket_final"].eq(bucket)].copy()
        count = int(len(bucket_rows))
        pct = round(count / float(total) * 100, 2) if total else 0.0
        lines.append(f"{bucket}: {count:,} ({pct}%)")
        if count:
            if mode == "molecule" and "match_molecule(s)" in bucket_rows.columns:
                _append_top_values(bucket_rows, "match_molecule(s)", "Molecule")
            elif mode == "match" and "match_quality" in bucket_rows.columns:
                _append_top_values(bucket_rows, "match_quality", "Match quality")

    remaining = [
        bucket
        for bucket in out_small.get("bucket_final", pd.Series(dtype=str)).dropna().unique()
        if bucket not in bucket_order
    ]
    for bucket in sorted(remaining):
        bucket_rows = out_small.loc[out_small["bucket_final"].eq(bucket)].copy()
        count = int(len(bucket_rows))
        pct = round(count / float(total) * 100, 2) if total else 0.0
        lines.append(f"{bucket}: {count:,} ({pct}%)")
        if count:
            if "why_final" in bucket_rows.columns and "reason_final" in bucket_rows.columns:
                grouped = (
                    bucket_rows.groupby(["why_final", "reason_final"], dropna=False)
                    .size()
                    .reset_index(name="n")
                    .sort_values(by=["n", "why_final", "reason_final"], ascending=[False, True, True])
                )
                for _, row in grouped.head(5).iterrows():
                    pct_local = round(row["n"] / float(count) * 100, 2) if count else 0.0
                    lines.append(f"    {row['why_final']}: {row['reason_final']}: {int(row['n']):,} ({pct_local}%)")

    return lines



def _write_summary_text(out_small: pd.DataFrame, out_csv: str) -> None:
    """Write multiple summary text files that slice the results by molecule or match quality."""
    base_dir = os.path.dirname(out_csv)
    summaries = [
        ("summary.txt", "default"),
        ("summary_molecule.txt", "molecule"),
        ("summary_match.txt", "match"),
    ]

    for filename, mode in summaries:
        summary_path = os.path.join(base_dir, filename)
        lines = _generate_summary_lines(out_small, mode)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

def write_outputs(
    out_df: pd.DataFrame,
    out_csv: str,
    *,
    timing_hook: Callable[[str, float], None] | None = None,
    skip_excel: bool = False,
) -> str:
    """Persist the canonical CSV output (and optionally XLSX) plus text summaries described in README."""

    def _timed(label: str, func: Callable[[], None]) -> float:
        elapsed = _run_with_spinner(label, func)
        if timing_hook:
            timing_hook(label, elapsed)
        return elapsed

    out_small = out_df.copy()
    list_to_pipe = ["fda_generics_list", "non_therapeutic_tokens"]
    json_columns = ["non_therapeutic_hits", "non_therapeutic_best"]

    def _join_list(value: object) -> str:
        if isinstance(value, (list, tuple, set)):
            parts = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    parts.append(text)
            return "|".join(parts)
        if value is None:
            return ""
        return str(value)

    def _jsonify(value: object) -> str:
        if isinstance(value, (list, dict)):
            if not value:
                return ""
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if value is None:
            return ""
        return str(value)

    for col in list_to_pipe:
        if col in out_small.columns:
            out_small[col] = out_small[col].map(_join_list)

    for col in json_columns:
        if col in out_small.columns:
            out_small[col] = out_small[col].map(_jsonify)
    if "match_basis" not in out_small.columns:
        out_small["match_basis"] = out_small.get("normalized", "")
    out_small = out_small[[c for c in OUTPUT_COLUMNS if c in out_small.columns]].copy()

    _timed("Write matched CSV", lambda: out_small.to_csv(out_csv, index=False, encoding="utf-8"))

    if not skip_excel:
        xlsx_out = os.path.splitext(out_csv)[0] + ".xlsx"

        def _to_excel():
            try:
                with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
                    out_small.to_excel(writer, index=False, sheet_name="matched")
                    ws = writer.sheets["matched"]
                    # Freeze the top row to keep headers visible during review.
                    ws.freeze_panes(1, 0)
                    nrows, ncols = out_small.shape
                    ws.autofilter(0, 0, nrows, ncols - 1)
            except Exception:
                with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
                    out_small.to_excel(writer, index=False, sheet_name="matched")
                    ws = writer.sheets["matched"]
                    try:
                        ws.freeze_panes = "A2"
                        ws.auto_filter.ref = ws.dimensions
                    except Exception:
                        pass

        _timed("Write Excel", _to_excel)

    def _write_unknowns():
        unknown = out_small.loc[
            out_small["unknown_kind"].ne("None") & out_small["unknown_words"].astype(str).ne(""),
            ["unknown_words"]
        ].copy()
        words = []
        known_tokens = _known_tokens()
        for s in unknown["unknown_words"]:
            for w in str(s).split("|"):
                w = w.strip()
                if not w:
                    continue
                if w.lower() in COMMON_UNKNOWN_STOPWORDS:
                    continue
                if w.lower() in known_tokens:
                    continue
                words.append(w)
        if words:
            unk_df = pd.DataFrame({"word": words})
            unk_df = unk_df.groupby("word").size().reset_index(name="count").sort_values("count", ascending=False)
            unk_path = os.path.join(os.path.dirname(out_csv), "unknown_words.csv")
            unk_df.to_csv(unk_path, index=False, encoding="utf-8")
            # Feeds resolve_unknowns.py to produce the missed_generics report highlighted in README.
    _timed("Write unknown words CSV", _write_unknowns)

    _timed("Write summary.txt", lambda: _write_summary_text(out_small, out_csv))

    return out_csv
