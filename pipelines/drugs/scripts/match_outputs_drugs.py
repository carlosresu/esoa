#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Polars-first outputs for matched eSOA records, summaries, and review artifacts."""

from __future__ import annotations
import sys, time
import glob
import json
import os
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set

import polars as pl
import pandas as pd  # TODO(polars): drop once Excel export is replaced with Polars-friendly path

from ..constants import PIPELINE_INPUTS_DIR, PIPELINE_OUTPUTS_DIR, PIPELINE_WHO_ATC_DIR, PROJECT_ROOT
from .reference_data_drugs import load_drugbank_generics, load_ignore_words

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
    "ITEM_NUMBER","esoa_idx","raw_text","parentheticals",
    "normalized","norm_compact","match_basis","match_basis_norm_basic",
    "probable_brands","did_brand_swap","brand_swap_added_generic","fda_dose_corroborated","fda_generics_list","drugbank_generics_list",
    "molecules_recognized","molecules_recognized_list","molecules_recognized_count",
    "dose_recognized","dosage_parsed_raw","dosage_parsed",
    "route_raw","form_raw","route_evidence_raw",
    "route","route_source","route_text","form","form_source","form_text",
    "form_ok","route_ok","route_form_imputations","route_evidence","reference_route_details",
    "present_in_pnf","present_in_annex","present_in_who","present_in_fda_generic","present_in_drugbank","probable_atc",
    "generic_id","generic_final","reference_source","reference_priority","molecule_token","pnf_hits_gids","pnf_hits_count","pnf_hits_tokens",
    "who_molecules_list","who_molecules","who_atc_codes_list","who_atc_codes","who_atc_count","who_atc_has_ddd","who_atc_adm_r","who_route_tokens","who_form_tokens",
    "selected_form","selected_route_allowed","selected_variant",
    "selected_dose_kind","selected_strength","selected_unit","selected_strength_mg",
    "selected_per_val","selected_per_unit","selected_ratio_mg_per_ml","selected_pct",
    "dose_sim",
    "non_therapeutic_summary","non_therapeutic_detail","non_therapeutic_tokens","non_therapeutic_hits","non_therapeutic_best",
    "unknown_kind","unknown_words_list","unknown_words",
    "qty_pnf","qty_who","qty_fda_drug","qty_drugbank","qty_fda_food","qty_unknown",
    "atc_code_final","drug_code_final","primary_code_final","confidence",
    "match_molecule(s)","match_quality","detail_final",
    "bucket_final","why_final","reason_final",
]

# Reserved for pipeline-specific tokens we know are safe to ignore.
COMMON_UNKNOWN_STOPWORDS: Set[str] = set(load_ignore_words())

FRIENDLY_MOLECULE_LABELS = {
    "ValidMoleculeWithDrugCodeInAnnex": "Annex generic (Drug Code)",
    "ValidBrandSwappedForGenericInAnnex": "Annex brand swap",
    "ValidMoleculeWithATCinPNF": "PNF generic (ATC)",
    "ValidBrandSwappedForGenericInPNF": "PNF brand swap",
    "ValidMoleculeWithATCinWHO/NotInPNF": "WHO-only generic (ATC)",
    "ValidBrandSwappedForMoleculeWithATCinWHO": "WHO brand swap",
    "ValidMoleculeNoATCinFDA/NotInPNF": "FDA generic (no ATC)",
    "ValidMoleculeInDrugBank": "DrugBank generic",
    "ValidMoleculeNoCodeInReference": "Reference generic w/o code",
    "NonTherapeuticCatalogOnly": "Non-therapeutic catalogue hit",
    "NonTherapeuticFoodWithUnknownTokens": "Non-therapeutic food + unknown tokens",
    "AllTokensUnknownTo_PNF_WHO_FDA": "All tokens unknown (PNF/WHO/FDA)",
    "RowFailedAllMatchingSteps": "No catalogue match",
}

FRIENDLY_MATCH_QUALITY_LABELS = {
    "auto_exact_dose_route_form": "Exact dose / route / form",
    "dose_mismatch_same_atc": "Dose mismatch (same drug code)",
    "dose_mismatch_varied_atc": "Dose mismatch (different drug codes)",
    "dose_mismatch": "Dose mismatch",
    "dose_conflicts_annex_drug_code": "Dose conflicts Annex drug code",
    "annex_drug_code_missing": "Annex drug code missing",
    "no_dose_available": "Dose missing",
    "no_dose_form_and_route_available": "Dose, form & route missing",
    "no_dose_and_form_available": "Dose & form missing",
    "no_form_and_route_available": "Form & route missing",
    "no_form_available": "Form missing",
    "form_mismatch": "Form mismatch",
    "route_mismatch": "Route mismatch",
    "route_form_mismatch": "Route & form mismatch",
    "contains_unknown_tokens": "Contains unknown tokens",
    "nontherapeutic_catalog_match": "Non-therapeutic catalogue hit",
    "nontherapeutic_and_unknown_tokens": "Non-therapeutic + unknown tokens",
    "no_reference_catalog_match": "No reference match",
    "review_required_metadata_insufficient": "Metadata insufficient",
    "who_does_not_provide_dose_info": "WHO missing dose info",
    "who_does_not_provide_route_info": "WHO missing route info",
    "candidate_ready": "Candidate ready",
    "candidate_ready_for_atc_assignment": "Candidate ready for ATC",
    "who_atc_assigned": "WHO ATC assigned",
    "fda_brand_linked": "FDA brand linked",
}

_ABBREVIATION_LOOKUP = {
    "pnf": "PNF",
    "who": "WHO",
    "fda": "FDA",
    "atc": "ATC",
    "ddd": "DDD",
}


def _prettify_label_fallback(raw: str) -> str:
    raw = raw.replace("(", " (").replace(")", ") ")
    parts = [part for part in raw.replace("/", " / ").split("_") if part]
    words: List[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        key = token.lower()
        if key in _ABBREVIATION_LOOKUP:
            words.append(_ABBREVIATION_LOOKUP[key])
        elif token.isupper():
            words.append(token)
        elif len(token) <= 3:
            words.append(token.upper())
        else:
            words.append(token.capitalize())
    label = " ".join(words).replace(" / ", "/").strip()
    return label or raw


def _friendly_label(value: str, mapping: dict[str, str]) -> str:
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed:
        return trimmed
    if trimmed.upper() == "N/A":
        return "N/A"
    if trimmed.startswith("PartiallyKnownTokensFrom_"):
        sources = [src for src in trimmed.split("_")[1:] if src]
        sources_display = "/".join(
            _ABBREVIATION_LOOKUP.get(src.lower(), src.upper() if len(src) <= 3 else src)
            for src in sources
        ) or "catalogues"
        return f"Partial tokens from {sources_display}"
    return mapping.get(trimmed, _prettify_label_fallback(trimmed))

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
    project_root = PROJECT_ROOT
    inputs_dir = PIPELINE_INPUTS_DIR

    # PNF prepared (Parquet-first)
    pnf_parquet = inputs_dir / "pnf_prepared.parquet"
    if pnf_parquet.is_file():
        try:
            df_pnf = pl.read_parquet(pnf_parquet, columns=["generic_name", "generic_normalized", "synonyms"])
            tokens.update(_extract_tokens(df_pnf.get_column("generic_normalized")))
            tokens.update(_extract_tokens(df_pnf.get_column("generic_name")))
            tokens.update(_extract_tokens(df_pnf.get_column("synonyms")))
        except Exception:
            pass

    # FDA brand map (latest)
    brand_maps = sorted(glob.glob(str(inputs_dir / "fda_drug_*.parquet")) or glob.glob(str(inputs_dir / "fda_brand_map_*.parquet")))
    if brand_maps:
        try:
            df_brand = pl.read_parquet(brand_maps[-1], columns=["brand_name", "generic_name"])
            tokens.update(_extract_tokens(df_brand.get_column("brand_name")))
            tokens.update(_extract_tokens(df_brand.get_column("generic_name")))
        except Exception:
            pass

    # FDA food catalog (optional)
    food_candidates = sorted(glob.glob(str(inputs_dir / "fda_food_*.parquet")))
    if not food_candidates:
        legacy_food = inputs_dir / "fda_food_products.parquet"
        if legacy_food.is_file():
            legacy_food.unlink(missing_ok=True)
    if food_candidates:
        food_catalog = Path(food_candidates[-1])
        try:
            df_food = pl.read_parquet(food_catalog, columns=["brand_name", "product_name", "company_name"])
            tokens.update(_extract_tokens(df_food.get_column("brand_name")))
            tokens.update(_extract_tokens(df_food.get_column("product_name")))
            tokens.update(_extract_tokens(df_food.get_column("company_name")))
        except Exception:
            pass

    # WHO molecules (latest)
    who_dir = PIPELINE_WHO_ATC_DIR
    who_files = sorted(glob.glob(str(who_dir / "who_atc_*_molecules.parquet")))
    if who_files:
        try:
            df_who = pl.read_parquet(who_files[-1], columns=["atc_name"])
            tokens.update(_extract_tokens(df_who.get_column("atc_name")))
        except Exception:
            pass

    tokens.update(load_ignore_words(project_root))
    _, drugbank_tokens, _, _, _ = load_drugbank_generics(project_root)
    tokens.update(drugbank_tokens)

    _KNOWN_TOKENS_CACHE = tokens
    return _KNOWN_TOKENS_CACHE


def _write_csv_and_parquet(frame: pl.DataFrame, csv_path: str) -> None:
    """Persist a Polars dataframe to both CSV and Parquet with matching stems."""
    frame.write_csv(csv_path)
    parquet_path = Path(csv_path).with_suffix(".parquet")
    frame.write_parquet(parquet_path)

def _generate_summary_lines(out_small: pl.DataFrame, mode: str) -> List[str]:
    """Produce human-readable distribution summaries for review files (Polars-first)."""
    total = out_small.height
    bucket_order = ["Auto-Accept", "Candidates", "Needs review", "Unknown"]

    def _col_as_list(df: pl.DataFrame, col: str, fallback: str) -> List[str]:
        if col not in df.columns:
            return [fallback] * df.height
        return [
            (val.strip() if isinstance(val, str) else str(val).strip()) or fallback
            for val in df.get_column(col).to_list()
        ]

    if mode == "default":
        lines: List[str] = []
        seen: Set[str] = set()

        def _emit_bucket(bucket: str, bucket_rows: pl.DataFrame) -> None:
            if bucket_rows.is_empty():
                return
            count = int(bucket_rows.height)
            pct_bucket = round(count / float(total) * 100, 2) if total else 0.0
            lines.append(f"{bucket}: {count:,} ({pct_bucket}%)")
            match_molecule = [
                _friendly_label(val, FRIENDLY_MOLECULE_LABELS)
                for val in _col_as_list(bucket_rows, "match_molecule(s)", "N/A")
            ]
            match_quality = [
                _friendly_label(val, FRIENDLY_MATCH_QUALITY_LABELS)
                for val in _col_as_list(bucket_rows, "match_quality", "N/A")
            ]
            counts: dict[tuple[str, str], int] = {}
            for mm, mq in zip(match_molecule, match_quality):
                key = (mm, mq)
                counts[key] = counts.get(key, 0) + 1
            for (mm, mq), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
                pct_total = round(n / float(total) * 100, 2) if total else 0.0
                lines.append(f"  {mm}: {mq}: {int(n):,} ({pct_total}%)")

        for bucket in bucket_order:
            bucket_rows = out_small.filter(pl.col("bucket_final") == bucket)
            _emit_bucket(bucket, bucket_rows)
            seen.add(bucket)

        extra_buckets = [
            b for b in out_small.get_column("bucket_final").drop_nulls().unique().to_list() if b not in seen
        ]
        for bucket in sorted(extra_buckets):
            bucket_rows = out_small.filter(pl.col("bucket_final") == bucket)
            _emit_bucket(bucket, bucket_rows)

        return lines

    lines = ["Distribution Summary"]

    def _append_top_values(bucket_df: pl.DataFrame, column: str, label: str, limit: int = 5) -> None:
        if column not in bucket_df.columns or bucket_df.is_empty():
            return
        series = bucket_df.get_column(column).fill_null("N/A").cast(pl.Utf8).str.strip_chars()
        series = series.fill_null("N/A").replace("", "N/A")
        values = [
            _friendly_label(val, FRIENDLY_MOLECULE_LABELS if column == "match_molecule(s)" else FRIENDLY_MATCH_QUALITY_LABELS)
            for val in series.to_list()
        ]
        counts: dict[str, int] = {}
        for val in values:
            counts[val] = counts.get(val, 0) + 1
        for val, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
            pct_total = round(cnt / float(total) * 100, 2) if total else 0.0
            lines.append(f"    {label}: {val}: {int(cnt):,} ({pct_total}%)")

    for bucket in bucket_order:
        bucket_rows = out_small.filter(pl.col("bucket_final") == bucket)
        count = int(bucket_rows.height)
        pct = round(count / float(total) * 100, 2) if total else 0.0
        lines.append(f"{bucket}: {count:,} ({pct}%)")
        if count:
            if mode == "molecule" and "match_molecule(s)" in bucket_rows.columns:
                _append_top_values(bucket_rows, "match_molecule(s)", "Molecule")
            elif mode == "match" and "match_quality" in bucket_rows.columns:
                _append_top_values(bucket_rows, "match_quality", "Match quality")

    remaining = [
        b for b in out_small.get_column("bucket_final").drop_nulls().unique().to_list() if b not in bucket_order
    ]
    for bucket in sorted(remaining):
        bucket_rows = out_small.filter(pl.col("bucket_final") == bucket)
        count = int(bucket_rows.height)
        pct = round(count / float(total) * 100, 2) if total else 0.0
        lines.append(f"{bucket}: {count:,} ({pct}%)")
        if count and {"why_final", "reason_final"} <= set(bucket_rows.columns):
            grouped = (
                bucket_rows
                .select("why_final", "reason_final")
                .group_by(["why_final", "reason_final"])
                .len()
                .sort(by=["len", "why_final", "reason_final"], descending=[True, False, False])
                .head(5)
                .to_dicts()
            )
            for row in grouped:
                pct_local = round(row["len"] / float(count) * 100, 2) if count else 0.0
                reason = _friendly_label(str(row["reason_final"]), FRIENDLY_MATCH_QUALITY_LABELS)
                lines.append(f"    {row['why_final']}: {reason}: {int(row['len']):,} ({pct_local}%)")

    return lines



def _write_summary_text(out_small: pl.DataFrame, out_csv: str) -> None:
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
    out_df: pl.DataFrame,
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

    out_small = out_df.clone()
    list_to_pipe = ["fda_generics_list", "drugbank_generics_list", "non_therapeutic_tokens"]
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
            out_small = out_small.with_columns(pl.col(col).map_elements(_join_list, return_dtype=pl.Utf8).alias(col))

    for col in json_columns:
        if col in out_small.columns:
            out_small = out_small.with_columns(pl.col(col).map_elements(_jsonify, return_dtype=pl.Utf8).alias(col))

    if "match_basis" not in out_small.columns:
        out_small = out_small.with_columns(pl.col("normalized").alias("match_basis"))

    out_small = out_small.select([c for c in OUTPUT_COLUMNS if c in out_small.columns])

    friendly_out = out_small
    friendly_column_mappings = {
        "match_molecule(s)": FRIENDLY_MOLECULE_LABELS,
        "match_quality": FRIENDLY_MATCH_QUALITY_LABELS,
        "reason_final": FRIENDLY_MATCH_QUALITY_LABELS,
    }
    for col, mapping in friendly_column_mappings.items():
        if col in friendly_out.columns:
            friendly_out = friendly_out.with_columns(
                pl.col(col).map_elements(lambda val: _friendly_label(val, mapping), return_dtype=pl.Utf8).alias(col)
            )

    friendly_out = friendly_out.rechunk()

    _timed("Write matched CSV/Parquet", lambda: _write_csv_and_parquet(friendly_out, out_csv))

    if not skip_excel:
        xlsx_out = os.path.splitext(out_csv)[0] + ".xlsx"

        def _to_excel():
            # TODO(polars): replace Excel path with Polars/pyarrow-only writer if needed.
            friendly_pd = friendly_out.to_pandas()
            try:
                with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
                    friendly_pd.to_excel(writer, index=False, sheet_name="matched")
                    ws = writer.sheets["matched"]
                    ws.freeze_panes(1, 0)
                    nrows, ncols = friendly_pd.shape
                    ws.autofilter(0, 0, nrows, ncols - 1)
            except Exception:
                with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
                    friendly_pd.to_excel(writer, index=False, sheet_name="matched")
                    ws = writer.sheets["matched"]
                    try:
                        ws.freeze_panes = "A2"
                        ws.auto_filter.ref = ws.dimensions
                    except Exception:
                        pass

        _timed("Write Excel", _to_excel)

    def _write_unknowns() -> None:
        if {"unknown_kind", "unknown_words"} <= set(out_small.columns):
            unknown = out_small.filter(
                (pl.col("unknown_kind") != "None") & (pl.col("unknown_words").cast(pl.Utf8).fill_null("") != "")
            ).select("unknown_words")
        else:
            unknown = pl.DataFrame({"unknown_words": []})
        words: List[str] = []
        known_tokens = _known_tokens()
        for s in unknown.get_column("unknown_words").to_list():
            for w in str(s).split("|"):
                w = w.strip()
                if not w:
                    continue
                lower = w.lower()
                if lower in COMMON_UNKNOWN_STOPWORDS:
                    continue
                if lower in known_tokens:
                    continue
                words.append(w)
        unk_path = os.path.join(os.path.dirname(out_csv), "unknown_words.csv")
        if words:
            unk_df = pl.DataFrame({"word": words})
            unk_df = (
                unk_df.group_by("word")
                .len()
                .sort(by="len", descending=True)
                .rename({"len": "count"})
            )
        else:
            unk_df = pl.DataFrame({"word": [], "count": []})
        _write_csv_and_parquet(unk_df, unk_path)
        # Feeds resolve_unknowns.py to produce the missed_generics report highlighted in README.
    _timed("Write unknown words CSV/Parquet", _write_unknowns)

    _timed("Write summary.txt", lambda: _write_summary_text(out_small, out_csv))

    return out_csv
