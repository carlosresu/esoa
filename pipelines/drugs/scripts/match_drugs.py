#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Polars/Parquet-first orchestration that chains feature building, scoring, and outputs."""

from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Callable, Dict, Any
import polars as pl

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    """Wrap a callable with a lightweight spinner to show progress inside module-level scripts."""
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

from .match_features_drugs import build_features
from .match_scoring_drugs import score_and_classify
from .match_outputs_drugs import write_outputs


def _ensure_columns(lf: pl.LazyFrame, defaults: Dict[str, Any]) -> pl.LazyFrame:
    """Add missing columns with provided default literals to keep downstream selects safe."""
    for name, default in defaults.items():
        if name not in lf.columns:
            lf = lf.with_columns(pl.lit(default).alias(name))
    return lf


def _prep_annex_synonyms(values: tuple[object, object, object]) -> str:
    """Build the ordered Annex synonym payload while deduplicating entries."""
    raw_description, normalized_description, generic_name = values
    candidates = []
    for value in (raw_description, normalized_description, generic_name):
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                candidates.append(trimmed)
    seen = set()
    ordered = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            ordered.append(cand)
    return "|".join(ordered)


def _assemble_reference_catalogue(annex_df: pl.DataFrame | pl.LazyFrame, pnf_df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Merge Annex F and PNF catalogues into a single reference frame with priority metadata."""
    annex_lf = annex_df.lazy() if isinstance(annex_df, pl.DataFrame) else annex_df
    pnf_lf = pnf_df.lazy() if isinstance(pnf_df, pl.DataFrame) else pnf_df

    annex_lf = _ensure_columns(
        annex_lf,
        {
            "raw_description": "",
            "normalized_description": "",
            "generic_name": "",
            "drug_code": "",
            "route_allowed": "",
            "form_token": "",
            "dose_kind": None,
            "strength": None,
            "unit": "",
            "per_val": None,
            "per_unit": "",
            "pct": None,
            "strength_mg": None,
            "ratio_mg_per_ml": None,
            "route_evidence": "",
        },
    )
    annex_with_synonyms = annex_lf.with_columns(
        pl.struct(
            [
                pl.col("raw_description"),
                pl.col("normalized_description"),
                pl.col("generic_name"),
            ]
        )
        .map_elements(
            lambda s: _prep_annex_synonyms(
                (s["raw_description"], s["normalized_description"], s["generic_name"])
            ),
            return_dtype=pl.Utf8,
        )
        .alias("synonyms")
    )
    annex_ref = annex_with_synonyms.select(
        pl.col("drug_code").cast(pl.Utf8).alias("generic_id"),
        pl.col("generic_name").fill_null("").cast(pl.Utf8).alias("generic_name"),
        pl.col("generic_name").fill_null("").cast(pl.Utf8).alias("generic_normalized"),
        pl.col("generic_name").fill_null("").cast(pl.Utf8).alias("raw_molecule"),
        pl.col("synonyms").fill_null("").cast(pl.Utf8).alias("synonyms"),
        pl.lit("").alias("atc_code"),
        pl.col("route_allowed").fill_null("").cast(pl.Utf8),
        pl.col("form_token").fill_null("").cast(pl.Utf8),
        pl.col("dose_kind"),
        pl.col("strength"),
        pl.col("unit"),
        pl.col("per_val"),
        pl.col("per_unit"),
        pl.col("pct"),
        pl.col("strength_mg"),
        pl.col("ratio_mg_per_ml"),
        pl.lit("annex_f").alias("source"),
        pl.lit(1).alias("source_priority"),
        pl.col("drug_code").cast(pl.Utf8),
        pl.col("drug_code").cast(pl.Utf8).alias("primary_code"),
        pl.col("route_evidence").fill_null("").cast(pl.Utf8).alias("route_evidence_reference"),
    )

    pnf_lf = _ensure_columns(
        pnf_lf,
        {
            "generic_id": "",
            "generic_name": "",
            "generic_normalized": "",
            "raw_molecule": "",
            "synonyms": "",
            "atc_code": "",
            "route_allowed": "",
            "form_token": "",
            "dose_kind": None,
            "strength": None,
            "unit": "",
            "per_val": None,
            "per_unit": "",
            "pct": None,
            "strength_mg": None,
            "ratio_mg_per_ml": None,
            "route_evidence_reference": "",
        },
    )
    pnf_ref = pnf_lf.select(
        pl.col("generic_id").cast(pl.Utf8).alias("generic_id"),
        pl.col("generic_name").fill_null("").cast(pl.Utf8).alias("generic_name"),
        pl.col("generic_normalized").fill_null("").cast(pl.Utf8).alias("generic_normalized"),
        pl.col("raw_molecule").fill_null("").cast(pl.Utf8).alias("raw_molecule"),
        pl.col("synonyms").fill_null("").cast(pl.Utf8).alias("synonyms"),
        pl.col("atc_code").fill_null("").cast(pl.Utf8).alias("atc_code"),
        pl.col("route_allowed").fill_null("").cast(pl.Utf8).alias("route_allowed"),
        pl.col("form_token").fill_null("").cast(pl.Utf8).alias("form_token"),
        pl.col("dose_kind"),
        pl.col("strength"),
        pl.col("unit"),
        pl.col("per_val"),
        pl.col("per_unit"),
        pl.col("pct"),
        pl.col("strength_mg"),
        pl.col("ratio_mg_per_ml"),
        pl.lit("pnf").alias("source"),
        pl.lit(2).alias("source_priority"),
        pl.lit("").alias("drug_code"),
        pl.col("atc_code").fill_null("").cast(pl.Utf8).alias("primary_code"),
        pl.col("route_evidence_reference").fill_null("").cast(pl.Utf8).alias("route_evidence_reference"),
    )

    combined = pl.concat(
        [annex_ref, pnf_ref],
        how="vertical",
        rechunk=True,
    )
    return combined.collect()


def match(
    annex_prepared_parquet: str,
    pnf_prepared_parquet: str,
    esoa_prepared_parquet: str,
    out_path: str = "esoa_matched.parquet",
    *,
    timing_hook: Callable[[str, float], None] | None = None,
    skip_excel: bool = False,
) -> str:
    """Run the feature build, scoring, and output-writing stages on prepared Parquet inputs (Polars-first)."""
    def _timed(label: str, func: Callable[[], None]) -> float:
        elapsed = _run_with_spinner(label, func)
        if timing_hook:
            timing_hook(label, elapsed)
        return elapsed

    # Load inputs
    # Use small mutable containers so closures can assign to outer scope by reference.
    reference_df = [None]
    esoa_df = [None]
    _timed(
        "Load reference catalogues",
        lambda: reference_df.__setitem__(
            0,
            _assemble_reference_catalogue(
                pl.scan_parquet(annex_prepared_parquet),
                pl.scan_parquet(pnf_prepared_parquet),
            ),
        ),
    )
    _timed(
        "Load eSOA prepared parquet",
        lambda: esoa_df.__setitem__(0, pl.scan_parquet(esoa_prepared_parquet).collect(streaming=True)),
    )

    # Build features — inner function prints its own sub-spinners; do not show outer spinner.
    # Feeding the matcher-specific feature engineering step ties together all reference data.
    features_df = build_features(reference_df[0], esoa_df[0], timing_hook=timing_hook)

    # Score & classify (returns Polars)
    out_df = [None]
    _timed("Score & classify", lambda: out_df.__setitem__(0, score_and_classify(features_df, reference_df[0])))

    # Write outputs — inner module prints its own sub-spinners; do not show outer spinner
    out_polars = out_df[0]
    target_path = Path(out_path)
    # match_outputs expects a CSV target; normalize to a CSV path while keeping the parquet stem predictable.
    out_csv = target_path.with_suffix(".csv") if target_path.suffix.lower() == ".parquet" else target_path
    out_parquet = str(out_csv.with_suffix(".parquet"))
    # Persist the outputs (CSV, XLSX, summaries) while capturing timing metrics.
    write_outputs(out_polars, str(out_csv), timing_hook=timing_hook, skip_excel=skip_excel)

    return out_parquet
