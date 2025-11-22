#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-processing helpers for PNF and eSOA inputs (Polars/Parquet-first)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import polars as pl

from .routes_forms_drugs import map_route_token, parse_form_from_text
from .dose_drugs import parse_dose_struct_from_text, to_mg, safe_ratio_mg_per_ml
from .text_utils_drugs import (
    clean_atc,
    extract_base_and_salts,
    normalize_text,
    serialize_salt_list,
    slug_id,
)
from .concurrency_drugs import maybe_parallel_map


def _calc_strength_mg(payload: tuple[object, object]) -> float | None:
    """Convert arbitrary (strength, unit) payloads into canonical mg where possible."""
    strength, unit = payload
    try:
        if strength is None or not isinstance(unit, str) or not unit:
            return None
        return to_mg(strength, unit)
    except Exception:
        return None


def _calc_ratio_mg_per_ml(payload: tuple[object, object, object, object, object]) -> float | None:
    """Derive mg/mL ratios only when the payload describes a ratio with ml denominator."""
    dose_kind, strength, unit, per_val, per_unit = payload
    if dose_kind == "ratio" and isinstance(per_unit, str) and per_unit.lower() == "ml":
        return safe_ratio_mg_per_ml(strength, unit, per_val)
    return None


def _write_csv_and_parquet(frame: pl.DataFrame, csv_path: str) -> None:
    """Persist a dataframe to CSV and Parquet using the same stem."""
    frame.write_csv(csv_path)
    parquet_path = Path(csv_path).with_suffix(".parquet")
    frame.write_parquet(parquet_path)


def _scan_input(path: str) -> pl.LazyFrame:
    """Scan CSV/Parquet into a LazyFrame (Parquet preferred)."""
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pl.scan_parquet(path)
    return pl.scan_csv(path)


def _prep_pnf(pnf_path: str) -> pl.DataFrame:
    """Load and normalize PNF inputs with Polars; return prepared frame."""
    pnf_scan = _scan_input(pnf_path)
    required_cols = ["Molecule", "Route", "ATC Code"]
    missing = [c for c in required_cols if c not in pnf_scan.columns]
    if missing:
        raise ValueError(f"pnf input missing required columns: {missing}")

    pnf = pnf_scan.with_columns(
        pl.col("Molecule").fill_null("").cast(pl.Utf8).alias("raw_molecule"),
        pl.col("Molecule").fill_null("").cast(pl.Utf8).str.strip().str.to_uppercase().alias("generic_name"),
        pl.col("Route").fill_null("").cast(pl.Utf8).alias("Route"),
        pl.col("ATC Code").fill_null("").cast(pl.Utf8).alias("ATC Code"),
    ).collect(streaming=True)

    molecule_list = pnf.get_column("generic_name").to_list()
    split_values = maybe_parallel_map(molecule_list, extract_base_and_salts)
    pnf = pnf.with_columns(
        pl.Series([base or original.strip().upper() for (base, _), original in zip(split_values, molecule_list)]).alias("generic_normalized"),
        pl.Series([serialize_salt_list(salts) for _, salts in split_values]).alias("salt_form"),
        pl.Series(maybe_parallel_map(molecule_list, slug_id)).alias("generic_id"),
        pl.lit("").alias("synonyms"),
    )

    route_values: List[str] = pnf.get_column("Route").to_list()
    atc_values: List[str] = pnf.get_column("ATC Code").to_list()
    pnf = pnf.with_columns(
        pl.Series(maybe_parallel_map(route_values, map_route_token)).alias("route_tokens"),
        pl.Series(maybe_parallel_map(atc_values, clean_atc)).alias("atc_code"),
    )

    text_cols = [c for c in ["Technical Specifications", "Specs", "Specification"] if c in pnf.columns]
    tech_vals = pnf.get_column(text_cols[0]).fill_null("").cast(pl.Utf8).to_list() if text_cols else [""] * pnf.height
    parse_src_list = [f"{gn} {tech}".strip() for gn, tech in zip(pnf.get_column("generic_normalized").to_list(), tech_vals)]
    parse_normalized = maybe_parallel_map(parse_src_list, normalize_text)
    pnf = pnf.with_columns(pl.Series(parse_normalized).alias("_parse_src"))

    parsed = maybe_parallel_map(parse_normalized, parse_dose_struct_from_text)
    pnf = pnf.with_columns(
        pl.Series([d.get("dose_kind") if isinstance(d, dict) else None for d in parsed]).alias("dose_kind"),
        pl.Series([d.get("strength") if isinstance(d, dict) else None for d in parsed]).alias("strength"),
        pl.Series([d.get("unit") if isinstance(d, dict) else None for d in parsed]).alias("unit"),
        pl.Series([d.get("per_val") if isinstance(d, dict) else None for d in parsed]).alias("per_val"),
        pl.Series([d.get("per_unit") if isinstance(d, dict) else None for d in parsed]).alias("per_unit"),
        pl.Series([d.get("pct") if isinstance(d, dict) else None for d in parsed]).alias("pct"),
        pl.Series(maybe_parallel_map(parse_normalized, parse_form_from_text)).alias("form_token"),
    )

    strength_inputs = list(zip(pnf.get_column("strength").to_list(), pnf.get_column("unit").to_list()))
    ratio_inputs = list(
        zip(
            pnf.get_column("dose_kind").to_list(),
            pnf.get_column("strength").to_list(),
            pnf.get_column("unit").to_list(),
            pnf.get_column("per_val").to_list(),
            pnf.get_column("per_unit").to_list(),
        )
    )
    pnf = pnf.with_columns(
        pl.Series(maybe_parallel_map(strength_inputs, _calc_strength_mg)).alias("strength_mg"),
        pl.Series(maybe_parallel_map(ratio_inputs, _calc_ratio_mg_per_ml)).alias("ratio_mg_per_ml"),
    )

    pnf = (
        pnf.explode("route_tokens")
        .with_columns(pl.col("route_tokens").alias("route_allowed"))
        .drop("route_tokens")
        .filter(pl.col("generic_name").str.strip() != "")
    )

    return pnf.select(
        "generic_id",
        "generic_name",
        "generic_normalized",
        "raw_molecule",
        "salt_form",
        "synonyms",
        "atc_code",
        "route_allowed",
        "form_token",
        "dose_kind",
        "strength",
        "unit",
        "per_val",
        "per_unit",
        "pct",
        "strength_mg",
        "ratio_mg_per_ml",
    )


def _prep_esoa(esoa_path: str) -> pl.DataFrame:
    """Load and normalize eSOA input with Polars; return prepared frame."""
    esoa_scan = _scan_input(esoa_path)
    if "DESCRIPTION" not in esoa_scan.columns:
        raise ValueError("eSOA input is missing required column: DESCRIPTION")
    return esoa_scan.with_columns(pl.col("DESCRIPTION").alias("raw_text")).select("raw_text").collect(streaming=True)


def prepare(pnf_path: str, esoa_path: str, outdir: str = ".") -> tuple[str, str]:
    """Normalize PNF and eSOA inputs, deriving helper columns and writing prepared Parquet/CSV (Polars-first)."""
    os.makedirs(outdir, exist_ok=True)

    pnf_prepared = _prep_pnf(pnf_path)
    esoa_prepared = _prep_esoa(esoa_path)

    pnf_out_parquet = os.path.join(outdir, "pnf_prepared.parquet")
    esoa_out_parquet = os.path.join(outdir, "esoa_prepared.parquet")

    _write_csv_and_parquet(pnf_prepared, pnf_out_parquet.replace(".parquet", ".csv"))
    _write_csv_and_parquet(esoa_prepared, esoa_out_parquet.replace(".parquet", ".csv"))

    return pnf_out_parquet, esoa_out_parquet
