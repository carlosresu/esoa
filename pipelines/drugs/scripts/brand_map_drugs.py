#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import ahocorasick  # type: ignore
import polars as pl

from .text_utils_drugs import (
    _base_name,
    _normalize_text_basic,
    extract_base_and_salts,
    normalize_compact,
    serialize_salt_list,
)


@dataclass
class BrandMatch:
    """Structured payload describing an FDA brand and its normalized attributes."""
    brand: str
    generic: str
    salt_form: str
    dosage_form: str
    route: str
    dosage_strength: str


def _split_generic_parts(value: str) -> Dict[str, Any]:
    """Return structured base/salt parts for a generic name."""
    base, salts = extract_base_and_salts(value)
    return {"base": base, "salts": salts}


def _latest_brandmap_path(inputs_dir: str) -> Optional[str]:
    """Return the newest brand map file path, supporting legacy naming schemes."""
    # Prefer renamed pattern
    pattern_new = os.path.join(inputs_dir, "fda_drug_*.parquet")
    candidates = glob.glob(pattern_new)
    if not candidates:
        # Backward-compatibility with old name
        pattern_old = os.path.join(inputs_dir, "brand_map_*.parquet")
        candidates = glob.glob(pattern_old)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def load_latest_brandmap(inputs_dir: str) -> Optional[pl.DataFrame]:
    """
    Load the most recent brand map Parquet into a Polars DataFrame, keeping transforms in Polars.

    Parquet is the canonical input; expected columns are filled with empty strings when absent and
    generic/salt parts are normalized for downstream matching.
    """
    path = _latest_brandmap_path(inputs_dir)
    if not path or not os.path.exists(path):
        return None
    try:
        lf = pl.scan_parquet(path)
        required_cols = ["brand_name", "generic_name", "dosage_form", "route", "dosage_strength", "salt_form"]
        lf = lf.with_columns([pl.lit("").alias(c) for c in required_cols if c not in lf.columns])
        lf = lf.with_columns(
            pl.col("brand_name").fill_null("").cast(pl.Utf8).alias("brand_name"),
            pl.col("generic_name").fill_null("").cast(pl.Utf8).alias("generic_name"),
            pl.col("dosage_form").fill_null("").cast(pl.Utf8).alias("dosage_form"),
            pl.col("route").fill_null("").cast(pl.Utf8).alias("route"),
            pl.col("dosage_strength").fill_null("").cast(pl.Utf8).alias("dosage_strength"),
            pl.col("salt_form").fill_null("").cast(pl.Utf8).alias("salt_form"),
        )
        lf = lf.with_columns(
            pl.col("generic_name")
            .map_elements(_split_generic_parts, return_dtype=pl.Struct({"base": pl.Utf8, "salts": pl.List(pl.Utf8)}))
            .alias("generic_parts")
        )
        lf = lf.with_columns(
            pl.when(
                pl.col("generic_parts").struct.field("base").is_not_null()
                & (pl.col("generic_parts").struct.field("base") != "")
            )
            .then(pl.col("generic_parts").struct.field("base"))
            .otherwise(pl.col("generic_name").str.strip_chars().str.to_uppercase())
            .alias("generic_name"),
            pl.col("generic_parts")
            .struct.field("salts")
            .map_elements(serialize_salt_list, return_dtype=pl.Utf8)
            .fill_null("")
            .alias("salt_form"),
        ).drop("generic_parts")
        return lf.collect()
    except Exception:
        return None


def build_brand_automata(brand_df: pl.DataFrame | pl.LazyFrame) -> Tuple[ahocorasick.Automaton, ahocorasick.Automaton, Dict[str, List[BrandMatch]]]:
    """
    Compile Aho–Corasick automatons and lookup tables for brand→generic substitutions.

    Expects Polars DataFrame/LazyFrame inputs from the Parquet-first pipeline and keeps normalization
    in Polars before materializing the automata.
    """
    lf = brand_df.lazy() if isinstance(brand_df, pl.DataFrame) else brand_df
    required_cols = ["brand_name", "generic_name", "dosage_form", "route", "dosage_strength", "salt_form"]
    lf = lf.with_columns([pl.lit("").alias(c) for c in required_cols if c not in lf.columns])
    lf_clean = lf.with_columns(
        pl.col("brand_name").fill_null("").cast(pl.Utf8).alias("brand_name"),
        pl.col("generic_name").fill_null("").cast(pl.Utf8).alias("generic_name"),
        pl.col("dosage_form").fill_null("").cast(pl.Utf8).alias("dosage_form"),
        pl.col("route").fill_null("").cast(pl.Utf8).alias("route"),
        pl.col("dosage_strength").fill_null("").cast(pl.Utf8).alias("dosage_strength"),
        pl.col("salt_form").fill_null("").cast(pl.Utf8).alias("salt_form"),
    )
    processed = (
        lf_clean.with_columns(
            pl.col("brand_name").map_elements(lambda b: _normalize_text_basic(_base_name(b)), return_dtype=pl.Utf8).alias("brand_norm"),
            pl.col("brand_name").map_elements(normalize_compact, return_dtype=pl.Utf8).alias("brand_compact"),
        )
        .filter((pl.col("brand_norm") != "") & (pl.col("generic_name") != ""))
        .collect()
    )
    A_norm = ahocorasick.Automaton()
    A_comp = ahocorasick.Automaton()
    mapping: Dict[str, List[BrandMatch]] = {}
    seen_norm = set()
    seen_comp = set()
    for r in processed.iter_rows(named=True):
        bn = str(r.get("brand_norm") or "").strip()
        bc = str(r.get("brand_compact") or "").strip()
        b = str(r.get("brand_name") or "").strip()
        g = str(r.get("generic_name") or "").strip()
        if not bn or not b or not g:
            continue
        dosage_form = str(r.get("dosage_form") or "").strip()
        route = str(r.get("route") or "").strip()
        dosage_strength = str(r.get("dosage_strength") or "").strip()
        salt_form = str(r.get("salt_form") or "").strip()
        # Map the normalized brand token to its full metadata payload.
        mapping.setdefault(bn, []).append(BrandMatch(
            brand=b,
            generic=g,
            salt_form=salt_form,
            dosage_form=dosage_form,
            route=route,
            dosage_strength=dosage_strength,
        ))
        if bn not in seen_norm:
            # Register the normalized token once for the standard automaton.
            A_norm.add_word(bn, bn); seen_norm.add(bn)
        if bc and bc not in seen_comp:
            # Register the compact token to capture hyphen/space variations.
            A_comp.add_word(bc, bn); seen_comp.add(bc)
    A_norm.make_automaton()
    A_comp.make_automaton()
    return A_norm, A_comp, mapping


def scan_brands(text_norm: str, text_comp: str,
                A_norm: ahocorasick.Automaton,
                A_comp: ahocorasick.Automaton) -> List[str]:
    """Return normalized brand keys detected via either normalized or compact searches."""
    found: Dict[str, int] = {}
    for _, bn in A_norm.iter(text_norm):
        # Record matches from the normalized automaton, tracking longest span per key.
        found[bn] = max(found.get(bn, 0), len(bn))
    for _, bn in A_comp.iter(text_comp):
        # Merge in compact hits to catch spacing-insensitive matches.
        found[bn] = max(found.get(bn, 0), len(bn))
    return [k for k, _ in sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))]


def fda_generics_set(brand_df: pl.DataFrame | pl.LazyFrame) -> Set[str]:
    """
    Return normalized base generic names present in FDA brand map (Polars/Parquet-first).
    """
    if isinstance(brand_df, pl.DataFrame):
        lf = brand_df.lazy()
    elif isinstance(brand_df, pl.LazyFrame):
        lf = brand_df
    else:
        return set()
    if "generic_name" not in lf.columns:
        return set()
    gens_df = (
        lf.select(pl.col("generic_name").fill_null("").cast(pl.Utf8).alias("generic_name"))
        .with_columns(
            pl.col("generic_name").map_elements(lambda g: _normalize_text_basic(_base_name(g)), return_dtype=pl.Utf8).alias("generic_norm")
        )
        .select("generic_norm")
        .drop_nulls()
        .filter(pl.col("generic_norm") != "")
        .unique()
        .collect()
    )
    return {g for g in gens_df.get_column("generic_norm").to_list() if g}
