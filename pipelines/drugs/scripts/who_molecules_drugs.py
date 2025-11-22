# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import polars as pl

from .text_utils_drugs import (
    _base_name,
    _normalize_text_basic,
    extract_base_and_salts,
    serialize_salt_list,
)


def load_who_molecules(who_parquet: str) -> Tuple[Dict[str, set], List[str], Dict[str, List[dict]]]:
    """Load WHO exports (Parquet/Polars-first), providing lookup dictionaries and candidate name lists."""
    who = pl.read_parquet(who_parquet)
    who = who.with_columns(
        pl.col("atc_name").fill_null("").cast(pl.Utf8).alias("atc_name"),
        pl.col("atc_code").fill_null("").cast(pl.Utf8).alias("atc_code"),
        pl.col("ddd"),
        pl.col("uom"),
        pl.col("adm_r"),
        pl.col("note"),
    )
    who = who.with_columns(
        pl.col("atc_name").map_elements(_base_name, return_dtype=pl.Utf8).alias("name_base"),
        pl.col("atc_name").map_elements(_normalize_text_basic, return_dtype=pl.Utf8).alias("name_norm"),
        pl.col("atc_name").map_elements(extract_base_and_salts, return_dtype=pl.Struct({"": pl.Utf8})).alias("_split_vals"),
    )
    split_vals = who.get_column("_split_vals").to_list()
    atc_names_list = who.get_column("atc_name").to_list()
    salt_forms = [serialize_salt_list(salts) for _, salts in split_vals]
    saltless_bases = [base if base else str(original).strip().upper() for (base, _), original in zip(split_vals, atc_names_list)]
    name_base_norm = [_normalize_text_basic(n) for n in who.get_column("name_base").to_list()]
    name_saltless_norm = [
        _normalize_text_basic(name) if name else _normalize_text_basic(str(original))
        for name, original in zip(saltless_bases, atc_names_list)
    ]

    who = who.with_columns(
        pl.Series(salt_forms).alias("salt_form"),
        pl.Series(saltless_bases).alias("name_saltless"),
        pl.Series(name_base_norm).alias("name_base_norm"),
        pl.Series(name_saltless_norm).alias("name_saltless_norm"),
    ).drop("_split_vals")

    codes_by_name = defaultdict(set)
    details_by_code: Dict[str, List[dict]] = defaultdict(list)
    for r in who.to_dicts():
        name_base_norm_val = r.get("name_base_norm")
        name_saltless_norm_val = r.get("name_saltless_norm")
        name_norm_val = r.get("name_norm")
        atc_code_val = r.get("atc_code")
        if name_base_norm_val:
            codes_by_name[name_base_norm_val].add(atc_code_val)
        if name_saltless_norm_val:
            codes_by_name[name_saltless_norm_val].add(atc_code_val)
        if name_norm_val:
            codes_by_name[name_norm_val].add(atc_code_val)

        details_by_code[atc_code_val].append(
            {
                "atc_name": r.get("atc_name"),
                "ddd": r.get("ddd"),
                "uom": r.get("uom"),
                "adm_r": r.get("adm_r"),
                "note": r.get("note"),
                "salt_form": r.get("salt_form"),
            }
        )

    candidate_names = sorted(
        set(name_norm_val for name_norm_val in who.get_column("name_norm").to_list())
        | set(name_base_norm)
        | set(name_saltless_norm),
        key=len,
        reverse=True,
    )
    candidate_names = [n for n in candidate_names if len(n) > 2]
    return codes_by_name, candidate_names, details_by_code


def detect_all_who_molecules(
    text: str,
    regex,
    codes_by_name,
    *,
    pre_normalized: str | None = None,
) -> Tuple[List[str], List[str]]:
    """Return normalized WHO molecule names and corresponding ATC codes found in text."""
    if not isinstance(text, str):
        return [], []
    nt = pre_normalized if pre_normalized is not None else _normalize_text_basic(text)
    names = []
    for m in regex.finditer(nt):
        detected = m.group(1)
        base = _base_name(detected)
        bn = _normalize_text_basic(base)
        # Track unique normalized molecule names in detection order.
        names.append(bn)
    names = list(dict.fromkeys(names))
    codes = sorted(set().union(*[codes_by_name.get(n, set()) for n in names])) if names else []
    return names, codes
