# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from .text_utils_drugs import (
    _base_name,
    _normalize_text_basic,
    extract_base_and_salts,
    serialize_salt_list,
)


def load_who_molecules(who_csv: str) -> Tuple[Dict[str, set], List[str], Dict[str, List[dict]]]:
    """Load WHO exports, providing lookup dictionaries and candidate name lists."""
    who = pd.read_csv(who_csv)
    who["name_base"] = who["atc_name"].fillna("").map(_base_name)
    who["name_norm"] = who["atc_name"].fillna("").map(_normalize_text_basic)
    split_vals = who["atc_name"].fillna("").map(extract_base_and_salts)
    who["salt_form"] = [serialize_salt_list(salts) for _, salts in split_vals]
    saltless_bases = [base if base else str(original).strip().upper() for (base, _), original in zip(split_vals, who["atc_name"].fillna("").astype(str))]
    who["name_saltless"] = saltless_bases
    who["name_base_norm"] = who["name_base"].map(_normalize_text_basic)
    who["name_saltless_norm"] = [
        _normalize_text_basic(name) if name else _normalize_text_basic(str(original))
        for name, original in zip(saltless_bases, who["atc_name"].fillna("").astype(str))
    ]

    codes_by_name = defaultdict(set)
    details_by_code: Dict[str, List[dict]] = defaultdict(list)
    for _, r in who.iterrows():
        # Store both base and fully normalized variants for lookup flexibility.
        codes_by_name[r["name_base_norm"]].add(r["atc_code"])
        codes_by_name[r["name_saltless_norm"]].add(r["atc_code"])
        codes_by_name[r["name_norm"]].add(r["atc_code"])

        details_by_code[r["atc_code"]].append(
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
        set(list(who["name_norm"]) + list(who["name_base_norm"]) + list(who["name_saltless_norm"])),
        key=len, reverse=True
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
