# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from .text_utils import _base_name, _normalize_text_basic


def load_who_molecules(who_csv: str) -> Tuple[Dict[str, set], List[str], Dict[str, List[dict]], Dict[str, str]]:
    """Load WHO exports, providing lookup dictionaries and candidate name lists."""
    who = pd.read_csv(who_csv)
    who["name_base"] = who["atc_name"].fillna("").map(_base_name)
    who["name_norm"] = who["atc_name"].fillna("").map(_normalize_text_basic)
    who["name_base_norm"] = who["name_base"].map(_normalize_text_basic)

    codes_by_name = defaultdict(set)
    details_by_code: Dict[str, List[dict]] = defaultdict(list)
    display_by_name: Dict[str, str] = {}
    for _, r in who.iterrows():
        # Store both base and fully normalized variants for lookup flexibility.
        codes_by_name[r["name_base_norm"]].add(r["atc_code"])
        codes_by_name[r["name_norm"]].add(r["atc_code"])
        if r["name_base_norm"] and r["name_base_norm"] not in display_by_name:
            display_by_name[r["name_base_norm"]] = r["atc_name"]
        if r["name_norm"] and r["name_norm"] not in display_by_name:
            display_by_name[r["name_norm"]] = r["atc_name"]

        details_by_code[r["atc_code"]].append(
            {
                "atc_name": r.get("atc_name"),
                "ddd": r.get("ddd"),
                "uom": r.get("uom"),
                "adm_r": r.get("adm_r"),
                "note": r.get("note"),
            }
        )

    candidate_names = sorted(
        set(list(who["name_norm"]) + list(who["name_base_norm"])),
        key=len, reverse=True
    )
    candidate_names = [n for n in candidate_names if len(n) > 2]
    return codes_by_name, candidate_names, details_by_code, display_by_name


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
