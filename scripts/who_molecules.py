# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from .text_utils import _base_name, _normalize_text_basic


def load_who_molecules(who_csv: str) -> Tuple[Dict[str, set], List[str], Dict[str, List[dict]]]:
    who = pd.read_csv(who_csv)
    who["name_base"] = who["atc_name"].fillna("").map(_base_name)
    who["name_norm"] = who["atc_name"].fillna("").map(_normalize_text_basic)
    who["name_base_norm"] = who["name_base"].map(_normalize_text_basic)

    codes_by_name = defaultdict(set)
    details_by_code: Dict[str, List[dict]] = defaultdict(list)
    for _, r in who.iterrows():
        codes_by_name[r["name_base_norm"]].add(r["atc_code"])
        codes_by_name[r["name_norm"]].add(r["atc_code"])

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
    return codes_by_name, candidate_names, details_by_code


def detect_all_who_molecules(
    text: str,
    regex,
    codes_by_name,
    *,
    pre_normalized: str | None = None,
) -> Tuple[List[str], List[str]]:
    if not isinstance(text, str):
        return [], []
    nt = pre_normalized if pre_normalized is not None else _normalize_text_basic(text)
    names = []
    for m in regex.finditer(nt):
        detected = m.group(1)
        base = _base_name(detected)
        bn = _normalize_text_basic(base)
        names.append(bn)
    names = list(dict.fromkeys(names))
    codes = sorted(set().union(*[codes_by_name.get(n, set()) for n in names])) if names else []
    return names, codes
