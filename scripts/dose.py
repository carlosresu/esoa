# ===============================
# File: scripts/dose.py
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, Optional

from .text_utils import safe_to_float

DOSAGE_PATTERNS = [
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\b",
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>1)?\s*(?P<per_unit>ml)\b",
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>\d+(?:[\.,]\d+)?)\s*(?P<per_unit>ml)\b",
    r"(?P<pct>\d+(?:[\.,]\d+)?)\s?%",
]
DOSAGE_REGEXES = [re.compile(p, flags=re.I) for p in DOSAGE_PATTERNS]


def parse_dose_struct_from_text(s_norm: str) -> Dict[str, Any]:
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_norm):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        if d.get("strength") and d.get("per_unit") == "ml":
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            return {"dose_kind": "ratio", "strength": float(d["strength"]), "unit": d["unit"].lower(),
                    "per_val": per_val, "per_unit": "ml"}
    for d in matches:
        if d.get("strength"):
            return {"dose_kind": "amount", "strength": float(d["strength"]), "unit": d["unit"].lower()}
    for d in matches:
        if d.get("pct"):
            return {"dose_kind": "percent", "pct": float(d["pct"])}
    return {}


def to_mg(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None or not isinstance(unit, str):
        return None
    u = unit.lower()
    if u == "mg":
        return value
    if u == "g":
        return value * 1000.0
    if u in ("mcg", "ug"):
        return value / 1000.0
    return None


def to_mg_match(value: float, unit: str):
    u = unit.lower()
    if u == "mg":
        return value
    if u == "g":
        return value * 1000.0
    if u in ("mcg", "ug"):
        return value / 1000.0
    return None


def safe_ratio_mg_per_ml(strength, unit, per_val):
    mg = to_mg(strength, unit)
    pv = safe_to_float(per_val)
    if mg is None or pv in (None, 0):
        return None
    return mg / pv


def extract_dosage(s_norm: str):
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_norm):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        if d.get("strength") and d.get("per_unit") == "ml":
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            return {"kind": "ratio", "strength": float(d["strength"]), "unit": d["unit"].lower(),
                    "per_val": per_val, "per_unit": "ml"}
    for d in matches:
        if d.get("strength"):
            return {"kind": "amount", "strength": float(d["strength"]), "unit": d["unit"].lower()}
    for d in matches:
        if d.get("pct"):
            return {"kind": "percent", "pct": float(d["pct"])}
    return None


def dose_similarity(esoa_dose: dict, pnf_row) -> float:
    if not esoa_dose:
        return 0.0
    kind = esoa_dose.get("kind")
    if kind == "amount":
        mg_esoa = to_mg_match(esoa_dose["strength"], esoa_dose["unit"])
        mg_pnf = pnf_row.get("strength_mg")
        if mg_esoa is None or mg_pnf is None or mg_pnf == 0:
            return 0.0
        rel_err = abs(mg_esoa - mg_pnf) / mg_pnf
        if rel_err < 0.001:
            return 1.0
        if rel_err <= 0.05:
            return 0.8
        if rel_err <= 0.10:
            return 0.6
        return 0.0
    if kind == "ratio":
        if pnf_row.get("dose_kind") != "ratio":
            return 0.0
        v_esoa = to_mg_match(esoa_dose["strength"], esoa_dose["unit"])
        if v_esoa is None:
            return 0.0
        ratio_esoa = v_esoa / float(esoa_dose.get("per_val", 1.0))
        ratio_pnf = pnf_row.get("ratio_mg_per_ml")
        if ratio_pnf in (None, 0):
            return 0.0
        rel_err = abs(ratio_esoa - ratio_pnf) / ratio_pnf
        if rel_err < 0.001:
            return 1.0
        if rel_err <= 0.05:
            return 0.8
        if rel_err <= 0.10:
            return 0.6
        return 0.0
    if kind == "percent":
        if pnf_row.get("dose_kind") != "percent":
            return 0.0
        pct_esoa = float(esoa_dose["pct"])
        pct_pnf = pnf_row.get("pct")
        if pct_pnf is None:
            return 0.0
        rel_err = abs(pct_esoa - pct_pnf) / max(pct_pnf, 1e-9)
        if rel_err < 0.001:
            return 1.0
        if rel_err <= 0.05:
            return 0.8
        if rel_err <= 0.10:
            return 0.6
        return 0.0
    return 0.0