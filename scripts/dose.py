# ===============================
# File: scripts/dose.py
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, Optional

from .text_utils import safe_to_float

PACK_RX = re.compile(r"\b(\d+)\s*(?:x|×)\s*(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|ug|iu)\b", re.I)
RATIO_RX_EXTRA = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)\s?(?P<num_unit>mg|g|mcg|ug)\s*/\s?(?P<den>\d+(?:[.,]\d+)?)\s?(?P<den_unit>ml|l)\b", re.I)
PER_UNIT_WORDS = r"(?:tab(?:let)?s?|cap(?:sule)?s?|sachet(?:s)?|drop(?:s)?|gtt|actuation(?:s)?|spray(?:s)?|puff(?:s)?)"

DOSAGE_PATTERNS = [
    # amount-only (e.g., 500 mg)
    r"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\b",
    # amount per mL or L (e.g., 5 mg/5 mL, 1 g/100 L)
    r"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>\d+(?:[.,]\d+)?)\s*(?P<per_unit>ml|l)\b",
    # amount per unit-dose nouns (tab/cap/sachet/drop/actuation/spray/puff)
    rf"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>1)?\s*(?P<per_unit>{PER_UNIT_WORDS})\b",
    # compact noun suffix (e.g., mg/tab, mg/cap)
    rf"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s*/\s?(?P<per_unit>{PER_UNIT_WORDS})\b",
    # percent (optionally w/v or w/w)
    r"(?P<pct>\d+(?:[.,]\d+)?)\s?%(?:\s?(?:w/v|w/w))?",
]
DOSAGE_REGEXES = [re.compile(p, flags=re.I) for p in DOSAGE_PATTERNS]


def _unmask_pack_strength(s_norm: str) -> str:
    """Convert '10 x 500 mg'/'10×500 mg' to just '500 mg' for dose parsing."""
    def repl(m: re.Match):
        amt = m.group(2)
        unit = m.group(3)
        return f"{amt}{unit}"
    return PACK_RX.sub(repl, s_norm)


def parse_dose_struct_from_text(s_norm: str) -> Dict[str, Any]:
    if not isinstance(s_norm, str) or not s_norm:
        return {}
    s_proc = _unmask_pack_strength(s_norm)
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_proc):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        if d.get("strength") and d.get("per_unit") in ("ml", "l"):
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            if d.get("per_unit", "").lower() == "l":
                per_val *= 1000.0
            return {"dose_kind": "ratio", "strength": float(d["strength"]), "unit": d["unit"].lower(),
                    "per_val": per_val, "per_unit": "ml"}
    for d in matches:
        if d.get("strength"):
            return {"dose_kind": "amount", "strength": float(d["strength"]), "unit": d["unit"].lower()}
    for d in matches:
        if d.get("pct"):
            return {"dose_kind": "percent", "pct": float(d["pct"])}
    m = RATIO_RX_EXTRA.search(s_proc)
    if m:
        num = float(m.group("num"))
        num_unit = m.group("num_unit").lower()
        den = float(m.group("den"))
        if m.group("den_unit").lower() == "l":
            den *= 1000.0
        return {"dose_kind": "ratio", "strength": num, "unit": num_unit, "per_val": den, "per_unit": "ml"}
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
    if not isinstance(s_norm, str) or not s_norm:
        return None
    s_proc = _unmask_pack_strength(s_norm)
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_proc):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        if d.get("strength") and d.get("per_unit") in ("ml", "l"):
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            if d.get("per_unit", "").lower() == "l":
                per_val *= 1000.0
            return {"kind": "ratio", "strength": float(d["strength"]), "unit": d["unit"].lower(),
                    "per_val": per_val, "per_unit": "ml"}
    for d in matches:
        if d.get("strength"):
            return {"kind": "amount", "strength": float(d["strength"]), "unit": d["unit"].lower()}
    for d in matches:
        if d.get("pct"):
            return {"kind": "percent", "pct": float(d["pct"])}
    m = RATIO_RX_EXTRA.search(s_proc)
    if m:
        num = float(m.group("num"))
        num_unit = m.group("num_unit").lower()
        den = float(m.group("den"))
        if m.group("den_unit").lower() == "l":
            den *= 1000.0
        return {"kind": "ratio", "strength": num, "unit": num_unit, "per_val": den, "per_unit": "ml"}
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