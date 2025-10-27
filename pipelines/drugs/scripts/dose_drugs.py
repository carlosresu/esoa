#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, Optional
from math import isclose

from .text_utils_drugs import safe_to_float

PACK_RX = re.compile(r"\b(\d+)\s*(?:x|×)\s*(\d+(?:[.,]\d+)?)\s*(mg|g|mcg|ug|iu)\b", re.I)
RATIO_RX_EXTRA = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)\s?(?P<num_unit>mg|g|mcg|ug)\s*/\s?(?P<den>\d+(?:[.,]\d+)?)\s?(?P<den_unit>ml|l)\b", re.I)
PER_UNIT_WORDS = r"(?:tab(?:let)?s?|cap(?:sule)?s?|sachet(?:s)?|drop(?:s)?|gtt|actuation(?:s)?|spray(?:s)?|puff(?:s)?)"

DOSAGE_PATTERNS = [
    # amount-only (e.g., 500 mg)
    r"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\b",
    # amount per mL or L (e.g., 5 mg/5 mL, 1 g/100 L, 150 mg/mL)
    r"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?:(?P<per_val>\d+(?:[.,]\d+)?)\s*)?(?P<per_unit>ml|l)\b",
    # amount per unit-dose nouns (tab/cap/sachet/drop/actuation/spray/puff)
    rf"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>1)?\s*(?P<per_unit>{PER_UNIT_WORDS})\b",
    # compact noun suffix (e.g., mg/tab, mg/cap)
    rf"(?P<strength>\d+(?:[.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s*/\s?(?P<per_unit>{PER_UNIT_WORDS})\b",
    # percent (optionally w/v or w/w)
    r"(?P<pct>\d+(?:[.,]\d+)?)\s?%(?:\s?(?:w/v|w/w))?",
]
DOSAGE_REGEXES = [re.compile(p, flags=re.I) for p in DOSAGE_PATTERNS]

_SPECIAL_AMOUNT_EQUIVALENCE = {
    # Extended/modified-release trimetazidine marketed as 60–80 mg capsules still correspond to
    # the 35 mg (base) strength present in the PNF. Accept values in that modified-release band.
    "trimetazidine": {
        "target_strength_mg": 35.0,
        "min_strength_mg": 55.0,
        "max_strength_mg": 90.0,
    },
}


def _unmask_pack_strength(s_norm: str) -> str:
    """Convert '10 x 500 mg'/'10×500 mg' to just '500 mg' for dose parsing."""
    def repl(m: re.Match):
        amt = m.group(2)
        unit = m.group(3)
        return f"{amt}{unit}"
    # Replace pack descriptors with a single strength to simplify downstream regexes.
    return PACK_RX.sub(repl, s_norm)


def parse_dose_struct_from_text(s_norm: str) -> Dict[str, Any]:
    """Extract a structured dose payload describing amounts, ratios (mg per mL or per unit-dose noun), packs (10×500 mg → 500 mg), and percent strengths, normalizing units as detailed in README."""
    if not isinstance(s_norm, str) or not s_norm:
        return {}
    s_proc = _unmask_pack_strength(s_norm)
    matches = []
    for rx in DOSAGE_REGEXES:
        # Collect all regex hits across the supported dose patterns.
        for m in rx.finditer(s_proc):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        strength = d.get("strength")
        unit = d.get("unit")
        per_unit = d.get("per_unit")
        if strength and per_unit in ("ml", "l") and unit:
            try:
                strength_val = float(strength)
            except (TypeError, ValueError):
                continue
            try:
                per_val_raw = d.get("per_val")
                per_val = float(per_val_raw) if per_val_raw not in (None, "") else 1.0
            except (TypeError, ValueError):
                per_val = 1.0
            per_unit_norm = str(per_unit).lower()
            if per_unit_norm == "l":
                per_val *= 1000.0
                per_unit_norm = "ml"
            # Return the first ratio-style match encountered.
            return {
                "dose_kind": "ratio",
                "strength": strength_val,
                "unit": str(unit).lower(),
                "per_val": per_val,
                "per_unit": per_unit_norm,
            }
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
    """Convert amount doses to milligrams when the unit is convertible."""
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
    """Helper used by match-time comparison for converting dose amounts to mg."""
    u = unit.lower()
    if u == "mg":
        return value
    if u == "g":
        return value * 1000.0
    if u in ("mcg", "ug"):
        return value / 1000.0
    return None


def safe_ratio_mg_per_ml(strength, unit, per_val):
    """Return the mg/mL equivalent for ratio doses when convertible."""
    mg = to_mg(strength, unit)
    pv = safe_to_float(per_val)
    if mg is None or pv in (None, 0):
        return None
    return mg / pv


def extract_dosage(s_norm: str):
    """Parse normalized text into a compact dosage dict tailored for features."""
    if not isinstance(s_norm, str) or not s_norm:
        return None
    s_proc = _unmask_pack_strength(s_norm)
    matches = []
    for rx in DOSAGE_REGEXES:
        # Gather candidates for amount/ratio/percent patterns within the string.
        for m in rx.finditer(s_proc):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        strength = d.get("strength")
        unit = d.get("unit")
        per_unit = d.get("per_unit")
        if strength and per_unit in ("ml", "l") and unit:
            try:
                strength_val = float(strength)
            except (TypeError, ValueError):
                continue
            try:
                per_val_raw = d.get("per_val")
                per_val = float(per_val_raw) if per_val_raw not in (None, "") else 1.0
            except (TypeError, ValueError):
                per_val = 1.0
            per_unit_norm = str(per_unit).lower()
            if per_unit_norm == "l":
                per_val *= 1000.0
                per_unit_norm = "ml"
            return {
                "kind": "ratio",
                "strength": strength_val,
                "unit": str(unit).lower(),
                "per_val": per_val,
                "per_unit": per_unit_norm,
            }
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
        # Normalize implicit litre denominators to mL for consistency.
        return {"kind": "ratio", "strength": num, "unit": num_unit, "per_val": den, "per_unit": "ml"}
    return None


def _eq(a: float, b: float) -> bool:
    """Exact equality with robust float check. Accept only true equality up to tiny machine epsilon.
    This enforces zero tolerance logically (e.g., 1 g == 1000 mg), but rejects 450 vs 500 mg.
    """
    # Use a very tight tolerance to avoid binary float artifacts while still enforcing exactness.
    return isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def dose_similarity(esoa_dose: dict, pnf_row) -> float:
    """Return 1.0 only for exact equality (after unit conversion); else 0.0.
    - Amounts compare mg against the PNF `strength_mg` with optional modified-release equivalence (trimetazidine 55–90 mg ↦ 35 mg base).
    - Ratios require the same mg/mL once litres are normalized to mL.
    - Percents accept only exact matches.
    """
    if not esoa_dose:
        return 0.0
    kind = esoa_dose.get("kind")
    if kind == "amount":
        mg_esoa = to_mg_match(esoa_dose["strength"], esoa_dose["unit"])
        mg_pnf = pnf_row.get("strength_mg")
        if mg_esoa is None or mg_pnf is None:
            return 0.0
        if _eq(mg_esoa, mg_pnf):
            return 1.0
        generic_id = pnf_row.get("generic_id")
        if isinstance(generic_id, str):
            special = _SPECIAL_AMOUNT_EQUIVALENCE.get(generic_id.strip().lower())
            if special and pnf_row.get("dose_kind") == "amount":
                target = special.get("target_strength_mg")
                min_val = special.get("min_strength_mg")
                max_val = special.get("max_strength_mg")
                if (
                    target is not None
                    and _eq(float(mg_pnf), float(target))
                    and min_val is not None
                    and max_val is not None
                    and min_val <= float(mg_esoa) <= max_val
                ):
                    # Allow documented equivalence ranges for modified-release capsules.
                    return 1.0
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
        # Compare mg/mL ratios with strict equality semantics.
        return 1.0 if _eq(ratio_esoa, ratio_pnf) else 0.0
    if kind == "percent":
        if pnf_row.get("dose_kind") != "percent":
            return 0.0
        pct_esoa = float(esoa_dose["pct"])
        pct_pnf = pnf_row.get("pct")
        if pct_pnf is None:
            return 0.0
        # Percent doses must match exactly once cast to floats.
        return 1.0 if _eq(pct_esoa, float(pct_pnf)) else 0.0
    return 0.0
