#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Builds dose/form-aware prepared inputs for the matcher:
- Reads:  pnf.csv, esoa.csv
- Writes: pnf_prepared.csv (one row per molecule × route × dose/form),
          esoa_prepared.csv
Usage:
  python prepare.py --pnf pnf.csv --esoa esoa.csv --outdir .
"""

import argparse
import os
import re
from typing import List, Optional, Dict, Any

import pandas as pd

# ---------- Normalization helpers ----------
def normalize_text(s: str) -> str:
    import unicodedata
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w%/+\.\- ]+", " ", s)
    s = s.replace("microgram", "mcg").replace("μg", "mcg").replace("µg", "mcg")
    s = s.replace("cc", "ml").replace("milli litre", "ml").replace("milliliter", "ml")
    s = s.replace("gm", "g").replace("gms", "g").replace("milligram", "mg")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def slug_id(name: str) -> str:
    base = normalize_text(str(name))
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")

def clean_atc(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return s.replace("\u00a0", " ").strip()

# ---------- Route & form mappings ----------
def map_route_token(r) -> List[str]:
    if not isinstance(r, str):
        return []
    r = r.strip()
    table = {
        "Oral:": ["oral"],
        "Oral/Tube feed:": ["oral"],
        "Inj.:": ["intravenous","intramuscular","subcutaneous"],
        "IV:": ["intravenous"],
        "IV/SC:": ["intravenous","subcutaneous"],
        "SC:": ["subcutaneous"],
        "Subdermal:": ["subcutaneous"],
        "Inhalation:": ["inhalation"],
        "Topical:": ["topical"],
        "Patch:": ["transdermal"],
        "Ophthalmic:": ["ophthalmic"],
        "Intraocular:": ["ophthalmic"],
        "Otic:": ["otic"],
        "Nasal:": ["nasal"],
        "Rectal:": ["rectal"],
        "Vaginal:": ["vaginal"],
        "Sublingual:": ["sublingual"],
        "Oral antiseptic:": ["oral"],
        "Oral/Inj.:": ["oral","intravenous","intramuscular","subcutaneous"],
    }
    return table.get(r, [])

FORM_TO_ROUTE = {
    "tablet": "oral", "tab": "oral", "capsule": "oral", "cap": "oral",
    "syrup": "oral", "suspension": "oral", "solution": "oral",
    "drop": "ophthalmic", "eye drop": "ophthalmic", "ear drop": "otic",
    "cream": "topical", "ointment": "topical", "gel": "topical", "lotion": "topical",
    "patch": "transdermal", "inhaler": "inhalation", "nebule": "inhalation",
    "ampoule": "intravenous", "vial": "intravenous", "inj": "intravenous",
    "suppository": "rectal"
}
FORM_WORDS = sorted(set(FORM_TO_ROUTE.keys()), key=len, reverse=True)

# ---------- Dosage parsing ----------
DOSAGE_PATTERNS = [
    # amount-only: "500 mg", "0.5 mg", "1 g", "200 mcg", "100 IU"
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\b",

    # ratio per 1 mL: "5 mg/ml", "5mg per ml"
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>1)?\s*(?P<per_unit>ml)\b",

    # ratio per X mL: "5 mg/5 mL", "120 mg per 5 ml"
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>\d+(?:[\.,]\d+)?)\s*(?P<per_unit>ml)\b",

    # percent: "2%", "0.05 %"
    r"(?P<pct>\d+(?:[\.,]\d+)?)\s?%",
]
DOSAGE_REGEXES = [re.compile(p, flags=re.I) for p in DOSAGE_PATTERNS]

def parse_form(s_norm: str) -> Optional[str]:
    for fw in FORM_WORDS:
        if re.search(rf"\b{re.escape(fw)}\b", s_norm):
            return fw
    return None

def parse_dose_struct(s_norm: str) -> Dict[str, Any]:
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_norm):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)

    # Prefer most-informative: ratio > amount > %
    for d in matches:
        if d.get("strength") and (d.get("per_unit") == "ml"):
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            return {
                "dose_kind": "ratio",
                "strength": float(d["strength"]),
                "unit": d["unit"].lower(),
                "per_val": per_val,
                "per_unit": "ml",
            }
    for d in matches:
        if d.get("strength"):
            return {
                "dose_kind": "amount",
                "strength": float(d["strength"]),
                "unit": d["unit"].lower(),
            }
    for d in matches:
        if d.get("pct"):
            return {
                "dose_kind": "percent",
                "pct": float(d["pct"]),
            }
    return {}

# ---------- Unit conversion & safe math ----------
def to_mg(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None or not isinstance(unit, str):
        return None
    u = unit.lower()
    if u == "mg": return value
    if u == "g":  return value * 1000.0
    if u in ("mcg","ug"): return value / 1000.0
    # IU is compound-specific; cannot convert safely → return None
    return None

def safe_to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        return float(x)
    except Exception:
        return None

def safe_ratio_mg_per_ml(strength, unit, per_val):
    """Return mg/mL or None if we cannot safely compute."""
    mg = to_mg(strength, unit)
    pv = safe_to_float(per_val)
    if mg is None or pv in (None, 0):
        return None
    return mg / pv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnf", required=True, help="Path to raw pnf.csv")
    ap.add_argument("--esoa", required=True, help="Path to raw esoa.csv")
    ap.add_argument("--outdir", default=".", help="Output folder")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- 1) Prepare PNF: PRESERVE DOSE + FORM VARIANTS ----------
    pnf = pd.read_csv(args.pnf)
    for col in ["Molecule", "Route", "ATC Code"]:
        if col not in pnf.columns:
            raise ValueError(f"pnf.csv is missing required column: {col}")

    pnf["generic_name"] = pnf["Molecule"].fillna("").astype(str)
    pnf["generic_id"]   = pnf["generic_name"].map(slug_id)
    pnf["synonyms"]     = ""

    # From Route column → route tokens
    pnf["route_tokens"] = pnf["Route"].map(map_route_token)
    pnf["atc_code"]     = pnf["ATC Code"].map(clean_atc)

    # Build a working text for dose/form parsing (Technical Specifications if present, else Molecule)
    text_cols = []
    for c in ["Technical Specifications", "Specs", "Specification"]:
        if c in pnf.columns:
            text_cols.append(c)
    if text_cols:
        pnf["_tech"] = pnf[text_cols[0]].fillna("")
    else:
        pnf["_tech"] = ""

    # Parse dose + form from tech text (fall back to generic_name for form hints)
    pnf["_parse_src"] = (
        (pnf["generic_name"].astype(str) + " " + pnf["_tech"].astype(str)).str.strip()
    ).map(normalize_text)

    parsed = pnf["_parse_src"].map(parse_dose_struct)
    pnf["dose_kind"] = parsed.map(lambda d: d.get("dose_kind"))
    pnf["strength"]  = parsed.map(lambda d: d.get("strength"))
    pnf["unit"]      = parsed.map(lambda d: d.get("unit"))
    pnf["per_val"]   = parsed.map(lambda d: d.get("per_val"))
    pnf["per_unit"]  = parsed.map(lambda d: d.get("per_unit"))
    pnf["pct"]       = parsed.map(lambda d: d.get("pct"))
    pnf["form_token"]= pnf["_parse_src"].map(parse_form)

    # Convenience: normalized mg and mg/ml (guarded)
    pnf["strength_mg"] = pnf.apply(
        lambda r: to_mg(r.get("strength"), r.get("unit"))
                  if (pd.notna(r.get("strength")) and isinstance(r.get("unit"), str) and r.get("unit"))
                  else None,
        axis=1
    )
    pnf["ratio_mg_per_ml"] = pnf.apply(
        lambda r: safe_ratio_mg_per_ml(r.get("strength"), r.get("unit"), r.get("per_val"))
                  if (r.get("dose_kind") == "ratio" and str(r.get("per_unit")).lower() == "ml")
                  else None,
        axis=1
    )

    # Explode by route_tokens so each row is molecule × single route × single dose/form
    exploded = pnf.explode("route_tokens", ignore_index=True)
    exploded.rename(columns={"route_tokens": "route_allowed"}, inplace=True)

    # Keep only meaningful rows (at least a name)
    keep = exploded[exploded["generic_name"].astype(bool)].copy()

    # Final projection
    pnf_prepared = keep[[
        "generic_id","generic_name","synonyms","atc_code",
        "route_allowed","form_token","dose_kind",
        "strength","unit","per_val","per_unit","pct",
        "strength_mg","ratio_mg_per_ml"
    ]].copy()

    # Write out
    pnf_out = os.path.join(args.outdir, "pnf_prepared.csv")
    pnf_prepared.to_csv(pnf_out, index=False, encoding="utf-8")
    print(f"Wrote {pnf_out} with {len(pnf_prepared):,} rows")

    # ---------- 2) Prepare eSOA ----------
    esoa = pd.read_csv(args.esoa)
    if "DESCRIPTION" not in esoa.columns:
        raise ValueError("esoa.csv is missing required column: DESCRIPTION")
    esoa_prepared = esoa.rename(columns={"DESCRIPTION": "raw_text"}).copy()

    esoa_out = os.path.join(args.outdir, "esoa_prepared.csv")
    esoa_prepared.to_csv(esoa_out, index=False, encoding="utf-8")
    print(f"Wrote {esoa_out} with {len(esoa_prepared):,} rows")

if __name__ == "__main__":
    main()
