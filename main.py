#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main.py — Dose/form-aware preparation + dose-aware matching.

Exports:
- prepare(pnf_csv, esoa_csv, outdir) -> (pnf_prepared_csv, esoa_prepared_csv)
- match(pnf_prepared_csv, esoa_prepared_csv, out_csv) -> out_csv
- run_all(pnf_csv, esoa_csv, outdir, out_csv) -> out_csv

CLI:
  python main.py --pnf pnf.csv --esoa esoa.csv --outdir . --out esoa_matched.csv
"""

import argparse
import os
import re
import math
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import unicodedata
import ahocorasick


# =========================
# Shared helpers
# =========================
def normalize_text(s: str) -> str:
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

def safe_to_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        return float(x)
    except Exception:
        return None

# Routes/forms
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
    "ampoule": "intravenous", "amp": "intravenous", "ampul": "intravenous",
    "vial": "intravenous", "inj": "intravenous",
    "suppository": "rectal"
}

FORM_WORDS = sorted(set(FORM_TO_ROUTE.keys()), key=len, reverse=True)

# Dosage regex
DOSAGE_PATTERNS = [
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\b",
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>1)?\s*(?P<per_unit>ml)\b",
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>\d+(?:[\.,]\d+)?)\s*(?P<per_unit>ml)\b",
    r"(?P<pct>\d+(?:[\.,]\d+)?)\s?%",
]
DOSAGE_REGEXES = [re.compile(p, flags=re.I) for p in DOSAGE_PATTERNS]

def parse_form_from_text(s_norm: str) -> Optional[str]:
    for fw in FORM_WORDS:
        if re.search(rf"\b{re.escape(fw)}\b", s_norm):
            return fw
    return None

def parse_dose_struct_from_text(s_norm: str) -> Dict[str, Any]:
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_norm):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        if d.get("strength") and d.get("per_unit") == "ml":
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            return {"dose_kind":"ratio","strength":float(d["strength"]),"unit":d["unit"].lower(),
                    "per_val":per_val,"per_unit":"ml"}
    for d in matches:
        if d.get("strength"):
            return {"dose_kind":"amount","strength":float(d["strength"]),"unit":d["unit"].lower()}
    for d in matches:
        if d.get("pct"):
            return {"dose_kind":"percent","pct":float(d["pct"])}
    return {}

def to_mg(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None or not isinstance(unit, str):
        return None
    u = unit.lower()
    if u == "mg": return value
    if u == "g":  return value * 1000.0
    if u in ("mcg","ug"): return value / 1000.0
    return None  # IU unknown

def safe_ratio_mg_per_ml(strength, unit, per_val):
    mg = to_mg(strength, unit)
    pv = safe_to_float(per_val)
    if mg is None or pv in (None, 0):
        return None
    return mg / pv


# =========================
# PREPARE
# =========================
def prepare(pnf_csv: str, esoa_csv: str, outdir: str = ".") -> tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)

    pnf = pd.read_csv(pnf_csv)
    for col in ["Molecule", "Route", "ATC Code"]:
        if col not in pnf.columns:
            raise ValueError(f"pnf.csv is missing required column: {col}")

    pnf["generic_name"] = pnf["Molecule"].fillna("").astype(str)
    pnf["generic_id"]   = pnf["generic_name"].map(slug_id)
    pnf["synonyms"]     = ""
    pnf["route_tokens"] = pnf["Route"].map(map_route_token)
    pnf["atc_code"]     = pnf["ATC Code"].map(clean_atc)

    # Where to parse dose/form from
    text_cols = [c for c in ["Technical Specifications", "Specs", "Specification"] if c in pnf.columns]
    pnf["_tech"] = pnf[text_cols[0]].fillna("") if text_cols else ""
    pnf["_parse_src"] = (pnf["generic_name"].astype(str) + " " + pnf["_tech"].astype(str)).str.strip().map(normalize_text)

    parsed = pnf["_parse_src"].map(parse_dose_struct_from_text)
    pnf["dose_kind"]  = parsed.map(lambda d: d.get("dose_kind"))
    pnf["strength"]   = parsed.map(lambda d: d.get("strength"))
    pnf["unit"]       = parsed.map(lambda d: d.get("unit"))
    pnf["per_val"]    = parsed.map(lambda d: d.get("per_val"))
    pnf["per_unit"]   = parsed.map(lambda d: d.get("per_unit"))
    pnf["pct"]        = parsed.map(lambda d: d.get("pct"))
    pnf["form_token"] = pnf["_parse_src"].map(parse_form_from_text)

    pnf["strength_mg"] = pnf.apply(
        lambda r: to_mg(r.get("strength"), r.get("unit"))
                  if (pd.notna(r.get("strength")) and isinstance(r.get("unit"), str) and r.get("unit"))
                  else None, axis=1)
    pnf["ratio_mg_per_ml"] = pnf.apply(
        lambda r: safe_ratio_mg_per_ml(r.get("strength"), r.get("unit"), r.get("per_val"))
                  if (r.get("dose_kind") == "ratio" and str(r.get("per_unit")).lower() == "ml")
                  else None, axis=1)

    exploded = pnf.explode("route_tokens", ignore_index=True)
    exploded.rename(columns={"route_tokens": "route_allowed"}, inplace=True)
    keep = exploded[exploded["generic_name"].astype(bool)].copy()

    pnf_prepared = keep[[
        "generic_id","generic_name","synonyms","atc_code",
        "route_allowed","form_token","dose_kind",
        "strength","unit","per_val","per_unit","pct",
        "strength_mg","ratio_mg_per_ml"
    ]].copy()

    pnf_out = os.path.join(outdir, "pnf_prepared.csv")
    pnf_prepared.to_csv(pnf_out, index=False, encoding="utf-8")
    print(f"[prepare] wrote {pnf_out} with {len(pnf_prepared):,} rows")

    esoa = pd.read_csv(esoa_csv)
    if "DESCRIPTION" not in esoa.columns:
        raise ValueError("esoa.csv is missing required column: DESCRIPTION")
    esoa_prepared = esoa.rename(columns={"DESCRIPTION": "raw_text"}).copy()
    esoa_out = os.path.join(outdir, "esoa_prepared.csv")
    esoa_prepared.to_csv(esoa_out, index=False, encoding="utf-8")
    print(f"[prepare] wrote {esoa_out} with {len(esoa_prepared):,} rows")

    return pnf_out, esoa_out


# =========================
# MATCH
# =========================
ROUTE_ALIASES = {
    "po": "oral", "per orem": "oral", "by mouth": "oral",
    "iv": "intravenous", "intravenous": "intravenous",
    "im": "intramuscular", "intramuscular": "intramuscular",
    "sc": "subcutaneous", "subcut": "subcutaneous",
    "sl": "sublingual", "bucc": "buccal",
    "topical": "topical", "cut": "topical", "dermal": "transdermal",
    "oph": "ophthalmic", "eye": "ophthalmic",
    "otic": "otic", "ear": "otic",
    "inh": "inhalation", "neb": "inhalation",
    "rectal": "rectal", "vaginal": "vaginal",
    "intrathecal": "intrathecal", "nasal": "nasal",
}

def extract_route_and_form(s_norm: str):
    route_found = None
    form_found = None
    evidence = []
    for fw in FORM_WORDS:
        if re.search(rf"\b{re.escape(fw)}\b", s_norm):
            form_found = fw
            evidence.append(f"form:{fw}")
            break
    for alias, route in ROUTE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", s_norm):
            route_found = route
            evidence.append(f"route:{alias}->{route}")
            break
    if not route_found and form_found in FORM_TO_ROUTE:
        route_found = FORM_TO_ROUTE[form_found]
        evidence.append(f"impute_route:{form_found}->{route_found}")
    return route_found, form_found, ";".join(evidence)

def extract_dosage(s_norm: str):
    matches = []
    for rx in DOSAGE_REGEXES:
        for m in rx.finditer(s_norm):
            d = {k: (v.replace(",", ".") if isinstance(v, str) else v) for k, v in m.groupdict().items()}
            matches.append(d)
    for d in matches:
        if d.get("strength") and d.get("per_unit") == "ml":
            per_val = float(d["per_val"]) if d.get("per_val") else 1.0
            return {"kind":"ratio","strength":float(d["strength"]), "unit":d["unit"].lower(),
                    "per_val":per_val, "per_unit":"ml"}
    for d in matches:
        if d.get("strength"):
            return {"kind":"amount","strength":float(d["strength"]), "unit":d["unit"].lower()}
    for d in matches:
        if d.get("pct"):
            return {"kind":"percent","pct":float(d["pct"])}
    return None

def build_molecule_automaton(pnf_df: pd.DataFrame) -> ahocorasick.Automaton:
    A = ahocorasick.Automaton()
    seen = set()
    for gid, gname in pnf_df[["generic_id","generic_name"]].drop_duplicates().itertuples(index=False):
        key = normalize_text(gname)
        if key and (gid, key) not in seen:
            A.add_word(key, (gid, key)); seen.add((gid, key))
    if "synonyms" in pnf_df.columns:
        for gid, syns in pnf_df[["generic_id","synonyms"]].itertuples(index=False):
            if isinstance(syns, str) and syns.strip():
                for s in syns.split("|"):
                    key = normalize_text(s)
                    if key and (gid, key) not in seen:
                        A.add_word(key, (gid, key)); seen.add((gid, key))
    A.make_automaton()
    return A

def to_mg_match(value: float, unit: str):
    u = unit.lower()
    if u == "mg": return value
    if u == "g": return value * 1000.0
    if u in ("mcg","ug"): return value / 1000.0
    return None

def dose_similarity(esoa_dose: dict, pnf_row: pd.Series) -> float:
    if not esoa_dose:
        return 0.0
    kind = esoa_dose.get("kind")
    if kind == "amount":
        mg_esoa = to_mg_match(esoa_dose["strength"], esoa_dose["unit"])
        mg_pnf = pnf_row.get("strength_mg")
        if mg_esoa is None or mg_pnf is None or mg_pnf == 0:
            return 0.0
        rel_err = abs(mg_esoa - mg_pnf) / mg_pnf
        if rel_err < 0.001: return 1.0
        if rel_err <= 0.05: return 0.8
        if rel_err <= 0.10: return 0.6
        return 0.0
    if kind == "ratio":
        if pnf_row.get("dose_kind") != "ratio":
            return 0.0
        v_esoa = to_mg_match(esoa_dose["strength"], esoa_dose["unit"])
        if v_esoa is None:
            return 0.0
        ratio_esoa = v_esoa / float(esoa_dose.get("per_val", 1.0))
        ratio_pnf  = pnf_row.get("ratio_mg_per_ml")
        if ratio_pnf in (None, 0):
            return 0.0
        rel_err = abs(ratio_esoa - ratio_pnf) / ratio_pnf
        if rel_err < 0.001: return 1.0
        if rel_err <= 0.05: return 0.8
        if rel_err <= 0.10: return 0.6
        return 0.0
    if kind == "percent":
        if pnf_row.get("dose_kind") != "percent":
            return 0.0
        pct_esoa = float(esoa_dose["pct"])
        pct_pnf  = pnf_row.get("pct")
        if pd.isna(pct_pnf):
            return 0.0
        rel_err = abs(pct_esoa - pct_pnf) / max(pct_pnf, 1e-9)
        if rel_err < 0.001: return 1.0
        if rel_err <= 0.05: return 0.8
        if rel_err <= 0.10: return 0.6
        return 0.0
    return 0.0

def looks_like_combination(s_norm: str, molecule_hits: set) -> bool:
    # Multiple distinct molecule hits => combination
    if len(molecule_hits) > 1:
        return True

    # Mask dosage ratios like "5 mg/ml", "120 mg / 5 ml" so their "/" doesn't trigger
    dosage_ratio_rx = re.compile(r"""
        \b
        \d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|ug|iu)   # numerator amount+unit
        \s*/\s*
        (?:\d+(?:[\.,]\d+)?\s*)?(?:ml|l)        # optional per-value + mL/L
        \b
    """, re.IGNORECASE | re.VERBOSE)
    s_masked = dosage_ratio_rx.sub(" <DOSE> ", s_norm)

    # If there's an explicit "with" or a plus sign, it's almost certainly a combo
    if re.search(r"\bwith\b", s_masked):
        return True
    if "+" in s_masked:
        return True

    # Slash between words (not numbers/units) indicates combo (e.g., amox/clav)
    if re.search(r"[a-z]\s*/\s*[a-z]", s_masked):
        return True

    return False

def match(pnf_prepared_csv: str, esoa_prepared_csv: str, out_csv: str = "esoa_matched.csv") -> str:
    pnf_df = pd.read_csv(pnf_prepared_csv)
    esoa_df = pd.read_csv(esoa_prepared_csv)

    required_pnf = {"generic_id","generic_name","synonyms","atc_code","route_allowed",
                    "form_token","dose_kind","strength","unit","per_val","per_unit","pct",
                    "strength_mg","ratio_mg_per_ml"}
    missing = required_pnf - set(pnf_df.columns)
    if missing:
        raise ValueError(f"{pnf_prepared_csv} missing columns: {missing}")
    if "raw_text" not in esoa_df.columns:
        raise ValueError(f"{esoa_prepared_csv} must contain a 'raw_text' column")

    A = build_molecule_automaton(pnf_df)

    df = esoa_df[["raw_text"]].copy()
    df["norm"] = df["raw_text"].map(normalize_text)

    generic_id, generic_name_hit, multi_hit = [], [], []
    for s in df["norm"]:
        hits = []
        for end_idx, (gid, key) in A.iter(s):
            start_idx = end_idx - len(key) + 1
            hits.append((start_idx, end_idx, gid, key))
        if not hits:
            generic_id.append(None); generic_name_hit.append(None); multi_hit.append(False); continue
        hits.sort(key=lambda x: x[1]-x[0], reverse=True)
        distinct_gids = {h[2] for h in hits}
        generic_id.append(hits[0][2]); generic_name_hit.append(hits[0][3])
        multi_hit.append(len(distinct_gids) > 1)

    df["generic_id"] = generic_id
    df["molecule_token"] = generic_name_hit
    df["multi_hit"] = multi_hit

    df["dosage_parsed"] = df["norm"].map(extract_dosage)
    df["route"], df["form"], df["route_evidence"] = zip(*df["norm"].map(extract_route_and_form))
    df["looks_combo"] = [looks_like_combination(s, {g} if g else set())
                         for s, g in zip(df["norm"], df["generic_id"])]

    df["bucket"] = np.where(df["looks_combo"], "Others:Combination",
                    np.where(df["generic_id"].isna(), "Others:Brand/NoGeneric", "Candidate"))

    df_cand = df.loc[df["bucket"].eq("Candidate")].merge(
        pnf_df, on="generic_id", how="left"
    )

    def route_ok(row):
        r = row["route"]; allowed = row.get("route_allowed")
        if pd.isna(r) or not r:
            return True
        if isinstance(allowed, str) and allowed:
            return r in allowed.split("|")
        return True

    df_cand = df_cand[df_cand.apply(route_ok, axis=1)]

    df_cand["dose_sim"] = df_cand.apply(lambda r: dose_similarity(r["dosage_parsed"], r), axis=1)
    df_cand["dose_sim"] = pd.to_numeric(df_cand["dose_sim"], errors="coerce").fillna(0.0)

    def pick_best(group: pd.DataFrame):
        if group.empty:
            return pd.Series({"atc_code_final": None, "atc_note":"Needs review: no route/dose match",
                              "selected_form": None, "selected_variant": None, "dose_sim": 0.0})
        g = group.sort_values(["dose_sim"], ascending=False)
        top = g.iloc[0]
        note = "OK" if float(top["dose_sim"]) >= 0.6 else "Needs review: weak dose match"
        strength = top.get("strength"); unit = top.get("unit") or ""
        per_val = top.get("per_val"); per_unit = top.get("per_unit") or ""
        pct = top.get("pct")
        variant = f'{top.get("dose_kind")}:{strength}{unit}' if pd.notna(strength) else str(top.get("dose_kind"))
        if pd.notna(per_val):
            try:
                pv_int = int(per_val)
            except Exception:
                pv_int = per_val
            variant += f'/{pv_int}{per_unit}'
        if pd.notna(pct):
            variant += f' {pct}%'
        return pd.Series({
            "atc_code_final": top["atc_code"] if isinstance(top["atc_code"], str) and top["atc_code"] else None,
            "atc_note": note,
            "selected_form": top.get("form_token"),
            "selected_variant": variant,
            "dose_sim": float(top.get("dose_sim", 0.0)),
        })

    best_by_idx = df_cand.groupby(level=0).apply(pick_best)
    df_candidates_resolved = df.loc[df["bucket"].eq("Candidate")].join(best_by_idx, how="left")

    def score_row(r):
        score = 0
        if pd.notna(r.get("generic_id")): score += 60
        if r.get("dosage_parsed"): score += 15
        if r.get("route_evidence"): score += 10
        if pd.notna(r.get("atc_code_final")) and r.get("atc_note") == "OK": score += 15
        sim = r.get("dose_sim")
        try:
            sim = float(sim)
        except Exception:
            sim = 0.0
        if math.isnan(sim):
            sim = 0.0
        score += int(max(0.0, min(1.0, sim)) * 10)
        if str(r.get("atc_note","")).startswith("Needs review"): score -= 15
        return score

    df_candidates_resolved["confidence"] = df_candidates_resolved.apply(score_row, axis=1)

    out = df.merge(
        df_candidates_resolved[["atc_code_final","atc_note","selected_form","selected_variant","dose_sim","confidence"]],
        left_index=True, right_index=True, how="left"
    )

    between_mask = (out["confidence"] >= 70) & (out["confidence"] <= 89)
    out["bucket"] = np.where(out["bucket"].eq("Candidate") & (out["confidence"]>=90), "Auto-Accept",
                     np.where(out["bucket"].eq("Candidate") & between_mask, "Needs review", out["bucket"]))

    out["normalized"] = out["norm"]
    out["why_flagged"] = out["atc_note"].fillna("")

    keep_cols = [
        "raw_text","normalized",
        "generic_id","molecule_token",
        "dosage_parsed","route","form",
        "selected_form","selected_variant","dose_sim",
        "atc_code_final","confidence","bucket","why_flagged"
    ]
    out = out[keep_cols]
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[match] wrote {out_csv} with {len(out):,} rows")
    return out_csv


# =========================
# RUN ALL
# =========================
def run_all(pnf_csv: str, esoa_csv: str, outdir: str = ".", out_csv: str = "esoa_matched.csv") -> str:
    pnf_prepared, esoa_prepared = prepare(pnf_csv, esoa_csv, outdir)
    return match(pnf_prepared, esoa_prepared, out_csv)


# =========================
# CLI
# =========================
def _cli():
    ap = argparse.ArgumentParser(description="Dose-aware drug matching pipeline")
    ap.add_argument("--pnf", required=False, default="pnf.csv")
    ap.add_argument("--esoa", required=False, default="esoa.csv")
    ap.add_argument("--outdir", required=False, default=".")
    ap.add_argument("--out", required=False, default="esoa_matched.csv")
    args = ap.parse_args()
    run_all(args.pnf, args.esoa, args.outdir, args.out)

if __name__ == "__main__":
    _cli()
