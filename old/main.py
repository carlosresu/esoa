#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runs the matcher with dose-aware picking:
- Reads:  pnf_prepared.csv (dose/form-aware), esoa_prepared.csv
- Writes: esoa_matched.csv
Usage:
  python main.py --pnf pnf_prepared.csv --esoa esoa_prepared.csv --out esoa_matched.csv
"""

import argparse
import pandas as pd
import numpy as np
import re
import unicodedata
import math

import ahocorasick
# from rapidfuzz import fuzz

# ---------- Normalization ----------
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

# ---------- Routes & forms ----------
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

# ---------- Dosage parsing for ESOA ----------
DOSAGE_PATTERNS = [
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\b",
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>1)?\s*(?P<per_unit>ml)\b",
    r"(?P<strength>\d+(?:[\.,]\d+)?)\s?(?P<unit>mg|g|mcg|ug|iu)\s?(?:/| per )\s?(?P<per_val>\d+(?:[\.,]\d+)?)\s*(?P<per_unit>ml)\b",
    r"(?P<pct>\d+(?:[\.,]\d+)?)\s?%",
]
DOSAGE_REGEXES = [re.compile(p, flags=re.I) for p in DOSAGE_PATTERNS]

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

# ---------- Aho-Corasick ----------
def build_molecule_automaton(pnf_df: pd.DataFrame) -> ahocorasick.Automaton:
    A = ahocorasick.Automaton()
    seen = set()
    for gid, gname in pnf_df[["generic_id","generic_name"]].drop_duplicates().itertuples(index=False):
        key = normalize_text(gname)
        if key and (gid, key) not in seen:
            A.add_word(key, (gid, key))
            seen.add((gid, key))
    if "synonyms" in pnf_df.columns:
        for gid, syns in pnf_df[["generic_id","synonyms"]].itertuples(index=False):
            if isinstance(syns, str) and syns.strip():
                for s in syns.split("|"):
                    key = normalize_text(s)
                    if key and (gid, key) not in seen:
                        A.add_word(key, (gid, key))
                        seen.add((gid, key))
    A.make_automaton()
    return A

# ---------- Unit conversion ----------
def to_mg(value: float, unit: str):
    u = unit.lower()
    if u == "mg": return value
    if u == "g": return value * 1000.0
    if u in ("mcg","ug"): return value / 1000.0
    return None  # IU unsupported generically

# ---------- Dose compatibility scoring ----------
def dose_similarity(esoa_dose: dict, pnf_row: pd.Series) -> float:
    if not esoa_dose:
        return 0.0

    kind = esoa_dose.get("kind")
    if kind == "amount":
        mg_esoa = to_mg(esoa_dose["strength"], esoa_dose["unit"])
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
        v_esoa = to_mg(esoa_dose["strength"], esoa_dose["unit"])
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
    if len(molecule_hits) > 1:
        return True
    if re.search(r"\bwith\b|\+|/", s_norm):
        return True
    return False

# ---------- Main pipeline ----------
def match_rows(df_esoa: pd.DataFrame, pnf_df: pd.DataFrame) -> pd.DataFrame:
    A = build_molecule_automaton(pnf_df)

    df = df_esoa.copy()
    df["norm"] = df["raw_text"].map(normalize_text)

    # AC scan
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
        generic_id.append(hits[0][2])
        generic_name_hit.append(hits[0][3])
        multi_hit.append(len(distinct_gids) > 1)

    df["generic_id"] = generic_id
    df["molecule_token"] = generic_name_hit
    df["multi_hit"] = multi_hit

    # Parse dose/route/form in ESOA row
    df["dosage_parsed"] = df["norm"].map(extract_dosage)
    df["route"], df["form"], df["route_evidence"] = zip(*df["norm"].map(extract_route_and_form))
    df["looks_combo"] = [looks_like_combination(s, {g} if g else set())
                         for s, g in zip(df["norm"], df["generic_id"])]

    # Initial bucket
    df["bucket"] = np.where(df["looks_combo"], "Others:Combination",
                    np.where(df["generic_id"].isna(), "Others:Brand/NoGeneric", "Candidate"))

    # Candidate subset join (retain dose/form variants!)
    df_cand = df.loc[df["bucket"].eq("Candidate")].merge(
        pnf_df, on="generic_id", how="left", suffixes=("","")
    )

    # Route screen (if ESOA route present, require it be allowed)
    def route_ok(row):
        r = row["route"]
        allowed = row.get("route_allowed")
        if pd.isna(r) or not r:
            return True
        if isinstance(allowed, str) and allowed:
            return r in allowed.split("|")
        return True

    df_cand = df_cand[df_cand.apply(route_ok, axis=1)]

    # Dose similarity & selection per ESOA row
    df_cand["dose_sim"] = df_cand.apply(lambda r: dose_similarity(r["dosage_parsed"], r), axis=1)
    # Ensure numeric & non-NaN
    df_cand["dose_sim"] = pd.to_numeric(df_cand["dose_sim"], errors="coerce").fillna(0.0)

    # Pick best candidate per ESOA row
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
            variant += f'/{int(per_val)}{per_unit}'
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

    # Confidence (robust to NaN)
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
        score += int(max(0.0, min(1.0, sim)) * 10)  # clamp to [0,10]
        if str(r.get("atc_note","")).startswith("Needs review"): score -= 15
        return score

    df_candidates_resolved["confidence"] = df_candidates_resolved.apply(score_row, axis=1)

    # Merge back
    out = df.merge(
        df_candidates_resolved[["atc_code_final","atc_note","selected_form","selected_variant","dose_sim","confidence"]],
        left_index=True, right_index=True, how="left"
    )

    # Final bucket
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
    return out[keep_cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pnf", default="pnf_prepared.csv")
    ap.add_argument("--esoa", default="esoa_prepared.csv")
    ap.add_argument("--out",  default="esoa_matched.csv")
    args = ap.parse_args()

    pnf_df = pd.read_csv(args.pnf)
    esoa_df = pd.read_csv(args.esoa)

    required_pnf = {"generic_id","generic_name","synonyms","atc_code","route_allowed",
                    "form_token","dose_kind","strength","unit","per_val","per_unit","pct",
                    "strength_mg","ratio_mg_per_ml"}
    missing = required_pnf - set(pnf_df.columns)
    if missing:
        raise ValueError(f"{args.pnf} missing columns: {missing}")
    if "raw_text" not in esoa_df.columns:
        raise ValueError(f"{args.esoa} must contain a 'raw_text' column")

    out = match_rows(esoa_df[["raw_text"]], pnf_df)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {args.out} with {len(out):,} rows")

if __name__ == "__main__":
    main()
