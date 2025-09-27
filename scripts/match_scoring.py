#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import List
import numpy as np, pandas as pd
from .dose import dose_similarity
from .text_utils import _base_name, _normalize_text_basic

def _route_ok(row: pd.Series) -> bool:
    r = row["route"]; allowed = row.get("route_allowed")
    if pd.isna(r) or not r: return True
    if isinstance(allowed, str) and allowed: return r in allowed.split("|")
    return True

def _pick_best(group: pd.DataFrame) -> pd.Series:
    if group.empty:
        return pd.Series({
            "atc_code_final": None, "dose_sim": 0.0, "form_ok": False, "route_ok": False,
            "match_quality": "unspecified", "selected_form": None, "selected_variant": None
        })
    esoa_form = group.iloc[0]["form"]; esoa_route = group.iloc[0]["route"]; esoa_dose = group.iloc[0]["dosage_parsed"]
    scored = []
    for _, row in group.iterrows():
        score = 0.0
        form_ok = bool(esoa_form and row.get("form_token") and esoa_form == row["form_token"])
        route_ok = bool(esoa_route and row.get("route_allowed") and esoa_route == row["route_allowed"])
        if form_ok: score += 40
        if route_ok: score += 30
        sim = dose_similarity(esoa_dose, row); score += sim * 30
        scored.append((score, sim, form_ok, route_ok, row))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, best_sim, best_form_ok, best_route_ok, best_row = scored[0]

    note = "OK"
    if best_sim < 1.0:
        note = "dose mismatch"
    if not best_form_ok and (note == "OK"):
        note = "no/poor form match"
    if not best_route_ok and (note == "OK"):
        note = "no/poor route match"

    strength = best_row.get("strength"); unit = best_row.get("unit") or ""
    per_val = best_row.get("per_val"); per_unit = best_row.get("per_unit") or ""; pct = best_row.get("pct")
    variant = f"{best_row.get('dose_kind')}:{strength}{unit}" if pd.notna(strength) else str(best_row.get("dose_kind"))
    if pd.notna(per_val):
        try: pv_int = int(per_val)
        except Exception: pv_int = per_val
        variant += f"/{pv_int}{per_unit}"
    if pd.notna(pct): variant += f" {pct}%"

    return pd.Series({
        "atc_code_final": best_row["atc_code"] if isinstance(best_row["atc_code"], str) and best_row["atc_code"] else None,
        "dose_sim": float(best_sim),
        "form_ok": bool(best_form_ok),
        "route_ok": bool(best_route_ok),
        "match_quality": note,
        "selected_form": best_row.get("form_token"),
        "selected_variant": variant,
    })

def _score_row(r: pd.Series) -> int:
    score = 0
    if pd.notna(r.get("generic_id")): score += 60
    if r.get("dosage_parsed"): score += 15
    if r.get("route_evidence"): score += 10
    if pd.notna(r.get("atc_code_final")): score += 15
    sim = r.get("dose_sim")
    try: sim = float(sim)
    except Exception: sim = 0.0
    if math.isnan(sim): sim = 0.0
    score += int(max(0.0, min(1.0, sim)) * 10)
    try:
        if r.get("did_brand_swap") and r.get("form_ok") and r.get("route_ok") and float(r.get("dose_sim", 0)) >= 1.0:
            score += 10
    except Exception:
        pass
    return score

def _union_molecules(row: pd.Series) -> List[str]:
    names = []
    for t in (row.get("pnf_hits_tokens") or []):
        if not isinstance(t, str): continue
        names.append(_normalize_text_basic(_base_name(t)))
    for t in (row.get("who_molecules_list") or []):
        if not isinstance(t, str): continue
        names.append(_normalize_text_basic(_base_name(t)))
    seen = set(); uniq = []
    for n in names:
        if not n or n in seen: continue
        seen.add(n); uniq.append(n)
    return uniq

def _mk_reason(series: pd.Series, default_ok: str) -> pd.Series:
    s = series.astype("string")
    s = s.fillna(default_ok)
    s = s.replace({"": default_ok, "unspecified": default_ok})
    return s.astype("string")

def _missing_combo(row: pd.Series) -> str:
    missing = []
    if not bool(row.get("dosage_parsed")):
        missing.append("dose")
    if not bool(row.get("form")):
        missing.append("form")
    if not bool(row.get("route")):
        missing.append("route")
    if not missing:
        return ""
    if len(missing) == 1:
        return f"no {missing[0]} available"
    if len(missing) == 2:
        return f"no {missing[0]} and {missing[1]} available"
    return "no dose, form, and route available"

def score_and_classify(features_df: pd.DataFrame, pnf_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()

    # Candidate rows that have PNF generic_id
    df_cand = df.loc[df["generic_id"].notna(), ["esoa_idx","generic_id","route","form","dosage_parsed"]].merge(pnf_df, on="generic_id", how="left")
    if not df_cand.empty:
        df_cand = df_cand[df_cand.apply(_route_ok, axis=1)]
        df_cand["dose_sim"] = df_cand.apply(lambda r: dose_similarity(r["dosage_parsed"], r), axis=1)
        df_cand["dose_sim"] = pd.to_numeric(df_cand["dose_sim"], errors="coerce").fillna(0.0)
        best_by_idx = df_cand.groupby("esoa_idx", sort=False).apply(_pick_best, include_groups=False)
        out = df.merge(best_by_idx, left_on="esoa_idx", right_index=True, how="left")
    else:
        out = df.copy()
        out[["atc_code_final","dose_sim","form_ok","route_ok","match_quality","selected_form","selected_variant"]] = [None, 0.0, False, False, "unspecified", None, None]

    # Confidence and molecules recognized
    out["confidence"] = out.apply(_score_row, axis=1)
    out["molecules_recognized_list"] = out.apply(_union_molecules, axis=1)
    out["molecules_recognized"] = out["molecules_recognized_list"].map(lambda xs: "|".join(xs) if xs else "")
    out["molecules_recognized_count"] = out["molecules_recognized_list"].map(lambda xs: len(xs or []))
    out["who_atc_count"] = out["who_atc_codes_list"].map(lambda xs: len(xs or []))

    # Probable ATC if absent in PNF but present in WHO
    out["probable_atc"] = np.where(~out["present_in_pnf"] & out["present_in_who"], out["who_atc_codes"], "")

    # Initialize tags
    out["bucket_final"] = ""
    out["why_final"] = ""
    out["reason_final"] = ""
    out["match_quality"] = ""
    out["match_molecule(s)"] = ""  # precise naming

    # Derive *match_molecule(s)* tags
    present_in_pnf = out["present_in_pnf"].astype(bool)
    present_in_who = out["present_in_who"].astype(bool)
    present_in_fda = out["present_in_fda_generic"].astype(bool)
    has_atc_in_pnf = out["atc_code_final"].astype(str).str.len().gt(0)

    out.loc[present_in_pnf & has_atc_in_pnf, "match_molecule(s)"] = "ValidMoleculeWithATCinPNF"
    out.loc[(~present_in_pnf) & present_in_who, "match_molecule(s)"] = "ValidMoleculeWithATCinWHO/NotInPNF"
    out.loc[(~present_in_pnf) & (~present_in_who) & present_in_fda, "match_molecule(s)"] = "ValidMoleculeNoATCinFDA/NotInPNF"

    # Brand-swapped variants
    out.loc[out["did_brand_swap"].astype(bool) & present_in_who, "match_molecule(s)"] = "ValidBrandSwappedForMoleculeWithATCinWHO"
    out.loc[out["did_brand_swap"].astype(bool) & present_in_pnf & has_atc_in_pnf, "match_molecule(s)"] = "ValidBrandSwappedForGenericInPNF"

    # AUTO-ACCEPT
    is_candidate_like = out["generic_id"].notna()
    form_ok_col = out["form_ok"].astype(bool)
    route_ok_col = out["route_ok"].astype(bool)
    auto_mask = is_candidate_like & present_in_pnf & has_atc_in_pnf & form_ok_col & route_ok_col
    out.loc[auto_mask, "bucket_final"] = "Auto-Accept"

    # Needs review
    needs_rev_mask = (out["bucket_final"] == "") & (is_candidate_like | present_in_who | present_in_fda)
    out.loc[needs_rev_mask, "bucket_final"] = "Needs review"
    out.loc[needs_rev_mask, "why_final"] = "Needs review"

    # Match quality for Needs review
    missing_strings = out.apply(lambda r: _missing_combo(r), axis=1).astype("string")
    dose_mismatch_general = (out["dose_sim"].astype(float) < 1.0) & out["dosage_parsed"].astype(bool)
    out.loc[needs_rev_mask & dose_mismatch_general, "match_quality"] = "dose mismatch"
    out.loc[needs_rev_mask & (~dose_mismatch_general) & (missing_strings.str.len() > 0), "match_quality"] = missing_strings
    out.loc[needs_rev_mask & (out["match_quality"] == ""), "match_quality"] = "unspecified"

    # Keep reason_final populated
    out.loc[needs_rev_mask, "reason_final"] = out.loc[needs_rev_mask, "match_quality"]
    out["reason_final"] = _mk_reason(out["reason_final"], "unspecified")

    # Others: Unknowns
    unknown_single = out["unknown_kind"].eq("Single - Unknown")
    unknown_multi_all = out["unknown_kind"].eq("Multiple - All Unknown")
    unknown_multi_some = out["unknown_kind"].eq("Multiple - Some Unknown")
    none_found = out["unknown_kind"].eq("None")
    fallback_unknown = (~is_candidate_like) & (~present_in_who) & (~present_in_fda) & (~none_found)

    def _annotate_unknown(s: str) -> str:
        if s == "Single - Unknown": return "Single - Unknown (unknown to PNF, WHO, FDA)"
        if s == "Multiple - All Unknown": return "Multiple - All Unknown (unknown to PNF, WHO, FDA)"
        if s == "Multiple - Some Unknown": return "Multiple - Some Unknown (some unknown to PNF, WHO, FDA)"
        return s

    for cond, reason in [
        (unknown_single | (fallback_unknown & unknown_single), "Single - Unknown"),
        (unknown_multi_all | (fallback_unknown & unknown_multi_all), "Multiple - All Unknown"),
        (unknown_multi_some | (fallback_unknown & unknown_multi_some), "Multiple - Some Unknown"),
    ]:
        mask = cond & (out["bucket_final"] == "")
        out.loc[mask, "bucket_final"] = "Others"
        out.loc[mask, "why_final"] = "Unknown"
        out.loc[mask, "reason_final"] = _annotate_unknown(reason)

    # Final safety net
    remaining = out["bucket_final"].eq("")
    out.loc[remaining, "bucket_final"] = "Needs review"
    out.loc[remaining, "why_final"] = "Needs review"
    out.loc[remaining, "reason_final"] = _mk_reason(out.loc[remaining, "match_quality"], "unspecified")

    # Dose recognized: N/A unless exact
    if "dose_recognized" in out.columns:
        out["dose_recognized"] = np.where(out["dose_sim"].astype(float) == 1.0, out["dose_recognized"], "N/A")

    return out
