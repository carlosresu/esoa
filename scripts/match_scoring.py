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
        return pd.Series({"atc_code_final": None, "match_note": "no route/dose match", "selected_form": None, "selected_variant": None, "dose_sim": 0.0})
    esoa_form = group.iloc[0]["form"]; esoa_route = group.iloc[0]["route"]; esoa_dose = group.iloc[0]["dosage_parsed"]
    scored = []
    for _, row in group.iterrows():
        score = 0.0
        if esoa_form and row.get("form_token") and esoa_form == row["form_token"]: score += 40
        if esoa_route and row.get("route_allowed") and esoa_route == row["route_allowed"]: score += 30
        sim = dose_similarity(esoa_dose, row); score += sim * 30
        scored.append((score, sim, row))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, best_sim, best_row = scored[0]
    note = "weak dose match" if best_sim >= 0.6 else "no/poor dose match"
    strength = best_row.get("strength"); unit = best_row.get("unit") or ""
    per_val = best_row.get("per_val"); per_unit = best_row.get("per_unit") or ""; pct = best_row.get("pct")
    variant = f"{best_row.get('dose_kind')}:{strength}{unit}" if pd.notna(strength) else str(best_row.get("dose_kind"))
    if pd.notna(per_val):
        try: pv_int = int(per_val)
        except Exception: pv_int = per_val
        variant += f"/{pv_int}{per_unit}"
    if pd.notna(pct): variant += f" {pct}%"
    return pd.Series({"atc_code_final": best_row["atc_code"] if isinstance(best_row["atc_code"], str) and best_row["atc_code"] else None, "match_note": note, "selected_form": best_row.get("form_token"), "selected_variant": variant, "dose_sim": float(best_sim)})

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

def score_and_classify(features_df: pd.DataFrame, pnf_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()
    df_cand = df.loc[df["bucket"].eq("Candidate"), ["esoa_idx","generic_id","route","form","dosage_parsed"]].merge(pnf_df, on="generic_id", how="left")
    if not df_cand.empty:
        df_cand = df_cand[df_cand.apply(_route_ok, axis=1)]
        df_cand["dose_sim"] = df_cand.apply(lambda r: dose_similarity(r["dosage_parsed"], r), axis=1)
        df_cand["dose_sim"] = pd.to_numeric(df_cand["dose_sim"], errors="coerce").fillna(0.0)
        best_by_idx = df_cand.groupby("esoa_idx", sort=False).apply(_pick_best, include_groups=False)
        out = df.merge(best_by_idx, left_on="esoa_idx", right_index=True, how="left")
    else:
        out = df.copy()
        out[["atc_code_final","match_note","selected_form","selected_variant","dose_sim"]] = [None, "no route/dose match", None, None, 0.0]
    out["confidence"] = out.apply(_score_row, axis=1)
    between_mask = (out["confidence"] >= 70) & (out["confidence"] <= 89)
    out["bucket"] = np.where(out["bucket"].eq("Candidate") & (out["confidence"] >= 90), "Auto-Accept", np.where(out["bucket"].eq("Candidate") & between_mask, "Candidate", out["bucket"]))
    pnf_basenorm_set = set(pnf_df["generic_name"].dropna().map(_base_name).map(_normalize_text_basic))
    out["present_in_pnf"] = out["pnf_hits_count"].astype(int).gt(0)
    out["present_in_who"] = out["who_atc_codes"].astype(str).str.len().gt(0)
    out["molecules_recognized_list"] = out.apply(_union_molecules, axis=1)
    out["molecules_recognized"] = out["molecules_recognized_list"].map(lambda xs: "|".join(xs) if xs else "")
    out["molecules_recognized_count"] = out["molecules_recognized_list"].map(lambda xs: len(xs or []))
    out["who_atc_count"] = out["who_atc_codes_list"].map(lambda xs: len(xs or []))
    out["probable_atc"] = np.where(~out["present_in_pnf"] & out["present_in_who"], out["who_atc_codes"], "")
    def classify_combo_reason(row: pd.Series) -> str:
        if row.get("combo_segments_total", 0) > 1:
            return (f"Combinations: Contains Unknown(s) — identified={row['combo_id_molecules']}, unknown_in_pnf={row['combo_unknown_in_pnf']}, unknown_in_who={row['combo_unknown_in_who']}")
        return "Combinations"
    final_bucket = out["bucket"].copy()
    final_why = pd.Series([""] * len(out), index=out.index)
    others_mask = final_bucket.eq("Others:Combinations")
    if others_mask.any():
        final_bucket.loc[others_mask] = "Others"
        final_why.loc[others_mask] = out.loc[others_mask].apply(classify_combo_reason, axis=1)
    cand_mask = final_bucket.eq("Candidate")
    final_bucket.loc[cand_mask] = "Needs review"
    cand_reason = out.loc[cand_mask, "match_note"].fillna("unspecified")
    final_why.loc[cand_mask] = "Candidate:" + cand_reason
    brand_mask = final_bucket.eq("BrandOnly/NoGeneric")
    final_bucket.loc[brand_mask] = "Needs review"
    final_why.loc[brand_mask] = "BrandOnly/NoGenericInPNF/NoGenericInWHO"
    any_who_codes = out["who_atc_codes"].astype(str).str.len().gt(0)
    any_not_in_pnf = out["who_molecules_list"].map(lambda names: any((_normalize_text_basic(_base_name(n)) not in pnf_basenorm_set) for n in (names or [])))
    vm_mask = any_who_codes & any_not_in_pnf
    if vm_mask.any():
        subreason = np.where(vm_mask & (out["pnf_hits_count"] >= 1) & out["looks_combo_final"], "2+ generics but only 1+ in PNF", "not in PNF")
        final_bucket.loc[vm_mask] = "Needs review"
        sr = pd.Series(subreason, index=out.index)[vm_mask]
        final_why.loc[vm_mask] = "ValidMoleculeWithATC/NotInPNF:" + sr
    out["bucket_final"] = final_bucket
    out["why_final"] = final_why.fillna("")
    return out
