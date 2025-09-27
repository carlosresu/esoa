#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from .dose import dose_similarity
from .text_utils import _base_name, _normalize_text_basic


def _mk_reason(series: pd.Series, default_ok: str) -> pd.Series:
    s = series.astype("string")
    s = s.fillna(default_ok)
    s = s.replace({"": default_ok, "unspecified": default_ok})
    return s.astype("string")


def _annotate_unknown(s: str) -> str:
    if s == "Single - Unknown":
        return "Single - Unknown (unknown to PNF, WHO, FDA)"
    if s == "Multiple - All Unknown":
        return "Multiple - All Unknown (unknown to PNF, WHO, FDA)"
    if s == "Multiple - Some Unknown":
        return "Multiple - Some Unknown (some unknown to PNF, WHO, FDA)"
    return s


def score_and_classify(features_df: pd.DataFrame, pnf_df: pd.DataFrame) -> pd.DataFrame:
    df = features_df.copy()

    df_cand = df.loc[df["generic_id"].notna(), ["esoa_idx", "generic_id", "route", "form", "dosage_parsed"]].merge(
        pnf_df, on="generic_id", how="left"
    )
    best_by_idx = pd.DataFrame()

    if not df_cand.empty:
        route_ok_mask: list[bool] = []
        route_exact_flags: list[bool] = []
        allowed_cache: dict[str, tuple[str, ...]] = {}
        for route_val, allowed_val in zip(df_cand["route"], df_cand["route_allowed"]):
            if pd.isna(route_val) or not route_val:
                route_ok_mask.append(True)
                route_exact_flags.append(False)
                continue
            if isinstance(allowed_val, str) and allowed_val:
                cached = allowed_cache.get(allowed_val)
                if cached is None:
                    cached = tuple(part.strip() for part in allowed_val.split("|") if part.strip())
                    allowed_cache[allowed_val] = cached
                route_ok = route_val in cached if cached else False
                route_ok_mask.append(route_ok)
                route_exact_flags.append(route_val == allowed_val)
            else:
                route_ok_mask.append(True)
                route_exact_flags.append(False)
        route_ok_series = pd.Series(route_ok_mask, index=df_cand.index)
        route_exact_series = pd.Series(route_exact_flags, index=df_cand.index)
        df_cand = df_cand.loc[route_ok_series].copy()

        if not df_cand.empty:
            route_exact_series = route_exact_series.loc[df_cand.index]
            form_series = df_cand["form"].fillna("").astype(str)
            form_token_series = df_cand["form_token"].fillna("").astype(str)
            form_ok_series = (form_series != "") & (form_token_series != "") & (form_series == form_token_series)

            pnf_records = df_cand[
                [
                    "dose_kind",
                    "strength_mg",
                    "ratio_mg_per_ml",
                    "pct",
                    "per_val",
                    "per_unit",
                    "strength",
                    "unit",
                ]
            ].to_dict("records")
            dose_sims = [
                dose_similarity(esoa_dose, pnf_row)
                for esoa_dose, pnf_row in zip(df_cand["dosage_parsed"], pnf_records)
            ]
            df_cand["dose_sim"] = pd.to_numeric(pd.Series(dose_sims), errors="coerce").fillna(0.0)

            df_cand["_form_ok"] = form_ok_series.to_numpy(dtype=bool)
            df_cand["_route_ok"] = route_exact_series.to_numpy(dtype=bool)
            df_cand["_score"] = (
                df_cand["_form_ok"].astype(int) * 40
                + df_cand["_route_ok"].astype(int) * 30
                + df_cand["dose_sim"].astype(float) * 30.0
            )

            best_rows = (
                df_cand.sort_values(["esoa_idx", "_score", "dose_sim"], ascending=[True, False, False])
                .drop_duplicates(subset=["esoa_idx"], keep="first")
                .set_index("esoa_idx")
            )

            if not best_rows.empty:
                match_quality = np.where(best_rows["dose_sim"] < 1.0, "dose mismatch", "OK")
                match_quality = np.where((match_quality == "OK") & (~best_rows["_form_ok"]), "no/poor form match", match_quality)
                match_quality = np.where((match_quality == "OK") & (~best_rows["_route_ok"]), "no/poor route match", match_quality)

                variants: list[str] = []
                for dose_kind, strength, unit, per_val, per_unit, pct in zip(
                    best_rows["dose_kind"],
                    best_rows["strength"],
                    best_rows["unit"],
                    best_rows["per_val"],
                    best_rows["per_unit"],
                    best_rows["pct"],
                ):
                    unit_val = unit if isinstance(unit, str) else ""
                    variant = str(dose_kind)
                    if pd.notna(strength):
                        variant = f"{dose_kind}:{strength}{unit_val}"
                    if pd.notna(per_val):
                        pv_display = per_val
                        try:
                            pv_display = int(per_val)
                        except Exception:
                            try:
                                pv_float = float(per_val)
                                if float(pv_float).is_integer():
                                    pv_display = int(pv_float)
                            except Exception:
                                pv_display = per_val
                        per_unit_val = per_unit if isinstance(per_unit, str) else ""
                        variant += f"/{pv_display}{per_unit_val}"
                    if pd.notna(pct):
                        variant += f" {pct}%"
                    variants.append(variant)

                atc_values = [
                    val if isinstance(val, str) and val else None
                    for val in best_rows["atc_code"]
                ]

                best_by_idx = pd.DataFrame(
                    {
                        "atc_code_final": atc_values,
                        "dose_sim": best_rows["dose_sim"].astype(float),
                        "form_ok": best_rows["_form_ok"].astype(bool),
                        "route_ok": best_rows["_route_ok"].astype(bool),
                        "match_quality": pd.Series(match_quality, index=best_rows.index),
                        "selected_form": best_rows["form_token"],
                        "selected_variant": variants,
                    },
                    index=best_rows.index,
                )

    if best_by_idx.empty:
        out = df.copy()
        out["atc_code_final"] = None
        out["dose_sim"] = 0.0
        out["form_ok"] = False
        out["route_ok"] = False
        out["match_quality"] = "unspecified"
        out["selected_form"] = None
        out["selected_variant"] = None
    else:
        out = df.merge(best_by_idx, left_on="esoa_idx", right_index=True, how="left")
        out["atc_code_final"] = out["atc_code_final"].where(out["atc_code_final"].notna(), None)
        out["dose_sim"] = out["dose_sim"].fillna(0.0)
        # Avoid FutureWarning on object-dtype downcasting by using pandas BooleanDtype
        out["form_ok"] = out["form_ok"].astype("boolean").fillna(False).astype(bool)
        out["route_ok"] = out["route_ok"].astype("boolean").fillna(False).astype(bool)
        out["match_quality"] = out["match_quality"].fillna("unspecified")
        out["selected_form"] = out["selected_form"].where(out["selected_form"].notna(), None)
        out["selected_variant"] = out["selected_variant"].where(out["selected_variant"].notna(), None)

    out["form_ok"] = out["form_ok"].astype(bool)
    out["route_ok"] = out["route_ok"].astype(bool)

    dose_present = pd.Series([bool(x) for x in out["dosage_parsed"]], index=out.index)
    form_present = pd.Series([bool(x) for x in out["form"]], index=out.index)
    route_present = pd.Series([bool(x) for x in out["route"]], index=out.index)
    route_evidence_present = pd.Series([bool(x) for x in out["route_evidence"]], index=out.index)
    generic_present = out["generic_id"].notna()
    atc_present = pd.Series([isinstance(x, str) and len(x) > 0 for x in out["atc_code_final"]], index=out.index)
    dose_sim_clipped = out["dose_sim"].fillna(0.0).astype(float).clip(0.0, 1.0)

    score_series = (
        generic_present.astype(int) * 60
        + dose_present.astype(int) * 15
        + route_evidence_present.astype(int) * 10
        + atc_present.astype(int) * 15
        + (dose_sim_clipped * 10).astype(int)
    )
    bonus_mask = (
        out["did_brand_swap"].astype(bool)
        & out["form_ok"]
        & out["route_ok"]
        & (out["dose_sim"].fillna(0.0).astype(float) >= 1.0)
    )
    score_series += bonus_mask.astype(int) * 10
    out["confidence"] = score_series.astype(int)

    union_lists: list[list[str]] = []
    for pnf_tokens, who_tokens in zip(out["pnf_hits_tokens"], out["who_molecules_list"]):
        names: list[str] = []
        if isinstance(pnf_tokens, list):
            for tok in pnf_tokens:
                if isinstance(tok, str):
                    names.append(_normalize_text_basic(_base_name(tok)))
        if isinstance(who_tokens, list):
            for tok in who_tokens:
                if isinstance(tok, str):
                    names.append(_normalize_text_basic(_base_name(tok)))
        seen: set[str] = set()
        uniq: list[str] = []
        for name in names:
            if name and name not in seen:
                seen.add(name)
                uniq.append(name)
        union_lists.append(uniq)
    out["molecules_recognized_list"] = union_lists
    out["molecules_recognized"] = ["|".join(xs) if xs else "" for xs in union_lists]
    out["molecules_recognized_count"] = [len(xs) for xs in union_lists]
    out["who_atc_count"] = [len(xs) if isinstance(xs, list) else 0 for xs in out["who_atc_codes_list"]]

    out["probable_atc"] = np.where(~out["present_in_pnf"] & out["present_in_who"], out["who_atc_codes"], "")

    out["bucket_final"] = ""
    out["why_final"] = ""
    out["reason_final"] = ""
    out["match_quality"] = ""
    out["match_molecule(s)"] = ""

    missing_strings: list[str] = []
    for has_dose, has_form, has_route in zip(dose_present, form_present, route_present):
        missing: list[str] = []
        if not has_dose:
            missing.append("dose")
        if not has_form:
            missing.append("form")
        if not has_route:
            missing.append("route")
        if not missing:
            missing_strings.append("")
        elif len(missing) == 1:
            missing_strings.append(f"no {missing[0]} available")
        elif len(missing) == 2:
            missing_strings.append(f"no {missing[0]} and {missing[1]} available")
        else:
            missing_strings.append("no dose, form, and route available")
    missing_series = pd.Series(missing_strings, index=out.index, dtype="string")

    present_in_pnf = out["present_in_pnf"].astype(bool)
    present_in_who = out["present_in_who"].astype(bool)
    present_in_fda = out["present_in_fda_generic"].astype(bool)
    has_atc_in_pnf = atc_present

    out.loc[present_in_pnf & has_atc_in_pnf, "match_molecule(s)"] = "ValidMoleculeWithATCinPNF"
    out.loc[(~present_in_pnf) & present_in_who, "match_molecule(s)"] = "ValidMoleculeWithATCinWHO/NotInPNF"
    out.loc[(~present_in_pnf) & (~present_in_who) & present_in_fda, "match_molecule(s)"] = "ValidMoleculeNoATCinFDA/NotInPNF"

    out.loc[out["did_brand_swap"].astype(bool) & present_in_who, "match_molecule(s)"] = "ValidBrandSwappedForMoleculeWithATCinWHO"
    out.loc[out["did_brand_swap"].astype(bool) & present_in_pnf & has_atc_in_pnf, "match_molecule(s)"] = "ValidBrandSwappedForGenericInPNF"

    is_candidate_like = out["generic_id"].notna()
    auto_mask = is_candidate_like & present_in_pnf & has_atc_in_pnf & out["form_ok"] & out["route_ok"]
    out.loc[auto_mask, "bucket_final"] = "Auto-Accept"

    needs_rev_mask = (out["bucket_final"] == "") & (is_candidate_like | present_in_who | present_in_fda)
    out.loc[needs_rev_mask, "bucket_final"] = "Needs review"
    out.loc[needs_rev_mask, "why_final"] = "Needs review"

    dose_mismatch_general = (out["dose_sim"].astype(float) < 1.0) & dose_present
    out.loc[needs_rev_mask & dose_mismatch_general, "match_quality"] = "dose mismatch"
    out.loc[needs_rev_mask & (~dose_mismatch_general) & (missing_series.str.len() > 0), "match_quality"] = missing_series
    out.loc[needs_rev_mask & (out["match_quality"] == ""), "match_quality"] = "unspecified"

    out.loc[needs_rev_mask, "reason_final"] = out.loc[needs_rev_mask, "match_quality"]
    out["reason_final"] = _mk_reason(out["reason_final"], "unspecified")

    unknown_single = out["unknown_kind"].eq("Single - Unknown")
    unknown_multi_all = out["unknown_kind"].eq("Multiple - All Unknown")
    unknown_multi_some = out["unknown_kind"].eq("Multiple - Some Unknown")
    none_found = out["unknown_kind"].eq("None")
    fallback_unknown = (~is_candidate_like) & (~present_in_who) & (~present_in_fda) & (~none_found)

    for cond, reason in [
        (unknown_single | (fallback_unknown & unknown_single), "Single - Unknown"),
        (unknown_multi_all | (fallback_unknown & unknown_multi_all), "Multiple - All Unknown"),
        (unknown_multi_some | (fallback_unknown & unknown_multi_some), "Multiple - Some Unknown"),
    ]:
        mask = cond & (out["bucket_final"] == "")
        out.loc[mask, "bucket_final"] = "Others"
        out.loc[mask, "why_final"] = "Unknown"
        out.loc[mask, "reason_final"] = _annotate_unknown(reason)

    remaining = out["bucket_final"].eq("")
    out.loc[remaining, "bucket_final"] = "Needs review"
    out.loc[remaining, "why_final"] = "Needs review"
    out.loc[remaining, "reason_final"] = _mk_reason(out.loc[remaining, "match_quality"], "unspecified")

    if "dose_recognized" in out.columns:
        out["dose_recognized"] = np.where(out["dose_sim"].astype(float) == 1.0, out["dose_recognized"], "N/A")

    return out
