#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scoring and selection logic for the eSOA ↔ PNF matching pipeline."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

# Policy constants that determine when a detected route/form pairing is acceptable
# for auto-acceptance.  These mirror the definitions explained in README.md.
APPROVED_ROUTE_FORMS: dict[str, set[str]] = {
    "oral": {"tablet", "capsule", "sachet", "suspension", "solution", "syrup", "suppository"},
    "nasal": {"solution"},
    "inhalation": {"solution", "dpi", "mdi"},
    "intravenous": {"solution", "ampule", "vial", "injection", "suspension"},
    "intramuscular": {"solution", "ampule", "vial", "injection", "suspension"},
    "subcutaneous": {"solution", "ampule", "vial", "injection", "suspension"},
    "rectal": {"suppository", "solution"},
    "ophthalmic": {"solution", "suspension", "ointment", "gel"},
    "topical": {"solution", "ointment", "gel", "lotion", "cream", "sachet"},
    "vaginal": {"cream", "suppository", "tablet"},
    "otic": {"solution"},
    "sublingual": {"tablet"},
    "transdermal": {"patch"},
}

FLAGGED_ROUTE_FORM_EXCEPTIONS: set[tuple[str, str]] = {
    ("oral", "vial"),
    ("oral", "ampule"),
    ("intravenous", "syrup"),
    ("intramuscular", "syrup"),
    ("subcutaneous", "syrup"),
    ("intravenous", "tablet"),
    ("intramuscular", "tablet"),
    ("subcutaneous", "tablet"),
    ("ophthalmic", "injection"),
}

# Default reasons when metadata is insufficient to determine the precise mismatch
DEFAULT_METADATA_GAP_REASON = "review_required_metadata_insufficient"
WHO_METADATA_GAP_REASON = "who_metadata_insufficient_review_required"

# WHO ATC administration route codes mapped to canonical route tokens
# (WHO ATC/DDD Index – Adm.R definitions)
# WHO and ATC metadata is used as a fallback when PNF coverage is missing; the
# mappings below mirror the feature builder so scoring decisions stay aligned.
WHO_ADM_ROUTE_MAP: dict[str, set[str]] = {
    "o": {"oral"},
    "oral": {"oral"},
    "chewing gum": {"oral"},
    "p": {"intravenous", "intramuscular", "subcutaneous"},  # parenteral umbrella
    "r": {"rectal"},
    "v": {"vaginal"},
    "n": {"nasal"},
    "sl": {"sublingual"},
    "td": {"transdermal"},
    "inhal": {"inhalation"},
    "inhal.aerosol": {"inhalation"},
    "inhal.powder": {"inhalation"},
    "inhal.solution": {"inhalation"},
    "instill.solution": {"ophthalmic"},
    "implant": {"subcutaneous"},
    "s.c. implant": {"subcutaneous"},
    "intravesical": {"intravesical"},
    "lamella": {"ophthalmic"},
    "ointment": {"topical"},
    "oral aerosol": {"inhalation"},
    "urethral": {"urethral"},
}

from .dose import dose_similarity
from .text_utils import _base_name, _normalize_text_basic


def _mk_reason(series: pd.Series, default_ok: str) -> pd.Series:
    """Standardize reason columns by filling unspecified entries with a default value."""
    s = series.astype("string")
    s = s.fillna(default_ok)
    s = s.replace({"": default_ok, "no_specific_reason_provided": default_ok})
    return s.astype("string")


def _annotate_unknown(s: str) -> str:
    """Append clarifying text for buckets that merely state "Unknown"."""
    if s == "Single - Unknown":
        return "Single - Unknown (unknown to PNF, WHO, FDA)"
    if s == "Multiple - All Unknown":
        return "Multiple - All Unknown (unknown to PNF, WHO, FDA)"
    if s == "Multiple - Some Unknown":
        return "Multiple - Some Unknown (some unknown to PNF, WHO, FDA)"
    return s


def score_and_classify(features_df: pd.DataFrame, pnf_df: pd.DataFrame) -> pd.DataFrame:
    """Score features, select best PNF candidates, and prepare audit columns using the policy described in README (route/form whitelist, dose equality, confidence weights, Auto-Accept gates)."""
    df = features_df.copy()

    def _format_variant(
        dose_kind,
        strength,
        unit,
        per_val,
        per_unit,
        pct,
    ) -> str:
        variant = str(dose_kind)
        if pd.notna(strength):
            unit_val = unit if isinstance(unit, str) else ""
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
        return variant

    def _normalize_route(value: object) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip().lower()

    def _normalize_form_token(value: object) -> str:
        if not isinstance(value, str):
            return ""
        value = value.strip().lower()
        return {
            "tab": "tablet",
            "tabs": "tablet",
            "tablet": "tablet",
            "chewing gum": "tablet",
            "cap": "capsule",
            "caps": "capsule",
            "capsule": "capsule",
            "capsulee": "capsule",
            "susp": "suspension",
            "suspension": "suspension",
            "syr": "syrup",
            "syrup": "syrup",
            "sol": "solution",
            "soln": "solution",
            "solution": "solution",
            "inhal.solution": "solution",
            "instill.solution": "solution",
            "lamella": "solution",
            "ointment": "ointment",
            "oint": "ointment",
            "gel": "gel",
            "cream": "cream",
            "lotion": "lotion",
            "patch": "patch",
            "supp": "suppository",
            "suppository": "suppository",
            "dpi": "dpi",
            "inhal.powder": "dpi",
            "mdi": "mdi",
            "inhal.aerosol": "mdi",
            "oral aerosol": "mdi",
            "ampu": "ampule",
            "ampul": "ampule",
            "ampule": "ampule",
            "ampoule": "ampule",
            "amp": "ampule",
            "vial": "vial",
            "inj": "injection",
            "injection": "injection",
            "implant": "solution",
            "s.c. implant": "solution",
        }.get(value, value)

    def _split_route_allowed(value: object) -> set[str]:
        if not isinstance(value, str):
            return set()
        return {part.strip().lower() for part in value.split("|") if part.strip()}

    def _format_dose_display(dose: dict | None) -> str | None:
        if not isinstance(dose, dict) or not dose:
            return None
        kind = dose.get("kind")
        if kind == "amount":
            strength = dose.get("strength")
            unit = dose.get("unit")
            if strength is not None and isinstance(unit, str):
                return f"{strength}{unit}"
        if kind == "ratio":
            strength = dose.get("strength")
            unit = dose.get("unit") or ""
            per_val = dose.get("per_val")
            per_unit = dose.get("per_unit") or ""
            if strength is not None and per_val is not None:
                return f"{strength}{unit}/{per_val}{per_unit}"
        if kind == "percent":
            pct = dose.get("pct")
            if pct is not None:
                return f"{pct}%"
        return None

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
                    # Split cached string once to avoid repeated work inside the loop.
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
                    "generic_id",
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
            # Attach dose similarity scores so downstream ranking can use them directly.
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
                match_quality = np.where(best_rows["dose_sim"] < 1.0, "dose_mismatch", "OK")
                match_quality = np.where((match_quality == "OK") & (~best_rows["_form_ok"]), "no_poor_form_match", match_quality)
                match_quality = np.where((match_quality == "OK") & (~best_rows["_route_ok"]), "no_poor_route_match", match_quality)

                variants: list[str] = [
                    _format_variant(dk, strength, unit, per_val, per_unit, pct)
                    for dk, strength, unit, per_val, per_unit, pct in zip(
                        best_rows["dose_kind"],
                        best_rows["strength"],
                        best_rows["unit"],
                        best_rows["per_val"],
                        best_rows["per_unit"],
                        best_rows["pct"],
                    )
                ]

                atc_values = [
                    val if isinstance(val, str) and val else None
                    for val in best_rows["atc_code"]
                ]

                best_rows = best_rows.assign(
                    atc_code_final=atc_values,
                    match_quality=pd.Series(match_quality, index=best_rows.index),
                    selected_form=best_rows["form_token"],
                    selected_variant=variants,
                    selected_route_allowed=best_rows["route_allowed"],
                    selected_dose_kind=best_rows["dose_kind"],
                    selected_strength_mg=best_rows["strength_mg"],
                    selected_strength=best_rows["strength"],
                    selected_unit=best_rows["unit"],
                    selected_per_val=best_rows["per_val"],
                    selected_per_unit=best_rows["per_unit"],
                    selected_ratio_mg_per_ml=best_rows["ratio_mg_per_ml"],
                    selected_pct=best_rows["pct"],
                )

                best_by_idx = best_rows[
                    [
                        "atc_code_final",
                        "dose_sim",
                        "_form_ok",
                        "_route_ok",
                        "match_quality",
                        "selected_form",
                        "selected_variant",
                        "selected_route_allowed",
                        "selected_dose_kind",
                        "selected_strength_mg",
                        "selected_strength",
                        "selected_unit",
                        "selected_per_val",
                        "selected_per_unit",
                        "selected_ratio_mg_per_ml",
                        "selected_pct",
                    ]
                ].rename(columns={"_form_ok": "form_ok", "_route_ok": "route_ok"})

    pnf_by_gid: dict[str, pd.DataFrame] = {gid: grp for gid, grp in pnf_df.groupby("generic_id")}

    if best_by_idx.empty:
        out = df.copy()
        out["atc_code_final"] = None
        out["dose_sim"] = 0.0
        out["form_ok"] = False
        out["route_ok"] = False
        out["match_quality"] = "no_specific_reason_provided"
        out["selected_form"] = None
        out["selected_variant"] = None
        out["selected_route_allowed"] = None
        out["selected_dose_kind"] = None
        out["selected_strength_mg"] = None
        out["selected_strength"] = None
        out["selected_unit"] = None
        out["selected_per_val"] = None
        out["selected_per_unit"] = None
        out["selected_ratio_mg_per_ml"] = None
        out["selected_pct"] = None
    else:
        out = df.merge(best_by_idx, left_on="esoa_idx", right_index=True, how="left")
        out["atc_code_final"] = out["atc_code_final"].where(out["atc_code_final"].notna(), None)
        out["dose_sim"] = out["dose_sim"].fillna(0.0)
        # Avoid FutureWarning on object-dtype downcasting by using pandas BooleanDtype
        out["form_ok"] = out["form_ok"].astype("boolean").fillna(False).astype(bool)
        out["route_ok"] = out["route_ok"].astype("boolean").fillna(False).astype(bool)
        out["match_quality"] = out["match_quality"].fillna("no_specific_reason_provided")
        out["selected_form"] = out["selected_form"].where(out["selected_form"].notna(), None)
        out["selected_variant"] = out["selected_variant"].where(out["selected_variant"].notna(), None)
        out["selected_route_allowed"] = out["selected_route_allowed"].where(out["selected_route_allowed"].notna(), None)
        out["selected_dose_kind"] = out["selected_dose_kind"].where(out["selected_dose_kind"].notna(), None)
        out["selected_strength_mg"] = out["selected_strength_mg"].where(out["selected_strength_mg"].notna(), None)
        out["selected_strength"] = out["selected_strength"].where(out["selected_strength"].notna(), None)
        out["selected_unit"] = out["selected_unit"].where(out["selected_unit"].notna(), None)
        out["selected_per_val"] = out["selected_per_val"].where(out["selected_per_val"].notna(), None)
        out["selected_per_unit"] = out["selected_per_unit"].where(out["selected_per_unit"].notna(), None)
        out["selected_ratio_mg_per_ml"] = out["selected_ratio_mg_per_ml"].where(out["selected_ratio_mg_per_ml"].notna(), None)
        out["selected_pct"] = out["selected_pct"].where(out["selected_pct"].notna(), None)

    out["form_ok"] = out["form_ok"].astype(bool)
    out["route_ok"] = out["route_ok"].astype(bool)

    def _parse_dose_obj(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):  # pragma: no cover - defensive
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    def _normalize_pnf_value(value):
        if pd.isna(value):
            return None
        return value

    def _recompute_dose(row):
        esoa_dose = _parse_dose_obj(row.get("dosage_parsed"))
        if not isinstance(esoa_dose, dict) or not esoa_dose:
            return row.get("dose_sim", 0.0)
        pnf_payload = {
            "dose_kind": _normalize_pnf_value(row.get("selected_dose_kind")),
            "strength_mg": _normalize_pnf_value(row.get("selected_strength_mg")),
            "ratio_mg_per_ml": _normalize_pnf_value(row.get("selected_ratio_mg_per_ml")),
            "pct": _normalize_pnf_value(row.get("selected_pct")),
            "per_val": _normalize_pnf_value(row.get("selected_per_val")),
            "per_unit": _normalize_pnf_value(row.get("selected_per_unit")),
            "strength": _normalize_pnf_value(row.get("selected_strength")),
            "unit": _normalize_pnf_value(row.get("selected_unit")),
        }
        try:
            return dose_similarity(esoa_dose, pnf_payload)
        except Exception:  # pragma: no cover - defensive fallback
            return row.get("dose_sim", 0.0)

    out["dose_sim"] = pd.to_numeric(out.apply(_recompute_dose, axis=1), errors="coerce").fillna(0.0)

    form_text = out["form"].copy()
    route_text = out["route"].copy()

    form_str = form_text.fillna("").astype(str).str.strip()
    route_str = route_text.fillna("").astype(str).str.strip()
    selected_form_str = out["selected_form"].fillna("").astype(str).str.strip()
    selected_route_str = out["selected_route_allowed"].fillna("").astype(str).str.strip()

    form_has_text = form_str != ""
    esoa_dose_objs = out["dosage_parsed"].map(_parse_dose_obj)
    esoa_dose_kinds = esoa_dose_objs.map(lambda d: d.get("kind") if isinstance(d, dict) else None)
    selected_dose_kind_str = out["selected_dose_kind"].fillna("").astype(str).str.lower()
    selected_form_norm = selected_form_str.str.lower()
    solid_forms = {"tablet", "capsule"}
    incompatible_form_infer = (
        esoa_dose_kinds.fillna("").astype(str).str.lower().eq("ratio")
        & selected_form_norm.isin(solid_forms)
    )
    form_can_infer = (~form_has_text) & (selected_form_str != "") & (~incompatible_form_infer)
    route_has_text = route_str != ""
    incompatible_route_infer = (
        esoa_dose_kinds.fillna("").astype(str).str.lower().eq("ratio")
        & selected_form_norm.isin(solid_forms)
    )
    route_can_infer = (~route_has_text) & (selected_route_str != "") & (~incompatible_route_infer)

    out["form_source"] = np.where(form_has_text, "text", np.where(form_can_infer, "pnf", ""))
    out["route_source"] = np.where(route_has_text, "text", np.where(route_can_infer, "pnf", ""))

    out["form_text"] = form_text
    out["route_text"] = route_text

    if form_can_infer.any():
        out.loc[form_can_infer, "form"] = selected_form_str[form_can_infer]

    if route_can_infer.any():
        out.loc[route_can_infer, "route"] = selected_route_str[route_can_infer]
        evidence_prefill = out["route_evidence"].fillna("").astype(str)
        inferred_evidence = [
            (f"pnf:{val}" if val else "")
            for val in selected_route_str[route_can_infer]
        ]
        out.loc[route_can_infer, "route_evidence"] = [
            "".join(filter(None, [orig, ";" if orig and add else "", add]))
            for orig, add in zip(evidence_prefill[route_can_infer], inferred_evidence)
        ]

    out["route_evidence"] = out["route_evidence"].fillna("")

    def _maybe_improve_selection(row: pd.Series) -> pd.Series:
        current_sim = float(row.get("dose_sim") or 0.0)
        esoa_dose = _parse_dose_obj(row.get("dosage_parsed"))
        if not isinstance(esoa_dose, dict) or not esoa_dose:
            return row
        generic_id = row.get("generic_id")
        if not isinstance(generic_id, str) or not generic_id:
            return row
        candidates = pnf_by_gid.get(generic_id)
        if candidates is None or candidates.empty:
            return row

        route_norm = _normalize_route(row.get("route")) or _normalize_route(row.get("route_text"))
        form_norm = _normalize_form_token(row.get("form")) or _normalize_form_token(row.get("form_text"))

        best_candidate = None
        best_sim = current_sim
        solid_forms = {"tablet", "capsule"}
        for _, candidate in candidates.iterrows():
            sim = dose_similarity(esoa_dose, candidate)
            route_tokens = _split_route_allowed(candidate.get("route_allowed"))
            if route_norm and route_tokens and route_norm not in route_tokens:
                continue
            cand_form_norm = _normalize_form_token(candidate.get("form_token"))
            prefer_current = False
            if sim > best_sim + 1e-9:
                prefer_current = True
            elif best_candidate is None:
                prefer_current = True
            elif abs(sim - best_sim) <= 1e-9 and esoa_dose.get("kind") == "ratio":
                best_form_norm = _normalize_form_token(best_candidate.get("form_token"))
                best_is_solid = best_form_norm in solid_forms
                current_is_solid = cand_form_norm in solid_forms
                if best_is_solid and not current_is_solid:
                    prefer_current = True
            if not prefer_current:
                continue
            best_candidate = candidate
            best_sim = sim

        if best_candidate is None:
            return row

        row = row.copy()
        row["selected_variant"] = _format_variant(
            best_candidate.get("dose_kind"),
            best_candidate.get("strength"),
            best_candidate.get("unit"),
            best_candidate.get("per_val"),
            best_candidate.get("per_unit"),
            best_candidate.get("pct"),
        )
        row["selected_form"] = best_candidate.get("form_token")
        row["selected_route_allowed"] = best_candidate.get("route_allowed")
        row["selected_dose_kind"] = best_candidate.get("dose_kind")
        row["selected_strength"] = best_candidate.get("strength")
        row["selected_unit"] = best_candidate.get("unit")
        row["selected_strength_mg"] = best_candidate.get("strength_mg")
        row["selected_per_val"] = best_candidate.get("per_val")
        row["selected_per_unit"] = best_candidate.get("per_unit")
        row["selected_ratio_mg_per_ml"] = best_candidate.get("ratio_mg_per_ml")
        row["selected_pct"] = best_candidate.get("pct")
        row["dose_sim"] = float(best_sim)

        if best_sim >= 1.0:
            friendly = _format_dose_display(esoa_dose)
            if friendly:
                row["dose_recognized"] = friendly

        if form_norm:
            cand_form_norm = _normalize_form_token(best_candidate.get("form_token"))
            row["form_ok"] = bool(cand_form_norm and cand_form_norm == form_norm)
        if route_norm:
            route_tokens = _split_route_allowed(best_candidate.get("route_allowed"))
            row["route_ok"] = bool((not route_tokens) or (route_norm in route_tokens))

        return row

    out = out.apply(_maybe_improve_selection, axis=1, result_type="expand")
    out["dose_sim"] = pd.to_numeric(out["dose_sim"], errors="coerce").fillna(0.0)

    form_values = out["form"].fillna("").astype(str).str.strip()
    selected_form_values = out["selected_form"].fillna("").astype(str).str.strip()
    route_values = out["route"].fillna("").astype(str).str.strip()

    form_ok_array = out["form_ok"].astype(bool).to_numpy()
    route_form_flags: list[str] = []
    route_form_invalid_flags: list[bool] = []

    for idx, route_raw, text_form_raw, pnf_form_raw in zip(
        range(len(out)), route_values, form_values, selected_form_values
    ):
        route_norm = _normalize_route(route_raw)
        text_form_norm = _normalize_form_token(text_form_raw)
        pnf_form_norm = _normalize_form_token(pnf_form_raw)
        approved_forms = APPROVED_ROUTE_FORMS.get(route_norm)
        invalid = False
        flag_message = ""

        def _is_allowed(form_norm: str) -> tuple[bool, bool]:
            if not form_norm:
                return True, False
            if approved_forms is None:
                return True, False
            if form_norm in approved_forms:
                return True, False
            if (route_norm, form_norm) in FLAGGED_ROUTE_FORM_EXCEPTIONS:
                return True, True
            return False, False

        text_allowed, text_flagged = _is_allowed(text_form_norm)
        pnf_allowed, pnf_flagged = _is_allowed(pnf_form_norm)

        invalid = bool(approved_forms) and ((text_form_norm and not text_allowed) or (pnf_form_norm and not pnf_allowed))

        if invalid:
            form_ok_array[idx] = False
            offending = text_form_norm if text_form_norm and not text_allowed else pnf_form_norm
            offending_display = offending or "unspecified"
            if route_norm:
                flag_message = f"invalid:{route_norm}={offending_display}"
            else:
                flag_message = f"invalid:{offending_display}"
        else:
            route_has_rules = approved_forms is not None
            if route_has_rules:
                if text_form_norm and pnf_form_norm and text_form_norm in approved_forms and pnf_form_norm in approved_forms:
                    form_ok_array[idx] = True
                    if text_form_norm != pnf_form_norm:
                        # Record when both sources are allowed but differ, so reviewers can spot overrides.
                        flag_message = f"accepted:{text_form_norm}={pnf_form_norm}"
                elif text_form_norm and text_form_norm in approved_forms and not pnf_form_norm:
                    form_ok_array[idx] = True
                elif pnf_form_norm and pnf_form_norm in approved_forms and not text_form_norm:
                    form_ok_array[idx] = True

            if (text_flagged or pnf_flagged):
                form_ok_array[idx] = True
            if (text_flagged or pnf_flagged) and not flag_message:
                flagged_form = text_form_norm if text_flagged else pnf_form_norm
                flagged_form_disp = flagged_form or "unspecified"
                # Communicate accepted-but-flagged exceptions for manual review.
                flag_message = f"flagged:{route_norm}={flagged_form_disp}" if route_norm else f"flagged:{flagged_form_disp}"

        route_form_flags.append(flag_message)
        route_form_invalid_flags.append(invalid)

    out["form_ok"] = pd.Series(form_ok_array, index=out.index)
    out["route_form_imputations"] = pd.Series(route_form_flags, index=out.index)
    route_form_invalid_mask = pd.Series(route_form_invalid_flags, index=out.index)

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
    # Confidence weights mirror README guidance: strong emphasis on generic (60), then dose/ATC proof points, with optional bonus for corroborated brand swaps.
    # Add a bonus for high-quality brand swaps that also satisfy form/route/dose checks.
    if "brand_swap_added_generic" in out.columns:
        brand_swap_added = out["brand_swap_added_generic"].astype(bool)
    else:
        brand_swap_added = out["did_brand_swap"].astype(bool)
    bonus_mask = (
        brand_swap_added
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
                    # Normalize PNF tokens before merging to avoid duplicates.
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
    out["detail_final"] = ""

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
    missing_series = missing_series.str.replace(r"[^0-9A-Za-z]+", "_", regex=True).str.strip("_")

    present_in_pnf = out["present_in_pnf"].astype(bool)
    present_in_who = out["present_in_who"].astype(bool)
    present_in_fda = out["present_in_fda_generic"].astype(bool)
    has_atc_in_pnf = atc_present
    who_route_lists = [tokens if isinstance(tokens, list) else [] for tokens in out.get("who_route_tokens", [[] for _ in range(len(out))])]
    who_route_sets = [set(tokens) for tokens in who_route_lists]
    who_route_info_available = pd.Series([bool(tokens) for tokens in who_route_sets], index=out.index)
    who_only_mask = present_in_who & (~present_in_pnf)

    out.loc[present_in_pnf & has_atc_in_pnf, "match_molecule(s)"] = "ValidMoleculeWithATCinPNF"
    out.loc[(~present_in_pnf) & present_in_who, "match_molecule(s)"] = "ValidMoleculeWithATCinWHO/NotInPNF"
    out.loc[(~present_in_pnf) & (~present_in_who) & present_in_fda, "match_molecule(s)"] = "ValidMoleculeNoATCinFDA/NotInPNF"

    pnf_without_atc = present_in_pnf & (~has_atc_in_pnf) & out["match_molecule(s)"].eq("")
    out.loc[pnf_without_atc, "match_molecule(s)"] = "ValidMoleculeNoATCinPNF"

    out.loc[brand_swap_added & present_in_who, "match_molecule(s)"] = "ValidBrandSwappedForMoleculeWithATCinWHO"
    out.loc[brand_swap_added & present_in_pnf & has_atc_in_pnf, "match_molecule(s)"] = "ValidBrandSwappedForGenericInPNF"

    def _unique_join(values: list[str]) -> str:
        seen: list[str] = []
        for val in values:
            if not isinstance(val, str):
                continue
            item = val.strip()
            if not item or item in seen:
                continue
            seen.append(item)
        return "|".join(seen)

    def _derive_generic_final(row: pd.Series) -> str:
        gid = row.get("generic_id")
        if isinstance(gid, str) and gid.strip():
            return gid.strip()

        who_list = row.get("who_molecules_list")
        if isinstance(who_list, list):
            joined = _unique_join([str(v) for v in who_list if isinstance(v, str)])
            if joined:
                return joined

        fda_list = row.get("fda_generics_list")
        if isinstance(fda_list, list):
            joined = _unique_join([str(v) for v in fda_list if isinstance(v, str)])
            if joined:
                return joined
        return ""

    out["generic_final"] = out.apply(_derive_generic_final, axis=1)

    is_candidate_like = out["generic_id"].notna()
    auto_mask = is_candidate_like & present_in_pnf & has_atc_in_pnf & out["form_ok"] & out["route_ok"]
    out.loc[auto_mask, "bucket_final"] = "Auto-Accept"

    needs_rev_mask = (out["bucket_final"] == "") & (is_candidate_like | present_in_who | present_in_fda)
    out.loc[needs_rev_mask, "bucket_final"] = "Needs review"
    out.loc[needs_rev_mask, "why_final"] = "Needs review"

    dose_mismatch_general = (out["dose_sim"].astype(float) < 1.0) & dose_present
    if "selected_variant" in out.columns:
        selected_variant_series = out["selected_variant"]
    else:
        selected_variant_series = pd.Series([None] * len(out), index=out.index)
    selected_variant_present = (
        selected_variant_series.fillna("").astype(str).str.strip().ne("")
    )
    who_has_ddd = False
    who_brand_swap = out["match_molecule(s)"].eq("ValidBrandSwappedForMoleculeWithATCinWHO")
    dose_mismatch_general = dose_mismatch_general & selected_variant_present
    out.loc[needs_rev_mask & dose_mismatch_general, "match_quality"] = "dose_mismatch"
    out.loc[needs_rev_mask & (~dose_mismatch_general) & (missing_series.str.len() > 0), "match_quality"] = missing_series

    form_norm = out["form"].map(_normalize_form_token)
    sel_form_norm = out["selected_form"].map(_normalize_form_token)
    form_source = out["form_source"].fillna("").astype(str).str.lower()
    form_conflict = (
        form_norm.ne("")
        & sel_form_norm.ne("")
        & (form_norm != sel_form_norm)
    )
    form_unreliable = form_source.ne("text") & sel_form_norm.ne("")

    route_norm = out["route"].map(_normalize_route)
    route_source = out["route_source"].fillna("").astype(str).str.lower()
    selected_allowed_sets = [
        _split_route_allowed(val)
        for val in out["selected_route_allowed"]
    ]
    who_allowed_sets = who_route_sets
    allowed_sets = []
    for sel_allowed, who_allowed, in_pnf, in_who in zip(
        selected_allowed_sets,
        who_allowed_sets,
        present_in_pnf,
        present_in_who,
    ):
        allowed = set(sel_allowed)
        if (not allowed) and in_who and not in_pnf and who_allowed:
            allowed = set(who_allowed)
        allowed_sets.append(allowed)

    route_conflict = [
        False if (who_only and not allowed)
        else bool(r) and ((not allowed) or (r not in allowed))
        for r, allowed, who_only in zip(route_norm, allowed_sets, who_only_mask)
    ]
    route_conflict = pd.Series(route_conflict, index=out.index)
    route_unreliable = route_source.ne("text") & (route_norm != "")

    route_ok_array = out["route_ok"].astype(bool).to_numpy()
    for idx, (who_only, allowed, route_value) in enumerate(zip(who_only_mask, allowed_sets, route_norm)):
        if who_only:
            if allowed:
                route_ok_array[idx] = route_value in allowed
            else:
                route_ok_array[idx] = True
    out["route_ok"] = pd.Series(route_ok_array, index=out.index)

    unresolved = needs_rev_mask & (out["match_quality"] == "")
    out.loc[unresolved & (form_conflict | form_unreliable), "match_quality"] = "form_mismatch"

    unresolved = needs_rev_mask & (out["match_quality"] == "")
    out.loc[unresolved & (route_conflict | route_unreliable), "match_quality"] = "route_mismatch"
    out.loc[needs_rev_mask & route_form_invalid_mask, "match_quality"] = "route_form_mismatch"
    unresolved = needs_rev_mask & (out["match_quality"] == "")
    out.loc[unresolved & who_only_mask, "match_quality"] = WHO_METADATA_GAP_REASON
    out.loc[unresolved & (~who_only_mask), "match_quality"] = DEFAULT_METADATA_GAP_REASON

    who_without_ddd = present_in_who & (~present_in_pnf)
    dose_related = out["match_quality"].str.contains("dose", case=False, na=False)
    out.loc[needs_rev_mask & who_without_ddd & dose_related, "match_quality"] = "who_does_not_provide_dose_info"
    who_without_route_info = present_in_who & (~present_in_pnf) & (~who_route_info_available)
    route_related = out["match_quality"].str.contains("route", case=False, na=False)
    out.loc[needs_rev_mask & who_without_route_info & route_related, "match_quality"] = "who_does_not_provide_route_info"

    out.loc[needs_rev_mask, "reason_final"] = out.loc[needs_rev_mask, "match_quality"]
    out["reason_final"] = _mk_reason(out["reason_final"], DEFAULT_METADATA_GAP_REASON)

    # Provide consistent quality tags for Auto-Accept rows so summaries add up cleanly.
    atc_counts = (
        pnf_df.dropna(subset=["generic_id", "atc_code"])
        .groupby("generic_id")["atc_code"]
        .nunique()
    )
    gid_single_atc = {gid: count == 1 for gid, count in atc_counts.items()}

    auto_rows = out["bucket_final"].eq("Auto-Accept")
    quality_blank = out["match_quality"].astype(str).str.strip().eq("")
    route_present_mask = out["route"].astype(str).str.strip().ne("")
    form_present_mask = out["form"].astype(str).str.strip().ne("")
    dose_values = out["dose_sim"].astype(float)
    gid_series = out["generic_id"].fillna("")
    is_single_atc = gid_series.map(lambda gid: bool(gid) and gid_single_atc.get(gid, False)).astype(bool)

    exact_auto = (
        auto_rows
        & quality_blank
        & route_present_mask
        & form_present_mask
        & (dose_values == 1.0)
    )
    out.loc[exact_auto, "match_quality"] = "auto_exact_dose_route_form"

    dose_mismatch_single_atc = (
        auto_rows
        & quality_blank
        & (dose_values < 1.0)
        & is_single_atc
    )
    out.loc[dose_mismatch_single_atc, "match_quality"] = "dose_mismatch_same_atc"

    dose_mismatch_multi_atc = (
        auto_rows
        & quality_blank
        & (dose_values < 1.0)
        & (~is_single_atc)
    )
    out.loc[dose_mismatch_multi_atc, "match_quality"] = "dose_mismatch_varied_atc"

    remaining_auto = auto_rows & out["match_quality"].astype(str).str.strip().eq("")
    out.loc[remaining_auto, "match_quality"] = "auto_exact_dose_route_form"

    unknown_single = out["unknown_kind"].eq("Single - Unknown")
    unknown_multi_all = out["unknown_kind"].eq("Multiple - All Unknown")
    unknown_multi_some = out["unknown_kind"].eq("Multiple - Some Unknown")
    none_found = out["unknown_kind"].eq("None")
    fallback_unknown = (~is_candidate_like) & (~present_in_who) & (~present_in_fda) & (~none_found)

    if "non_therapeutic_summary" in out.columns:
        nonthera_summary = out["non_therapeutic_summary"].fillna("").astype(str)
        nonthera_mask = nonthera_summary.str.strip().ne("")
    else:
        nonthera_summary = pd.Series(["" for _ in range(len(out))], index=out.index)
        nonthera_mask = pd.Series([False for _ in range(len(out))], index=out.index)

    def _annotate_unknown_with_presence(base_reason: str, idx: pd.Index) -> pd.Series:
        canonical = _annotate_unknown(base_reason)
        if "(" in canonical and canonical.endswith(")"):
            head, tail = canonical.split("(", 1)
            base_text = head.strip()
            default_detail = tail.rsplit(")", 1)[0].strip()
        else:
            base_text = canonical
            default_detail = "unknown to PNF, WHO, FDA"
        labels = []
        for i in idx:
            sources = []
            if present_in_pnf[i]:
                sources.append("PNF")
            if present_in_who[i]:
                sources.append("WHO")
            if present_in_fda[i]:
                sources.append("FDA")
            if sources:
                detail = f"known in {'/'.join(sources)}; remaining tokens unknown"
            else:
                detail = default_detail
            labels.append(f"{base_text} ({detail})")
        return pd.Series(labels, index=idx)

    for cond, reason in [
        (unknown_single | (fallback_unknown & unknown_single), "Single - Unknown"),
        (unknown_multi_all | (fallback_unknown & unknown_multi_all), "Multiple - All Unknown"),
        (unknown_multi_some | (fallback_unknown & unknown_multi_some), "Multiple - Some Unknown"),
    ]:
        mask = cond & (out["bucket_final"] == "")
        if not mask.any():
            continue

        if reason == "Multiple - Some Unknown":
            # Keep partial-unknown rows in the Needs review bucket; only annotate later via detail_final.
            continue

        # Only fall back to the FDA food catalog when no molecule was identified in
        # PNF, WHO, or the FDA drug mappings. This preserves the desired
        # prioritization order (PNF → WHO → FDA drug → FDA food).
        nonthera_here = (
            mask
            & nonthera_mask
            & (~present_in_pnf)
            & (~present_in_who)
            & (~present_in_fda)
        )
        others_mask = mask & (~nonthera_mask)

        if nonthera_here.any():
            out.loc[nonthera_here, "bucket_final"] = "Others"
            out.loc[nonthera_here, "why_final"] = "Non-Therapeutic Medical Products"
            reason_tag = "non_therapeutic_detected"
            out.loc[nonthera_here, "reason_final"] = reason_tag
            out.loc[nonthera_here, "match_molecule(s)"] = "NonTherapeuticCatalogOnly"
            out.loc[nonthera_here, "match_quality"] = "nontherapeutic_catalog_match"

        if others_mask.any():
            out.loc[others_mask, "bucket_final"] = "Others"
            out.loc[others_mask, "why_final"] = "Unknown"
            out.loc[others_mask, "reason_final"] = _annotate_unknown_with_presence(reason, out.loc[others_mask].index)
            out.loc[others_mask, "match_molecule(s)"] = "AllTokensUnknownTo_PNF_WHO_FDA"
            out.loc[others_mask, "match_quality"] = "N/A"

    if "unknown_words_list" in out.columns:
        unknown_tokens_col = out["unknown_words_list"]
    else:
        unknown_tokens_col = pd.Series([[] for _ in range(len(out))], index=out.index)

    def _unknown_count(value: object) -> int:
        if isinstance(value, (list, tuple, set)):
            return sum(1 for tok in value if isinstance(tok, str) and tok.strip())
        return 0

    unknown_counts = unknown_tokens_col.map(_unknown_count)

    nonthera_label = nonthera_summary.fillna("").astype(str).replace({"nan": ""})

    detail_values: list[str] = []
    for pos, idx in enumerate(out.index):
        descriptors: list[str] = []
        count_unknown = unknown_counts.iat[pos] if pos < len(unknown_counts) else 0
        if count_unknown:
            descriptors.append(f"Unknown tokens: {count_unknown}")
        nonthera_flag = nonthera_label.at[idx] if idx in nonthera_label.index else ""
        if nonthera_flag:
            descriptors.append("Matches FDA food/non-therapeutic catalog")
        detail_values.append("; ".join(descriptors))
    out["detail_final"] = detail_values

    remaining = out["bucket_final"].eq("")
    out.loc[remaining, "bucket_final"] = "Needs review"
    out.loc[remaining, "why_final"] = "Needs review"
    out.loc[remaining, "reason_final"] = _mk_reason(out.loc[remaining, "match_quality"], DEFAULT_METADATA_GAP_REASON)

    residual_molecule = out["match_molecule(s)"].astype(str).str.strip().eq("")
    residual_quality = out["match_quality"].astype(str).str.strip().eq("")
    has_nonthera = nonthera_label.str.strip().ne("")
    unknown_some = out["unknown_kind"].eq("Multiple - Some Unknown")
    unknown_any = out["unknown_kind"].isin([
        "Single - Unknown",
        "Multiple - All Unknown",
        "Multiple - Some Unknown",
    ])

    mask = residual_molecule & has_nonthera & unknown_some
    if mask.any():
        out.loc[mask, "match_molecule(s)"] = "NonTherapeuticFoodWithUnknownTokens"
        out.loc[mask & residual_quality, "match_quality"] = "nontherapeutic_and_unknown_tokens"

    mask = residual_molecule & has_nonthera & (~unknown_any)
    if mask.any():
        out.loc[mask, "match_molecule(s)"] = "NonTherapeuticFoodNoMolecule"
        out.loc[mask & residual_quality, "match_quality"] = "nontherapeutic_catalog_match"

    mask = residual_molecule & (~has_nonthera) & unknown_some
    if mask.any():
        source_labels: list[tuple[int, str]] = []
        route_to_others: list[int] = []
        for pos, idx in enumerate(out.index):
            if not mask.iat[pos]:
                continue
            sources: list[str] = []
            if present_in_pnf.iat[pos]:
                sources.append("PNF")
            if present_in_who.iat[pos]:
                sources.append("WHO")
            if present_in_fda.iat[pos]:
                sources.append("FDA")
            if sources:
                suffix = "_".join(sources)
                source_labels.append((idx, f"PartiallyKnownTokensFrom_{suffix}"))
            else:
                route_to_others.append(idx)
        if source_labels:
            idxs, labels = zip(*source_labels)
            out.loc[list(idxs), "match_molecule(s)"] = list(labels)
        if route_to_others:
            out.loc[route_to_others, "bucket_final"] = "Others"
            out.loc[route_to_others, "why_final"] = "Unknown"
            out.loc[route_to_others, "reason_final"] = "all_tokens_unknown"
            out.loc[route_to_others, "match_molecule(s)"] = "AllTokensUnknownTo_PNF_WHO_FDA"
            out.loc[route_to_others, "match_quality"] = "N/A"

    mask = residual_molecule & (~has_nonthera) & (~unknown_any)
    if mask.any():
        out.loc[mask, "match_molecule(s)"] = "NoReferenceCatalogMatches"
        out.loc[mask & residual_quality, "match_quality"] = "no_reference_catalog_match"

    if "dose_recognized" in out.columns:
        out["dose_recognized"] = np.where(out["dose_sim"].astype(float) == 1.0, out["dose_recognized"], "N/A")

    return out
