#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Polars-only scoring and selection logic mirroring the legacy pandas behavior."""

from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import polars as pl

from .dose_drugs import dose_similarity

# Policy constants (mirrors original scorer)
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

DEFAULT_METADATA_GAP_REASON = "review_required_metadata_insufficient"
WHO_METADATA_GAP_REASON = "who_metadata_insufficient_review_required"
SOLID_FORMS: set[str] = {"tablet", "capsule"}


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


def _format_variant(dose_kind: object, strength: object, unit: object, per_val: object, per_unit: object, pct: object) -> str:
    variant = str(dose_kind) if dose_kind is not None else ""
    try:
        strength_val = float(strength) if strength is not None else None
    except Exception:
        strength_val = None
    if strength_val is not None and isinstance(unit, str):
        unit_val = unit if isinstance(unit, str) else ""
        variant = f"{dose_kind}:{strength_val:g}{unit_val}"
    try:
        per_val_num = float(per_val) if per_val is not None else None
    except Exception:
        per_val_num = None
    if per_val_num is not None:
        per_unit_val = per_unit if isinstance(per_unit, str) else ""
        variant += f"/{per_val_num:g}{per_unit_val}"
    try:
        pct_val = float(pct) if pct is not None else None
    except Exception:
        pct_val = None
    if pct_val is not None:
        variant += f" {pct_val:g}%"
    return variant


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


def _parse_dose_obj(value: object) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _unique_join(values: Iterable[str]) -> str:
    seen: list[str] = []
    for val in values:
        if not isinstance(val, str):
            continue
        item = val.strip()
        if not item or item in seen:
            continue
        seen.append(item)
    return "|".join(seen)


def _derive_generic_final_row(row: Mapping[str, Any]) -> str:
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
    drugbank_list = row.get("drugbank_generics_list")
    if isinstance(drugbank_list, list):
        joined = _unique_join([str(v) for v in drugbank_list if isinstance(v, str)])
        if joined:
            return joined
    return ""


def _score_candidate(esoa_row: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[float, bool, bool, float]:
    route_norm = _normalize_route(esoa_row.get("route") or esoa_row.get("route_text"))
    allowed = _split_route_allowed(candidate.get("route_allowed"))
    route_ok = not route_norm or (not allowed or route_norm in allowed)

    form_norm = _normalize_form_token(esoa_row.get("form") or esoa_row.get("form_text"))
    cand_form_norm = _normalize_form_token(candidate.get("form_token"))
    form_ok = bool(form_norm and cand_form_norm and form_norm == cand_form_norm)

    esoa_dose = _parse_dose_obj(esoa_row.get("dosage_parsed"))
    pnf_payload = {
        "dose_kind": candidate.get("dose_kind"),
        "strength_mg": candidate.get("strength_mg"),
        "ratio_mg_per_ml": candidate.get("ratio_mg_per_ml"),
        "pct": candidate.get("pct"),
        "per_val": candidate.get("per_val"),
        "per_unit": candidate.get("per_unit"),
        "strength": candidate.get("strength"),
        "unit": candidate.get("unit"),
    }
    try:
        dose_sim = float(dose_similarity(esoa_dose, pnf_payload)) if esoa_dose else 0.0
    except Exception:
        dose_sim = 0.0

    score = (40.0 if form_ok else 0.0) + (30.0 if route_ok else 0.0) + (dose_sim * 30.0)
    return score, form_ok, route_ok, dose_sim


def _select_best_candidate(esoa_row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    best_priority = 9999
    best_dose = -1.0
    for cand in candidates:
        score, form_ok, route_ok, dose_sim = _score_candidate(esoa_row, cand)
        try:
            priority_val = int(cand.get("source_priority"))
        except Exception:
            priority_val = 99
        prefer = False
        if score > best_score + 1e-9:
            prefer = True
        elif abs(score - best_score) <= 1e-9:
            if priority_val < best_priority:
                prefer = True
            elif priority_val == best_priority and dose_sim > best_dose + 1e-9:
                prefer = True
            elif priority_val == best_priority and abs(dose_sim - best_dose) <= 1e-9:
                if esoa_row.get("dosage_parsed") and esoa_row.get("dosage_parsed", {}).get("kind") == "ratio":
                    best_form_norm = _normalize_form_token(best.get("form_token")) if best else ""
                    cand_form_norm = _normalize_form_token(cand.get("form_token"))
                    if best_form_norm in SOLID_FORMS and cand_form_norm not in SOLID_FORMS:
                        prefer = True
        if not prefer:
            continue
        best_score = score
        best_priority = priority_val
        best_dose = dose_sim
        best = dict(cand)
        best["_score"] = score
        best["_form_ok"] = form_ok
        best["_route_ok"] = route_ok
        best["dose_sim"] = dose_sim
    return best


def _validate_route_form(route_raw: object, text_form_raw: object, selected_form_raw: object) -> tuple[bool, str, bool]:
    """Check route/form whitelist, mirroring legacy route_form_imputations behavior."""
    route_norm = _normalize_route(route_raw)
    text_form_norm = _normalize_form_token(text_form_raw)
    pnf_form_norm = _normalize_form_token(selected_form_raw)
    approved_forms = APPROVED_ROUTE_FORMS.get(route_norm)
    flag_message = ""
    invalid = False

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
        offending = text_form_norm if text_form_norm and not text_allowed else pnf_form_norm
        offending_display = offending or "unspecified"
        flag_message = f"invalid:{route_norm}={offending_display}" if route_norm else f"invalid:{offending_display}"
        return False, flag_message, True

    if text_flagged or pnf_flagged:
        flagged_form = text_form_norm if text_flagged else pnf_form_norm
        flagged_form_disp = flagged_form or "unspecified"
        flag_message = f"flagged:{route_norm}={flagged_form_disp}" if route_norm else f"flagged:{flagged_form_disp}"
        return True, flag_message, False

    if approved_forms and text_form_norm and pnf_form_norm and text_form_norm in approved_forms and pnf_form_norm in approved_forms:
        if text_form_norm != pnf_form_norm:
            flag_message = f"accepted:{text_form_norm}={pnf_form_norm}"
        return True, flag_message, False

    if approved_forms:
        if text_form_norm and text_form_norm in approved_forms:
            return True, flag_message, False
        if pnf_form_norm and pnf_form_norm in approved_forms:
            return True, flag_message, False

    return True, flag_message, False


def _match_molecule_label(row: Mapping[str, Any], best: Mapping[str, Any] | None) -> str:
    present_in_pnf = bool(row.get("present_in_pnf"))
    present_in_annex = bool(row.get("present_in_annex"))
    present_in_who = bool(row.get("present_in_who"))
    present_in_fda = bool(row.get("present_in_fda_generic"))
    present_in_drugbank = bool(row.get("present_in_drugbank"))
    brand_swap_added = bool(row.get("brand_swap_added_generic"))
    has_code = bool(best and (best.get("primary_code") or best.get("atc_code") or best.get("drug_code")))

    if brand_swap_added and present_in_who:
        return "ValidBrandSwappedForMoleculeWithATCinWHO"
    if brand_swap_added and present_in_pnf and present_in_annex and has_code:
        return "ValidBrandSwappedForGenericInAnnex"
    if brand_swap_added and present_in_pnf and has_code:
        return "ValidBrandSwappedForGenericInPNF"
    if present_in_pnf and present_in_annex and has_code:
        return "ValidMoleculeWithDrugCodeInAnnex"
    if present_in_pnf and has_code:
        return "ValidMoleculeWithATCinPNF"
    if present_in_who:
        return "ValidMoleculeWithATCinWHO/NotInPNF"
    if present_in_fda:
        return "ValidMoleculeNoATCinFDA/NotInPNF"
    if present_in_drugbank:
        return "ValidMoleculeInDrugBank"
    if present_in_pnf:
        return "ValidMoleculeNoCodeInReference"
    return ""


def score_and_classify(features_df: pl.DataFrame | pl.LazyFrame, pnf_df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Score features against PNF candidates using the legacy policy, implemented without pandas."""
    fdf = features_df.collect(streaming=True) if isinstance(features_df, pl.LazyFrame) else features_df.clone()
    pdf = pnf_df.collect(streaming=True) if isinstance(pnf_df, pl.LazyFrame) else pnf_df.clone()

    # Precompute per-generic lookups for tie-breaks
    candidate_map: Dict[str, List[dict[str, Any]]] = {}
    for row in pdf.to_dicts():
        gid = row.get("generic_id")
        gid_norm = str(gid).strip() if gid is not None else ""
        if gid_norm:
            candidate_map.setdefault(gid_norm, []).append(row)

    # Track single ATC/drug code flags
    gid_single_atc = {}
    gid_single_drug = {}
    if {"generic_id", "atc_code"} <= set(pdf.columns):
        atc_counts = pdf.drop_nulls(["generic_id", "atc_code"]).group_by("generic_id").agg(pl.col("atc_code").n_unique())
        gid_single_atc = {row["generic_id"]: bool(row["atc_code"]) and row["atc_code"] == 1 for row in atc_counts.to_dicts()}
    if {"generic_id", "drug_code"} <= set(pdf.columns):
        drug_counts = pdf.drop_nulls(["generic_id", "drug_code"]).group_by("generic_id").agg(pl.col("drug_code").n_unique())
        gid_single_drug = {row["generic_id"]: bool(row["drug_code"]) and row["drug_code"] == 1 for row in drug_counts.to_dicts()}

    out_rows: list[dict[str, Any]] = []
    for row in fdf.to_dicts():
        base = dict(row)
        esoa_idx = base.get("esoa_idx")
        gid = base.get("generic_id")
        gid_norm = str(gid).strip() if gid is not None else ""

        best = _select_best_candidate(base, candidate_map.get(gid_norm, []))
        route_form_note = ""
        route_form_invalid = False
        if best:
            base["selected_form"] = best.get("form_token", "") or ""
            base["selected_route_allowed"] = best.get("route_allowed", "") or ""
            base["selected_variant"] = _format_variant(
                best.get("dose_kind"),
                best.get("strength"),
                best.get("unit"),
                best.get("per_val"),
                best.get("per_unit"),
                best.get("pct"),
            )
            base["selected_dose_kind"] = best.get("dose_kind")
            base["selected_strength"] = best.get("strength")
            base["selected_unit"] = best.get("unit")
            base["selected_strength_mg"] = best.get("strength_mg")
            base["selected_per_val"] = best.get("per_val")
            base["selected_per_unit"] = best.get("per_unit")
            base["selected_ratio_mg_per_ml"] = best.get("ratio_mg_per_ml")
            base["selected_pct"] = best.get("pct")
            base["form_ok"] = bool(best.get("_form_ok"))
            base["route_ok"] = bool(best.get("_route_ok"))
            base["dose_sim"] = float(best.get("dose_sim") or 0.0)
            base["reference_source"] = best.get("source", "")
            try:
                base["reference_priority"] = int(best.get("source_priority"))
            except Exception:
                base["reference_priority"] = 99
            base["drug_code_final"] = best.get("drug_code", "") or ""
            base["primary_code_final"] = best.get("primary_code", "") or best.get("atc_code", "") or ""
            base["atc_code_final"] = best.get("atc_code", "") or base.get("probable_atc", "")
            base["reference_route_details"] = best.get("route_evidence_reference", "") or ""
            base["_score"] = float(best.get("_score") or 0.0)
            if base.get("dose_sim", 0.0) >= 1.0:
                friendly_dose = _format_dose_display(_parse_dose_obj(base.get("dosage_parsed")))
                if friendly_dose:
                    base["dose_recognized"] = friendly_dose

            form_ok_val, route_form_note, route_form_invalid = _validate_route_form(
                base.get("route") or base.get("route_text"),
                base.get("form") or base.get("form_text"),
                base.get("selected_form"),
            )
            if route_form_invalid:
                base["form_ok"] = False
            elif form_ok_val:
                base["form_ok"] = True
            if route_form_note:
                base["route_form_imputations"] = route_form_note
        else:
            base.setdefault("form_ok", False)
            base.setdefault("route_ok", False)
            base["dose_sim"] = float(base.get("dose_sim") or 0.0)
            base["reference_source"] = base.get("reference_source", "") or ""
            base["reference_priority"] = base.get("reference_priority", 99)
            base["drug_code_final"] = base.get("drug_code_final", "") or ""
            base["primary_code_final"] = base.get("primary_code_final", "") or ""
            base["atc_code_final"] = base.get("atc_code_final", "") or base.get("probable_atc", "")
            base["reference_route_details"] = base.get("reference_route_details", "") or ""

        base["route_form_invalid"] = route_form_invalid

        # Match labels and bucket selection (mirrors legacy behavior closely)
        base["match_molecule(s)"] = _match_molecule_label(base, best)
        qty_unknown = int(base.get("qty_unknown") or 0)
        nonthera = str(base.get("non_therapeutic_summary") or "").strip()
        dose_present = bool(base.get("dosage_parsed"))
        form_present = bool(base.get("form"))
        route_present = bool(base.get("route"))
        has_unknown_kind = str(base.get("unknown_kind") or "") in {"Single - Unknown", "Multiple - All Unknown", "Multiple - Some Unknown"}
        has_unknowns = qty_unknown > 0 or has_unknown_kind
        has_candidate = best is not None
        has_drug_code = bool(best and (best.get("drug_code") or best.get("primary_code") or best.get("atc_code")))
        dose_sim_val = float(base.get("dose_sim") or 0.0)

        bucket_final = "Unknown"
        match_quality = "no_reference_catalog_match"
        reason_final = "no_reference_catalog_match"

        if has_candidate and has_drug_code and base.get("form_ok") and base.get("route_ok") and dose_present and dose_sim_val >= 1.0 and not has_unknowns:
            bucket_final = "Auto-Accept"
            match_quality = "auto_exact_dose_route_form"
            reason_final = "Auto-Accept"
        elif has_candidate and has_drug_code:
            mq = "dose_mismatch" if dose_present and dose_sim_val < 1.0 else "candidate_ready"
            bucket_final = "Candidates"
            match_quality = mq
            reason_final = mq if mq != "candidate_ready" else "candidate_ready_for_atc_assignment"
        elif has_candidate:
            bucket_final = "Needs review"
            base_reason = "route_form_mismatch" if route_form_invalid else "annex_drug_code_missing"
            match_quality = base_reason
            reason_final = base_reason
        elif has_unknowns:
            bucket_final = "Unknown"
            match_quality = "contains_unknown_tokens"
            reason_final = "contains_unknown_tokens"
        elif route_form_invalid:
            bucket_final = "Unknown"
            match_quality = "route_form_mismatch"
            reason_final = "route_form_mismatch"
        else:
            missing: list[str] = []
            if not dose_present:
                missing.append("dose")
            if not form_present:
                missing.append("form")
            if not route_present:
                missing.append("route")
            if missing:
                if len(missing) == 3:
                    missing_label = "no_dose_form_and_route_available"
                elif len(missing) == 2:
                    missing_label = f"no_{missing[0]}_and_{missing[1]}_available"
                else:
                    missing_label = f"no_{missing[0]}_available"
                bucket_final = "Unknown"
                match_quality = missing_label
                reason_final = missing_label
            else:
                bucket_final = "Unknown"
                match_quality = "no_reference_catalog_match"
                reason_final = "no_reference_catalog_match"

        # Route mismatch adjustments (legacy metadata gaps)
        if bucket_final in {"Candidates", "Needs review"} and route_form_invalid:
            match_quality = "route_form_mismatch"
            reason_final = "route_form_mismatch"

        # Dose conflict labels for auto bucket (single vs multi code)
        if bucket_final == "Auto-Accept" and dose_sim_val < 1.0:
            is_single_drug = bool(gid_norm) and gid_single_drug.get(gid_norm, False)
            match_quality = "dose_mismatch_same_atc" if is_single_drug else "dose_mismatch_varied_atc"
            reason_final = match_quality

        # WHO gaps (approximation)
        present_in_who = bool(base.get("present_in_who"))
        present_in_pnf = bool(base.get("present_in_pnf"))
        if bucket_final in {"Candidates", "Needs review"} and present_in_who and not present_in_pnf:
            if "route" in match_quality and not base.get("who_route_tokens"):
                match_quality = "who_does_not_provide_route_info"
                reason_final = match_quality
            if "dose" in match_quality and dose_sim_val < 1.0:
                match_quality = "who_does_not_provide_dose_info"
                reason_final = match_quality

        # Unknown + non-thera adjustments
        if nonthera:
            if qty_unknown and match_quality == "contains_unknown_tokens":
                match_quality = "nontherapeutic_and_unknown_tokens"
                reason_final = "nontherapeutic_and_unknown_tokens"
            elif bucket_final == "Unknown" and match_quality in {"no_reference_catalog_match", "contains_unknown_tokens"}:
                match_quality = "nontherapeutic_catalog_match"
                reason_final = "nontherapeutic_catalog_match"

        base["bucket_final"] = bucket_final
        base["why_final"] = bucket_final
        base["match_quality"] = match_quality
        base["reason_final"] = reason_final

        # Detail + generic_final
        details: list[str] = []
        if qty_unknown:
            details.append(f"Unknown tokens: {qty_unknown}")
        if nonthera:
            details.append("Matches FDA food/non-therapeutic catalog")
        base["detail_final"] = "; ".join(details) if details else "N/A"
        base["generic_final"] = _derive_generic_final_row(base)

        # Confidence (normalized score)
        score_val = float(base.get("_score") or 0.0)
        base["confidence"] = max(0.0, min(1.0, score_val / 100.0)) if score_val else 0.0

        # Friendly cleanup for unknown detail (mirrors collapse)
        if base["bucket_final"] == "Unknown":
            detail = base.get("detail_final", "N/A") or "N/A"
            if isinstance(detail, str):
                parts = [seg.strip() for seg in detail.split(";") if seg.strip()]
                if any("Unknown tokens:" in seg for seg in parts):
                    base["detail_final"] = "ContainsUnknown(s)"
                elif not parts:
                    base["detail_final"] = "N/A"

        # Auto bucket: ensure dose_recognized if exact
        if "dose_recognized" in base and base.get("dose_sim", 0.0) != 1.0:
            base["dose_recognized"] = "N/A"

        out_rows.append(base)

    return pl.DataFrame(out_rows)
