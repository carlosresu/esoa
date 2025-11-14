#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize Annex F drug descriptions into the prepared schema used by the matcher."""
from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from pipelines.drugs.scripts.aho_drugs import build_molecule_automata, scan_pnf_all
from pipelines.drugs.scripts.combos_drugs import SALT_TOKENS
from pipelines.drugs.scripts.dose_drugs import parse_dose_struct_from_text, safe_ratio_mg_per_ml, to_mg
from pipelines.drugs.scripts.generic_normalization import normalize_generic
from pipelines.drugs.scripts.match_features_drugs import (
    WHO_ADM_ROUTE_MAP,
    WHO_FORM_TO_CANONICAL,
    WHO_UOM_TO_CANONICAL,
    load_latest_who_file,
)
from pipelines.drugs.scripts.pnf_aliases_drugs import SPECIAL_GENERIC_ALIASES, expand_generic_aliases
from pipelines.drugs.scripts.pnf_partial_drugs import PnfTokenIndex
from pipelines.drugs.scripts.reference_data_drugs import load_drugbank_generics
from pipelines.drugs.scripts.routes_forms_drugs import extract_route_and_form
from pipelines.drugs.scripts.text_utils_drugs import (
    _base_name,
    _normalize_text_basic,
    extract_base_and_salts,
    normalize_compact,
    normalize_text,
    serialize_salt_list,
    slug_id,
)
from pipelines.drugs.scripts.who_molecules_drugs import detect_all_who_molecules, load_who_molecules


def _dedupe_preserve(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for val in values:
        key = val.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _build_pnf_context(pnf_df: pd.DataFrame) -> Dict[str, Any]:
    if not {"generic_id", "generic_name"}.issubset(pnf_df.columns):
        raise ValueError("PNF prepared CSV must contain generic_id and generic_name columns")
    df = pnf_df.copy()
    for col in ("salt_form", "synonyms", "form_token", "route_allowed"):
        if col not in df.columns:
            df[col] = ""
    df["salt_form"] = df["salt_form"].fillna("")
    df["synonyms"] = df["synonyms"].fillna("")
    df["form_token"] = df["form_token"].fillna("")
    df["route_allowed"] = df["route_allowed"].fillna("")

    name_by_gid: Dict[str, str] = {}
    salt_by_gid: Dict[str, str] = {}
    form_by_gid: Dict[str, List[str]] = {}
    routes_by_gid: Dict[str, List[str]] = {}
    alias_lookup: Dict[str, Tuple[str, str]] = {}

    for row in df.itertuples(index=False):
        gid = str(row.generic_id)
        if not gid:
            continue
        name = str(row.generic_name or "").strip().upper()
        if gid not in name_by_gid and name:
            name_by_gid[gid] = name
        salt = str(row.salt_form or "").strip().upper()
        if gid not in salt_by_gid and salt:
            salt_by_gid[gid] = salt
        form = str(row.form_token or "").strip().lower()
        if form:
            bucket = form_by_gid.setdefault(gid, [])
            if form not in bucket:
                bucket.append(form)
        route = str(row.route_allowed or "").strip().lower()
        if route:
            bucket = routes_by_gid.setdefault(gid, [])
            if route not in bucket:
                bucket.append(route)

    def _register_alias(generic_id: str, alias: str, canonical: str) -> None:
        norm = _normalize_text_basic(alias)
        if not norm or norm in alias_lookup:
            return
        alias_lookup[norm] = (generic_id, canonical)

    for gid, gname in name_by_gid.items():
        alias_set = expand_generic_aliases(gname)
        alias_set |= SPECIAL_GENERIC_ALIASES.get(gid, set())
        syn_series = df.loc[df["generic_id"] == gid, "synonyms"]
        for syn in syn_series:
            if not isinstance(syn, str):
                continue
            for alias in syn.split("|"):
                alias_set |= expand_generic_aliases(alias)
        for alias in alias_set:
            _register_alias(gid, alias, gname)

    automata_norm, automata_comp = build_molecule_automata(df)
    partial_index = PnfTokenIndex().build_from_pnf(df)

    return {
        "df": df,
        "A_norm": automata_norm,
        "A_comp": automata_comp,
        "partial_index": partial_index,
        "name_by_gid": name_by_gid,
        "salt_by_gid": salt_by_gid,
        "form_by_gid": form_by_gid,
        "routes_by_gid": routes_by_gid,
        "alias_lookup": alias_lookup,
    }


def _build_who_context(who_csv: Path) -> Dict[str, Any]:
    codes_by_name, candidate_names, details_by_code = load_who_molecules(str(who_csv))
    regex = None
    if candidate_names:
        pattern = r"\b(" + "|".join(map(re.escape, candidate_names)) + r")\b"
        regex = re.compile(pattern)

    frame = pd.read_csv(who_csv)
    name_lookup: Dict[str, str] = {}
    for raw_name in frame.get("atc_name", pd.Series([], dtype=str)):
        if not isinstance(raw_name, str):
            continue
        base = _base_name(raw_name)
        display = base.strip().upper() or raw_name.strip().upper()
        norm_key = _normalize_text_basic(base)
        if norm_key and norm_key not in name_lookup:
            name_lookup[norm_key] = display
        full_norm = _normalize_text_basic(raw_name)
        if full_norm and full_norm not in name_lookup:
            name_lookup[full_norm] = display

    return {
        "regex": regex,
        "codes_by_name": codes_by_name,
        "details_by_code": details_by_code,
        "name_lookup": name_lookup,
    }


def _build_drugbank_context(project_root: Optional[Path] = None) -> Dict[str, Any]:
    (
        normalized_names,
        token_pool,
        token_index,
        display_lookup,
        salt_lookup,
    ) = load_drugbank_generics(project_root)
    return {
        "normalized_names": normalized_names,
        "token_pool": token_pool,
        "token_index": token_index,
        "display_lookup": display_lookup,
        "salt_lookup": salt_lookup,
    }


def _detect_pnf(norm_text: str, compact_text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    gids: List[str] = []
    tokens: List[str] = []
    if norm_text:
        raw_gids, raw_tokens = scan_pnf_all(norm_text, compact_text, ctx["A_norm"], ctx["A_comp"])
        for gid, token in zip(raw_gids, raw_tokens):
            base_norm = _normalize_text_basic(_base_name(token))
            words = [w for w in base_norm.split() if w]
            if words and all(word in SALT_TOKENS for word in words):
                continue
            if gid not in gids:
                gids.append(gid)
                tokens.append(token)

    if not gids:
        partial = ctx["partial_index"].best_partial_in_text(norm_text)
        if partial:
            gid, _matched = partial
            gids = [gid]

    names = [ctx["name_by_gid"].get(gid, "").strip().upper() for gid in gids if ctx["name_by_gid"].get(gid)]
    routes = _dedupe_preserve(
        route for gid in gids for route in ctx["routes_by_gid"].get(gid, []) if route
    )
    forms = _dedupe_preserve(
        form for gid in gids for form in ctx["form_by_gid"].get(gid, []) if form
    )
    salt_forms = _dedupe_preserve(
        ctx["salt_by_gid"].get(gid, "").strip().upper() for gid in gids if ctx["salt_by_gid"].get(gid)
    )
    return {
        "gids": gids,
        "names": names,
        "routes": routes,
        "forms": forms,
        "salt_forms": salt_forms,
    }


def _detect_who(raw_text: str, norm_text: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    regex = ctx["regex"]
    if not regex:
        return {"names": [], "codes": [], "routes": [], "forms": [], "route_display": "", "salt_forms": []}
    names, codes = detect_all_who_molecules(raw_text, regex, ctx["codes_by_name"], pre_normalized=norm_text)
    display_names = [
        ctx["name_lookup"].get(name, name.strip().upper()) for name in names if name
    ]
    route_tokens: Set[str] = set()
    form_tokens: Set[str] = set()
    adm_r_values: Set[str] = set()
    salt_forms: Set[str] = set()
    for code in codes:
        for detail in ctx["details_by_code"].get(code, []):
            adm_val = detail.get("adm_r")
            if isinstance(adm_val, str) and adm_val.strip():
                adm_key = adm_val.strip().lower()
                mapped = WHO_ADM_ROUTE_MAP.get(adm_key)
                if mapped:
                    route_tokens.update(mapped)
                    adm_r_values.update(mapped)
                else:
                    route_tokens.add(adm_key)
                    adm_r_values.add(adm_key)
                form_tokens.update(WHO_FORM_TO_CANONICAL.get(adm_key, set()))
            uom_val = detail.get("uom")
            if isinstance(uom_val, str) and uom_val.strip():
                form_tokens.update(WHO_UOM_TO_CANONICAL.get(uom_val.strip().lower(), set()))
            salt = detail.get("salt_form")
            if isinstance(salt, str) and salt.strip():
                salt_forms.add(salt.strip().upper())
    return {
        "names": _dedupe_preserve(display_names),
        "codes": codes,
        "routes": sorted(route_tokens),
        "forms": sorted(form_tokens),
        "route_display": "|".join(sorted(adm_r_values)),
        "salt_forms": sorted(salt_forms),
    }


def _detect_drugbank(norm_text: str, ctx: Dict[str, Any]) -> List[str]:
    token_index = ctx.get("token_index")
    if not token_index:
        return []
    norm_basic = _normalize_text_basic(norm_text)
    if not norm_basic:
        return []
    tokens = norm_basic.split()
    matches: List[str] = []
    seen: Set[str] = set()

    def _record(candidate_norm: str) -> None:
        if not candidate_norm:
            return
        compact = candidate_norm.replace(" ", "")
        dedupe_key = compact or candidate_norm
        if dedupe_key in seen:
            return
        display = (
            ctx["display_lookup"].get(candidate_norm)
            or ctx["display_lookup"].get(compact)
            or candidate_norm
        )
        seen.add(dedupe_key)
        matches.append(display.strip().upper())

    token_count = len(tokens)
    for pos, token in enumerate(tokens):
        candidates = token_index.get(token)
        if not candidates:
            continue
        for cand_tokens in candidates:
            length = len(cand_tokens)
            if length == 0 or pos + length > token_count:
                continue
            if tuple(tokens[pos : pos + length]) == cand_tokens:
                candidate_norm = " ".join(cand_tokens)
                _record(candidate_norm)

    for start in range(token_count):
        for end in range(start + 2, min(token_count, start + 6) + 1):
            compact = "".join(tokens[start:end])
            if compact in ctx["display_lookup"]:
                _record(compact)

    whole_compact = norm_basic.replace(" ", "")
    if whole_compact in ctx["display_lookup"]:
        _record(whole_compact)

    return matches


def _normalize_description(raw: str) -> str:
    return normalize_text(raw)


def _select_generic(generated: List[str]) -> str:
    ordered = _dedupe_preserve(name.strip().upper() for name in generated if name)
    if ordered:
        return " + ".join(ordered)
    return ""


def _choose_generic_id(
    chosen_name: str,
    pnf_matches: Dict[str, Any],
    alias_lookup: Dict[str, Tuple[str, str]],
) -> str:
    gids = pnf_matches["gids"]
    if len(gids) == 1:
        return gids[0]
    if len(gids) > 1:
        return slug_id(chosen_name or "_".join(gids))
    norm_key = _normalize_text_basic(chosen_name)
    if norm_key and norm_key in alias_lookup:
        return alias_lookup[norm_key][0]
    return slug_id(chosen_name)


def _final_salt_form(
    salt_tokens: Sequence[str],
    generic_name: str,
    reference_fallbacks: Sequence[str],
) -> str:
    keep: List[str] = []
    gen_tokens = {tok.lower() for tok in generic_name.split()}
    for salt in salt_tokens:
        token = salt.strip()
        if not token:
            continue
        if token.lower() in gen_tokens:
            continue
        keep.append(token.upper())
    if keep:
        return serialize_salt_list(keep)
    for ref in reference_fallbacks:
        clean = str(ref).strip().upper()
        if clean:
            return clean
    return ""


def _resolve_route(
    text_route: Optional[str],
    who_result: Dict[str, Any],
    pnf_matches: Dict[str, Any],
    evidence_parts: List[str],
) -> str:
    route = ""
    if pnf_matches["routes"]:
        if text_route and text_route in pnf_matches["routes"]:
            route = text_route
        else:
            route = pnf_matches["routes"][0]
        evidence_parts.append(f"pnf:{route}")
    elif who_result["routes"]:
        route = who_result["routes"][0]
        disp = who_result["route_display"]
        evidence_parts.append(f"who_adm_r:{disp or route}")
    elif text_route:
        route = text_route
    return route


def _resolve_form(
    text_form: Optional[str],
    who_result: Dict[str, Any],
    pnf_matches: Dict[str, Any],
) -> str:
    if pnf_matches["forms"]:
        return pnf_matches["forms"][0]
    if who_result["forms"]:
        return who_result["forms"][0]
    if text_form:
        return text_form
    return ""


def _prepare_record(
    row: pd.Series,
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    raw_description = str(row.get("Drug Description", "") or "").strip()
    drug_code = str(row.get("Drug Code", "") or "").strip()
    normalized_description = _normalize_description(raw_description)
    norm_compact = normalize_compact(raw_description)

    dose_struct = parse_dose_struct_from_text(normalized_description)
    dose_kind = dose_struct.get("dose_kind")
    strength = dose_struct.get("strength")
    unit = dose_struct.get("unit")
    per_val = dose_struct.get("per_val")
    per_unit = dose_struct.get("per_unit")
    pct = dose_struct.get("pct")
    strength_mg = to_mg(strength, unit) if dose_kind == "amount" else None
    ratio_mg_per_ml = (
        safe_ratio_mg_per_ml(strength, unit, per_val)
        if dose_kind == "ratio" and str(per_unit).lower() == "ml"
        else None
    )

    text_route, text_form, route_evidence_text = extract_route_and_form(normalized_description)
    evidence_parts: List[str] = []
    if route_evidence_text:
        evidence_parts.append(route_evidence_text)

    pnf_matches = _detect_pnf(normalized_description, norm_compact, ctx["pnf"])
    who_result = _detect_who(raw_description, normalized_description, ctx["who"])
    drugbank_hits = _detect_drugbank(normalized_description, ctx["drugbank"])

    base_text, salt_tokens = extract_base_and_salts(raw_description)
    reference_names: List[str] = []
    reference_names.extend(pnf_matches["names"])
    if not reference_names:
        reference_names.extend(who_result["names"])
    if not reference_names:
        reference_names.extend(drugbank_hits)
    if not reference_names:
        norm_generic = normalize_generic(raw_description).strip().upper()
        if norm_generic:
            reference_names.append(norm_generic)
        elif base_text:
            reference_names.append(base_text.strip().upper())
        else:
            reference_names.append(raw_description.strip().upper())

    generic_name = _select_generic(reference_names)
    generic_id = _choose_generic_id(generic_name, pnf_matches, ctx["pnf"]["alias_lookup"])

    salt_form = _final_salt_form(
        salt_tokens,
        generic_name,
        pnf_matches["salt_forms"] + who_result["salt_forms"],
    )

    route_allowed = _resolve_route(text_route, who_result, pnf_matches, evidence_parts)
    form_token = _resolve_form(text_form, who_result, pnf_matches)
    route_evidence = ";".join(part for part in evidence_parts if part).strip(";")

    return {
        "drug_code": drug_code,
        "raw_description": raw_description,
        "normalized_description": normalized_description,
        "generic_name": generic_name,
        "generic_id": generic_id,
        "salt_form": salt_form,
        "route_allowed": route_allowed,
        "route_evidence": route_evidence,
        "form_token": form_token,
        "dose_kind": dose_kind,
        "strength": strength,
        "unit": unit,
        "per_val": per_val,
        "per_unit": per_unit,
        "pct": pct,
        "strength_mg": strength_mg,
        "ratio_mg_per_ml": ratio_mg_per_ml,
    }


def _resolve_inputs_dir(path: str | Path) -> Path:
    candidates = [
        Path(path).expanduser(),
        Path.cwd() / Path(path),
        Path(__file__).resolve().parent / Path(path),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Input file not found: {path}")


def _prepare_annex(
    annex_csv: Path,
    ctx: Dict[str, Any],
) -> pd.DataFrame:
    annex = pd.read_csv(annex_csv)
    if "Drug Description" not in annex.columns or "Drug Code" not in annex.columns:
        raise ValueError("Annex CSV must contain 'Drug Code' and 'Drug Description' columns")
    records = [_prepare_record(row, ctx) for _, row in annex.iterrows()]
    return pd.DataFrame.from_records(records)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_inputs = Path("inputs") / "drugs"
    parser = argparse.ArgumentParser(
        description="Normalize Annex F descriptions using PNF/WHO/DrugBank references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annex", default=str(default_inputs / "annex_f.csv"), help="Raw Annex F CSV path")
    parser.add_argument(
        "--pnf-prepared",
        default=str(default_inputs / "pnf_prepared.csv"),
        help="Prepared PNF CSV path (output of pipelines.drugs.scripts.prepare_drugs)",
    )
    parser.add_argument(
        "--who",
        default=None,
        help="WHO ATC molecules CSV (defaults to latest who_atc_*_molecules.csv under inputs/drugs/)",
    )
    parser.add_argument(
        "--output",
        default=str(default_inputs / "annex_f_prepared.csv"),
        help="Destination for normalized Annex F CSV",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parent
    annex_path = _resolve_inputs_dir(args.annex)
    pnf_path = _resolve_inputs_dir(args.pnf_prepared)
    pnf_df = pd.read_csv(pnf_path)
    pnf_ctx = _build_pnf_context(pnf_df)

    who_path = Path(args.who).expanduser() if args.who else None
    if who_path is None:
        resolved = load_latest_who_file(str(project_root))
        if not resolved:
            raise FileNotFoundError("Unable to locate WHO ATC molecules CSV; provide --who explicitly.")
        who_path = Path(resolved)
    if not who_path.is_file():
        raise FileNotFoundError(f"WHO ATC CSV not found: {who_path}")
    who_ctx = _build_who_context(who_path)

    drugbank_ctx = _build_drugbank_context(project_root)

    context = {
        "pnf": pnf_ctx,
        "who": who_ctx,
        "drugbank": drugbank_ctx,
    }
    prepared = _prepare_annex(annex_path, context)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote {len(prepared)} rows to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
