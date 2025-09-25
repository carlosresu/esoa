#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import glob, os, re
from typing import Tuple, Optional, List, Dict, Set
import numpy as np, pandas as pd
from .aho import build_molecule_automata, scan_pnf_all
from .combos import looks_like_combination, split_combo_segments
from .routes_forms import extract_route_and_form
from .text_utils import _base_name, _normalize_text_basic, normalize_text, extract_parenthetical_phrases
from .who_molecules import detect_all_who_molecules, load_who_molecules
from .brand_map import load_latest_brandmap, build_brand_automata, fda_generics_set
from .pnf_partial import PnfTokenIndex  # token-boundary partial PNF matcher

DOSE_OR_UNIT_RX = re.compile(r"(?:(\b\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ug|iu|lsu|ml|l|%)(?:\b|/))|(\b\d+(?:[.,]\d+)?\b))", re.I)
GENERIC_TOKEN_RX = re.compile(r"[a-z]+", re.I)


def _friendly_dose(d: dict) -> str:
    if not d: return ""
    kind = d.get("kind") or d.get("dose_kind")
    if kind == "amount": return f"{d.get('strength')}{d.get('unit','')}"
    if kind == "ratio":
        pv = d.get("per_val", 1)
        try: pv = int(pv)
        except Exception: pass
        return f"{d.get('strength')}{d.get('unit','')}/{pv}{d.get('per_unit','')}"
    if kind == "percent": return f"{d.get('pct')}%"
    return ""


def _segment_norm(seg: str) -> str:
    s = _normalize_text_basic(_base_name(seg))
    s = DOSE_OR_UNIT_RX.sub(" ", s)
    s = re.sub(r"\b(?:per|with|and)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_latest_who_dir(root_dir: str) -> str | None:
    who_dir = os.path.join(root_dir, "dependencies", "atcd", "output")
    candidates = glob.glob(os.path.join(who_dir, "who_atc_*_molecules.csv"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def _select_generic_for_brand(matches_for_brand: List, pnf_name_to_gid: Dict[str, Tuple[str, str]],
                              esoa_form: Optional[str], friendly_dose: str) -> Optional[str]:
    if not matches_for_brand:
        return None

    def _score(m):
        sc = 0
        if friendly_dose and getattr(m, "dosage_strength", None) and friendly_dose.lower() in (m.dosage_strength or "").lower():
            sc += 2
        if esoa_form and getattr(m, "dosage_form", None) and esoa_form.lower() == (m.dosage_form or "").lower():
            sc += 1
        gen_base = _normalize_text_basic(_base_name(m.generic))
        if gen_base in pnf_name_to_gid:
            sc += 3
        return sc

    sorted_matches = sorted(matches_for_brand, key=_score, reverse=True)
    return sorted_matches[0].generic if sorted_matches else None


def _build_match_basis_single(norm_text: str, norm_compact: str, brand_A_norm, brand_A_comp,
                              brandmap_lookup: Dict[str, List], pnf_name_to_gid: Dict[str, Tuple[str, str]],
                              esoa_form: Optional[str], friendly_dose: str) -> Tuple[str, bool]:
    found_keys: List[str] = []
    lengths: Dict[str, int] = {}
    for _, bn in brand_A_norm.iter(norm_text):
        found_keys.append(bn); lengths[bn] = max(lengths.get(bn, 0), len(bn))
    for _, bn in brand_A_comp.iter(norm_compact):
        found_keys.append(bn); lengths[bn] = max(lengths.get(bn, 0), len(bn))
    if not found_keys:
        return norm_text, False

    uniq_keys = list(dict.fromkeys(found_keys))
    uniq_keys.sort(key=lambda k: (-lengths.get(k, len(k)), k))

    out = norm_text
    replaced_any = False
    for bn in uniq_keys:
        options = brandmap_lookup.get(bn, [])
        chosen_generic = _select_generic_for_brand(options, pnf_name_to_gid, esoa_form, friendly_dose)
        if not chosen_generic:
            continue
        gd_norm = normalize_text(chosen_generic)
        new_out = re.sub(rf"\b{re.escape(bn)}\b", gd_norm, out)
        if new_out != out:
            replaced_any = True
            out = new_out
    out = re.sub(r"\s+", " ", out).strip()
    return out, replaced_any


def _tokenize_unknowns(s_norm: str) -> List[str]:
    return [m.group(0) for m in GENERIC_TOKEN_RX.finditer(s_norm)]


def build_features(pnf_df: pd.DataFrame, esoa_df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[str], Set[str], Set[str]]:
    required_pnf = {"generic_id","generic_name","synonyms","atc_code","route_allowed","form_token","dose_kind","strength","unit","per_val","per_unit","pct","strength_mg","ratio_mg_per_ml"}
    missing = required_pnf - set(pnf_df.columns)
    if missing: raise ValueError(f"pnf_prepared.csv missing columns: {missing}")
    if "raw_text" not in esoa_df.columns: raise ValueError("esoa_prepared.csv must contain a 'raw_text' column")

    pnf_name_to_gid: Dict[str, Tuple[str, str]] = {}
    for gid, gname in pnf_df[["generic_id","generic_name"]].drop_duplicates().itertuples(index=False):
        key = _normalize_text_basic(_base_name(str(gname)))
        if key and key not in pnf_name_to_gid:
            pnf_name_to_gid[key] = (gid, gname)
    pnf_name_set: Set[str] = set(pnf_name_to_gid.keys())

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    who_file = load_latest_who_dir(root_dir)
    codes_by_name, candidate_names = ({}, [])
    who_name_set: Set[str] = set()
    who_regex = None
    if who_file and os.path.exists(who_file):
        codes_by_name, candidate_names = load_who_molecules(who_file)
        who_name_set = set(codes_by_name.keys())
        who_regex = re.compile(r"\b(" + "|".join(map(re.escape, candidate_names)) + r")\b") if candidate_names else None

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    inputs_dir = os.path.join(project_root, "inputs")
    brand_df = load_latest_brandmap(inputs_dir)
    has_brandmap = brand_df is not None and not brand_df.empty
    if has_brandmap:
        B_norm, B_comp, brand_lookup = build_brand_automata(brand_df)
        fda_gens = fda_generics_set(brand_df)
    else:
        B_norm = B_comp = None
        brand_lookup = {}
        fda_gens = set()

    A_norm, A_comp = build_molecule_automata(pnf_df)

    pnf_partial_idx = PnfTokenIndex().build_from_pnf(pnf_df)

    df = esoa_df[["raw_text"]].copy()
    df["esoa_idx"] = df.index
    df["normalized"] = df["raw_text"].map(normalize_text)
    df["norm_compact"] = df["normalized"].map(lambda s: re.sub(r"[ \-]", "", s))

    from .dose import extract_dosage as _extract_dosage
    df["dosage_parsed_raw"] = df["normalized"].map(_extract_dosage)
    df["dose_recognized"] = df["dosage_parsed_raw"].map(_friendly_dose)
    df["route_raw"], df["form_raw"], df["route_evidence_raw"] = zip(*df["normalized"].map(extract_route_and_form))

    if has_brandmap:
        mb_list, swapped = [], []
        for norm, comp, form, friendly in zip(df["normalized"], df["norm_compact"], df["form_raw"], df["dose_recognized"]):
            mb, did_swap = _build_match_basis_single(norm, comp, B_norm, B_comp, brand_lookup, pnf_name_to_gid, form, friendly)
            mb_list.append(mb); swapped.append(did_swap)
        df["match_basis"] = mb_list
        df["did_brand_swap"] = swapped
    else:
        df["match_basis"] = df["normalized"]
        df["did_brand_swap"] = False

    df["dosage_parsed"] = df["match_basis"].map(_extract_dosage)
    df["route"], df["form"], df["route_evidence"] = zip(*df["match_basis"].map(extract_route_and_form))

    # Silent per-row loop (no tqdm)
    primary_gid, primary_token, pnf_hits_gids, pnf_hits_tokens, pnf_hits_count = [], [], [], [], []
    for s_norm, s_comp in zip(df["match_basis"], df["norm_compact"]):
        gids, tokens = scan_pnf_all(s_norm, s_comp, A_norm, A_comp)
        pnf_hits_gids.append(gids); pnf_hits_tokens.append(tokens); pnf_hits_count.append(len(gids))
        if gids: primary_gid.append(gids[0]); primary_token.append(tokens[0])
        else: primary_gid.append(None); primary_token.append(None)

    df["pnf_hits_gids"] = pnf_hits_gids; df["pnf_hits_tokens"] = pnf_hits_tokens; df["pnf_hits_count"] = pnf_hits_count
    df["generic_id"] = primary_gid; df["molecule_token"] = primary_token

    partial_gids: List[Optional[str]] = [None] * len(df)
    partial_tokens: List[Optional[str]] = [None] * len(df)
    for i, (gid, txt) in enumerate(zip(df["generic_id"].tolist(), df["match_basis"].tolist())):
        if gid is not None:
            continue
        res = pnf_partial_idx.best_partial_in_text(str(txt))
        if res:
            pgid, matched_span = res
            partial_gids[i] = pgid
            partial_tokens[i] = matched_span

    mask_partial = pd.Series([g is not None for g in partial_gids])
    if mask_partial.any():
        df.loc[mask_partial, "generic_id"] = [g for g in partial_gids if g is not None]
        df.loc[mask_partial, "molecule_token"] = [t for t in partial_tokens if t is not None]
        df.loc[mask_partial, "pnf_hits_count"] = df.loc[mask_partial, "pnf_hits_count"].fillna(0).astype(int) + 1

    if who_regex:
        who_names_all, who_atc_all = [], []
        for txt in df["match_basis"].tolist():
            names, codes = detect_all_who_molecules(txt, who_regex, codes_by_name)
            who_names_all.append(names); who_atc_all.append(sorted(codes))
        df["who_molecules_list"] = who_names_all; df["who_atc_codes_list"] = who_atc_all
        df["who_molecules"] = df["who_molecules_list"].map(lambda xs: "|".join(xs) if xs else "")
        df["who_atc_codes"] = df["who_atc_codes_list"].map(lambda xs: "|".join(xs) if xs else "")
    else:
        df["who_molecules_list"] = [[] for _ in range(len(df))]
        df["who_atc_codes_list"] = [[] for _ in range(len(df))]
        df["who_molecules"] = ""; df["who_atc_codes"] = ""

    def _known_generic_tokens(text_norm: str) -> List[str]:
        s = _segment_norm(text_norm)
        toks = _tokenize_unknowns(s)
        out = []
        for t in toks:
            if t in pnf_name_set or t in who_name_set or t in fda_gens:
                out.append(t)
        seen=set(); res=[]
        for t in out:
            if t not in seen:
                seen.add(t); res.append(t)
        return res

    known_counts = df["match_basis"].map(lambda s: len(_known_generic_tokens(s)))
    df["combo_known_generics_count"] = known_counts
    df["looks_combo_final"] = df["combo_known_generics_count"].ge(2)
    df["combo_reason"] = np.where(df["looks_combo_final"], "combo/known-generics>=2", "single/heuristic")

    def _unknown_kind_and_list(text_norm: str) -> Tuple[str, List[str]]:
        s = _segment_norm(text_norm)
        all_toks = _tokenize_unknowns(s)
        unknowns = []
        for t in all_toks:
            if (t not in pnf_name_set) and (t not in who_name_set) and (t not in fda_gens):
                unknowns.append(t)
        seen=set(); unknowns_uniq=[]
        for t in unknowns:
            if t not in seen:
                seen.add(t); unknowns_uniq.append(t)
        if not unknowns_uniq:
            return "None", []
        if len(unknowns_uniq) == len(all_toks):
            if len(unknowns_uniq) == 1:
                return "Single - Unknown", unknowns_uniq
            return "Multiple - All Unknown", unknowns_uniq
        return ("Multiple - Some Unknown", unknowns_uniq)

    kinds, lists_ = zip(*df["match_basis"].map(_unknown_kind_and_list))
    df["unknown_kind"] = kinds
    df["unknown_words_list"] = lists_
    df["unknown_words"] = df["unknown_words_list"].map(lambda xs: "|".join(xs) if xs else "")

    df["present_in_pnf"] = df["pnf_hits_count"].astype(int).gt(0)
    df["present_in_who"] = df["who_atc_codes"].astype(str).str.len().gt(0)
    df["present_in_fda_generic"] = df["match_basis"].map(lambda s: any(tok in fda_gens for tok in _tokenize_unknowns(_segment_norm(s))))

    return df, pnf_name_set, who_name_set, fda_gens
