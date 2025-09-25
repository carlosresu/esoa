#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import glob, os, re
from typing import Tuple
import numpy as np, pandas as pd
from .aho import build_molecule_automata, scan_pnf_all
from .combos import looks_like_combination, split_combo_segments
from .routes_forms import extract_route_and_form
from .text_utils import _base_name, _normalize_text_basic, normalize_text, extract_parenthetical_phrases
from .who_molecules import detect_all_who_molecules, load_who_molecules

DOSE_OR_UNIT_RX = re.compile(r"(?:(\b\d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|ug|iu|lsu|ml|l|%)(?:\b|/))|(\b\d+(?:[\.,]\d+)?\b))", re.I)

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

def _analyze_combo_segments(row: pd.Series, pnf_name_set: set, who_name_set: set):
    segs = split_combo_segments(row.get("normalized", ""))
    norms = [_segment_norm(s) for s in segs if s]
    if not norms: return 0,0,0,0,0
    in_pnf = [n in pnf_name_set for n in norms]
    in_who = [n in who_name_set for n in norms]
    identified = sum(1 for pnff, whof in zip(in_pnf, in_who) if pnff or whof)
    unknown_in_pnf = sum(1 for pnff, whof in zip(in_pnf, in_who) if (not pnff) and whof)
    unknown_in_who = sum(1 for pnff, whof in zip(in_pnf, in_who) if pnff and (not whof))
    unknown_in_both = sum(1 for pnff, whof in zip(in_pnf, in_who) if (not pnff) and (not whof))
    return len(norms), identified, unknown_in_pnf, unknown_in_who, unknown_in_both

def load_latest_who_dir(root_dir: str) -> str | None:
    who_dir = os.path.join(root_dir, "dependencies", "atcd", "output")
    candidates = glob.glob(os.path.join(who_dir, "who_atc_*_molecules.csv"))
    return max(candidates, key=os.path.getmtime) if candidates else None

def build_features(pnf_df: pd.DataFrame, esoa_df: pd.DataFrame) -> Tuple[pd.DataFrame, set, set]:
    required_pnf = {"generic_id","generic_name","synonyms","atc_code","route_allowed","form_token","dose_kind","strength","unit","per_val","per_unit","pct","strength_mg","ratio_mg_per_ml"}
    missing = required_pnf - set(pnf_df.columns)
    if missing: raise ValueError(f"pnf_prepared.csv missing columns: {missing}")
    if "raw_text" not in esoa_df.columns: raise ValueError("esoa_prepared.csv must contain a 'raw_text' column")
    A_norm, A_comp = build_molecule_automata(pnf_df)
    df = esoa_df[["raw_text"]].copy()
    df["esoa_idx"] = df.index
    df["norm"] = df["raw_text"].map(normalize_text)
    df["norm_compact"] = df["norm"].map(lambda s: re.sub(r"[ \-]", "", s))
    df["probable_brands_list"] = df["raw_text"].map(extract_parenthetical_phrases)
    df["probable_brands"] = df["probable_brands_list"].map(lambda xs: "|".join(xs) if xs else "")
    primary_gid, primary_token, pnf_hits_gids, pnf_hits_tokens, pnf_hits_count = [], [], [], [], []
    for s_norm, s_comp in zip(df["norm"], df["norm_compact"]):
        gids, tokens = scan_pnf_all(s_norm, s_comp, A_norm, A_comp)
        pnf_hits_gids.append(gids); pnf_hits_tokens.append(tokens); pnf_hits_count.append(len(gids))
        if gids: primary_gid.append(gids[0]); primary_token.append(tokens[0])
        else: primary_gid.append(None); primary_token.append(None)
    df["pnf_hits_gids"] = pnf_hits_gids; df["pnf_hits_tokens"] = pnf_hits_tokens; df["pnf_hits_count"] = pnf_hits_count
    df["generic_id"] = primary_gid; df["molecule_token"] = primary_token
    from .dose import extract_dosage as _extract_dosage
    df["dosage_parsed"] = df["norm"].map(_extract_dosage)
    df["dose_recognized"] = df["dosage_parsed"].map(_friendly_dose)
    df["dose_kind_detected"] = df["dosage_parsed"].map(lambda d: (d or {}).get("kind") or (d or {}).get("dose_kind") or "")
    df["route"], df["form"], df["route_evidence"] = zip(*df["norm"].map(extract_route_and_form))
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    who_file = load_latest_who_dir(root_dir)
    who_name_set = set()
    if who_file and os.path.exists(who_file):
        codes_by_name, candidate_names = load_who_molecules(who_file)
        who_name_set = set(codes_by_name.keys())
        regex = re.compile(r"\b(" + "|".join(map(re.escape, candidate_names)) + r")\b")
        who_names_all, who_atc_all = [], []
        for txt in df["norm"].tolist():
            names, codes = detect_all_who_molecules(txt, regex, codes_by_name)
            who_names_all.append(names); who_atc_all.append(sorted(codes))
        df["who_molecules_list"] = who_names_all; df["who_atc_codes_list"] = who_atc_all
        df["who_molecules"] = df["who_molecules_list"].map(lambda xs: "|".join(xs) if xs else "")
        df["who_atc_codes"] = df["who_atc_codes_list"].map(lambda xs: "|".join(xs) if xs else "")
    else:
        df["who_molecules_list"] = [[] for _ in range(len(df))]
        df["who_atc_codes_list"] = [[] for _ in range(len(df))]
        df["who_molecules"] = ""; df["who_atc_codes"] = ""
    looks_raw = [looks_like_combination(s, p_cnt, len(w_names)) for s, p_cnt, w_names in zip(df["norm"], df["pnf_hits_count"], df["who_molecules_list"])]
    df["looks_combo_raw"] = looks_raw; df["looks_combo_final"] = looks_raw
    df["combo_reason"] = np.where(df["looks_combo_final"], "combo/heuristic", "single/heuristic")
    df["bucket"] = np.where(df["looks_combo_final"], "Others:Combinations", np.where(df["generic_id"].isna(), "BrandOnly/NoGeneric", "Candidate"))
    pnf_name_set = set(pnf_df["generic_name"].dropna().map(_base_name).map(_normalize_text_basic))
    (df["combo_segments_total"], df["combo_id_molecules"], df["combo_unknown_in_pnf"], df["combo_unknown_in_who"], df["combo_unknown_in_both"]) = zip(*df.apply(lambda r: _analyze_combo_segments(r, pnf_name_set, who_name_set) if r.get("looks_combo_final") else (0,0,0,0,0), axis=1))
    df["normalized"] = df["norm"]
    return df, pnf_name_set, who_name_set
