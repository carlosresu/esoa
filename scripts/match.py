
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import math
import os
import re
from typing import Tuple, List

import numpy as np
import pandas as pd

from .aho import build_molecule_automata, scan_pnf_all
from .combos import looks_like_combination, split_combo_segments
from .dose import dose_similarity
from .routes_forms import extract_route_and_form
from .text_utils import _base_name, _normalize_text_basic, normalize_text, extract_parenthetical_phrases
from .who_molecules import detect_all_who_molecules, load_who_molecules

def _friendly_dose(d: dict) -> str:
    if not d:
        return ""
    kind = d.get("kind") or d.get("dose_kind")
    if kind == "amount":
        return f"{d.get('strength')}{d.get('unit','')}"
    if kind == "ratio":
        pv = d.get("per_val", 1)
        try:
            pv = int(pv)
        except Exception:
            pass
        return f"{d.get('strength')}{d.get('unit','')}/{pv}{d.get('per_unit','')}"
    if kind == "percent":
        return f"{d.get('pct')}%"
    return ""

def match(pnf_prepared_csv: str, esoa_prepared_csv: str, out_csv: str = "esoa_matched.csv") -> str:
    pnf_df = pd.read_csv(pnf_prepared_csv)
    esoa_df = pd.read_csv(esoa_prepared_csv)

    required_pnf = {"generic_id", "generic_name", "synonyms", "atc_code", "route_allowed",
                    "form_token", "dose_kind", "strength", "unit", "per_val", "per_unit", "pct",
                    "strength_mg", "ratio_mg_per_ml"}
    missing = required_pnf - set(pnf_df.columns)
    if missing:
        raise ValueError(f"{pnf_prepared_csv} missing columns: {missing}")
    if "raw_text" not in esoa_df.columns:
        raise ValueError(f"{esoa_prepared_csv} must contain a 'raw_text' column")

    A_norm, A_comp = build_molecule_automata(pnf_df)

    df = esoa_df[["raw_text"]].copy()
    df["esoa_idx"] = df.index
    df["norm"] = df["raw_text"].map(normalize_text)
    df["norm_compact"] = df["norm"].map(lambda s: re.sub(r"[ \-]", "", s))

    # Parenthetical probable brands before normalization loss
    df["probable_brands_list"] = df["raw_text"].map(extract_parenthetical_phrases)
    df["probable_brands"] = df["probable_brands_list"].map(lambda xs: "|".join(xs) if xs else "")

    primary_gid, primary_token, pnf_hits_gids, pnf_hits_tokens, pnf_hits_count = [], [], [], [], []
    for s_norm, s_comp in zip(df["norm"], df["norm_compact"]):
        gids, tokens = scan_pnf_all(s_norm, s_comp, A_norm, A_comp)
        pnf_hits_gids.append(gids)
        pnf_hits_tokens.append(tokens)
        pnf_hits_count.append(len(gids))
        if gids:
            primary_gid.append(gids[0])
            primary_token.append(tokens[0])
        else:
            primary_gid.append(None)
            primary_token.append(None)

    df["pnf_hits_gids"] = pnf_hits_gids
    df["pnf_hits_tokens"] = pnf_hits_tokens
    df["pnf_hits_count"] = pnf_hits_count
    df["generic_id"] = primary_gid
    df["molecule_token"] = primary_token

    # Dose & route/form
    from .dose import extract_dosage as _extract_dosage
    df["dosage_parsed"] = df["norm"].map(_extract_dosage)
    df["dose_recognized"] = df["dosage_parsed"].map(_friendly_dose)
    df["route"], df["form"], df["route_evidence"] = zip(*df["norm"].map(extract_route_and_form))

    # WHO molecules
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    who_dir = os.path.join(root_dir, "dependencies", "atcd", "output")
    candidates = glob.glob(os.path.join(who_dir, "who_atc_*_molecules.csv"))
    who_file = max(candidates, key=os.path.getmtime) if candidates else None

    who_name_set = set()
    if who_file and os.path.exists(who_file):
        print(f"[not_in_pnf] Using WHO molecules file: {who_file}")
        codes_by_name, candidate_names = load_who_molecules(who_file)
        who_name_set = set(codes_by_name.keys())
        regex = re.compile(r"\b(" + "|".join(map(re.escape, candidate_names)) + r")\b")
        who_names_all, who_atc_all = [], []
        for txt in df["norm"].tolist():
            names, codes = detect_all_who_molecules(txt, regex, codes_by_name)
            who_names_all.append(names)
            who_atc_all.append(sorted(codes))
        df["who_molecules_list"] = who_names_all
        df["who_atc_codes_list"] = who_atc_all
        df["who_molecules"] = df["who_molecules_list"].map(lambda xs: "|".join(xs) if xs else "")
        df["who_atc_codes"] = df["who_atc_codes_list"].map(lambda xs: "|".join(xs) if xs else "")
    else:
        print("[not_in_pnf] WHO molecules file not found, skipping.")
        df["who_molecules_list"] = [[] for _ in range(len(df))]
        df["who_atc_codes_list"] = [[] for _ in range(len(df))]
        df["who_molecules"] = ""
        df["who_atc_codes"] = ""

    # Combination logic
    looks_raw = [
        looks_like_combination(s, p_cnt, len(w_names))
        for s, p_cnt, w_names in zip(df["norm"], df["pnf_hits_count"], df["who_molecules_list"])
    ]
    df["looks_combo_raw"] = looks_raw
    df["looks_combo_final"] = looks_raw
    df["combo_reason"] = np.where(df["looks_combo_final"], "combo/heuristic", "single/heuristic")

    # Initial bucket
    df["bucket"] = np.where(
        df["looks_combo_final"], "Others:Combinations",
        np.where(df["generic_id"].isna(), "BrandOnly/NoGeneric", "Candidate"),
    )

    # --- Combo segment analysis ---
    DOSE_OR_UNIT_RX = re.compile(r"(?:(\b\d+(?:[\.,]\d+)?\s*(?:mg|g|mcg|ug|iu|lsu|ml|l|%)(?:\b|/))|(\b\d+(?:[\.,]\d+)?\b))",
                                 re.IGNORECASE)

    def _segment_norm(seg: str) -> str:
        s = _normalize_text_basic(_base_name(seg))
        s = DOSE_OR_UNIT_RX.sub(" ", s)
        s = re.sub(r"\b(?:per|with|and)\b", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def analyze_combo_segments(row: pd.Series):
        segs = split_combo_segments(row.get("normalized", ""))
        norms = [_segment_norm(s) for s in segs if s]
        in_pnf = [n in set(pnf_df["generic_name"].map(_base_name).map(_normalize_text_basic)) for n in norms]
        in_who = [n in who_name_set for n in norms]
        id_count = sum(1 for pnff, whof in zip(in_pnf, in_who) if pnff or whof)
        unk_in_pnf = sum(1 for pnff, whof in zip(in_pnf, in_who) if not pnff and whof)
        unk_in_who = sum(1 for pnff, whof in zip(in_pnf, in_who) if pnff and not whof)
        return len(norms), id_count, unk_in_pnf, unk_in_who

    df["combo_segments_total"], df["combo_id_molecules"], df["combo_unknown_in_pnf"], df["combo_unknown_in_who"] =         zip(*df.apply(lambda r: analyze_combo_segments(r) if r.get("looks_combo_final") else (0,0,0,0), axis=1))

    # Candidate subset for dose/route scoring
    df_cand = df.loc[df["bucket"].eq("Candidate"), ["esoa_idx", "generic_id", "route", "form", "dosage_parsed"]].merge(
        pnf_df, on="generic_id", how="left"
    )

    def route_ok(row):
        r = row["route"]
        allowed = row.get("route_allowed")
        if pd.isna(r) or not r:
            return True
        if isinstance(allowed, str) and allowed:
            return r in allowed.split("|")
        return True

    if not df_cand.empty:
        df_cand = df_cand[df_cand.apply(route_ok, axis=1)]
        df_cand["dose_sim"] = df_cand.apply(lambda r: dose_similarity(r["dosage_parsed"], r), axis=1)
        df_cand["dose_sim"] = pd.to_numeric(df_cand["dose_sim"], errors="coerce").fillna(0.0)
    else:
        df_cand = pd.DataFrame(columns=["esoa_idx", "dose_sim"])

    def pick_best(group: pd.DataFrame):
        if group.empty:
            return pd.Series({
                "atc_code_final": None,
                "match_note": "no route/dose match",
                "selected_form": None,
                "selected_variant": None,
                "dose_sim": 0.0,
            })
        esoa_form = group.iloc[0]["form"]
        esoa_route = group.iloc[0]["route"]
        esoa_dose = group.iloc[0]["dosage_parsed"]

        scored = []
        for _, row in group.iterrows():
            score = 0.0
            if esoa_form and row.get("form_token") and esoa_form == row["form_token"]:
                score += 40
            if esoa_route and row.get("route_allowed") and esoa_route == row["route_allowed"]:
                score += 30
            sim = dose_similarity(esoa_dose, row)
            score += sim * 30
            scored.append((score, sim, row))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, best_sim, best_row = scored[0]

        note = "weak dose match" if best_sim >= 0.6 else "no/poor dose match"

        strength = best_row.get("strength"); unit = best_row.get("unit") or ""
        per_val = best_row.get("per_val"); per_unit = best_row.get("per_unit") or ""
        pct = best_row.get("pct")
        variant = f"{best_row.get('dose_kind')}:{strength}{unit}" if pd.notna(strength) else str(best_row.get("dose_kind"))
        if pd.notna(per_val):
            try:
                pv_int = int(per_val)
            except Exception:
                pv_int = per_val
            variant += f"/{pv_int}{per_unit}"
        if pd.notna(pct):
            variant += f" {pct}%"

        return pd.Series({
            "atc_code_final": best_row["atc_code"] if isinstance(best_row["atc_code"], str) and best_row["atc_code"] else None,
            "match_note": note,
            "selected_form": best_row.get("form_token"),
            "selected_variant": variant,
            "dose_sim": float(best_sim),
        })

    if not df_cand.empty:
        best_by_idx = df_cand.groupby("esoa_idx", sort=False).apply(pick_best, include_groups=False)
        out = df.merge(best_by_idx, left_on="esoa_idx", right_index=True, how="left")
    else:
        out = df.copy()
        out[["atc_code_final", "match_note", "selected_form", "selected_variant", "dose_sim"]] = [None, "no route/dose match", None, None, 0.0]

    if "match_note" not in out.columns:
        out["match_note"] = np.where(out["bucket"].eq("Candidate"), "no route/dose match", "")

    def score_row(r):
        score = 0
        if pd.notna(r.get("generic_id")):
            score += 60
        if r.get("dosage_parsed"):
            score += 15
        if r.get("route_evidence"):
            score += 10
        if pd.notna(r.get("atc_code_final")):
            score += 15
        sim = r.get("dose_sim")
        try:
            sim = float(sim)
        except Exception:
            sim = 0.0
        if math.isnan(sim):
            sim = 0.0
        score += int(max(0.0, min(1.0, sim)) * 10)
        return score

    out["confidence"] = out.apply(score_row, axis=1)

    between_mask = (out["confidence"] >= 70) & (out["confidence"] <= 89)
    out["bucket"] = np.where(out["bucket"].eq("Candidate") & (out["confidence"] >= 90), "Auto-Accept",
                       np.where(out["bucket"].eq("Candidate") & between_mask, "Candidate", out["bucket"]))

    out["normalized"] = out["norm"]

    # Present in PNF/WHO flags & molecules recognized
    pnf_basenorm_set = set(pnf_df["generic_name"].dropna().map(_base_name).map(_normalize_text_basic))

    out["present_in_pnf"] = out["pnf_hits_count"].astype(int).gt(0)
    out["present_in_who"] = out["who_atc_codes"].astype(str).str.len().gt(0)

    def _union_molecules(row) -> List[str]:
        names = []
        for t in (row.get("pnf_hits_tokens") or []):
            if not isinstance(t, str):
                continue
            names.append(_normalize_text_basic(_base_name(t)))
        for t in (row.get("who_molecules_list") or []):
            if not isinstance(t, str):
                continue
            names.append(_normalize_text_basic(_base_name(t)))
        seen = set(); uniq = []
        for n in names:
            if not n or n in seen:
                continue
            seen.add(n); uniq.append(n)
        return uniq

    out["molecules_recognized_list"] = out.apply(_union_molecules, axis=1)
    out["molecules_recognized"] = out["molecules_recognized_list"].map(lambda xs: "|".join(xs) if xs else "")

    out["probable_atc"] = np.where(~out["present_in_pnf"] & out["present_in_who"], out["who_atc_codes"], "")

    def classify_combo_reason(row: pd.Series) -> str:
        if row.get("combo_segments_total", 0) > 1:
            return (
                f"Combinations: Contains Unknown(s) — "
                f"identified={row['combo_id_molecules']}, "
                f"unknown_in_pnf={row['combo_unknown_in_pnf']}, "
                f"unknown_in_who={row['combo_unknown_in_who']}"
            )
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
    final_why.loc[brand_mask] = "BrandOnly/NoGenericInPNForWHO"

    any_who_codes = out["who_atc_codes"].astype(str).str.len().gt(0)
    any_not_in_pnf = out["who_molecules_list"].map(
        lambda names: any((n not in pnf_basenorm_set) for n in (names or []))
    )
    vm_mask = any_who_codes & any_not_in_pnf
    if vm_mask.any():
        subreason = np.where(
            vm_mask & (out["pnf_hits_count"] >= 1) & out["looks_combo_final"],
            "2+ generics but only 1+ in PNF",
            "not in PNF",
        )
        final_bucket.loc[vm_mask] = "Needs review"
        sr = pd.Series(subreason, index=out.index)[vm_mask]
        final_why.loc[vm_mask] = "ValidMoleculeWithATC/NotInPNF:" + sr

    out["bucket_final"] = final_bucket
    out["why_final"] = final_why.fillna("")

    out_small = out[[
        "raw_text", "normalized",
        "molecules_recognized", "probable_brands", "dose_recognized",
        "route", "form",
        "present_in_pnf", "present_in_who", "probable_atc",
        "pnf_hits_tokens", "who_molecules", "who_atc_codes",
        "route_evidence", "looks_combo_raw", "looks_combo_final", "combo_reason",
        "atc_code_final", "confidence", "bucket_final", "why_final",
    ]].copy()

    out_small.to_csv(out_csv, index=False, encoding="utf-8")
    total = len(out_small)
    print(f"[match] wrote {out_csv} with {total:,} rows")

    xlsx_out = os.path.splitext(out_csv)[0] + ".xlsx"
    try:
        with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
            out_small.to_excel(writer, index=False, sheet_name="matched")
            ws = writer.sheets["matched"]
            ws.freeze_panes(1, 0)
            nrows, ncols = out_small.shape
            ws.autofilter(0, 0, nrows, ncols - 1)
    except Exception:
        with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
            out_small.to_excel(writer, index=False, sheet_name="matched")
            ws = writer.sheets["matched"]
            try:
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions
            except Exception:
                pass
    print(f"[match] wrote {xlsx_out} with filters and frozen header")

    grouped = (
        out_small.groupby(["bucket_final", "why_final"], dropna=False)
                 .size()
                 .reset_index(name="n")
    )
    grouped["pct"] = (grouped["n"] / float(total) * 100).round(2)

    print("\n[summary] Distribution")
    aa = grouped[grouped["bucket_final"].eq("Auto-Accept")]["n"].sum()
    aa_pct = round(aa / float(total) * 100, 2) if total else 0.0
    print(f"  - Auto-Accept: {aa:,} ({aa_pct}%)")

    nr = grouped[grouped["bucket_final"].eq("Needs review")].copy()
    if not nr.empty:
        def fam_order(s):
            if s.startswith("ValidMoleculeWithATC/NotInPNF:"):
                return (0, s)
            if s.startswith("Candidate:"):
                return (1, s)
            if s == "BrandOnly/NoGenericInPNForWHO":
                return (2, s)
            return (3, s)
        nr = nr.sort_values(by=["why_final"], key=lambda c: c.map(fam_order))
        print(f"  - Needs review:")
        for _, row in nr.iterrows():
            print(f"      • {row['why_final']}: {row['n']:,} ({row['pct']}%)")

    oth = grouped[grouped["bucket_final"].eq("Others")].copy()
    if not oth.empty:
        total_oth = oth["n"].sum()
        total_oth_pct = round(total_oth / float(total) * 100, 2) if total else 0.0
        print(f"  - Others:")
        for _, row in oth.sort_values("n", ascending=False).iterrows():
            label = row["why_final"] if row["why_final"] else "Combinations"
            print(f"      • {label}: {row['n']:,} ({row['pct']}%)")

    return out_csv
