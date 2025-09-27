#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, time, glob, os, re
from typing import Tuple, Optional, List, Dict, Set, Callable
import numpy as np, pandas as pd
from .aho import build_molecule_automata, scan_pnf_all
from .combos import SALT_TOKENS, looks_like_combination, split_combo_segments
from .routes_forms import extract_route_and_form
from .text_utils import _base_name, _normalize_text_basic, normalize_text, extract_parenthetical_phrases, STOPWORD_TOKENS
from .who_molecules import detect_all_who_molecules, load_who_molecules
from .brand_map import load_latest_brandmap, build_brand_automata, fda_generics_set
from .pnf_partial import PnfTokenIndex

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    import threading
    done = threading.Event()
    err = []

    def worker():
        try:
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"; i = 0
    while not done.is_set():
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r{frames[i % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        time.sleep(0.1); i += 1
    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return elapsed

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

def _tokenize_unknowns(s_norm: str) -> List[str]:
    return [m.group(0) for m in GENERIC_TOKEN_RX.finditer(s_norm)]

def build_features(
    pnf_df: pd.DataFrame,
    esoa_df: pd.DataFrame,
    *,
    timing_hook: Callable[[str, float], None] | None = None,
) -> pd.DataFrame:
    def _timed(label: str, func: Callable[[], None]) -> float:
        elapsed = _run_with_spinner(label, func)
        if timing_hook:
            timing_hook(label, elapsed)
        return elapsed

    # 1) Validate inputs
    def _validate():
        required_pnf = {
            "generic_id","generic_name","synonyms","atc_code","route_allowed","form_token",
            "dose_kind","strength","unit","per_val","per_unit","pct","strength_mg","ratio_mg_per_ml"
        }
        missing = required_pnf - set(pnf_df.columns)
        if missing:
            raise ValueError(f"pnf_prepared.csv missing columns: {missing}")
        if "raw_text" not in esoa_df.columns:
            raise ValueError("esoa_prepared.csv must contain a 'raw_text' column")
    _timed("Validate inputs", _validate)

    # 2) Map normalized PNF names to gid + original name
    pnf_name_to_gid: Dict[str, Tuple[str, str]] = {}
    def _pnf_map():
        for gid, gname in pnf_df[["generic_id","generic_name"]].drop_duplicates().itertuples(index=False):
            key = _normalize_text_basic(_base_name(str(gname)))
            if key and key not in pnf_name_to_gid:
                pnf_name_to_gid[key] = (gid, gname)
    _timed("Index PNF names", _pnf_map)
    pnf_name_set: Set[str] = set(pnf_name_to_gid.keys())

    # 3) WHO molecules (names + regex)
    codes_by_name, candidate_names = ({}, [])
    who_name_set: Set[str] = set()
    who_regex = [None]
    def _load_who():
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        who_file = load_latest_who_dir(root_dir)
        if who_file and os.path.exists(who_file):
            cbn, cand = load_who_molecules(who_file)
            codes_by_name.update(cbn)
            candidate_names.extend(cand)
            who_name_set.update(cbn.keys())
            who_regex[0] = re.compile(r"\b(" + "|".join(map(re.escape, candidate_names)) + r")\b") if candidate_names else None
    _timed("Load WHO molecules", _load_who)
    who_regex = who_regex[0]

    # 4) Brand map & FDA generics
    brand_df = [None]
    def _load_brand():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        inputs_dir = os.path.join(project_root, "inputs")
        brand_df[0] = load_latest_brandmap(inputs_dir)
    _timed("Load FDA brand map", _load_brand)
    has_brandmap = brand_df[0] is not None and not brand_df[0].empty
    if has_brandmap:
        B_norm = [None]; B_comp = [None]; brand_lookup = [{}]; fda_gens = [set()]
        _timed("Build brand automata", lambda: (
            (lambda r: (B_norm.__setitem__(0, r[0]), B_comp.__setitem__(0, r[1]), brand_lookup.__setitem__(0, r[2])))(build_brand_automata(brand_df[0]))
        ))
        _timed("Index FDA generics", lambda: fda_gens.__setitem__(0, fda_generics_set(brand_df[0])))
        B_norm, B_comp, brand_lookup, fda_gens = B_norm[0], B_comp[0], brand_lookup[0], fda_gens[0]
    else:
        B_norm = B_comp = None
        brand_lookup = {}
        fda_gens = set()

    # 5) PNF automata + partial index
    A_norm = [None]; A_comp = [None]
    _timed("Build PNF automata", lambda: (
        (lambda r: (A_norm.__setitem__(0, r[0]), A_comp.__setitem__(0, r[1])))(build_molecule_automata(pnf_df))
    ))
    pnf_partial_idx = [None]
    _timed("Build PNF partial index", lambda: pnf_partial_idx.__setitem__(0, PnfTokenIndex().build_from_pnf(pnf_df)))
    A_norm, A_comp, pnf_partial_idx = A_norm[0], A_comp[0], pnf_partial_idx[0]

    # 6) Base ESOA frame and text normalization
    df = [None]
    def _mk_base():
        tmp = esoa_df[["raw_text"]].copy()
        tmp["parentheticals"] = tmp["raw_text"].map(extract_parenthetical_phrases)
        tmp["esoa_idx"] = tmp.index
        tmp["normalized"] = tmp["raw_text"].map(normalize_text)
        tmp["norm_compact"] = tmp["normalized"].map(lambda s: re.sub(r"[ \-]", "", s))
        df.append(tmp)
    _timed("Normalize ESOA text", _mk_base)
    df = df[-1]

    # 7) Dose/route/form on original normalized text
    def _dose_route_form_raw():
        from .dose import extract_dosage as _extract_dosage
        df["dosage_parsed_raw"] = df["normalized"].map(_extract_dosage)
        df["dose_recognized"] = df["dosage_parsed_raw"].map(_friendly_dose)
        df["route_raw"], df["form_raw"], df["route_evidence_raw"] = zip(*df["normalized"].map(extract_route_and_form))
    _timed("Parse dose/route/form (raw)", _dose_route_form_raw)

    # 8) Brand → Generic swap (if brand map available)
    def _brand_swap():
        if not has_brandmap:
            df["match_basis"] = df["normalized"]
            df["did_brand_swap"] = False
            df["fda_dose_corroborated"] = False
            return
        mb_list, swapped = [], []
        fda_hits = []
        for norm, comp, form, friendly, parens in zip(
            df["normalized"], df["norm_compact"], df["form_raw"], df["dose_recognized"], df["parentheticals"]
        ):
            # Inline selection/scoring identical to prior implementation
            found_keys: List[str] = []
            lengths: Dict[str, int] = {}
            for _, bn in B_norm.iter(norm):  # type: ignore
                found_keys.append(bn); lengths[bn] = max(lengths.get(bn, 0), len(bn))
            for _, bn in B_comp.iter(comp):  # type: ignore
                found_keys.append(bn); lengths[bn] = max(lengths.get(bn, 0), len(bn))
            if not found_keys:
                mb_list.append(norm); swapped.append(False); fda_hits.append(False); continue
            uniq_keys = list(dict.fromkeys(found_keys))
            uniq_keys.sort(key=lambda k: (-lengths.get(k, len(k)), k))

            out = norm; replaced_any = False
            for bn in uniq_keys:
                options = brand_lookup.get(bn, [])
                chosen_generic = None
                if options:
                    # score
                    def _score(m):
                        sc = 0
                        if friendly and getattr(m, "dosage_strength", None) and friendly.lower() in (m.dosage_strength or "").lower(): sc += 2
                        if form and getattr(m, "dosage_form", None) and form.lower() == (m.dosage_form or "").lower(): sc += 1
                        gen_base = _normalize_text_basic(_base_name(m.generic))
                        if gen_base in pnf_name_to_gid: sc += 3
                        return sc
                    options = sorted(options, key=_score, reverse=True)
                    chosen_generic = options[0].generic if options else None
                if not chosen_generic:
                    continue
                gd_norm = normalize_text(chosen_generic)
                new_out = re.sub(rf"\b{re.escape(bn)}\b", gd_norm, out)
                if new_out != out:
                    replaced_any = True
                    out = new_out
            out = re.sub(r"\s+", " ", out).strip()
            # FDA dose corroboration
            fda_hit = False
            if options and friendly:
                ds = getattr(options[0], "dosage_strength", "") or ""
                if ds and friendly.lower() in ds.lower():
                    fda_hit = True
            mb_list.append(out); swapped.append(replaced_any); fda_hits.append(fda_hit)
        df["match_basis"] = mb_list
        df["did_brand_swap"] = swapped
        df["fda_dose_corroborated"] = fda_hits
    _timed("Apply brand→generic swaps", _brand_swap)
    df["match_basis_norm_basic"] = df["match_basis"].map(_normalize_text_basic)

    # 9) Dose/route/form on match_basis
    def _dose_route_form_basis():
        from .dose import extract_dosage as _extract_dosage
        df["dosage_parsed"] = df["match_basis"].map(_extract_dosage)
        df["route"], df["form"], df["route_evidence"] = zip(*df["match_basis"].map(extract_route_and_form))
    _timed("Parse dose/route/form (basis)", _dose_route_form_basis)

    # 10) PNF hits (Aho-Corasick) on match_basis
    def _pnf_hits():
        primary_gid, primary_token, pnf_hits_gids, pnf_hits_tokens, pnf_hits_count = [], [], [], [], []
        for s_norm, s_comp in zip(df["match_basis"], df["norm_compact"]):
            gids, tokens = scan_pnf_all(s_norm, s_comp, A_norm, A_comp)
            if tokens:
                salt_flags = []
                for tok in tokens:
                    base_tok = _normalize_text_basic(_base_name(tok))
                    words = base_tok.split()
                    salt_flags.append(bool(words) and all(word in SALT_TOKENS for word in words))
                if any(not flag for flag in salt_flags):
                    gids = [g for g, is_salt in zip(gids, salt_flags) if not is_salt]
                    tokens = [t for t, is_salt in zip(tokens, salt_flags) if not is_salt]
            pnf_hits_gids.append(gids); pnf_hits_tokens.append(tokens); pnf_hits_count.append(len(gids))
            if gids: primary_gid.append(gids[0]); primary_token.append(tokens[0])
            else: primary_gid.append(None); primary_token.append(None)
        df["pnf_hits_gids"] = pnf_hits_gids; df["pnf_hits_tokens"] = pnf_hits_tokens; df["pnf_hits_count"] = pnf_hits_count
        df["generic_id"] = primary_gid; df["molecule_token"] = primary_token
    _timed("Scan PNF hits", _pnf_hits)

    # 11) Partial PNF fallback
    def _pnf_partial():
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
    _timed("Partial PNF fallback", _pnf_partial)

    # 12) WHO molecule detection
    def _who_detect():
        if who_regex:
            who_names_all, who_atc_all = [], []
            for txt, txt_norm in zip(df["match_basis"].tolist(), df["match_basis_norm_basic"].tolist()):
                names, codes = detect_all_who_molecules(txt, who_regex, codes_by_name, pre_normalized=txt_norm)
                who_names_all.append(names)
                who_atc_all.append(sorted(codes))
            df["who_molecules_list"] = who_names_all
            df["who_atc_codes_list"] = who_atc_all
            df["who_molecules"] = df["who_molecules_list"].map(lambda xs: "|".join(xs) if xs else "")
            df["who_atc_codes"] = df["who_atc_codes_list"].map(lambda xs: "|".join(xs) if xs else "")
        else:
            df["who_molecules_list"] = [[] for _ in range(len(df))]
            df["who_atc_codes_list"] = [[] for _ in range(len(df))]
            df["who_molecules"] = ""
            df["who_atc_codes"] = ""
    _timed("Detect WHO molecules", _who_detect)

    # 13) Combination detection helpers
    def _combo_feats():
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
    _timed("Compute combo features", _combo_feats)

    # 14) Unknown tokens extraction
    def _unknowns():
        def _token_set_from_phrase_list(values: object) -> set[str]:
            tokens: set[str] = set()
            if isinstance(values, list):
                for phrase in values:
                    if isinstance(phrase, str) and phrase:
                        tokens.update(_tokenize_unknowns(_segment_norm(phrase)))
            elif isinstance(values, str) and values:
                tokens.update(_tokenize_unknowns(_segment_norm(values)))
            return tokens

        pnf_token_lookup: set[str] = set()
        for name in pnf_name_set:
            pnf_token_lookup.update(_tokenize_unknowns(name))

        who_token_lookup: set[str] = set()
        for name in who_name_set:
            who_token_lookup.update(_tokenize_unknowns(name))

        fda_token_lookup: set[str] = set()
        for name in fda_gens:
            fda_token_lookup.update(_tokenize_unknowns(name))

        def _unknown_kind_and_list(text_norm: str, matched_tokens: set[str]) -> Tuple[str, List[str]]:
            s = _segment_norm(text_norm)
            all_toks = _tokenize_unknowns(s)
            unknowns: list[str] = []
            for t in all_toks:
                if (
                    t not in matched_tokens
                    and t not in pnf_token_lookup
                    and t not in who_token_lookup
                    and t not in fda_token_lookup
                    and t not in STOPWORD_TOKENS
                ):
                    unknowns.append(t)
            seen: set[str] = set()
            unknowns_uniq: list[str] = []
            for t in unknowns:
                if t not in seen:
                    seen.add(t)
                    unknowns_uniq.append(t)
            if not unknowns_uniq:
                return "None", []
            if len(unknowns_uniq) == len(all_toks):
                if len(unknowns_uniq) == 1:
                    return "Single - Unknown", unknowns_uniq
                return "Multiple - All Unknown", unknowns_uniq
            return "Multiple - Some Unknown", unknowns_uniq

        results: list[Tuple[str, List[str]]] = []
        for text_norm, p_hits, who_hits in zip(df["match_basis"], df["pnf_hits_tokens"], df["who_molecules_list"]):
            matched_tokens = _token_set_from_phrase_list(p_hits) | _token_set_from_phrase_list(who_hits)
            results.append(_unknown_kind_and_list(text_norm, matched_tokens))

        if results:
            kinds, lists_ = zip(*results)
        else:
            kinds, lists_ = (), ()
        df["unknown_kind"] = list(kinds)
        df["unknown_words_list"] = list(lists_)
        df["unknown_words"] = df["unknown_words_list"].map(lambda xs: "|".join(xs) if xs else "")
    _timed("Extract unknown tokens", _unknowns)

    # 15) Presence flags
    def _presence_flags():
        df["present_in_pnf"] = df["pnf_hits_count"].astype(int).gt(0)
        df["present_in_who"] = df["who_atc_codes"].astype(str).str.len().gt(0)
        df["present_in_fda_generic"] = df["match_basis"].map(lambda s: any(tok in fda_gens for tok in _tokenize_unknowns(_segment_norm(s))))
    _timed("Compute presence flags", _presence_flags)

    return df
