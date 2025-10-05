#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature engineering stage responsible for every signal used by scoring."""

from __future__ import annotations
import sys, time, glob, os, re
from collections import defaultdict
from typing import Tuple, Optional, List, Dict, Set, Callable
import difflib
import numpy as np, pandas as pd
import ahocorasick  # type: ignore
from .aho import build_molecule_automata, scan_pnf_all
from .combos import SALT_TOKENS, looks_like_combination, split_combo_segments
from .routes_forms import extract_route_and_form
from .text_utils import (
    _base_name,
    _normalize_text_basic,
    normalize_text,
    extract_parenthetical_phrases,
    STOPWORD_TOKENS,
)
from .who_molecules import detect_all_who_molecules, load_who_molecules
from .brand_map import load_latest_brandmap, build_brand_automata, fda_generics_set
from .pnf_aliases import expand_generic_aliases, SPECIAL_GENERIC_ALIASES, apply_spelling_rules
from .pnf_partial import PnfTokenIndex

# Reference dictionaries describing canonical route/form mappings leveraged
# throughout the feature builder.  Keeping them in one place simplifies policy
# reviews when clinicians update acceptable substitutions.
WHO_ADM_ROUTE_MAP: dict[str, set[str]] = {
    "o": {"oral"},
    "oral": {"oral"},
    "chewing gum": {"oral"},
    "p": {"intravenous", "intramuscular", "subcutaneous"},
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

WHO_FORM_TO_CANONICAL: dict[str, set[str]] = {
    "chewing gum": {"tablet"},
    "inhal.aerosol": {"mdi"},
    "inhal.powder": {"dpi"},
    "inhal.solution": {"solution"},
    "instill.solution": {"solution"},
    "oral aerosol": {"mdi"},
    "ointment": {"ointment"},
    "lamella": {"solution"},
    "implant": {"solution"},
    "s.c. implant": {"solution"},
}

FOOD_FORM_KEYWORDS: dict[str, set[str]] = {
    "capsule": {"capsule", "cap"},
    "tablet": {"tablet", "tab"},
    "syrup": {"syrup"},
    "solution": {"solution", "sol"},
    "suspension": {"suspension", "susp"},
    "powder": {"powder"},
    "gel": {"gel"},
    "lotion": {"lotion"},
    "cream": {"cream"},
    "ointment": {"ointment", "oint"},
    "spray": {"spray"},
    "drops": {"drops", "drop"},
    "drink": {"drink", "beverage"},
}

FOOD_ROUTE_KEYWORDS: dict[str, set[str]] = {
    "oral": {"oral", "po", "by mouth"},
    "topical": {"topical", "dermal"},
    "nasal": {"nasal"},
    "ophthalmic": {"ophthalmic", "eye"},
    "otic": {"otic", "ear"},
    "inhalation": {"inhalation", "inhaled"},
}

WHO_UOM_TO_CANONICAL: dict[str, set[str]] = {
    "tablet": {"tablet"},
    "ml": {"solution", "suspension"},
    "mg": {"tablet", "capsule", "solution", "injection", "suspension"},
    "g": {"tablet", "capsule", "solution", "injection", "suspension"},
    "mcg": {"tablet", "capsule", "solution", "injection", "suspension"},
    "mmol": {"tablet", "solution"},
    "u": {"tablet", "solution", "injection"},
    "mu": {"tablet", "solution", "injection"},
    "tu": {"tablet", "solution", "injection"},
    "lsu": {"tablet", "solution"},
}

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    """Mirror the CLI spinner locally so the feature builder stays self-contained."""
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
    """Render a human-friendly dose string from a structured dose payload."""
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
    """Normalize a combo segment by stripping doses, stopwords, and punctuation."""
    s = _normalize_text_basic(_base_name(seg))
    s = DOSE_OR_UNIT_RX.sub(" ", s)
    s = re.sub(r"\b(?:per|with|and)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_latest_who_dir(root_dir: str) -> str | None:
    """Locate the newest WHO ATC export living under the project dependencies folder."""
    who_dir = os.path.join(root_dir, "dependencies", "atcd", "output")
    candidates = glob.glob(os.path.join(who_dir, "who_atc_*_molecules.csv"))
    return max(candidates, key=os.path.getmtime) if candidates else None

def _tokenize_unknowns(s_norm: str) -> List[str]:
    """Break normalized candidate strings into tokens for unknown-word diagnostics."""
    return [m.group(0) for m in GENERIC_TOKEN_RX.finditer(s_norm)]

def build_features(
    pnf_df: pd.DataFrame,
    esoa_df: pd.DataFrame,
    *,
    timing_hook: Callable[[str, float], None] | None = None,
) -> pd.DataFrame:
    """Derive the expansive feature frame used downstream for scoring: validation, vocabulary indexing, brand swaps, re-parsed dose/route/form, molecule detection, combo flags, and unknown token extraction."""
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
    pnf_alias_keys: List[str] = []
    pnf_trigram_holder: List[Dict[str, Set[str]] | None] = [None]
    def _generate_trigrams(value: str) -> Set[str]:
        segment = re.sub(r"\s+", "", value)
        if not segment:
            return set()
        if len(segment) <= 3:
            return {segment}
        return {segment[i:i+3] for i in range(len(segment) - 2)}

    def _pnf_map():
        trigram_map: Dict[str, Set[str]] = defaultdict(set)
        for gid, gname in pnf_df[["generic_id","generic_name"]].drop_duplicates().itertuples(index=False):
            alias_set = expand_generic_aliases(str(gname))
            alias_set |= SPECIAL_GENERIC_ALIASES.get(gid, set())
            syns = pnf_df.loc[pnf_df["generic_id"] == gid, "synonyms"].dropna().astype(str)
            for syn in syns:
                alias_set |= expand_generic_aliases(syn)
            for alias in alias_set:
                key = _normalize_text_basic(alias)
                if key and key not in pnf_name_to_gid:
                    # Store the first-seen generic mapping so duplicates do not overwrite earlier entries.
                    pnf_name_to_gid[key] = (gid, gname)
                if key:
                    for tri in _generate_trigrams(key):
                        if tri:
                            trigram_map[tri].add(key)
        pnf_alias_keys.clear()
        pnf_alias_keys.extend(sorted(pnf_name_to_gid.keys()))
        pnf_trigram_holder[0] = {tri: values for tri, values in trigram_map.items()}
    _timed("Index PNF names", _pnf_map)
    pnf_name_set: Set[str] = set(pnf_name_to_gid.keys())
    pnf_tokens: Set[str] = set()
    for name in pnf_name_set:
        pnf_tokens.update(_tokenize_unknowns(name))
    pnf_trigram_index: Dict[str, Set[str]] = pnf_trigram_holder[0] or {}

    # 3) WHO molecules (names + regex)
    codes_by_name, candidate_names = ({}, [])
    who_details_by_code: Dict[str, List[dict]] = {}
    who_name_set: Set[str] = set()
    who_regex_box = [None]
    def _load_who():
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        who_file = load_latest_who_dir(root_dir)
        if who_file and os.path.exists(who_file):
            cbn, cand, details = load_who_molecules(who_file)
            codes_by_name.update(cbn)
            candidate_names.extend(cand)
            who_name_set.update(cbn.keys())
            who_details_by_code.update(details)
            if candidate_names:
                # Compile a single regex that captures any candidate name at word boundaries.
                who_regex_box[0] = re.compile(r"\b(" + "|".join(map(re.escape, candidate_names)) + r")\b")
    _timed("Load WHO molecules", _load_who)
    who_regex = who_regex_box[0]
    who_tokens: Set[str] = set()
    for name in who_name_set:
        who_tokens.update(_tokenize_unknowns(name))

    # 4) Brand map & FDA generics
    brand_df = [None]
    def _load_brand():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        inputs_dir = os.path.join(project_root, "inputs")
        # Attempt to load the newest brand map for downstream substitutions.
        brand_df[0] = load_latest_brandmap(inputs_dir)
    _timed("Load FDA brand map", _load_brand)
    has_brandmap = brand_df[0] is not None and not brand_df[0].empty
    brand_token_lookup: Set[str] = set()
    if has_brandmap:
        B_norm = [None]; B_comp = [None]; brand_lookup = [{}]; fda_gens = [set()]
        _timed("Build brand automata", lambda: (
            (lambda r: (B_norm.__setitem__(0, r[0]), B_comp.__setitem__(0, r[1]), brand_lookup.__setitem__(0, r[2])))(build_brand_automata(brand_df[0]))
        ))
        _timed("Index FDA generics", lambda: fda_gens.__setitem__(0, fda_generics_set(brand_df[0])))
        B_norm, B_comp, brand_lookup, fda_gens = B_norm[0], B_comp[0], brand_lookup[0], fda_gens[0]

        brand_series = brand_df[0].get("brand_name")
        if brand_series is not None:
            for name in brand_series.fillna("").astype(str):
                norm = _normalize_text_basic(_base_name(name))
                if norm:
                    brand_token_lookup.update(_tokenize_unknowns(norm))
    else:
        B_norm = B_comp = None
        brand_lookup = {}
        fda_gens = set()
        brand_token_lookup = set()

    fda_tokens: Set[str] = set()
    for name in fda_gens:
        fda_tokens.update(_tokenize_unknowns(name))

    therapeutic_tokens: Set[str] = pnf_tokens | who_tokens | fda_tokens

    # 4b) FDA food/non-therapeutic catalog (optional, may not exist locally)
    nonthera_lookup: Dict[str, List[Dict[str, str]]] = {}
    nonthera_token_lookup: Dict[str, Set[str]] = {}
    nonthera_loaded = [False]
    nonthera_automaton = [None]

    def _load_nontherapeutic_catalog():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        inputs_dir = os.path.join(project_root, "inputs")
        catalog_path = os.path.join(inputs_dir, "fda_food_products.csv")
        if not os.path.exists(catalog_path):
            return
        try:
            df_catalog = pd.read_csv(catalog_path)
        except Exception:
            return
        if df_catalog.empty:
            return

        keys: Set[str] = set()
        for row in df_catalog.fillna("").to_dict("records"):
            brand = str(row.get("brand_name", "") or "").strip()
            product = str(row.get("product_name", "") or "").strip()
            company = str(row.get("company_name", "") or "").strip()
            regno = str(row.get("registration_number", "") or "").strip()

            for field_name, raw_value in (("brand_name", brand), ("product_name", product)):
                if not raw_value:
                    continue
                norm = _normalize_text_basic(_base_name(raw_value))
                if not norm:
                    continue
                tokens_raw = {m.group(0).lower() for m in GENERIC_TOKEN_RX.finditer(norm)}
                filtered_tokens = {
                    tok
                    for tok in tokens_raw
                    if len(tok) >= 3
                    and tok not in therapeutic_tokens
                    and tok not in STOPWORD_TOKENS
                    and tok not in brand_token_lookup
                }
                if not filtered_tokens:
                    continue
                keys.add(norm)
                detail = {
                    "brand_name": brand,
                    "product_name": product,
                    "company_name": company,
                    "registration_number": regno,
                    "match_field": field_name,
                    "match_value": raw_value,
                }
                nonthera_lookup.setdefault(norm, []).append(detail)
                existing = nonthera_token_lookup.setdefault(norm, set())
                existing.update(filtered_tokens)

        if keys:
            auto = ahocorasick.Automaton()
            for key in keys:
                auto.add_word(key, key)
            auto.make_automaton()
            nonthera_automaton[0] = auto
            nonthera_loaded[0] = True

    _timed("Load FDA non-therapeutic catalog", _load_nontherapeutic_catalog)

    # 5) PNF automata + partial index
    A_norm = [None]; A_comp = [None]
    _timed("Build PNF automata", lambda: (
        (lambda r: (A_norm.__setitem__(0, r[0]), A_comp.__setitem__(0, r[1])))(build_molecule_automata(pnf_df))
    ))
    pnf_partial_idx = [None]
    _timed("Build PNF partial index", lambda: pnf_partial_idx.__setitem__(0, PnfTokenIndex().build_from_pnf(pnf_df)))
    A_norm, A_comp, pnf_partial_idx = A_norm[0], A_comp[0], pnf_partial_idx[0]
    for key, (gid, _display) in pnf_name_to_gid.items():
        if key:
            pnf_partial_idx.add(gid, key)

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

    def _detect_non_therapeutic():
        if not nonthera_loaded[0] or not nonthera_automaton[0]:
            df["non_therapeutic_hits"] = [[] for _ in range(len(df))]
            df["non_therapeutic_tokens"] = [[] for _ in range(len(df))]
            df["non_therapeutic_summary"] = ["" for _ in range(len(df))]
            df["non_therapeutic_detail"] = ["" for _ in range(len(df))]
            return

        auto = nonthera_automaton[0]
        hits_col: List[List[Dict[str, str]]] = []
        tokens_col: List[List[str]] = []
        summary_col: List[str] = []
        detail_col: List[str] = []

        for norm_text in df["normalized"].astype(str).tolist():
            if not norm_text:
                hits_col.append([])
                tokens_col.append([])
                summary_col.append("")
                detail_col.append("")
                continue

            matched_keys: Set[str] = set()
            for end_idx, key in auto.iter(norm_text):  # type: ignore[attr-defined]
                if key not in nonthera_token_lookup:
                    continue
                start_idx = end_idx - len(key) + 1
                if start_idx > 0 and norm_text[start_idx - 1].isalnum():
                    continue
                if end_idx + 1 < len(norm_text) and norm_text[end_idx + 1].isalnum():
                    continue
                matched_keys.add(key)
            if not matched_keys:
                hits_col.append([])
                tokens_col.append([])
                summary_col.append("")
                detail_col.append("")
                continue

            details: List[Dict[str, str]] = []
            token_set: Set[str] = set()
            summary_parts: List[str] = []
            for key in matched_keys:
                for entry in nonthera_lookup.get(key, []):
                    details.append(entry)
                    tokens_for_key = nonthera_token_lookup.get(key)
                    if tokens_for_key:
                        token_set.update(tokens_for_key)
                    display = (entry.get("match_value") or entry.get("brand_name") or entry.get("product_name") or key).strip()
                    regno = (entry.get("registration_number") or "").strip()
                    if regno:
                        summary_parts.append(f"{display} [Reg {regno}]")
                    else:
                        summary_parts.append(display)

            unique_parts = list(dict.fromkeys(part for part in summary_parts if part))
            summary = "Matches FDA PH food catalog: " + "; ".join(unique_parts) if unique_parts else ""

            hits_col.append(details)
            tokens_col.append(sorted(token_set))
            summary_col.append(summary)
            detail_col.append(summary)

        df["non_therapeutic_hits"] = hits_col
        df["non_therapeutic_tokens"] = tokens_col
        df["non_therapeutic_summary"] = summary_col
        df["non_therapeutic_detail"] = detail_col

    _timed("Detect non-therapeutic brands", _detect_non_therapeutic)

    def _score_food_entry(entry: Dict[str, str], norm_text: str, form_raw: Optional[str], route_raw: Optional[str]) -> float:
        norm_lower = norm_text.lower()
        norm_tokens = set(GENERIC_TOKEN_RX.findall(norm_lower))
        score = 0.0

        def _token_score(text: Optional[str], weight: float) -> float:
            if not text:
                return 0.0
            text_lower = text.lower()
            tokens = set(GENERIC_TOKEN_RX.findall(text_lower))
            overlap = len(tokens & norm_tokens)
            bonus = weight * 2 if text_lower and text_lower in norm_lower else 0.0
            return weight * overlap + bonus

        brand = entry.get("brand_name", "")
        product = entry.get("product_name", "")
        company = entry.get("company_name", "")
        combined = " ".join(filter(None, [brand, product, company])).lower()

        score += _token_score(brand, 3.0)
        score += _token_score(product, 2.0)
        score += _token_score(company, 1.0)

        form_lower = (form_raw or "").lower()
        for kw, candidates in FOOD_FORM_KEYWORDS.items():
            if kw in combined:
                if any(cand in form_lower for cand in candidates):
                    score += 4.0
                elif any(cand in norm_lower for cand in candidates):
                    score += 2.0

        route_lower = (route_raw or "").lower()
        for kw, routes in FOOD_ROUTE_KEYWORDS.items():
            if kw in combined and route_lower in routes:
                score += 3.0

        for num in re.findall(r"\d+(?:\.\d+)?", combined):
            if num and num in norm_lower:
                score += 1.0

        if entry.get("registration_number"):
            score += 1.0

        return score

    def _choose_best_nonthera() -> None:
        if "non_therapeutic_hits" not in df.columns:
            df["non_therapeutic_best"] = [{} for _ in range(len(df))]
            return

        existing_summary = df.get("non_therapeutic_summary", pd.Series(["" for _ in range(len(df))])).tolist()
        best_entries: List[Dict[str, str]] = []
        summaries: List[str] = []
        details: List[str] = []

        for idx, (hits, norm_text, form_raw, route_raw) in enumerate(
            zip(
                df["non_therapeutic_hits"],
                df["normalized"],
                df.get("form_raw", [None] * len(df)),
                df.get("route_raw", [None] * len(df)),
            )
        ):
            if not hits:
                best_entries.append({})
                summaries.append(existing_summary[idx] if idx < len(existing_summary) else "")
                details.append("")
                continue

            scores = []
            for entry in hits:
                scores.append((entry, _score_food_entry(entry, str(norm_text), form_raw, route_raw)))
            scores.sort(key=lambda item: (-item[1], item[0].get("brand_name", ""), item[0].get("product_name", "")))
            best_entry, best_score = scores[0] if scores else ({}, 0.0)

            if best_score <= 0.0:
                best_entries.append(best_entry)
                summaries.append(existing_summary[idx] if idx < len(existing_summary) else "")
                details.append("")
                continue

            display = best_entry.get("brand_name") or best_entry.get("product_name") or best_entry.get("company_name") or "FDA Food item"
            detail_parts = []
            if best_entry.get("product_name") and best_entry.get("product_name").strip().lower() != display.strip().lower():
                detail_parts.append(best_entry["product_name"].strip())
            if best_entry.get("company_name"):
                detail_parts.append(best_entry["company_name"].strip())
            if best_entry.get("registration_number"):
                detail_parts.append(f"Reg {best_entry['registration_number'].strip()}")
            detail = "; ".join(part for part in detail_parts if part)
            detail_string = f"FDA Food match: {display}" + (f" ({detail})" if detail else "")

            best_entries.append(best_entry)
            summaries.append("non_therapeutic_detected")
            details.append(detail_string)

        df["non_therapeutic_best"] = best_entries
        df["non_therapeutic_summary"] = summaries
        df["non_therapeutic_detail"] = details

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
            df["probable_brands"] = ""
            df["fda_generics_list"] = [[] for _ in range(len(df))]
            return
        mb_list, swapped = [], []
        fda_hits = []
        probable_brand_hits: List[str] = []
        fda_generics_lists: List[List[str]] = []

        def _clean_generic_for_swap(name: str) -> str:
            base = _base_name(name)
            base = re.sub(r"\(.*?\)", " ", base)
            base = re.split(r"\bas\b", base, 1)[0]
            base = re.sub(r"\s+", " ", base).strip()
            # Strips parenthetical brand annotations so swaps don't rewrite contextual notes (README: avoid touching parentheses when generic already present).
            return base or _base_name(name)

        for norm, comp, form, friendly, parens in zip(
            df["normalized"], df["norm_compact"], df["form_raw"], df["dose_recognized"], df["parentheticals"]
        ):
            # Scan each row for brand hits across both normalized and compact automatons.
            # Inline selection/scoring identical to prior implementation
            found_keys: List[str] = []
            lengths: Dict[str, int] = {}
            row_brand_labels: set[str] = set()
            row_generics_raw: List[str] = []
            for _, bn in B_norm.iter(norm):  # type: ignore
                found_keys.append(bn); lengths[bn] = max(lengths.get(bn, 0), len(bn))
            for _, bn in B_comp.iter(comp):  # type: ignore
                found_keys.append(bn); lengths[bn] = max(lengths.get(bn, 0), len(bn))
            if not found_keys:
                mb_list.append(norm); swapped.append(False); fda_hits.append(False); probable_brand_hits.append(""); fda_generics_lists.append([]); continue
            uniq_keys = list(dict.fromkeys(found_keys))
            # Prioritize longer matches first to favor more specific brand tokens.
            uniq_keys.sort(key=lambda k: (-lengths.get(k, len(k)), k))

            out = norm; replaced_any = False
            primary_option = None
            for bn in uniq_keys:
                options = brand_lookup.get(bn, [])
                chosen_generic = None
                if options:
                    for opt in options:
                        if getattr(opt, "brand", None):
                            row_brand_labels.add(str(opt.brand))
                    # score
                    def _score(m):
                        sc = 0
                        if friendly and getattr(m, "dosage_strength", None) and friendly.lower() in (m.dosage_strength or "").lower(): sc += 2
                        if form and getattr(m, "dosage_form", None) and form.lower() == (m.dosage_form or "").lower(): sc += 1
                        gen_base = _normalize_text_basic(_base_name(m.generic))
                        if gen_base in pnf_name_to_gid: sc += 3
                        return sc
                    # Prefer candidates that align on dose, form, and PNF membership.
                    options = sorted(options, key=_score, reverse=True)
                    if options:
                        primary_option = options[0]
                        chosen_generic = primary_option.generic
                if not chosen_generic:
                    continue
                clean_generic = _clean_generic_for_swap(chosen_generic)
                gd_norm = normalize_text(clean_generic)
                if not gd_norm:
                    gd_norm = normalize_text(chosen_generic)

                canonical_generic = _normalize_text_basic(_base_name(chosen_generic))
                if canonical_generic:
                    row_generics_raw.append(canonical_generic)

                norm_basic_current = _normalize_text_basic(out)
                generic_basic = _normalize_text_basic(clean_generic)
                root_token = generic_basic.split()[0] if generic_basic else ""
                tokens_set = set(norm_basic_current.split())
                has_generic_already = False
                if root_token and root_token in tokens_set:
                    has_generic_already = True
                elif generic_basic and generic_basic in norm_basic_current:
                    has_generic_already = True
                elif generic_basic in pnf_name_set and generic_basic in tokens_set:
                    has_generic_already = True

                if has_generic_already:
                    new_out = re.sub(rf"\b{re.escape(bn)}\b", "", out)
                    new_out = re.sub(r"\s+", " ", new_out).strip()
                    if new_out != out:
                        replaced_any = True
                        out = new_out
                    continue

                new_out = re.sub(rf"\b{re.escape(bn)}\b", gd_norm, out)
                if new_out != out:
                    replaced_any = True
                    out = new_out
            out = re.sub(r"\s+", " ", out).strip()
            # FDA dose corroboration
            fda_hit = False
            if primary_option and friendly:
                ds = getattr(primary_option, "dosage_strength", "") or ""
                if ds and friendly.lower() in ds.lower():
                    fda_hit = True
            if not row_brand_labels and uniq_keys:
                row_brand_labels.update(uniq_keys)
            labels_joined = "|".join(sorted(row_brand_labels)) if row_brand_labels else ""
            mb_list.append(out); swapped.append(replaced_any); fda_hits.append(fda_hit); probable_brand_hits.append(labels_joined)
            uniq_generics: List[str] = []
            for g in row_generics_raw:
                if g and g not in uniq_generics:
                    uniq_generics.append(g)
            fda_generics_lists.append(uniq_generics)
        df["match_basis"] = mb_list
        df["did_brand_swap"] = swapped
        df["fda_dose_corroborated"] = fda_hits
        df["probable_brands"] = probable_brand_hits
        df["fda_generics_list"] = fda_generics_lists
    _timed("Apply brand→generic swaps", _brand_swap)

    _timed("Summarize non-therapeutic", _choose_best_nonthera)
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

    def _pnf_fuzzy():
        if not pnf_alias_keys or not pnf_trigram_index:
            return
        for idx, row in df.loc[df["generic_id"].isna()].iterrows():
            basis = row.get("match_basis_norm_basic") or ""
            if not isinstance(basis, str) or not basis:
                continue
            tokens = [tok for tok in basis.split() if tok]
            if not tokens:
                continue
            candidate_forms_map: Dict[str, str] = {}

            def _register(form: str, origin: str) -> None:
                norm = form.strip()
                if not norm:
                    return
                if norm not in candidate_forms_map:
                    candidate_forms_map[norm] = origin

            max_window = min(4, len(tokens))
            for window in range(1, max_window + 1):
                for start in range(0, len(tokens) - window + 1):
                    segment_tokens = tokens[start:start + window]
                    if not segment_tokens:
                        continue
                    candidate = " ".join(segment_tokens)
                    if len(candidate.replace(" ", "")) < 3:
                        continue
                    _register(candidate, candidate)
                    for variant in apply_spelling_rules(candidate):
                        _register(variant, candidate)
                    if len(segment_tokens) > 1:
                        sorted_variant = " ".join(sorted(segment_tokens))
                        _register(sorted_variant, candidate)
                        for variant in apply_spelling_rules(sorted_variant):
                            _register(variant, candidate)

            if not candidate_forms_map:
                continue

            for form, origin in list(candidate_forms_map.items()):
                compact = form.replace(" ", "")
                if compact:
                    _register(compact, origin)

            form_trigram_cache: Dict[str, Set[str]] = {}
            for form in list(candidate_forms_map.keys()):
                trigrams = _generate_trigrams(form)
                if trigrams:
                    form_trigram_cache[form] = trigrams
                else:
                    candidate_forms_map.pop(form, None)

            if not form_trigram_cache:
                continue

            candidate_counts: Dict[str, int] = {}
            for form, tris in form_trigram_cache.items():
                for tri in tris:
                    for key in pnf_trigram_index.get(tri, set()):
                        candidate_counts[key] = candidate_counts.get(key, 0) + 1

            if not candidate_counts:
                continue

            sorted_candidates = sorted(candidate_counts.items(), key=lambda kv: kv[1], reverse=True)[:50]
            best_score = 0.0
            best_gid: Optional[str] = None
            best_span = ""
            phonetic_cache: Dict[str, Set[str]] = {}

            for key, _overlap in sorted_candidates:
                gid, _display = pnf_name_to_gid.get(key, (None, None))
                if gid is None:
                    continue
                key_tris = _generate_trigrams(key)
                if not key_tris:
                    continue
                for form, tris in form_trigram_cache.items():
                    if len(form) >= 2 and len(key) >= 2 and form[:2] != key[:2]:
                        continue
                    edit_sim = difflib.SequenceMatcher(None, form, key).ratio()
                    if edit_sim < 0.72:
                        continue
                    union = tris | key_tris
                    if not union:
                        continue
                    jaccard = len(tris & key_tris) / len(union)
                    if form not in phonetic_cache:
                        phonetic_cache[form] = apply_spelling_rules(form)
                    domain_bonus = 0.1 if key in phonetic_cache[form] else 0.0
                    score = edit_sim * 0.6 + jaccard * 0.3 + domain_bonus
                    if score > best_score:
                        best_score = score
                        best_gid = gid
                        best_span = candidate_forms_map.get(form, form)

            if best_score >= 0.88 and best_gid:
                df.at[idx, "generic_id"] = best_gid
                df.at[idx, "molecule_token"] = best_span
                current_gids = df.at[idx, "pnf_hits_gids"] if "pnf_hits_gids" in df.columns else None
                if isinstance(current_gids, list):
                    if best_gid not in current_gids:
                        current_gids.append(best_gid)
                    df.at[idx, "pnf_hits_gids"] = current_gids
                else:
                    df.at[idx, "pnf_hits_gids"] = [best_gid]
                current_tokens = df.at[idx, "pnf_hits_tokens"] if "pnf_hits_tokens" in df.columns else None
                if isinstance(current_tokens, list):
                    if best_span not in current_tokens:
                        current_tokens.append(best_span)
                    df.at[idx, "pnf_hits_tokens"] = current_tokens
                else:
                    df.at[idx, "pnf_hits_tokens"] = [best_span]
                try:
                    current_count = int(df.at[idx, "pnf_hits_count"])
                except Exception:
                    current_count = 0
                df.at[idx, "pnf_hits_count"] = current_count + 1 if current_count >= 0 else 1
    _timed("Fuzzy PNF fallback", _pnf_fuzzy)

    # 12) WHO molecule detection
    def _who_detect():
        if who_regex is not None:
            who_names_all, who_atc_all = [], []
            who_ddd_flags: List[bool] = []
            who_adm_r_cols: List[str] = []
            who_route_cols: List[List[str]] = []
            who_form_cols: List[List[str]] = []
            for txt, txt_norm in zip(df["match_basis"].tolist(), df["match_basis_norm_basic"].tolist()):
                names, codes = detect_all_who_molecules(txt, who_regex, codes_by_name, pre_normalized=txt_norm)
                who_names_all.append(names)
                sorted_codes = sorted(codes)
                who_atc_all.append(sorted_codes)

                has_ddd = False
                adm_r_values: set[str] = set()
                route_tokens: set[str] = set()
                form_tokens: set[str] = set()
                for code in sorted_codes:
                    for detail in who_details_by_code.get(code, []):
                        ddd_val = detail.get("ddd")
                        if pd.notna(ddd_val) and str(ddd_val).strip():
                            has_ddd = True
                        uom_val = detail.get("uom")
                        if pd.notna(uom_val):
                            uom_key = str(uom_val).strip().lower()
                            form_tokens.update(WHO_UOM_TO_CANONICAL.get(uom_key, set()))
                        adm_r_val = detail.get("adm_r")
                        if pd.notna(adm_r_val):
                            adm_r_text = str(adm_r_val).strip()
                            if adm_r_text:
                                adm_key = adm_r_text.lower()
                                mapped_routes = WHO_ADM_ROUTE_MAP.get(adm_key)
                                if mapped_routes:
                                    route_tokens.update(mapped_routes)
                                else:
                                    route_tokens.add(adm_key)
                                form_tokens.update(WHO_FORM_TO_CANONICAL.get(adm_key, set()))
                if route_tokens:
                    adm_r_values.update(route_tokens)
                who_ddd_flags.append(has_ddd)
                who_adm_r_cols.append("|".join(sorted(adm_r_values)) if adm_r_values else "")
                who_route_cols.append(sorted(route_tokens))
                who_form_cols.append(sorted(form_tokens))

            df["who_molecules_list"] = who_names_all
            df["who_atc_codes_list"] = who_atc_all
            df["who_molecules"] = df["who_molecules_list"].map(lambda xs: "|".join(xs) if xs else "")
            df["who_atc_codes"] = df["who_atc_codes_list"].map(lambda xs: "|".join(xs) if xs else "")
            df["who_atc_has_ddd"] = who_ddd_flags
            df["who_atc_adm_r"] = who_adm_r_cols
            df["who_route_tokens"] = who_route_cols
            df["who_form_tokens"] = who_form_cols
        else:
            df["who_molecules_list"] = [[] for _ in range(len(df))]
            df["who_atc_codes_list"] = [[] for _ in range(len(df))]
            df["who_molecules"] = ""
            df["who_atc_codes"] = ""
            df["who_atc_has_ddd"] = False
            df["who_atc_adm_r"] = ""
            df["who_route_tokens"] = [[] for _ in range(len(df))]
            df["who_form_tokens"] = [[] for _ in range(len(df))]
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

        def _unknown_kind_and_list(
            text_norm: str,
            matched_tokens: set[str],
            nonthera_tokens: Set[str],
        ) -> Tuple[str, List[str]]:
            s = _segment_norm(text_norm)
            all_toks = _tokenize_unknowns(s)
            unknowns: list[str] = []
            for t in all_toks:
                if (
                    t not in matched_tokens
                    and t not in pnf_tokens
                    and t not in who_tokens
                    and t not in fda_tokens
                    and t not in STOPWORD_TOKENS
                    and t not in nonthera_tokens
                    and t not in brand_token_lookup
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
        if "non_therapeutic_tokens" in df.columns:
            nonthera_seq = df["non_therapeutic_tokens"].tolist()
        else:
            nonthera_seq = [[] for _ in range(len(df))]

        for text_norm, p_hits, who_hits, nonthera_tokens in zip(
            df["match_basis"],
            df["pnf_hits_tokens"],
            df["who_molecules_list"],
            nonthera_seq,
        ):
            matched_tokens = _token_set_from_phrase_list(p_hits) | _token_set_from_phrase_list(who_hits)
            tokens_for_row: Set[str]
            if isinstance(nonthera_tokens, (list, tuple, set)):
                tokens_for_row = {str(t) for t in nonthera_tokens if isinstance(t, str)}
            else:
                tokens_for_row = set()
            results.append(_unknown_kind_and_list(text_norm, matched_tokens, tokens_for_row))

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
