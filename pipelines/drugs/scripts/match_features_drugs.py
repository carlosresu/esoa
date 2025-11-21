#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature engineering stage responsible for every signal used by scoring."""

from __future__ import annotations
import sys, time, glob, os, re
from collections import defaultdict
from typing import Tuple, Optional, List, Dict, Set, Callable, Any
import difflib
import numpy as np, pandas as pd
import ahocorasick  # type: ignore
from ..constants import PIPELINE_INPUTS_DIR, PIPELINE_WHO_ATC_DIR
from .aho_drugs import build_molecule_automata, scan_pnf_all
from .concurrency_drugs import maybe_parallel_map, resolve_worker_count
from .combos_drugs import SALT_TOKENS
from .routes_forms_drugs import extract_route_and_form
from .text_utils_drugs import (
    _base_name,
    _normalize_text_basic,
    normalize_text,
    strip_after_as,
    extract_parenthetical_phrases,
    STOPWORD_TOKENS,
)
from .reference_data_drugs import load_drugbank_generics, load_ignore_words
from .who_molecules_drugs import detect_all_who_molecules, load_who_molecules
from .brand_map_drugs import load_latest_brandmap, build_brand_automata, fda_generics_set
from .pnf_aliases_drugs import (
    expand_generic_aliases,
    SPECIAL_GENERIC_ALIASES,
    apply_spelling_rules,
    SALT_FORM_SUFFIXES,
)
from .pnf_partial_drugs import PnfTokenIndex

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

_BRAND_SWAP_CONTEXT: Dict[str, Any] | None = None
_WHO_DETECT_CONTEXT: Dict[str, Any] | None = None
_FUZZY_CONTEXT: Dict[str, Any] | None = None

FUZZY_SCORE_THRESHOLD = 0.88
FUZZY_SOURCE_PRIMARY: Dict[str, int] = {"annex": 0, "pnf": 1, "who": 1, "drugbank": 1, "fda": 1}
FUZZY_SOURCE_TIE: Dict[str, int] = {"annex": 0, "pnf": 1, "who": 2, "drugbank": 3, "fda": 4}


def _generate_trigrams(value: str) -> Set[str]:
    """Return overlapping trigram set for a normalized string."""
    segment = re.sub(r"\s+", "", value)
    if not segment:
        return set()
    if len(segment) <= 3:
        return {segment}
    return {segment[i : i + 3] for i in range(len(segment) - 2)}


def _string_or_empty(value: object) -> str:
    """Return a safe string representation without propagating NaN/None."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _clean_generic_for_swap(name: str) -> str:
    """Strip parenthetical or descriptive suffixes before swapping brands."""
    base = _base_name(name)
    base = re.sub(r"\(.*?\)", " ", base)
    base = re.split(r"\bas\b", base, 1)[0]
    base = re.sub(r"\s+", " ", base).strip()
    return base or _base_name(name)


def _prepare_brand_context_from_records(
    records: List[Dict[str, Any]],
    pnf_name_set: Set[str],
    drugbank_name_set: Set[str],
    fda_generic_set: Set[str],
) -> Dict[str, Any] | None:
    """Build brand automata inside worker processes from serialized records."""
    if not records:
        return None
    brand_df = pd.DataFrame.from_records(records)
    if brand_df.empty:
        return None
    norm_auto, comp_auto, brand_lookup = build_brand_automata(brand_df)
    return {
        "A_norm": norm_auto,
        "A_comp": comp_auto,
        "brand_lookup": brand_lookup,
        "pnf_name_set": set(pnf_name_set),
        "drugbank_name_set": set(drugbank_name_set),
        "fda_generic_set": set(fda_generic_set),
    }


def _fuzzy_worker_init(payload: Dict[str, Any]) -> None:
    """Bootstrap heavy fuzzy matching artifacts inside worker processes."""
    global _FUZZY_CONTEXT
    if not payload:
        _FUZZY_CONTEXT = None
        return
    ctx: Dict[str, Any] = {
        "trigram_index": {tri: set(vals) for tri, vals in payload.get("trigram_index", {}).items()},
        "entries": {str(k): list(v) for k, v in payload.get("entries", {}).items()},
        "non_molecule_tokens": set(payload.get("non_molecule_tokens", [])),
        "salt_tokens": set(payload.get("salt_tokens", [])),
        "dose_suffixes": tuple(payload.get("dose_suffixes", DOSE_UNIT_SUFFIXES)),
        "therapeutic_tokens": set(payload.get("therapeutic_tokens", [])),
        "brand_token_lookup": set(payload.get("brand_token_lookup", [])),
        "source_primary": dict(payload.get("source_primary", FUZZY_SOURCE_PRIMARY)),
        "source_tie": dict(payload.get("source_tie", FUZZY_SOURCE_TIE)),
        "score_threshold": float(payload.get("score_threshold", FUZZY_SCORE_THRESHOLD)),
    }
    ctx["skip_tokens"] = set(ctx["non_molecule_tokens"]) | set(ctx["brand_token_lookup"])
    _FUZZY_CONTEXT = ctx


def _fuzzy_worker(task: Tuple[object, str]) -> Tuple[object, Optional[List[Dict[str, Any]]]]:
    """Entry point for process pool evaluation of fuzzy matches."""
    idx, basis = task
    ctx = _FUZZY_CONTEXT or {}
    match = _fuzzy_core(basis, ctx)
    return idx, match


def _fuzzy_core(basis: str, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Best-effort fuzzy match of normalized basis text against reference catalogues."""
    if not basis:
        return None
    trigram_index: Dict[str, Set[str]] = context.get("trigram_index") or {}
    entries_map: Dict[str, List[Dict[str, Any]]] = context.get("entries") or {}
    if not trigram_index or not entries_map:
        return None

    non_molecule_tokens: Set[str] = context.get("non_molecule_tokens", set())
    salt_tokens: Set[str] = context.get("salt_tokens", set())
    dose_suffixes: Tuple[str, ...] = context.get("dose_suffixes", DOSE_UNIT_SUFFIXES)
    therapeutic_tokens: Set[str] = context.get("therapeutic_tokens", set())
    brand_token_lookup: Set[str] = context.get("brand_token_lookup", set())
    skip_tokens: Set[str] = context.get("skip_tokens") or (set(non_molecule_tokens) | set(brand_token_lookup))
    source_primary: Dict[str, int] = context.get("source_primary", FUZZY_SOURCE_PRIMARY)
    source_tie: Dict[str, int] = context.get("source_tie", FUZZY_SOURCE_TIE)
    score_threshold: float = context.get("score_threshold", FUZZY_SCORE_THRESHOLD)

    candidate_tokens = _tokenize_unknowns(basis)
    if not candidate_tokens:
        candidate_tokens = [tok for tok in basis.split() if tok]

    looks_suspicious = False
    for token in candidate_tokens:
        tok_lower = token.lower()
        if len(tok_lower) < 3:
            continue
        if tok_lower in skip_tokens:
            continue
        if tok_lower in therapeutic_tokens:
            continue
        if tok_lower in salt_tokens and len(tok_lower) <= 4:
            continue
        if tok_lower.isdigit():
            continue
        tris = _generate_trigrams(tok_lower)
        if not tris:
            continue
        if any(tri in trigram_index for tri in tris):
            looks_suspicious = True
            break

    if not looks_suspicious:
        return None

    def _token_is_noise(token: str) -> bool:
        if not token:
            return True
        if token in non_molecule_tokens:
            return True
        if token in salt_tokens and len(token) <= 4:
            return True
        lower = token.lower()
        if lower.isdigit():
            return True
        stripped = lower.replace(".", "")
        if stripped.isdigit():
            return True
        if any(ch.isdigit() for ch in lower):
            for suffix in dose_suffixes:
                if lower.endswith(suffix):
                    return True
        return False

    def _segment_is_noise(segment_tokens: List[str]) -> bool:
        filtered = [tok for tok in segment_tokens if tok]
        if not filtered:
            return True
        if all(_token_is_noise(tok) for tok in filtered):
            return True
        return False

    tokens = [tok for tok in basis.split() if tok]
    if not tokens:
        return None

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
            if _segment_is_noise(segment_tokens):
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
        return None

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
        return None

    candidate_counts: Dict[str, int] = {}
    for form, tris in form_trigram_cache.items():
        for tri in tris:
            for key in trigram_index.get(tri, set()):
                candidate_counts[key] = candidate_counts.get(key, 0) + 1

    if not candidate_counts:
        return None

    sorted_candidates = sorted(candidate_counts.items(), key=lambda kv: kv[1], reverse=True)[:50]
    best_by_source: Dict[str, Dict[str, Any]] = {}
    phonetic_cache: Dict[str, Set[str]] = {}

    for key, _overlap in sorted_candidates:
        entry_list = entries_map.get(key, [])
        if not entry_list:
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
            if score < score_threshold:
                continue
            span = candidate_forms_map.get(form, form)
            for entry in entry_list:
                source = entry.get("source", "pnf")
                current = best_by_source.get(source)
                if current and current["score"] >= score:
                    continue
                best_by_source[source] = {
                    "source": source,
                    "score": score,
                    "span": span,
                    "key": key,
                    "gid": entry.get("gid"),
                    "display": entry.get("display") or key,
                    "metadata": entry.get("metadata") or {},
                }

    if not best_by_source:
        return None

    matches = list(best_by_source.values())
    matches.sort(
        key=lambda m: (
            source_primary.get(m["source"], 99),
            -m["score"],
            source_tie.get(m["source"], 99),
            m["display"],
        )
    )
    return matches
    return None


def _brand_swap_worker_init(payload: Dict[str, Any]) -> None:
    """Initializer for process pool brand swaps."""
    global _BRAND_SWAP_CONTEXT
    records = payload.get("brand_records") or []
    pnf_names = payload.get("pnf_name_set") or []
    drugbank_names = payload.get("drugbank_name_set") or []
    fda_generics = payload.get("fda_generic_set") or []
    _BRAND_SWAP_CONTEXT = _prepare_brand_context_from_records(
        records,
        set(pnf_names),
        set(drugbank_names),
        set(fda_generics),
    )


def _brand_swap_core(
    norm: object,
    comp: object,
    form: object,
    friendly: object,
    context: Dict[str, Any] | None,
) -> Tuple[str, bool, bool, bool, str, List[str]]:
    """Shared implementation for both serial and parallel brand swaps."""
    if not context:
        return (_string_or_empty(norm), False, False, False, "", [])

    auto_norm = context.get("A_norm")
    auto_comp = context.get("A_comp")
    brand_lookup = context.get("brand_lookup") or {}
    pnf_name_set: Set[str] = context.get("pnf_name_set", set())
    drugbank_name_set: Set[str] = context.get("drugbank_name_set", set())
    fda_generic_set: Set[str] = context.get("fda_generic_set", set())

    norm_text = _string_or_empty(norm)
    comp_text = _string_or_empty(comp)
    form_lower = _string_or_empty(form).lower()
    friendly_text = _string_or_empty(friendly)
    friendly_lower = friendly_text.lower()

    if not auto_norm or not auto_comp or not brand_lookup:
        return (norm_text, False, False, False, "", [])

    found_keys: List[str] = []
    lengths: Dict[str, int] = {}
    if norm_text:
        for _, bn in auto_norm.iter(norm_text):  # type: ignore[attr-defined]
            found_keys.append(bn)
            lengths[bn] = max(lengths.get(bn, 0), len(bn))
    if comp_text:
        for _, bn in auto_comp.iter(comp_text):  # type: ignore[attr-defined]
            found_keys.append(bn)
            lengths[bn] = max(lengths.get(bn, 0), len(bn))
    if not found_keys:
        return (norm_text, False, False, False, "", [])

    uniq_keys = list(dict.fromkeys(found_keys))
    uniq_keys.sort(key=lambda k: (-lengths.get(k, len(k)), k))

    out = norm_text
    replaced_any = False
    inserted_generic = False
    primary_option = None
    row_brand_labels: Set[str] = set()
    row_generics_raw: List[str] = []

    for bn in uniq_keys:
        options = brand_lookup.get(bn, [])
        if options:
            for opt in options:
                if getattr(opt, "brand", None):
                    brand_label = str(opt.brand)
                    brand_norm = _normalize_text_basic(_base_name(brand_label))
                    if brand_norm and (
                        brand_norm in pnf_name_set
                        or brand_norm in drugbank_name_set
                        or brand_norm in fda_generic_set
                    ):
                        continue
                    row_brand_labels.add(brand_label)

        chosen_generic = None
        local_primary = None

        if options:
            def _score(match) -> int:
                sc = 0
                if friendly_lower and getattr(match, "dosage_strength", None):
                    strength = (match.dosage_strength or "").lower()
                    if friendly_lower in strength:
                        sc += 2
                if form_lower and getattr(match, "dosage_form", None):
                    if form_lower == (match.dosage_form or "").lower():
                        sc += 1
                gen_base = _normalize_text_basic(_base_name(match.generic))
                if gen_base in pnf_name_set:
                    sc += 3
                return sc
            sorted_options = sorted(options, key=_score, reverse=True)
            if sorted_options:
                local_primary = sorted_options[0]
                chosen_generic = local_primary.generic

        if local_primary:
            primary_option = local_primary

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
        generic_compact = generic_basic.replace(" ", "") if generic_basic else ""
        current_compact = norm_basic_current.replace(" ", "")

        has_generic_already = False
        if root_token and root_token in tokens_set:
            has_generic_already = True
        elif generic_basic and generic_basic in norm_basic_current:
            has_generic_already = True
        elif generic_compact and generic_compact in current_compact:
            has_generic_already = True
        elif generic_basic in pnf_name_set and generic_basic in tokens_set:
            has_generic_already = True
        elif generic_basic and generic_basic in drugbank_name_set and generic_basic in norm_basic_current:
            has_generic_already = True

        if has_generic_already:
            continue

        new_out = re.sub(rf"\b{re.escape(bn)}\b", gd_norm, out)
        if new_out != out:
            replaced_any = True
            inserted_generic = True
            out = new_out

    out = re.sub(r"\s+", " ", out).strip()

    fda_hit = False
    if primary_option and friendly_lower:
        ds = getattr(primary_option, "dosage_strength", "") or ""
        if ds and friendly_lower in ds.lower():
            fda_hit = True

    if not row_brand_labels and uniq_keys:
        for key in uniq_keys:
            key_norm = _normalize_text_basic(_base_name(key))
            if key_norm and (
                key_norm in pnf_name_set
                or key_norm in drugbank_name_set
                or key_norm in fda_generic_set
            ):
                continue
            row_brand_labels.add(key)
    labels_joined = "|".join(sorted(row_brand_labels)) if row_brand_labels else ""

    uniq_generics: List[str] = []
    seen_generics: Set[str] = set()
    for g in row_generics_raw:
        if g and g not in seen_generics:
            uniq_generics.append(g)
            seen_generics.add(g)

    return (out, bool(replaced_any), bool(inserted_generic), bool(fda_hit), labels_joined, uniq_generics)


def _brand_swap_worker(row: Tuple[object, object, object, object]) -> Tuple[str, bool, bool, bool, str, List[str]]:
    """ProcessPool worker entry point for brand swapping."""
    return _brand_swap_core(row[0], row[1], row[2], row[3], _BRAND_SWAP_CONTEXT)


def _who_worker_init(payload: Dict[str, Any]) -> None:
    """Initializer for WHO detection workers."""
    global _WHO_DETECT_CONTEXT
    pattern = payload.get("pattern")
    regex = re.compile(pattern) if pattern else None
    codes_source = payload.get("codes_by_name") or {}
    details = payload.get("details_by_code") or {}
    # Ensure sets are reconstructed to avoid accidental mutation.
    codes_map = {k: set(v) for k, v in codes_source.items()}
    _WHO_DETECT_CONTEXT = {
        "regex": regex,
        "codes_by_name": codes_map,
        "details_by_code": details,
    }


def _who_detect_core(
    text: object,
    text_norm: object,
    context: Dict[str, Any] | None,
) -> Tuple[List[str], List[str], bool, str, List[str], List[str]]:
    """Shared WHO detection logic used in both serial and parallel execution."""
    if not context:
        return ([], [], False, "", [], [])
    regex = context.get("regex")
    if not regex:
        return ([], [], False, "", [], [])
    codes_by_name = context.get("codes_by_name") or {}
    details_by_code = context.get("details_by_code") or {}

    source_text = _string_or_empty(text)
    norm_text = _string_or_empty(text_norm)

    names, codes = detect_all_who_molecules(source_text, regex, codes_by_name, pre_normalized=norm_text)
    sorted_codes = sorted(codes)

    has_ddd = False
    adm_r_values: Set[str] = set()
    route_tokens: Set[str] = set()
    form_tokens: Set[str] = set()

    for code in sorted_codes:
        for detail in details_by_code.get(code, []):
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

    adm_r_display = "|".join(sorted(adm_r_values)) if adm_r_values else ""
    return (
        names,
        sorted_codes,
        has_ddd,
        adm_r_display,
        sorted(route_tokens),
        sorted(form_tokens),
    )


def _who_worker(row: Tuple[object, object]) -> Tuple[List[str], List[str], bool, str, List[str], List[str]]:
    """ProcessPool worker entry point for WHO detection."""
    return _who_detect_core(row[0], row[1], _WHO_DETECT_CONTEXT)

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
DOSE_UNIT_SUFFIXES: tuple[str, ...] = (
    "mg",
    "g",
    "mcg",
    "ug",
    "iu",
    "lsu",
    "ml",
    "l",
    "%",
    "mgml",
    "mcgml",
    "gml",
    "unit",
    "units",
)
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

def load_latest_who_file(root_dir: str) -> str | None:
    """Return the freshest WHO ATC CSV path, preferring the pipeline-specific inputs directory."""
    who_candidates = glob.glob(os.path.join(str(PIPELINE_WHO_ATC_DIR), "who_atc_*_molecules.csv"))
    if who_candidates:
        return max(who_candidates, key=os.path.getmtime)
    legacy_dir = os.path.join(root_dir, "dependencies", "atcd", "output")
    legacy_candidates = glob.glob(os.path.join(legacy_dir, "who_atc_*_molecules.csv"))
    if legacy_candidates:
        return max(legacy_candidates, key=os.path.getmtime)
    return None

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

    ignore_words = load_ignore_words()
    stopword_tokens_all: Set[str] = set(STOPWORD_TOKENS) | ignore_words

    def _normalize_token(token: str) -> str:
        norm = _normalize_text_basic(token)
        return norm if norm else ""

    form_noise_tokens: Set[str] = set()
    route_noise_tokens: Set[str] = set()
    for values in FOOD_FORM_KEYWORDS.values():
        for raw in values:
            norm = _normalize_token(raw)
            if norm:
                form_noise_tokens.add(norm)

    for raw in WHO_ADM_ROUTE_MAP.keys():
        norm = _normalize_token(raw)
        if norm:
            route_noise_tokens.add(norm)
    for values in WHO_ADM_ROUTE_MAP.values():
        for raw in values:
            norm = _normalize_token(raw)
            if norm:
                route_noise_tokens.add(norm)

    salt_suffix_tokens: Set[str] = set()
    for raw in SALT_FORM_SUFFIXES:
        norm = _normalize_token(raw)
        if norm:
            salt_suffix_tokens.add(norm)

    salt_tokens: Set[str] = set()
    for raw in SALT_TOKENS:
        norm = _normalize_token(raw)
        if norm:
            salt_tokens.add(norm)

    non_molecule_tokens: Set[str] = (
        stopword_tokens_all
        | form_noise_tokens
        | route_noise_tokens
        | salt_suffix_tokens
        | {
            "vial",
            "vials",
            "ampule",
            "ampoule",
            "sachet",
            "sachets",
            "prefilled",
            "prefill",
            "drop",
            "drops",
            "dose",
            "doses",
            "unit",
            "units",
        }
    )

    # 1) Validate inputs
    def _validate():
        required_pnf = {
            "generic_id",
            "generic_name",
            "synonyms",
            "atc_code",
            "route_allowed",
            "form_token",
            "dose_kind",
            "strength",
            "unit",
            "per_val",
            "per_unit",
            "pct",
            "strength_mg",
            "ratio_mg_per_ml",
            "source",
            "source_priority",
            "drug_code",
            "primary_code",
            "route_evidence_reference",
        }
        missing = required_pnf - set(pnf_df.columns)
        if missing:
            raise ValueError(f"reference catalogue missing columns: {missing}")
        if "raw_text" not in esoa_df.columns:
            raise ValueError("esoa_prepared.csv must contain a 'raw_text' column")
    _timed("Validate inputs", _validate)

    gid_to_source: Dict[str, str] = {}
    gid_to_priority: Dict[str, int] = {}
    gid_to_primary_code: Dict[str, str] = {}
    gid_to_drug_code: Dict[str, str] = {}
    gid_to_route_ref: Dict[str, str] = {}
    for row in (
        pnf_df[
            [
                "generic_id",
                "source",
                "source_priority",
                "primary_code",
                "drug_code",
                "route_evidence_reference",
            ]
        ]
        .dropna(subset=["generic_id"])
        .drop_duplicates()
        .itertuples(index=False)
    ):
        gid = str(row.generic_id)
        if not gid:
            continue
        if gid not in gid_to_source:
            gid_to_source[gid] = str(row.source) if isinstance(row.source, str) else ""
        if gid not in gid_to_priority:
            try:
                gid_to_priority[gid] = int(row.source_priority)
            except Exception:
                gid_to_priority[gid] = 99
        if gid not in gid_to_primary_code:
            code = str(row.primary_code) if isinstance(row.primary_code, str) else ""
            gid_to_primary_code[gid] = code
        if gid not in gid_to_drug_code:
            drug_code = str(row.drug_code) if isinstance(row.drug_code, str) else ""
            gid_to_drug_code[gid] = drug_code
        if gid not in gid_to_route_ref:
            ref = str(row.route_evidence_reference) if isinstance(row.route_evidence_reference, str) else ""
            gid_to_route_ref[gid] = ref

    # 2) Map normalized PNF names to gid + original name
    pnf_name_to_gid: Dict[str, Tuple[str, str]] = {}

    def _pnf_map():
        name_col = "generic_normalized" if "generic_normalized" in pnf_df.columns else "generic_name"
        for gid, gname in pnf_df[["generic_id", name_col]].drop_duplicates().itertuples(index=False):
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
    _timed("Index PNF names", _pnf_map)
    pnf_name_set: Set[str] = set(pnf_name_to_gid.keys())
    pnf_tokens: Set[str] = set()
    for name in pnf_name_set:
        pnf_tokens.update(_tokenize_unknowns(name))

    # 3) WHO molecules (names + regex)
    codes_by_name, candidate_names = ({}, [])
    who_details_by_code: Dict[str, List[dict]] = {}
    who_name_set: Set[str] = set()
    who_regex_box = [None]
    def _load_who():
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        who_file = load_latest_who_file(root_dir)
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
        inputs_dir = str(PIPELINE_INPUTS_DIR)
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

    (
        drugbank_name_set,
        drugbank_tokens,
        drugbank_token_index,
        drugbank_display_lookup,
        drugbank_salt_lookup,
    ) = load_drugbank_generics()
    drugbank_name_set = set(drugbank_name_set)
    drugbank_tokens = set(drugbank_tokens)
    therapeutic_tokens: Set[str] = pnf_tokens | who_tokens | fda_tokens | drugbank_tokens

    fuzzy_context_box: List[Dict[str, Any] | None] = [None]

    def _ensure_fuzzy_context() -> Dict[str, Any]:
        if fuzzy_context_box[0] is not None:
            return fuzzy_context_box[0]  # type: ignore[return-value]

        entries_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        def _add_entry(norm_key: str, entry: Dict[str, Any]) -> None:
            if not norm_key:
                return
            entries_map[norm_key].append(entry)

        for key, (gid, gname) in pnf_name_to_gid.items():
            if not key:
                continue
            source_label = (gid_to_source.get(gid) or "").strip().lower()
            source_tag = "annex" if source_label == "annex_f" else "pnf"
            metadata = {
                "primary_code": gid_to_primary_code.get(gid, ""),
                "drug_code": gid_to_drug_code.get(gid, ""),
                "route_ref": gid_to_route_ref.get(gid, ""),
                "source": gid_to_source.get(gid, ""),
            }
            _add_entry(
                key,
                {
                    "source": source_tag,
                    "gid": gid,
                    "display": gname,
                    "metadata": metadata,
                },
            )

        def _codes_for_name(name: str, norm_key: str) -> List[str]:
            candidates = codes_by_name.get(name) or codes_by_name.get(norm_key) or codes_by_name.get(name.lower())
            if isinstance(candidates, (set, list, tuple)):
                return [str(val) for val in sorted({str(v) for v in candidates if str(v)})]
            if candidates:
                val = str(candidates)
                return [val] if val else []
            return []

        for name in who_name_set:
            norm_key = _normalize_text_basic(name)
            if not norm_key:
                continue
            codes_list = _codes_for_name(name, norm_key)
            _add_entry(
                norm_key,
                {
                    "source": "who",
                    "gid": None,
                    "display": name,
                    "metadata": {"codes": codes_list},
                },
            )

        for name in drugbank_name_set:
            norm_key = _normalize_text_basic(name)
            if not norm_key:
                continue
            _add_entry(
                norm_key,
                {
                    "source": "drugbank",
                    "gid": None,
                    "display": name,
                    "metadata": {},
                },
            )

        for name in fda_gens:
            norm_key = _normalize_text_basic(name)
            if not norm_key:
                continue
            _add_entry(
                norm_key,
                {
                    "source": "fda",
                    "gid": None,
                    "display": name,
                    "metadata": {},
                },
            )

        trigram_map: Dict[str, Set[str]] = defaultdict(set)
        for key in entries_map.keys():
            for tri in _generate_trigrams(key):
                if tri:
                    trigram_map[tri].add(key)

        trigram_index_lists = {tri: sorted(keys) for tri, keys in trigram_map.items()}
        trigram_index_sets = {tri: set(keys) for tri, keys in trigram_map.items()}

        context = {
            "entries": {k: [dict(entry) for entry in v] for k, v in entries_map.items()},
            "trigram_index": trigram_index_lists,
            "trigram_index_sets": trigram_index_sets,
            "source_primary": dict(FUZZY_SOURCE_PRIMARY),
            "source_tie": dict(FUZZY_SOURCE_TIE),
            "score_threshold": FUZZY_SCORE_THRESHOLD,
        }
        fuzzy_context_box[0] = context
        return context  # type: ignore[return-value]

    # 4b) FDA food/non-therapeutic catalog (optional, may not exist locally)
    nonthera_lookup: Dict[str, List[Dict[str, str]]] = {}
    nonthera_token_lookup: Dict[str, Set[str]] = {}
    nonthera_loaded = [False]
    nonthera_automaton = [None]

    def _load_nontherapeutic_catalog():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        inputs_dir = str(PIPELINE_INPUTS_DIR)
        food_candidates = sorted((PIPELINE_INPUTS_DIR).glob("fda_food_*.csv"))
        if not food_candidates:
            legacy_path = PIPELINE_INPUTS_DIR / "fda_food_products.csv"
            if legacy_path.exists():
                legacy_path.unlink(missing_ok=True)
            return
        catalog_path = str(food_candidates[-1])
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
                    and tok not in stopword_tokens_all
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
        base_cols = ["raw_text"]
        if "ITEM_NUMBER" in esoa_df.columns:
            base_cols = ["ITEM_NUMBER"] + base_cols
        tmp = esoa_df[base_cols].copy()
        raw_values = tmp["raw_text"].tolist()
        tmp["parentheticals"] = maybe_parallel_map(raw_values, extract_parenthetical_phrases)
        tmp["esoa_idx"] = tmp.index
        normalized_full_values = maybe_parallel_map(raw_values, normalize_text)
        normalized_generic_values = [strip_after_as(val) for val in normalized_full_values]
        tmp["normalized_full"] = normalized_full_values
        tmp["normalized"] = normalized_generic_values
        tmp["norm_compact"] = [re.sub(r"[ \-]", "", s) for s in normalized_generic_values]
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

        norm_series = df.get("normalized_full", df["normalized"])
        for norm_text in norm_series.astype(str).tolist():
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

        norm_series = df.get("normalized_full", df["normalized"])
        for idx, (hits, norm_text, form_raw, route_raw) in enumerate(
            zip(
                df["non_therapeutic_hits"],
                norm_series,
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
        from .dose_drugs import extract_dosage as _extract_dosage
        norm_values = df.get("normalized_full", df["normalized"]).tolist()
        dosage_parsed = maybe_parallel_map(norm_values, _extract_dosage)
        df["dosage_parsed_raw"] = dosage_parsed
        df["dose_recognized"] = [_friendly_dose(item) for item in dosage_parsed]
        route_triplets = maybe_parallel_map(norm_values, extract_route_and_form)
        if route_triplets:
            routes, forms, evidences = zip(*route_triplets)
            df["route_raw"] = list(routes)
            df["form_raw"] = list(forms)
            df["route_evidence_raw"] = list(evidences)
        else:
            df["route_raw"] = []
            df["form_raw"] = []
            df["route_evidence_raw"] = []
    _timed("Parse dose/route/form (raw)", _dose_route_form_raw)

    # 8) Brand → Generic swap (if brand map available)
    def _brand_swap():
        if not has_brandmap:
            df["match_basis"] = df["normalized"]
            df["did_brand_swap"] = False
            df["brand_swap_added_generic"] = False
            df["fda_dose_corroborated"] = False
            df["probable_brands"] = ""
            df["fda_generics_list"] = [[] for _ in range(len(df))]
            return
        normalized_values = df["normalized"].tolist()
        compact_values = df["norm_compact"].tolist()
        form_values = df["form_raw"].tolist() if "form_raw" in df.columns else [""] * len(df)
        dose_values = df["dose_recognized"].tolist()
        payload = list(zip(normalized_values, compact_values, form_values, dose_values))

        worker_count = resolve_worker_count(task_size=len(payload))
        local_context = {
            "A_norm": B_norm,
            "A_comp": B_comp,
            "brand_lookup": brand_lookup,
            "pnf_name_set": pnf_name_set,
            "drugbank_name_set": drugbank_name_set,
            "fda_generic_set": set(fda_gens),
        }

        if worker_count <= 1:
            results = [
                _brand_swap_core(norm, comp, form, friendly, local_context)
                for norm, comp, form, friendly in payload
            ]
        else:
            brand_records = brand_df[0].fillna("").to_dict("records")
            init_payload = {
                "brand_records": brand_records,
                "pnf_name_set": list(pnf_name_set),
                "drugbank_name_set": list(drugbank_name_set),
                "fda_generic_set": list(fda_gens),
            }
            results = maybe_parallel_map(
                payload,
                _brand_swap_worker,
                max_workers=worker_count,
                initializer=_brand_swap_worker_init,
                initargs=(init_payload,),
                parallel_threshold=1000,
                chunksize=400,
            )

        if results:
            mb_list, swapped, swap_inserted_flags, fda_hits, probable_brand_hits, fda_generics_lists = zip(*results)
        else:
            mb_list = swapped = swap_inserted_flags = fda_hits = probable_brand_hits = fda_generics_lists = ()

        df["match_basis"] = list(mb_list)
        df["did_brand_swap"] = [bool(x) for x in swapped]
        df["brand_swap_added_generic"] = [bool(x) for x in swap_inserted_flags]
        df["fda_dose_corroborated"] = [bool(x) for x in fda_hits]
        df["probable_brands"] = list(probable_brand_hits)
        df["fda_generics_list"] = [list(x) for x in fda_generics_lists]
    _timed("Apply brand→generic swaps", _brand_swap)

    def _drugbank_hits():
        if not drugbank_token_index:
            df["drugbank_generics_list"] = [[] for _ in range(len(df))]
            df["present_in_drugbank"] = False
            return
        hits_list: List[List[str]] = []
        present_flags: List[bool] = []
        for basis in df["match_basis"]:
            norm = _normalize_text_basic(str(basis))
            if not norm:
                hits_list.append([])
                present_flags.append(False)
                continue
            tokens = norm.split()
            token_count = len(tokens)
            matches: List[str] = []
            seen_keys: Set[str] = set()

            def _record_match(candidate_norm: str) -> None:
                if not candidate_norm:
                    return
                compact = candidate_norm.replace(" ", "")
                dedupe_key = compact or candidate_norm
                if dedupe_key in seen_keys:
                    return
                display = (
                    drugbank_display_lookup.get(candidate_norm)
                    or drugbank_display_lookup.get(compact)
                    or candidate_norm
                )
                seen_keys.add(dedupe_key)
                matches.append(display)

            for pos, token in enumerate(tokens):
                candidates = drugbank_token_index.get(token)
                if not candidates:
                    continue
                for cand_tokens in candidates:
                    length = len(cand_tokens)
                    if length == 0 or pos + length > token_count:
                        continue
                    if tuple(tokens[pos:pos + length]) == cand_tokens:
                        candidate_norm = " ".join(cand_tokens)
                        _record_match(candidate_norm)

            # Fallback for spaced tokens whose reference form is compacted (e.g., acetyl + cysteine => acetylcysteine).
            for start in range(token_count):
                for end in range(start + 2, min(token_count, start + 6) + 1):
                    segment_compact = "".join(tokens[start:end])
                    if segment_compact in drugbank_display_lookup:
                        _record_match(segment_compact)

            # Whole-string compact lookup as final fallback.
            whole_compact = norm.replace(" ", "")
            if whole_compact in drugbank_display_lookup:
                _record_match(whole_compact)

            hits_list.append(matches)
            present_flags.append(bool(matches))
        df["drugbank_generics_list"] = hits_list
        df["present_in_drugbank"] = present_flags

    _timed("Detect DrugBank generics", _drugbank_hits)

    _timed("Summarize non-therapeutic", _choose_best_nonthera)
    df["match_basis_norm_basic"] = df["match_basis"].map(_normalize_text_basic)

    # 9) Dose/route/form on match_basis
    def _dose_route_form_basis():
        from .dose_drugs import extract_dosage as _extract_dosage
        basis_values = df["match_basis"].tolist()
        dosage_parsed = maybe_parallel_map(basis_values, _extract_dosage)
        df["dosage_parsed"] = dosage_parsed
        route_triplets = maybe_parallel_map(basis_values, extract_route_and_form)
        if route_triplets:
            routes, forms, evidences = zip(*route_triplets)
            df["route"] = list(routes)
            df["form"] = list(forms)
            df["route_evidence"] = list(evidences)
        else:
            df["route"] = []
            df["form"] = []
            df["route_evidence"] = []
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
            if gids:
                ordered = sorted(
                    zip(gids, tokens),
                    key=lambda pair: (gid_to_priority.get(str(pair[0]), 99), str(pair[0])),
                )
                gids = [g for g, _ in ordered]
                tokens = [t for _, t in ordered]
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
        if who_regex is None:
            df["who_molecules_list"] = [[] for _ in range(len(df))]
            df["who_atc_codes_list"] = [[] for _ in range(len(df))]
            df["who_molecules"] = ""
            df["who_atc_codes"] = ""
            df["who_atc_has_ddd"] = False
            df["who_atc_adm_r"] = ""
            df["who_route_tokens"] = [[] for _ in range(len(df))]
            df["who_form_tokens"] = [[] for _ in range(len(df))]
            return

        basis_values = df["match_basis"].tolist()
        norm_basic_values = df["match_basis_norm_basic"].tolist()
        payload = list(zip(basis_values, norm_basic_values))

        worker_count = resolve_worker_count(task_size=len(payload))
        local_context = {
            "regex": who_regex,
            "codes_by_name": codes_by_name,
            "details_by_code": who_details_by_code,
        }

        if worker_count <= 1:
            results = [
                _who_detect_core(text, norm_text, local_context)
                for text, norm_text in payload
            ]
        else:
            init_payload = {
                "pattern": who_regex.pattern,
                "codes_by_name": {k: list(v) for k, v in codes_by_name.items()},
                "details_by_code": who_details_by_code,
            }
            results = maybe_parallel_map(
                payload,
                _who_worker,
                max_workers=worker_count,
                initializer=_who_worker_init,
                initargs=(init_payload,),
                parallel_threshold=1000,
                chunksize=400,
            )

        if results:
            (
                who_names_all,
                who_atc_all,
                who_ddd_flags,
                who_adm_r_cols,
                who_route_cols,
                who_form_cols,
            ) = map(list, zip(*results))
        else:
            who_names_all = who_atc_all = who_adm_r_cols = []
            who_route_cols = who_form_cols = []
            who_ddd_flags = []

        pnf_counts = pd.to_numeric(
            df.get("pnf_hits_count", pd.Series([0] * len(df))),
            errors="coerce",
        ).fillna(0).astype(int)
        skip_who = pnf_counts.gt(0).tolist()
        if any(skip_who):
            who_names_all = [list() if skip else list(names) for names, skip in zip(who_names_all, skip_who)]
            who_atc_all = [list() if skip else list(codes) for codes, skip in zip(who_atc_all, skip_who)]
            who_ddd_flags = [False if skip else bool(flag) for flag, skip in zip(who_ddd_flags, skip_who)]
            who_adm_r_cols = ["" if skip else adm for adm, skip in zip(who_adm_r_cols, skip_who)]
            who_route_cols = [list() if skip else list(routes) for routes, skip in zip(who_route_cols, skip_who)]
            who_form_cols = [list() if skip else list(forms) for forms, skip in zip(who_form_cols, skip_who)]

        df["who_molecules_list"] = who_names_all
        df["who_atc_codes_list"] = who_atc_all
        df["who_molecules"] = df["who_molecules_list"].map(lambda xs: "|".join(xs) if xs else "")
        df["who_atc_codes"] = df["who_atc_codes_list"].map(lambda xs: "|".join(xs) if xs else "")
        df["who_atc_has_ddd"] = who_ddd_flags
        df["who_atc_adm_r"] = who_adm_r_cols
        df["who_route_tokens"] = who_route_cols
        df["who_form_tokens"] = who_form_cols
    _timed("Detect WHO molecules", _who_detect)

    def _fuzzy_reference():
        context = _ensure_fuzzy_context()
        entries_map: Dict[str, List[Dict[str, Any]]] = context.get("entries", {})
        trigram_index_lists: Dict[str, List[str]] = context.get("trigram_index", {})
        trigram_index_sets: Dict[str, Set[str]] = context.get("trigram_index_sets", {})
        if not entries_map or not trigram_index_lists:
            if "fuzzy_matches" not in df.columns:
                df["fuzzy_matches"] = [[] for _ in range(len(df))]
            return

        df["fuzzy_matches"] = [[] for _ in range(len(df))]

        unresolved_mask = df["generic_id"].isna()
        if not unresolved_mask.any():
            return

        if trigram_index_sets:
            trigram_lookup = trigram_index_sets
        else:
            trigram_lookup = {tri: set(keys) for tri, keys in trigram_index_lists.items()}
        skip_tokens = set(non_molecule_tokens) | set(brand_token_lookup)
        salt_skip = set(salt_tokens)

        def _looks_suspicious(norm_basis: str) -> bool:
            if not norm_basis:
                return False
            candidate_tokens = _tokenize_unknowns(norm_basis)
            if not candidate_tokens:
                candidate_tokens = [tok for tok in norm_basis.split() if tok]
            for token in candidate_tokens:
                tok_lower = token.lower()
                if len(tok_lower) < 3:
                    continue
                if tok_lower in skip_tokens:
                    continue
                if tok_lower in therapeutic_tokens:
                    continue
                if tok_lower in salt_skip and len(tok_lower) <= 4:
                    continue
                if tok_lower.isdigit():
                    continue
                tris = _generate_trigrams(tok_lower)
                if not tris:
                    continue
                if any(tri in trigram_lookup for tri in tris):
                    return True
            return False

        tasks: List[Tuple[object, str]] = []
        unresolved_indices = df.index[unresolved_mask]
        for idx in unresolved_indices:
            basis = df.at[idx, "match_basis_norm_basic"]
            if not isinstance(basis, str) or not basis:
                continue
            if not _looks_suspicious(basis):
                continue
            tasks.append((idx, basis))

        if not tasks:
            return

        shared_context = {
            "trigram_index": trigram_lookup if trigram_lookup else {tri: set(keys) for tri, keys in trigram_index_lists.items()},
            "entries": entries_map,
            "non_molecule_tokens": non_molecule_tokens,
            "salt_tokens": salt_tokens,
            "dose_suffixes": DOSE_UNIT_SUFFIXES,
            "therapeutic_tokens": therapeutic_tokens,
            "brand_token_lookup": brand_token_lookup,
            "source_primary": context.get("source_primary", FUZZY_SOURCE_PRIMARY),
            "source_tie": context.get("source_tie", FUZZY_SOURCE_TIE),
            "score_threshold": context.get("score_threshold", FUZZY_SCORE_THRESHOLD),
        }

        worker_count = resolve_worker_count(task_size=len(tasks))
        if worker_count <= 1:
            results = [(idx, _fuzzy_core(basis, shared_context)) for idx, basis in tasks]
        else:
            init_payload = {
                "trigram_index": trigram_index_lists,
                "entries": {k: [dict(entry) for entry in v] for k, v in entries_map.items()},
                "non_molecule_tokens": list(non_molecule_tokens),
                "salt_tokens": list(salt_tokens),
                "dose_suffixes": list(DOSE_UNIT_SUFFIXES),
                "therapeutic_tokens": list(therapeutic_tokens),
                "brand_token_lookup": list(brand_token_lookup),
                "source_primary": context.get("source_primary", FUZZY_SOURCE_PRIMARY),
                "source_tie": context.get("source_tie", FUZZY_SOURCE_TIE),
                "score_threshold": context.get("score_threshold", FUZZY_SCORE_THRESHOLD),
            }
            results = maybe_parallel_map(
                tasks,
                _fuzzy_worker,
                max_workers=worker_count,
                initializer=_fuzzy_worker_init,
                initargs=(init_payload,),
                parallel_threshold=200,
                chunksize=64,
            )

        for idx, match_list in results:
            matches: List[Dict[str, Any]] = match_list or []
            df.at[idx, "fuzzy_matches"] = matches
            if not matches:
                continue

            best = matches[0]
            best_gid = best.get("gid")
            if best_gid is not None:
                current_gid = df.at[idx, "generic_id"]
                if current_gid is None or (isinstance(current_gid, float) and pd.isna(current_gid)):
                    df.at[idx, "generic_id"] = best_gid
                    span = best.get("span")
                    if span:
                        df.at[idx, "molecule_token"] = span

            current_gids_raw = df.at[idx, "pnf_hits_gids"] if "pnf_hits_gids" in df.columns else None
            current_tokens_raw = df.at[idx, "pnf_hits_tokens"] if "pnf_hits_tokens" in df.columns else None
            current_gids = list(current_gids_raw) if isinstance(current_gids_raw, list) else []
            current_tokens = list(current_tokens_raw) if isinstance(current_tokens_raw, list) else []

            for match in matches:
                source = str(match.get("source", "")).lower()
                display = match.get("display") or ""
                span = match.get("span") or ""
                gid = match.get("gid")
                metadata = match.get("metadata") or {}

                if gid:
                    gid_str = str(gid)
                    if gid_str and gid_str not in current_gids:
                        current_gids.append(gid_str)
                    if span and span not in current_tokens:
                        current_tokens.append(span)

                if source == "who":
                    names_list = df.at[idx, "who_molecules_list"] if "who_molecules_list" in df.columns else []
                    if not isinstance(names_list, list):
                        names_list = []
                    codes_list = df.at[idx, "who_atc_codes_list"] if "who_atc_codes_list" in df.columns else []
                    if not isinstance(codes_list, list):
                        codes_list = []
                    if display and display not in names_list:
                        names_list = names_list + [display]
                    for code in metadata.get("codes", []):
                        if code and code not in codes_list:
                            codes_list.append(code)
                    df.at[idx, "who_molecules_list"] = names_list
                    df.at[idx, "who_atc_codes_list"] = codes_list
                    df.at[idx, "who_molecules"] = "|".join(names_list) if names_list else ""
                    df.at[idx, "who_atc_codes"] = "|".join(codes_list) if codes_list else ""
                elif source == "drugbank":
                    db_list = df.at[idx, "drugbank_generics_list"] if "drugbank_generics_list" in df.columns else []
                    if not isinstance(db_list, list):
                        db_list = []
                    if display and display not in db_list:
                        db_list = db_list + [display]
                    df.at[idx, "drugbank_generics_list"] = db_list
                    if "present_in_drugbank" in df.columns:
                        df.at[idx, "present_in_drugbank"] = bool(db_list)
                elif source == "fda":
                    fda_list = df.at[idx, "fda_generics_list"] if "fda_generics_list" in df.columns else []
                    if not isinstance(fda_list, list):
                        fda_list = []
                    if display and display not in fda_list:
                        fda_list = fda_list + [display]
                    df.at[idx, "fda_generics_list"] = fda_list

            if current_gids:
                df.at[idx, "pnf_hits_gids"] = current_gids
                df.at[idx, "pnf_hits_count"] = len(current_gids)
            if current_tokens:
                df.at[idx, "pnf_hits_tokens"] = current_tokens

    _timed("Fuzzy reference fallback", _fuzzy_reference)

    # 13) Unknown tokens extraction
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
                    and t not in drugbank_tokens
                    and t not in stopword_tokens_all
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

        if "fuzzy_matches" in df.columns:
            fuzzy_seq = df["fuzzy_matches"].tolist()
        else:
            fuzzy_seq = [[] for _ in range(len(df))]

        for text_norm, p_hits, who_hits, fuzzy_hits, nonthera_tokens in zip(
            df["match_basis"],
            df["pnf_hits_tokens"],
            df["who_molecules_list"],
            fuzzy_seq,
            nonthera_seq,
        ):
            matched_tokens = _token_set_from_phrase_list(p_hits) | _token_set_from_phrase_list(who_hits)
            if isinstance(fuzzy_hits, list):
                for match in fuzzy_hits:
                    if isinstance(match, dict):
                        span = match.get("span")
                        display = match.get("display")
                        if isinstance(span, str) and span:
                            matched_tokens.update(_tokenize_unknowns(_segment_norm(span)))
                        if isinstance(display, str) and display:
                            matched_tokens.update(_tokenize_unknowns(_segment_norm(display)))
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
        if "drugbank_generics_list" in df.columns:
            df["present_in_drugbank"] = df["drugbank_generics_list"].map(lambda xs: bool(xs))
        else:
            df["present_in_drugbank"] = False
        df["reference_source"] = df["generic_id"].map(lambda gid: gid_to_source.get(str(gid), "") if gid is not None else "")
        df["reference_priority"] = df["generic_id"].map(lambda gid: gid_to_priority.get(str(gid), 99) if gid is not None else 99)
        df["reference_primary_code"] = df["generic_id"].map(lambda gid: gid_to_primary_code.get(str(gid), "") if gid is not None else "")
        df["reference_drug_code"] = df["generic_id"].map(lambda gid: gid_to_drug_code.get(str(gid), "") if gid is not None else "")
        df["reference_route_details"] = df["generic_id"].map(lambda gid: gid_to_route_ref.get(str(gid), "") if gid is not None else "")
        df["present_in_annex"] = df["reference_source"].str.lower().eq("annex_f") & df["generic_id"].notna()
    _timed("Compute presence flags", _presence_flags)

    return df
