#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalize Annex F drug catalogue entries into a structured CSV that mirrors the
prepared PNF schema (dose, route, form) while surfacing the Annex F Drug Code as
the primary identifier.

The heuristics favour safe fallbacks: we only infer routes when the dosage form
or packaging strongly implies a modality (e.g., ampules/vials → intravenous,
tablets/capsules → oral).  Remaining ambiguities are left blank so downstream
reviewers can resolve them explicitly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..constants import PIPELINE_INPUTS_DIR, PROJECT_ROOT
from .dose_drugs import parse_dose_struct_from_text, safe_ratio_mg_per_ml, to_mg
from .routes_forms_drugs import FORM_TO_ROUTE, extract_route_and_form, parse_form_from_text
from .text_utils_drugs import (
    detect_as_boundary,
    extract_base_and_salts,
    extract_parenthetical_phrases,
    normalize_text,
    serialize_salt_list,
    slug_id,
    strip_after_as,
)
from .combos_drugs import SALT_TOKENS, looks_like_combination, split_combo_segments

# Containers observed in Annex F (canonical form -> token variants)
# Recognized containers observed in Annex F (canonical form -> token variants).
CONTAINER_ALIASES = {
    "ampule": {"ampule", "ampul", "ampoule", "amp", "ampu"},
    "vial": {"vial", "vialx"},
    "bottle": {"bottle", "bot", "bottl"},
    "bag": {"bag", "bagxx"},
    "can": {"can"},
    "sachet": {"sachet", "sachets"},
    "capsule": {"capsule", "capsules", "cap", "caps"},
    "tablet": {"tablet", "tablets", "tab", "tabs", "tabx"},
    "tube": {"tube", "tub", "tubes"},
    "drops": {"drop", "drops"},
    "patch": {"patch", "patches"},
    "syringe": {"syringe", "syringes"},
    "nebule": {"nebule", "nebules"},
}

# Map volume/weight tokens to canonical units.
UNIT_ALIASES = {
    "ml": {"ml", "milliliter", "milliliters"},
    "l": {"l", "liter", "litre", "liters", "litres"},
    "g": {"g", "gram", "grams"},
    "mg": {"mg"},
    "mcg": {"mcg", "ug"},
}

# Reverse lookups for quick normalization during token scans.
UNIT_NORMAL = {alias: base for base, aliases in UNIT_ALIASES.items() for alias in aliases}
CONTAINER_NORMAL = {alias: base for base, aliases in CONTAINER_ALIASES.items() for alias in aliases}

# Tokens that should never be stripped from molecule names when building the Annex F generic.
SALT_WHITELIST = set(SALT_TOKENS)
SALT_TOKEN_WORDS = {token.lower() for token in SALT_TOKENS}
for phrase in SALT_TOKENS:
    for part in normalize_text(phrase).split():
        SALT_TOKEN_WORDS.add(part)

# Additional words that add no value to the canonical molecule string.
GENERIC_STOPWORDS = {
    "ml",
    "l",
    "mg",
    "mcg",
    "ug",
    "g",
    "iu",
    "lsu",
    "per",
    "as",
    "with",
    "and",
    "w",
    "v",
    "w/v",
    "w/w",
    "x",
    "per",
    "dose",
    "doses",
    "unit",
    "units",
    "meter",
    "metered",
    "count",
    "counts",
    "sol",
    "soln",
    "susp",
    "syr",
    "usp",
    "bp",
    "ep",
    "nf",
}

GENERIC_STOPWORDS |= {  # forms/vehicles that do not affect the molecule identity
    "solution",
    "suspension",
    "syrup",
    "powder",
    "cream",
    "ointment",
    "gel",
    "lotion",
    "spray",
    "drops",
    "drop",
    "nebule",
    "neb",
    "inhaler",
}

# Compile regexes once so the preparation pass stays fast enough for CLI usage.
# Token regex helpers reused during parsing.
DIGIT_ONLY_RX = re.compile(r"^\d+(?:\.\d+)?$")
UNIT_FRAGMENT_RX = re.compile(r"(?:mg|mcg|ug|g|iu|lsu|ml|l|%)", re.I)

PNF_PRIORITY = 0
WHO_PRIORITY = 1
DRUGBANK_GENERIC_PRIORITY = 2
FDA_BRAND_PRIORITY = 3
DRUGBANK_BRAND_PRIORITY = 4
FDA_FOOD_PRIORITY = 5

MAX_SINGLE_TOKEN_START_INDEX = 4
GENERIC_SINGLE_TOKEN_BLACKLIST = {
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "chloride",
    "iodine",
    "iron",
    "zinc",
    "solution",
    "suspension",
    "tablet",
    "capsule",
    "cream",
    "ointment",
    "powder",
    "syrup",
    "drops",
    "drop",
    "gel",
    "spray",
    "lotion",
    "ampule",
    "vial",
    "bag",
    "bottle",
    "vitamins",
    "trace",
    "elements",
    "intravenous",
    "intramuscular",
    "subcutaneous",
}

COMBO_KEYWORDS = {
    "combined",
    "combination",
    "compound",
    "coformulated",
    "co-formulated",
    "coformulation",
}

COMBO_ALPHA_PLUS_RX = re.compile(r"[a-z]\s*\+\s*[a-z]")
COMBO_ALPHA_SLASH_RX = re.compile(r"[a-z]\s*/\s*[a-z]")
VITAMIN_COMPLEX_RX = re.compile(r"\bvitamin\s+b\d+(?:\s+b\d+)+")
VITAMIN_B_TOKEN_RX = re.compile(r"\b(b\d+)\b", re.I)
DOSE_DIGIT_RX = re.compile(r"\d")

EXTRA_TOKEN_STOPWORDS = (
    GENERIC_STOPWORDS
    | {
        "as",
        "per",
        "contains",
        "containing",
        "include",
        "including",
        "compound",
        "combination",
        "combined",
        "combo",
        "plus",
        "and",
        "with",
        "without",
        "coformulated",
        "co-formulated",
        "ratio",
        "preparation",
        "formula",
        "formulation",
        "applicator",
        "applicators",
        "dropper",
        "droppers",
        "device",
        "devices",
        "kit",
        "kits",
        "unit",
        "units",
        "meter",
        "metered",
        "count",
        "counts",
    }
    | set(CONTAINER_NORMAL.keys())
    | set(UNIT_NORMAL.keys())
    | SALT_TOKEN_WORDS
)

@dataclass(frozen=True)
class _NameEntry:
    tokens: Tuple[str, ...]
    canonical: str
    priority: int
    source: str


@dataclass(frozen=True)
class _ResolvedMatch:
    entry: _NameEntry
    start: int
    end: int


class _ReferenceNameResolver:
    """Token-based matching between Annex F free text and reference catalogues."""

    def __init__(self) -> None:
        self.token_index: Dict[str, List[_NameEntry]] = {}
        self._seen: set[Tuple[Tuple[str, ...], str]] = set()

    def register(self, label: str, canonical: Optional[str], priority: int, source: str) -> None:
        canonical = (canonical or label).strip()
        if not canonical:
            return
        tokens = tuple(normalize_text(label).split())
        if not tokens:
            return
        key = (tokens, canonical.lower())
        if key in self._seen:
            return
        self._seen.add(key)
        entry = _NameEntry(tokens=tokens, canonical=canonical, priority=priority, source=source)
        bucket = self.token_index.setdefault(tokens[0], [])
        bucket.append(entry)

    def resolve(self, raw_text: str) -> Optional[_ResolvedMatch]:
        tokens = tuple(normalize_text(raw_text).split())
        if not tokens:
            return None
        best: Optional[_NameEntry] = None
        best_start = -1
        # Prefer longest span, then highest priority (lower number), then earliest position.
        best_score: Tuple[int, int, int] = (-1, 0, 0)
        for idx, tok in enumerate(tokens):
            bucket = self.token_index.get(tok)
            if not bucket:
                continue
            for entry in bucket:
                span = len(entry.tokens)
                if idx + span > len(tokens):
                    continue
                if entry.tokens != tokens[idx : idx + span]:
                    continue
                score = (span, -entry.priority, -idx)
                if score > best_score:
                    best = entry
                    best_score = score
                    best_start = idx
        if not best:
            return None
        start_idx = best_start if best_start >= 0 else 0
        end_idx = start_idx + len(best.tokens) if best else start_idx
        return _ResolvedMatch(entry=best, start=start_idx, end=end_idx)

    def resolve_all(self, raw_text: str) -> List[_ResolvedMatch]:
        tokens = tuple(normalize_text(raw_text).split())
        if not tokens:
            return []
        matches: List[_ResolvedMatch] = []
        seen_spans: set[Tuple[int, int, str]] = set()
        for idx, tok in enumerate(tokens):
            bucket = self.token_index.get(tok)
            if not bucket:
                continue
            for entry in bucket:
                span = len(entry.tokens)
                if idx + span > len(tokens):
                    continue
                if entry.tokens != tokens[idx : idx + span]:
                    continue
                key = (idx, idx + span, entry.canonical.lower())
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                matches.append(_ResolvedMatch(entry=entry, start=idx, end=idx + span))
        matches.sort(key=lambda m: (m.start, -len(m.entry.tokens), m.entry.priority))
        return matches


def _token_has_letters(token: str) -> bool:
    return bool(token and re.search(r"[a-z]", token))


def _token_is_substance(token: str) -> bool:
    tok = (token or "").lower()
    if not _token_has_letters(tok):
        return False
    if tok in EXTRA_TOKEN_STOPWORDS:
        return False
    if tok in CONTAINER_NORMAL or tok in UNIT_NORMAL:
        return False
    if tok in {"-", "+", "/"}:
        return False
    return True


def _first_dose_token_index(tokens: Sequence[str]) -> int:
    for idx, tok in enumerate(tokens):
        if not tok:
            continue
        low = tok.lower()
        if low in UNIT_NORMAL or low in {"mg", "mcg", "ug", "g", "iu", "lsu", "ml", "l", "%"}:
            return idx
        if low in {"per", "ratio"}:
            return idx
        if DOSE_DIGIT_RX.search(tok):
            return idx
    return len(tokens)


def _combo_indicator_present(normalized_desc: str, tokens: Sequence[str]) -> bool:
    text = normalized_desc
    if COMBO_ALPHA_PLUS_RX.search(text):
        return True
    if COMBO_ALPHA_SLASH_RX.search(text):
        return True
    if any(tok in COMBO_KEYWORDS for tok in tokens):
        return True
    if VITAMIN_COMPLEX_RX.search(text):
        return True
    return False


def _name_looks_combo(name: str) -> bool:
    norm = normalize_text(name)
    if COMBO_ALPHA_PLUS_RX.search(norm):
        return True
    if COMBO_ALPHA_SLASH_RX.search(norm):
        return True
    if " with " in norm:
        return True
    if " and " in norm:
        return True
    if any(keyword in norm for keyword in COMBO_KEYWORDS):
        return True
    return False


def _has_unmatched_substances(
    normalized_desc: str,
    match: Optional[_ResolvedMatch],
) -> bool:
    if not match:
        return False
    tokens = normalized_desc.split()
    limit = _first_dose_token_index(tokens)
    match_range = set(range(match.start, match.end))
    for idx, tok in enumerate(tokens[:limit]):
        if idx in match_range:
            continue
        if tok in {"+", "/"}:
            continue
        if _token_is_substance(tok):
            return True
    return False


def _read_csv(path: Path, usecols: Iterable[str]) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, usecols=list(usecols), dtype=str).fillna("")
    except Exception:
        return None
    return frame


def _register_column_values(
    resolver: _ReferenceNameResolver,
    path: Path,
    column: str,
    priority: int,
    source: str,
) -> bool:
    frame = _read_csv(path, [column])
    if frame is None or column not in frame.columns:
        return False
    for value in frame[column].dropna().astype(str):
        clean = value.strip()
        if not clean:
            continue
        resolver.register(clean, clean, priority, source)
    return True


def _register_alias_pairs(
    resolver: _ReferenceNameResolver,
    path: Path,
    alias_col: str,
    canonical_col: str,
    priority: int,
    source: str,
) -> None:
    frame = _read_csv(path, [alias_col, canonical_col])
    if frame is None:
        return
    for _, row in frame.iterrows():
        alias = str(row.get(alias_col, "") or "").strip()
        canonical = str(row.get(canonical_col, "") or "").strip()
        if not canonical:
            continue
        resolver.register(canonical, canonical, max(priority - 1, 0), source)
        if alias:
            resolver.register(alias, canonical, priority, source)


CUSTOM_ALIAS_ENTRIES: Tuple[Tuple[str, str, int], ...] = (
    ("Vitamins Intravenous Fat-Soluble", "Vitamins Intravenous, Fat-Soluble", 0),
    ("Vitamins Intravenous Trace Elements", "Vitamins Intravenous, Trace Elements", 0),
    ("Vitamins Intravenous Water-Soluble", "Vitamins Intravenous, Water-Soluble", 0),
    ("Vitamins Intravenous", "Vitamins Intravenous", 1),
    ("Dextrose in Lactated Ringer's Solution", "Dextrose in Lactated Ringer's Solution", 0),
)


@lru_cache(maxsize=1)
def _reference_name_resolver() -> _ReferenceNameResolver:
    resolver = _ReferenceNameResolver()

    # PNF prepared (preferred) then fallback to raw PNF.
    pnf_prepared = PIPELINE_INPUTS_DIR / "pnf_prepared.csv"
    if not _register_column_values(resolver, pnf_prepared, "generic_name", PNF_PRIORITY, "pnf_prepared"):
        _register_column_values(
            resolver,
            PIPELINE_INPUTS_DIR / "pnf.csv",
            "Molecule",
            PNF_PRIORITY,
            "pnf_raw",
        )

    # WHO ATC molecule lists (allow multiple dated files).
    for path in sorted(PIPELINE_INPUTS_DIR.glob("who_atc*_molecules.csv")):
        _register_column_values(resolver, path, "atc_name", WHO_PRIORITY, f"who:{path.name}")

    # FDA brand map(s) contribute both generic and brand aliases.
    for path in sorted(PIPELINE_INPUTS_DIR.glob("fda_brand_map*.csv")):
        _register_alias_pairs(
            resolver,
            path,
            "brand_name",
            "generic_name",
            FDA_BRAND_PRIORITY,
            f"fda:{path.name}",
        )

    # DrugBank generics/brands (prefer freshly exported dependencies, then inputs copies).
    drugbank_candidates = [
        PROJECT_ROOT / "dependencies" / "drugbank_generics" / "output" / "drugbank_generics.csv",
        PIPELINE_INPUTS_DIR / "drugbank_generics.csv",
        PIPELINE_INPUTS_DIR / "generics.csv",
    ]
    for path in drugbank_candidates:
        _register_column_values(
            resolver,
            path,
            "generic",
            DRUGBANK_GENERIC_PRIORITY,
            f"drugbank:{path.name}",
        )

    drugbank_brand_candidates = [
        PROJECT_ROOT / "dependencies" / "drugbank_generics" / "output" / "drugbank_brands.csv",
        PIPELINE_INPUTS_DIR / "drugbank_brands.csv",
    ]
    for path in drugbank_brand_candidates:
        _register_alias_pairs(
            resolver,
            path,
            "brand",
            "generic",
            DRUGBANK_BRAND_PRIORITY,
            f"drugbank_brand:{path.name}",
        )

    food_catalog = PIPELINE_INPUTS_DIR / "fda_food_products.csv"
    _register_column_values(
        resolver,
        food_catalog,
        "brand_name",
        FDA_FOOD_PRIORITY,
        "fda_food_brand",
    )
    _register_column_values(
        resolver,
        food_catalog,
        "product_name",
        FDA_FOOD_PRIORITY,
        "fda_food_product",
    )

    for alias, canonical, priority in CUSTOM_ALIAS_ENTRIES:
        resolver.register(alias, canonical, priority, "custom")

    return resolver


def _fallback_generic_name(raw_desc: str, normalized_desc: str, as_index: Optional[int]) -> str:
    """Strip dose/pack cues to surface the base Annex F molecule name when references miss."""
    if not isinstance(raw_desc, str):
        return ""
    norm = normalized_desc if normalized_desc else normalize_text(raw_desc)
    norm = norm.replace("+", " + ").replace("/", " / ")
    tokens_source = norm.split()
    if as_index is not None and as_index > 0:
        tokens_source = tokens_source[:as_index]
    tokens: list[str] = []
    for tok in tokens_source:
        if not tok or tok == "/":
            continue
        if DIGIT_ONLY_RX.fullmatch(tok):
            continue
        if tok in GENERIC_STOPWORDS and tok not in SALT_WHITELIST:
            continue
        if tok in CONTAINER_NORMAL:
            continue
        if tok in UNIT_NORMAL:
            continue
        if UNIT_FRAGMENT_RX.search(tok) and not tok.isalpha() and tok.lower() not in SALT_WHITELIST:
            continue
        if re.search(r"\d", tok):
            if not re.search(r"[a-z]", tok) or tok[0].isdigit():
                continue
        tokens.append(tok)

    cleaned: list[str] = []
    prev_plus = False
    for tok in tokens:
        if tok == "+":
            if not cleaned or prev_plus:
                continue
            prev_plus = True
            cleaned.append(tok)
            continue
        prev_plus = False
        cleaned.append(tok)

    while cleaned and cleaned[-1] == "+":
        cleaned.pop()
    while cleaned and cleaned[0] == "+":
        cleaned.pop(0)
    if not cleaned:
        # Fall back to the raw uppercase name when stripping produced nothing.
        return raw_desc.strip().upper()
    return re.sub(r"\s+\+\s+", " + ", " ".join(cleaned)).upper()


def _looks_like_vitamin_combo(norm_text: str) -> bool:
    return bool(VITAMIN_COMPLEX_RX.search(norm_text))


def _combo_fragments(raw_desc: str, normalized_desc: str) -> List[str]:
    fragments: List[str] = []
    seen: set[str] = set()
    for snippet in extract_parenthetical_phrases(raw_desc):
        norm = normalize_text(snippet)
        if not norm or norm in seen:
            continue
        if looks_like_combination(norm, 0, 0) or _looks_like_vitamin_combo(norm):
            fragments.append(norm)
            seen.add(norm)

    def _needs_full_scan(text: str) -> bool:
        if not text:
            return False
        if looks_like_combination(text, 0, 0):
            return True
        if _looks_like_vitamin_combo(text):
            return True
        if any(keyword in text for keyword in COMBO_KEYWORDS):
            return True
        return False

    if normalized_desc and _needs_full_scan(normalized_desc) and normalized_desc not in seen:
        fragments.append(normalized_desc)
    return fragments


def _segment_has_molecule_hint(segment: str) -> bool:
    for tok in segment.split():
        if not tok:
            continue
        if DIGIT_ONLY_RX.fullmatch(tok):
            continue
        if tok in UNIT_NORMAL or tok in CONTAINER_NORMAL:
            continue
        if tok in GENERIC_STOPWORDS:
            continue
        if re.search(r"[a-z]", tok):
            return True
    return False


def _expand_component_aliases(
    name: str,
    salts: List[str],
    raw_source: Optional[str] = None,
) -> List[Tuple[str, List[str]]]:
    upper = (name or "").strip().upper()
    if not upper:
        return []
    source = (raw_source or name or "").strip().upper()
    b_tokens = [match.upper() for match in VITAMIN_B_TOKEN_RX.findall(source)]
    if upper.startswith("VITAMIN") and len(b_tokens) >= 2:
        return [(f"VITAMIN {token}", []) for token in b_tokens]
    return [(upper, salts)]


def _segment_component_entries(segment: str, resolver: _ReferenceNameResolver) -> List[Tuple[str, List[str]]]:
    if not segment or not _segment_has_molecule_hint(segment):
        return []
    tokens = segment.split()
    matches = resolver.resolve_all(segment)
    components: List[Tuple[str, List[str]]] = []
    seen_spans: set[Tuple[int, int]] = set()
    seen_names: set[str] = set()
    for match in matches:
        canonical_raw = match.entry.canonical or ""
        canonical_norm = normalize_text(canonical_raw)
        if not _segment_has_molecule_hint(canonical_norm):
            continue
        if "+" in canonical_norm or "/" in canonical_norm or " with " in canonical_norm:
            continue
        canonical_tokens = canonical_norm.split()
        if canonical_tokens and all(
            tok in SALT_TOKEN_WORDS or tok in UNIT_NORMAL or tok in CONTAINER_NORMAL for tok in canonical_tokens
        ):
            continue
        span = (match.start, match.end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        snippet_tokens = tokens[match.start : match.end]
        snippet_text = " ".join(snippet_tokens)
        base, salts = extract_base_and_salts(snippet_text or canonical_raw)
        source_text = snippet_text or canonical_raw
        for expanded_name, expanded_salts in _expand_component_aliases(base or canonical_raw, salts, source_text):
            key = expanded_name.strip().upper()
            if not key:
                continue
            key_norm = normalize_text(expanded_name)
            if not _segment_has_molecule_hint(key_norm):
                continue
            key_tokens = key_norm.split()
            if key_tokens and all(tok in SALT_TOKEN_WORDS for tok in key_tokens):
                continue
            if key in seen_names:
                continue
            seen_names.add(key)
            components.append((key, expanded_salts))
    if len(components) > 1:
        has_specific_vitamins = any(name.startswith("VITAMIN ") for name, _ in components)
        if has_specific_vitamins:
            components = [comp for comp in components if comp[0] != "VITAMIN"]
    if not components:
        base, salts = extract_base_and_salts(segment)
        components = _expand_component_aliases(base, salts, segment) if base else []
    return components


def _combo_components_from_text(
    raw_desc: str,
    normalized_desc: str,
    resolver: _ReferenceNameResolver,
) -> Tuple[List[str], List[str]]:
    fragments = _combo_fragments(raw_desc, normalized_desc)
    if not fragments:
        return [], []
    components: List[Tuple[str, List[str]]] = []
    seen_names: set[str] = set()
    for fragment in fragments:
        segments = split_combo_segments(fragment) or [fragment]
        frag_components: List[Tuple[str, List[str]]] = []
        for segment in segments:
            frag_components.extend(_segment_component_entries(segment, resolver))
        for name, salts in frag_components:
            key = name.strip().upper()
            if not key or key in seen_names:
                continue
            seen_names.add(key)
            components.append((key, salts))
        if len(components) >= 2:
            break
    if len(components) < 2:
        return [], []
    salt_tokens: List[str] = []
    for _, salts in components:
        salt_tokens.extend(salts)
    ordered_names = [name for name, _ in components]
    return ordered_names, salt_tokens


def _looks_like_vitamin_combo(norm_text: str) -> bool:
    return bool(VITAMIN_COMPLEX_RX.search(norm_text))


def _vitamin_descriptor(normalized_desc: str) -> Optional[str]:
    if "vitamins intravenous" not in normalized_desc:
        return None
    if "trace elements" in normalized_desc:
        return "Trace Elements"
    if "fat soluble" in normalized_desc:
        return "Fat-Soluble"
    if "water soluble" in normalized_desc:
        return "Water-Soluble"
    return None
def _accept_resolved_match(match: _ResolvedMatch, as_index: Optional[int]) -> bool:
    entry = match.entry
    tokens = entry.tokens
    if as_index is not None and as_index > 0 and match.start >= as_index:
        return False
    if len(tokens) > 1:
        return True
    token = tokens[0]
    if not re.search(r"[a-z]", token):
        return False
    if token in GENERIC_SINGLE_TOKEN_BLACKLIST:
        return False
    if token in GENERIC_STOPWORDS:
        return False
    if match.start > MAX_SINGLE_TOKEN_START_INDEX:
        return False
    if match.start <= 2:
        return True
    return False


def _derive_generic_name(
    raw_desc: str,
    normalized_desc: str,
    resolver: _ReferenceNameResolver,
) -> Tuple[str, str, List[str]]:
    custom_descriptor = _vitamin_descriptor(normalized_desc)
    tokens_full = normalized_desc.split()
    combo_hint = _combo_indicator_present(normalized_desc, tokens_full)
    trimmed_norm = strip_after_as(normalized_desc)
    norm_for_matching = (
        normalized_desc
        if combo_hint
        else (trimmed_norm if trimmed_norm else normalized_desc)
    )
    as_index_full = detect_as_boundary(normalized_desc)
    as_index_trimmed = detect_as_boundary(norm_for_matching)
    base_name_from_text, salts_from_text = extract_base_and_salts(norm_for_matching)
    combo_components, combo_salts = _combo_components_from_text(raw_desc, normalized_desc, resolver)
    if combo_components:
        salt_tokens = combo_salts.copy()
        for tok in salts_from_text:
            if tok not in salt_tokens:
                salt_tokens.append(tok)
        generic_combo_name = " + ".join(combo_components).upper()
        if custom_descriptor and custom_descriptor.lower() not in generic_combo_name.lower():
            generic_combo_name = f"{generic_combo_name} ({custom_descriptor})"
        return generic_combo_name, "combo_fallback", salt_tokens
    resolved = resolver.resolve(raw_desc) if isinstance(raw_desc, str) else None
    match_source = "fallback"
    if resolved and not _accept_resolved_match(resolved, as_index_full):
        resolved = None
    if resolved and combo_hint and not _name_looks_combo(resolved.entry.canonical):
        if _has_unmatched_substances(normalized_desc, resolved):
            resolved = None
    if resolved:
        name = resolved.entry.canonical
        match_source = resolved.entry.source
        resolved_norm_tokens = normalize_text(name).split()
        if base_name_from_text and resolved_norm_tokens and all(tok in SALT_TOKEN_WORDS for tok in resolved_norm_tokens):
            name = base_name_from_text
            match_source = "fallback_preferred"
    else:
        name = _fallback_generic_name(raw_desc, norm_for_matching, as_index_trimmed)
        match_source = "fallback"
    base_name, salts = extract_base_and_salts(name)
    generic_name = base_name or name
    salt_tokens = salts or salts_from_text
    if custom_descriptor and custom_descriptor.lower() not in generic_name.lower():
        generic_name = f"{generic_name} ({custom_descriptor})"
    return generic_name.upper(), match_source, salt_tokens


PACK_FREETEXT_SKIP = {
    "sterile",
    "sterilized",
    "solution",
    "suspension",
    "syrup",
    "powder",
    "cream",
    "ointment",
    "gel",
    "lotion",
    "topical",
    "oral",
    "for",
    "concentrate",
    "drops",
    "drop",
}


def _scan_packaging(tokens: Iterable[str]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Walk tokens from the tail to capture '[qty] [unit] [container]' patterns."""
    tokens = list(tokens)
    qty: Optional[float] = None
    unit: Optional[str] = None
    container: Optional[str] = None

    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i]
        base_container = CONTAINER_NORMAL.get(tok)
        if base_container:
            container = base_container
            # Try to consume unit + quantity immediately preceding the container token.
            idx = i - 1
            # Skip filler words between the container and the numeric/unit pair (e.g., "solution").
            while idx >= 0 and tokens[idx] in PACK_FREETEXT_SKIP:
                idx -= 1
            if idx >= 0:
                unit_candidate = UNIT_NORMAL.get(tokens[idx])
                if unit_candidate:
                    unit = unit_candidate
                    if idx - 1 >= 0:
                        try:
                            qty = float(tokens[idx - 1])
                        except ValueError:
                            qty = None
                    continue
            # Alternate pattern: quantity directly before the container without explicit unit.
            if idx >= 0:
                try:
                    qty = float(tokens[idx])
                except ValueError:
                    qty = None
            break

    return qty, unit, container


def _normalize_pack_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    if unit == "l":
        return "ml"
    return unit


def _deduce_route_form(
    normalized: str,
    pack_container: Optional[str],
) -> Tuple[Optional[str], Optional[str], str]:
    """Combine text-derived clues with packaging hints to infer form + route."""
    form_primary = parse_form_from_text(normalized)
    route_primary, form_secondary, base_evidence = extract_route_and_form(normalized)
    evidence_parts = [part for part in base_evidence.split(";") if part]
    evidence_seen = set(evidence_parts)

    def _add_evidence(tag: str) -> None:
        if not tag or tag in evidence_seen:
            return
        evidence_seen.add(tag)
        evidence_parts.append(tag)

    form_token = form_primary or form_secondary
    route_token = route_primary
    norm_text = normalized.lower()

    def _register_form(form: str, route: Optional[str], reason: str) -> None:
        nonlocal form_token, route_token
        if form and not form_token:
            form_token = form
            _add_evidence(f"{reason}:form={form}")
        if route and (not route_token or (form and form.startswith("eye"))):
            route_token = route
            _add_evidence(f"{reason}:route={route}")

    keyword_forms: tuple[tuple[str, str, Optional[str]], ...] = (
        ("eye drops", "eye drops", "ophthalmic"),
        ("eye drop", "eye drops", "ophthalmic"),
        ("ear drops", "ear drops", "otic"),
        ("ear drop", "ear drops", "otic"),
        ("nasal drops", "nasal drops", "nasal"),
        ("nasal drop", "nasal drops", "nasal"),
        ("oral drops", "oral drops", "oral"),
        ("ovule", "ovule", "vaginal"),
        ("ovules", "ovule", "vaginal"),
        ("shampoo", "shampoo", "topical"),
        ("soap", "soap", "topical"),
        ("wash", "wash", "topical"),
        ("granules", "granule", "oral"),
        ("granule", "granule", "oral"),
        ("lozenge", "lozenge", "oral"),
        ("mouthwash", "mouthwash", "oral"),
    )

    if not form_token:
        for token, canonical, route in keyword_forms:
            if token in norm_text:
                _register_form(canonical, route or FORM_TO_ROUTE.get(canonical), f"keyword:{token}")
                break

    plural_map = (
        ("solutions", "solution"),
        ("suspensions", "suspension"),
        ("syrups", "syrup"),
        ("lotions", "lotion"),
        ("creams", "cream"),
        ("ointments", "ointment"),
    )
    if not form_token:
        for token, canonical in plural_map:
            if token in norm_text:
                _register_form(canonical, FORM_TO_ROUTE.get(canonical), f"keyword:{token}->{canonical}")
                break

    if not form_token and "solution" in norm_text:
        _register_form("solution", FORM_TO_ROUTE.get("solution"), "keyword:solution")
    if not form_token and "suspension" in norm_text:
        _register_form("suspension", FORM_TO_ROUTE.get("suspension"), "keyword:suspension")
    if not form_token and "syrup" in norm_text:
        _register_form("syrup", FORM_TO_ROUTE.get("syrup"), "keyword:syrup")

    # Packaging overrides: ampules/vials are injectable, nebules inhalation, etc.
    if pack_container in {"ampule", "vial"}:
        form_token = pack_container
        if route_token not in {"intravenous", "intramuscular", "subcutaneous"}:
            route_token = "intravenous"
            _add_evidence(f"packaging->{pack_container}:intravenous")
    elif pack_container == "nebule":
        form_token = "nebule"
        route_token = "inhalation"
        _add_evidence("packaging->nebule:inhalation")
    elif pack_container == "syringe":
        route_token = "intravenous"
        _add_evidence("packaging->syringe:intravenous")
    elif pack_container == "drops":
        if "eye" in norm_text:
            _register_form("eye drops", "ophthalmic", "packaging:drops_eye")
        elif "ear" in norm_text:
            _register_form("ear drops", "otic", "packaging:drops_ear")
        elif "nasal" in norm_text:
            _register_form("nasal drops", "nasal", "packaging:drops_nasal")
        else:
            _register_form("oral drops", "oral", "packaging:drops_oral")
    elif pack_container in {"bottle", "bag"}:
        if "intravenous" in norm_text or "iv" in norm_text.split():
            route_token = "intravenous"
            _add_evidence("text->intravenous")
        elif form_token in {"solution"}:
            # Large-volume solutions in bottles/bags typically indicate parenteral use.
            route_token = "intravenous"
            _add_evidence(f"packaging->{pack_container}:assume_intravenous")

    if not route_token and form_token in FORM_TO_ROUTE:
        route_token = FORM_TO_ROUTE[form_token]
        _add_evidence(f"form_impute->{form_token}:{route_token}")

    return route_token, form_token, ";".join(evidence_parts)


def prepare_annex_f(input_csv: str, output_csv: str) -> str:
    """Entry point used by CLI/automation to normalize Annex F into CSV form."""
    source_path = Path(input_csv)
    if not source_path.is_file():
        raise FileNotFoundError(f"Annex F CSV not found: {input_csv}")

    frame = pd.read_csv(source_path, dtype=str).fillna("")
    if "Drug Code" not in frame.columns or "Drug Description" not in frame.columns:
        raise ValueError("annex_f.csv must contain 'Drug Code' and 'Drug Description' columns")

    resolver = _reference_name_resolver()
    records = []
    for raw_code, raw_desc in frame[["Drug Code", "Drug Description"]].itertuples(index=False):
        desc = (raw_desc or "").strip()
        norm = normalize_text(desc)
        tokens = norm.split()
        pack_qty, pack_unit, pack_container = _scan_packaging(tokens)
        if pack_unit == "l" and pack_qty is not None and not math.isnan(pack_qty):
            pack_qty *= 1000.0
        pack_unit = _normalize_pack_unit(pack_unit)

        parsed_dose = parse_dose_struct_from_text(norm)
        dose_kind = parsed_dose.get("dose_kind")
        strength = parsed_dose.get("strength")
        unit = parsed_dose.get("unit")
        per_val = parsed_dose.get("per_val")
        per_unit = parsed_dose.get("per_unit")
        pct = parsed_dose.get("pct")

        strength_mg = to_mg(strength, unit) if strength is not None else None
        ratio_mg_per_ml = (
            safe_ratio_mg_per_ml(strength, unit, per_val)
            if dose_kind == "ratio" and strength is not None
            else None
        )

        route_allowed, form_token, route_evidence = _deduce_route_form(norm, pack_container)

        generic_name, generic_source, salt_tokens = _derive_generic_name(desc, norm, resolver)
        generic_id = slug_id(generic_name) if generic_name else ""
        salt_form = serialize_salt_list(salt_tokens)

        records.append(
            {
                "drug_code": str(raw_code).strip(),
                "generic_id": generic_id,
                "generic_name": generic_name,
                "generic_source": generic_source,
                "salt_form": salt_form,
                "raw_description": desc,
                "normalized_description": norm,
                "dose_kind": dose_kind,
                "strength": strength,
                "unit": unit,
                "per_val": per_val,
                "per_unit": per_unit,
                "pct": pct,
                "strength_mg": strength_mg,
                "ratio_mg_per_ml": ratio_mg_per_ml,
                "route_allowed": route_allowed or "",
                "form_token": form_token or "",
                "route_evidence": route_evidence,
                "pack_quantity": pack_qty,
                "pack_unit": pack_unit,
                "pack_container": pack_container or "",
            }
        )

    out_frame = pd.DataFrame.from_records(records)
    out_frame.to_csv(output_csv, index=False, encoding="utf-8")
    return str(Path(output_csv).resolve())


def main() -> None:
    """CLI wrapper to allow `python -m pipelines.drugs.scripts.prepare_annex_f_drugs` execution."""
    inputs_dir = PIPELINE_INPUTS_DIR
    input_csv = inputs_dir / "annex_f.csv"
    output_csv = inputs_dir / "annex_f_prepared.csv"
    path = prepare_annex_f(str(input_csv), str(output_csv))
    print(f"Wrote Annex F prepared CSV: {path}")


if __name__ == "__main__":
    main()
