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
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..constants import PIPELINE_INPUTS_DIR, PROJECT_ROOT
from .dose_drugs import parse_dose_struct_from_text, safe_ratio_mg_per_ml, to_mg
from .routes_forms_drugs import FORM_TO_ROUTE, ROUTE_ALIASES, extract_route_and_form, parse_form_from_text
from .text_utils_drugs import (
    SPECIAL_SALT_TOKENS,
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
    "gas",
    "agent",
    "agents",
    "forming",
    "nebule",
    "neb",
    "inhaler",
    "ampule",
    "amp",
    "ampul",
    "ampoule",
    "vial",
    "vials",
    "bottle",
    "bottl",
    "bott",
    "bottles",
    "bag",
    "bags",
    "pack",
    "packs",
    "box",
    "boxes",
    "kit",
    "kits",
    "sachet",
    "sachets",
    "pouch",
    "container",
    "jar",
    "jars",
    "intravenous",
    "intramuscular",
    "subcutaneous",
    "intrathecal",
    "parenteral",
    "oral",
    "topical",
    "dermal",
    "cutaneous",
    "ophthalmic",
    "otic",
    "nasal",
    "inhalation",
    "transdermal",
    "sublingual",
    "buccal",
    "injection",
    "injectable",
    "infusion",
    "eye",
    "ear",
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

RINGER_TERMS = {"RINGER", "RINGERS", "RINGER'S"}
RINGER_SOLUTION_RX = re.compile(r"ringer(?:'s|s|\s+s)?\s+solution", re.IGNORECASE)
SYNONYM_SKIP_PREFIXES = ("AS ", "AS/", "FOR ", "WITH ", "IN ", "OF ", "TO ", "ON ")
SYNONYM_SKIP_TERMS = {"SOLUTION"}

GENERIC_ROUTE_FORM_TOKENS = (
    {token.upper() for token in FORM_TO_ROUTE}
    | {alias.upper() for alias in ROUTE_ALIASES}
    | {route.upper() for route in ROUTE_ALIASES.values()}
    | {container.upper() for container in set(CONTAINER_NORMAL.values())}
    | {
        "INJECTION",
        "INFUSION",
        "INJECTABLE",
        "DILUENT",
        "CONCENTRATE",
        "CONCENTRATED",
        "STERILE",
    }
)
GENERIC_MEASUREMENT_SUFFIXES = {
    "mg",
    "mcg",
    "ug",
    "ml",
    "l",
    "g",
    "iu",
    "lsu",
    "meq",
    "meqs",
    "cc",
    "pct",
    "cl",
    "kg",
}

PLACEHOLDER_TOKENS = {
    "balanced",
    "split",
    "group",
    "adult",
    "pedia",
    "pediatric",
    "agent",
    "agents",
    "pouch",
    "cartridge",
    "diluent",
    "dispersible",
    "chewable",
    "equiv",
    "equivalent",
    "standard",
    "standardized",
    "dose",
    "doses",
    "mci",
    "drugs",
    "medicines",
    "episode",
    "care",
    "liquid",
    "concentrate",
}

PREFIX_KEYWORDS = {
    "agent",
    "vaccine",
    "dialysate",
    "hemodialysis",
    "irrigating",
    "admixture",
    "solution",
    "maintenance",
    "replacement",
    "intravenous",
    "gas",
    "balanced",
    "diluent",
}

FORM_BREAK_TOKENS = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "syrup",
    "suspension",
    "drop",
    "drops",
    "ointment",
    "cream",
    "powder",
    "gel",
}

MEASUREMENT_COMPONENT_HINTS = {"elemental", "equivalent", "equiv", "approximate", "radioactive"}

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


def _token_key(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (token or "").lower())


def _looks_like_measurement(token: str) -> bool:
    tok = (token or "").lower()
    if not tok:
        return False
    if tok in UNIT_NORMAL or tok in {"ml", "l", "g", "mg", "mcg", "iu", "lsu", "meq", "meqs", "pct"}:
        return True
    if "%" in tok:
        return True
    if DIGIT_ONLY_RX.fullmatch(tok):
        return True
    return any(ch.isdigit() for ch in tok)


def _token_is_substance(token: str, allow_salts: bool = False) -> bool:
    tok = (token or "").lower()
    if not _token_has_letters(tok):
        return False
    if tok in EXTRA_TOKEN_STOPWORDS:
        if not (allow_salts and tok in SALT_TOKEN_WORDS):
            return False
    if tok in CONTAINER_NORMAL or tok in UNIT_NORMAL:
        return False
    if tok in {"-", "+", "/"}:
        return False
    return True


def _is_placeholder_name(name: str) -> bool:
    tokens = [tok for tok in normalize_text(name).split() if tok]
    if not tokens:
        return True
    cleaned = [tok for tok in tokens if tok not in {"and", "with", "in", "plus", "to", "of"}]
    if not cleaned:
        return True
    return all(tok in PLACEHOLDER_TOKENS or len(tok) == 1 for tok in cleaned)


def _raw_prefix_before_packaging(raw_desc: str) -> str:
    if not isinstance(raw_desc, str):
        return ""
    words = raw_desc.strip().split()
    prefix: List[str] = []
    prev_norm = ""
    for word in words:
        norm = normalize_text(word)
        tok = norm.split()[0] if norm else ""
        tok_lower = tok.lower()
        if tok == "+":
            if "diluent" in raw_desc.lower():
                prefix.append("+")
                continue
            if prefix:
                break
            continue
        if _looks_like_measurement(tok_lower):
            if "%" in word or "%" in tok_lower or prev_norm == "ph":
                prefix.append(word)
            prev_norm = tok_lower
            continue
        if tok_lower in CONTAINER_NORMAL or tok_lower in {"bag", "bottle", "ampule", "vial", "drum", "gallon", "kit", "set", "pack", "box"}:
            break
        if tok_lower in {"ml", "l", "g", "mg", "mcg", "iu", "lsu"}:
            break
        if tok_lower in {"approx", "approximately"}:
            continue
        if tok_lower in {"dose", "doses"}:
            continue
        if tok_lower in {"per"}:
            continue
        prefix.append(word)
        prev_norm = tok_lower
    return " ".join(prefix).strip(" ,-/")


def _restore_possessives(text: str) -> str:
    updated = re.sub(r"\bRINGER S\b", "RINGER'S", text)
    updated = re.sub(r"\bRINGERS\b", "RINGER'S", updated)
    return updated


def _maybe_restore_parenthetical(raw_desc: str, candidate: str) -> str:
    if "+" in candidate:
        return candidate
    phrases = extract_parenthetical_phrases(raw_desc)
    if not phrases:
        return candidate
    cand_norm = normalize_text(candidate)
    leading_raw = raw_desc.split("(")[0].strip()
    leading = re.sub(r"[\s0-9/%.]+$", "", leading_raw).strip() or leading_raw
    leading_norm = normalize_text(leading)
    for phrase in phrases:
        phrase_norm = normalize_text(phrase)
        if not phrase_norm:
            continue
        if phrase_norm.startswith("as "):
            continue
        if cand_norm == phrase_norm or (cand_norm and cand_norm in phrase_norm):
            base = leading if leading else candidate
            return f"{base.strip()} ({phrase.strip()})"
        if leading and cand_norm == leading_norm:
            return f"{leading.strip()} ({phrase.strip()})"
        if phrase_norm and cand_norm.endswith(f" {phrase_norm}") and leading:
            return f"{leading.strip()} ({phrase.strip()})"
        if phrase_norm and phrase_norm in cand_norm and leading:
            return f"{leading.strip()} ({phrase.strip()})"
    return candidate


def _collect_in_phrase_tokens(tokens: Sequence[str], reverse: bool) -> List[str]:
    collected: List[str] = []
    sequence = reversed(tokens) if reverse else tokens
    for tok in sequence:
        if not tok:
            continue
        if tok in {"in", "and", "with"}:
            if collected:
                break
            if reverse:
                continue
            break
        if tok in {"+", "/"}:
            if collected:
                break
            continue
        if tok in CONTAINER_NORMAL or tok in UNIT_NORMAL or tok in FORM_BREAK_TOKENS:
            break
        if _looks_like_measurement(tok):
            if not collected:
                continue
            break
        if not _token_is_substance(tok, allow_salts=True):
            if collected:
                break
            continue
        collected.append(tok.upper())
    if reverse:
        collected.reverse()
    return collected


def _compose_in_phrase_name(normalized_desc: str) -> Optional[str]:
    tokens = normalized_desc.split()
    for idx, tok in enumerate(tokens):
        if tok != "in":
            continue
        left = _collect_in_phrase_tokens(tokens[:idx], reverse=True)
        right = _collect_in_phrase_tokens(tokens[idx + 1 :], reverse=False)
        if left and right:
            return f"{' '.join(left)} IN {' '.join(right)}"
    return None


def _filter_measurement_components(
    components: List[Tuple[str, List[str]]],
    normalized_desc: str,
) -> List[Tuple[str, List[str]]]:
    if not components:
        return components
    lowered = normalized_desc
    filtered: List[Tuple[str, List[str]]] = []
    for name, salts in components:
        key = normalize_text(name)
        if key in {"iron", "calcium"} and f"elemental {key}" in lowered:
            continue
        filtered.append((name, salts))
    return filtered


def _normalize_component_key(name: str) -> str:
    norm = normalize_text(name)
    norm = re.sub(r"ic acid\b", "ate", norm)
    norm = re.sub(r"\bacid\b", "", norm).strip()
    tokens = [tok for tok in norm.split() if tok and tok not in SALT_TOKEN_WORDS]
    if tokens:
        norm = " ".join(tokens)
    return norm


def _dedupe_component_names(components: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    seen: set[str] = set()
    deduped: List[Tuple[str, List[str]]] = []
    for name, salts in components:
        key = _normalize_component_key(name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, salts))
    return deduped


def _maybe_merge_primary_salt(generic_name: str, salt_tokens: List[str]) -> Tuple[str, List[str]]:
    # Keep salts in their dedicated column rather than merging into the generic label.
    return generic_name, salt_tokens


def _clean_repeated_words(name: str) -> str:
    tokens = name.split()
    cleaned: List[str] = []
    prev = ""
    for tok in tokens:
        if tok == prev and tok not in {"PH"}:
            continue
        cleaned.append(tok)
        prev = tok
    return " ".join(cleaned)


def _token_should_strip_from_generic_name(token: str) -> bool:
    token = token.strip(".,;:-()\"'")
    if not token:
        return True
    upper_token = token.upper()
    if upper_token in GENERIC_ROUTE_FORM_TOKENS:
        return True
    lower_token = token.lower()
    if lower_token in GENERIC_STOPWORDS:
        return True
    if lower_token in UNIT_NORMAL:
        return True
    if lower_token and lower_token[0].isdigit():
        return True
    if "%" in lower_token:
        return True
    if "/" in lower_token and any(ch.isdigit() for ch in lower_token):
        return True
    if ":" in lower_token and any(ch.isdigit() for ch in lower_token):
        return True
    if any(lower_token.endswith(suffix) for suffix in GENERIC_MEASUREMENT_SUFFIXES):
        if re.search(r"\d", lower_token):
            return True
    return False


def _strip_generic_name_extras(name: str) -> str:
    tokens = re.sub(r"([+/])", r" \1 ", name).split()
    filtered: List[str] = []
    prev_raw = ""
    for tok in tokens:
        if tok in {"+", "/"}:
            filtered.append(tok)
            prev_raw = tok
            continue
        upper_tok = tok.upper().strip(".,;:-()\"'")
        if upper_tok == "SOLUTION" and _is_ringer_token(prev_raw):
            filtered.append(tok)
            prev_raw = tok
            continue
        if _token_should_strip_from_generic_name(tok):
            prev_raw = tok
            continue
        filtered.append(tok)
        prev_raw = tok

    cleaned: List[str] = []
    for tok in filtered:
        if tok in {"+", "/"}:
            if not cleaned or cleaned[-1] in {"+", "/"}:
                continue
            cleaned.append(tok)
            continue
        cleaned.append(tok)
    if cleaned and cleaned[-1] in {"+", "/"}:
        cleaned.pop()

    sanitized = " ".join(cleaned).strip()
    sanitized = re.sub(r"\s*\+\s*", " + ", sanitized)
    sanitized = re.sub(r"\s*/\s*", " / ", sanitized)
    sanitized = sanitized.strip(" ,;:-")
    if not sanitized:
        return name.strip().upper()
    return sanitized.upper()


def _normalize_token(token: str) -> str:
    return re.sub(r"[^A-Z0-9']+", "", (token or "").upper())


def _is_ringer_token(token: str) -> bool:
    return _normalize_token(token) in RINGER_TERMS


def _filter_synonyms(synonyms: Sequence[str]) -> List[str]:
    filtered: List[str] = []
    seen: set[str] = set()
    for candidate in synonyms:
        norm = (candidate or "").strip()
        if not norm:
            continue
        upper = norm.upper()
        if upper in seen:
            continue
        if upper in SYNONYM_SKIP_TERMS:
            continue
        if any(upper.startswith(prefix) for prefix in SYNONYM_SKIP_PREFIXES):
            continue
        if re.fullmatch(r"[\d./%]+", upper):
            continue
        filtered.append(upper)
        seen.add(upper)
    return filtered


def _extract_parenthetical_synonyms(name: str) -> Tuple[str, List[str]]:
    buffer: list[str] = []
    synonyms: List[str] = []
    idx = 0
    while idx < len(name):
        char = name[idx]
        if char == ")":
            idx += 1
            continue
        if char == "(":
            close = name.find(")", idx + 1)
            if close == -1:
                break
            snippet = name[idx + 1 : close].strip()
            if snippet:
                synonyms.append(snippet)
            idx = close + 1
            continue
        buffer.append(char)
        idx += 1
    cleaned = re.sub(r"\s+", " ", "".join(buffer)).strip(" ,;:-")
    return cleaned, _filter_synonyms(synonyms)


def _split_generic_variant(name: str) -> Tuple[str, str]:
    if not name:
        return "", ""
    parts = [part.strip() for part in name.split(",", 1)]
    base = parts[0] if parts else ""
    variant = parts[1] if len(parts) > 1 else ""
    return base, variant


def _clean_variant_text(variant: str) -> str:
    tokens = variant.split()
    cleaned: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        stripped = tok.strip(".,;:-)")
        if not stripped:
            continue
        if "%" in stripped:
            continue
        if re.search(r"\d", stripped):
            continue
        cleaned.append(stripped.upper())
    return " ".join(cleaned).strip()


def _split_generic_components(name: str) -> Tuple[str, str, List[str]]:
    stripped, synonyms = _extract_parenthetical_synonyms(name)
    base, variant = _split_generic_variant(stripped)
    base = base.strip()
    variant = variant.strip()
    variant = _clean_variant_text(variant)
    if not base and stripped:
        base = stripped
    return base.upper(), variant.upper(), synonyms


def _apply_suffix_fixes(name: str) -> str:
    updated = re.sub(r"\s+FOR ADULT\b", " (ADULT)", name)
    updated = re.sub(r"\s+FOR PEDIA\b", " (PEDIA)", updated)
    updated = re.sub(r"\s+FOR PEDIATRIC\b", " (PEDIATRIC)", updated)
    updated = re.sub(r"\s+\+\s+PH\b", " PH", updated)
    return updated


def _should_use_prefix_name(candidate: str, prefix: str, match_source: str) -> bool:
    if not prefix:
        return False
    prefix_norm = normalize_text(prefix)
    if not prefix_norm:
        return False
    if " as " in prefix_norm:
        return False
    candidate_norm = normalize_text(candidate)
    if " IN " in candidate.upper():
        return False
    if _is_placeholder_name(candidate):
        return True
    if match_source == "combo_fallback" and any(keyword in prefix_norm for keyword in PREFIX_KEYWORDS):
        return True
    cand_tokens = candidate_norm.split()
    prefix_tokens = prefix_norm.split()
    if len(cand_tokens) <= 1 and len(prefix_tokens) > 1:
        return True
    prefix_kw = {kw for kw in PREFIX_KEYWORDS if kw in prefix_norm}
    candidate_kw = {kw for kw in PREFIX_KEYWORDS if kw in candidate_norm}
    if len(prefix_tokens) > len(cand_tokens) and prefix_kw - candidate_kw:
        return True
    if "(" in prefix and "(" not in candidate:
        return True
    return False


def _refine_generic_display(
    raw_desc: str,
    normalized_desc: str,
    generic_name: str,
    match_source: str,
) -> str:
    candidate = generic_name
    in_phrase = _compose_in_phrase_name(normalized_desc)
    if in_phrase:
        if candidate.upper().endswith(" IN"):
            candidate = in_phrase
        elif _is_placeholder_name(candidate):
            candidate = in_phrase
    prefix_name = _raw_prefix_before_packaging(raw_desc)
    if _should_use_prefix_name(candidate, prefix_name, match_source):
        candidate = prefix_name
    elif prefix_name:
        prefix_norm = normalize_text(prefix_name)
        cand_norm = normalize_text(candidate)
        if "diluent" in prefix_norm and "diluent" not in cand_norm:
            candidate = prefix_name
        if "/" in prefix_name and "/" not in candidate:
            candidate = prefix_name
    candidate = _maybe_restore_parenthetical(raw_desc, candidate)
    candidate = _restore_possessives(candidate)
    candidate = _apply_suffix_fixes(candidate)
    candidate = _clean_repeated_words(candidate)
    return candidate.strip().upper()


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
    ("Alpha-Tocopherol", "Alpha-Tocopherol", 0),
    ("Alpha Tocopherol", "Alpha-Tocopherol", 0),
    ("Alpha-Tocopherol Vitamin E", "Alpha-Tocopherol", 0),
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
        if norm.startswith("as "):
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
        letters_only = re.sub(r"[^a-z]", "", tok.lower())
        if letters_only and letters_only in UNIT_NORMAL:
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
        if any(unit in canonical_norm for unit in ("mcg", "mg", "ml", "iu", "mci")):
            continue
        if any(tok in FORM_BREAK_TOKENS for tok in canonical_norm.split()):
            continue
        canonical_tokens = canonical_norm.split()
        if canonical_tokens and len(canonical_tokens) == 1 and canonical_tokens[0] in SALT_TOKEN_WORDS:
            continue
        if canonical_tokens and all(tok in UNIT_NORMAL or tok in CONTAINER_NORMAL for tok in canonical_tokens):
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
            if key_tokens and len(key_tokens) == 1 and key_tokens[0] in SALT_TOKEN_WORDS:
                continue
            if key_tokens and all(tok in UNIT_NORMAL or tok in CONTAINER_NORMAL for tok in key_tokens):
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
        base_check = base
        if base_check and any(unit in base_check.lower() for unit in ("mcg", "mg", "ml", "iu", "mci")):
            base_check = ""
        components = _expand_component_aliases(base_check, salts, segment) if base_check else []
    return components


def _prune_combo_components(components: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    if not components:
        return []
    normalized = [normalize_text(name).split() for name, _ in components]
    keep: List[Tuple[str, List[str]]] = []
    for idx, tokens in enumerate(normalized):
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            continue
        if not _segment_has_molecule_hint(" ".join(tokens)):
            continue
        if all(tok in {"as", "and", "or", "salt", "salts"} for tok in tokens):
            continue
        if len(tokens) == 1 and tokens[0] in SALT_TOKEN_WORDS:
            continue
        token_set = set(tokens)
        is_subset = False
        for j, other in enumerate(normalized):
            if idx == j:
                continue
            other_tokens = [tok for tok in other if tok]
            if not other_tokens:
                continue
            other_set = set(other_tokens)
            if token_set and token_set < other_set:
                is_subset = True
                break
        if not is_subset:
            keep.append(components[idx])
    return keep


def _shared_combo_tokens(names: Sequence[str]) -> set[str]:
    def _fingerprint(token: str) -> Optional[str]:
        if not token:
            return None
        stripped = re.sub(r"[^a-z]", "", token.lower())
        if not stripped or len(stripped) < 4:
            return None
        return stripped[:6]

    token_sets: List[set[str]] = []
    for name in names:
        tokens = [tok for tok in normalize_text(name).split() if _token_is_substance(tok)]
        fingerprints = {_fingerprint(tok) for tok in tokens if _fingerprint(tok)}
        if not fingerprints:
            return set()
        token_sets.append(fingerprints)
    if not token_sets:
        return set()
    shared = set(token_sets[0])
    for tokens in token_sets[1:]:
        shared &= tokens
    return shared


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
    components = _prune_combo_components(components)
    components = _filter_measurement_components(components, normalized_desc)
    components = _dedupe_component_names(components)
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


def _append_all_in_one_synonyms(
    raw_desc: str,
    normalized_desc: str,
    synonyms: List[str],
) -> List[str]:
    if not raw_desc or not normalized_desc:
        return synonyms
    norm_lower = normalized_desc.lower()
    if "all-in-one" not in norm_lower or "admixture" not in norm_lower:
        return synonyms
    for snippet in extract_parenthetical_phrases(raw_desc):
        candidate = snippet.strip().upper()
        if candidate and candidate not in synonyms:
            synonyms.append(candidate)
    return synonyms


def _canonicalize_enteral_nutrition(name: str, normalized_desc: str) -> str:
    norm = (normalized_desc or "").lower()
    if "enteral nutrition" in norm and "disease" in norm:
        return "ENTERAL NUTRITION DISEASE-SPECIFIC"
    return name


def _canonicalize_vaccine_name(name: str, normalized_desc: str) -> str:
    text = (normalized_desc or "").lower()
    if "vaccine" not in text:
        return name
    if "bcg" in text:
        return "BCG VACCINE"
    if "diphtheria" in text and "tetanus" in text:
        return "DIPHTHERIA-TETANUS TOXOIDS"
    if "live attenuated" in text or "opv" in text or "oral polio" in text:
        return "LIVE ATTENUATED VACCINE"
    if "polio" in text and ("inactivated" in text or "ipv" in text) and "live" not in text:
        return "INACTIVATED POLIOMYELITIS VACCINE"
    if "influenza" in text:
        return "INFLUENZA POLYVALENT VACCINE"
    if "hepatitis a" in text:
        return "HEPATITIS A INACTIVATED VACCINE"
    if "hepatitis b" in text:
        return "HEPATITIS B VACCINE"
    if "meningococcal" in text and "polysaccharide" in text:
        return "MENINGOCOCCAL POLYSACCHARIDE"
    if ("human papillomavirus" in text or "hpv" in text) and "quadrivalent" in text:
        return "HUMAN PAPILLOMAVIRUS QUADRIVALENT RECOMBINANT VACCINE"
    if "tetanus toxoid" in text and "diphtheria" not in text:
        return "TETANUS TOXOID"
    if "tuberculin" in text:
        return "TUBERCULIN"
    if "yellow fever" in text:
        return "YELLOW FEVER VACCINE"
    if "typhoid" in text:
        return "TYPHOID VACCINE"
    return name


def _restore_special_salt_suffixes(name: str, raw_desc: str) -> str:
    raw_text = (raw_desc or "").lower()
    if not raw_text or not any(salt in raw_text for salt in SPECIAL_SALT_TOKENS):
        return name
    components = [comp.strip() for comp in re.split(r"\s*\+\s*", name)]
    raw_segments = [seg.strip() for seg in re.split(r"\s*\+\s*", raw_desc or "")]
    updated: List[str] = []
    for comp, raw_segment in zip_longest(components, raw_segments, fillvalue=""):
        comp_text = (comp or "").strip()
        if not comp_text:
            continue
        comp_norm = normalize_text(comp_text)
        raw_norm = normalize_text(raw_segment)
        suffixes: List[str] = []
        if comp_norm and raw_norm:
            for salt in SPECIAL_SALT_TOKENS:
                if salt in raw_norm and salt not in comp_norm:
                    upper = salt.upper()
                    if upper not in comp_text:
                        suffixes.append(upper)
        segment = comp_text
        if suffixes:
            segment = f"{segment} {' '.join(suffixes)}"
        updated.append(segment.strip())
    return " + ".join(updated)


def _apply_special_name_overrides(
    name: str,
    normalized_desc: str,
    raw_desc: str,
    synonyms: List[str],
) -> Tuple[str, List[str]]:
    name = _restore_special_salt_suffixes(name, raw_desc)
    synonyms = _append_all_in_one_synonyms(raw_desc, normalized_desc, synonyms)
    name = _canonicalize_enteral_nutrition(name, normalized_desc)
    name = _canonicalize_vaccine_name(name, normalized_desc)
    return name, synonyms
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
) -> Tuple[str, str, List[str], str, List[str]]:
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
    shared_combo_tokens = _shared_combo_tokens(combo_components) if combo_components else set()
    ratio_pattern = bool(re.search(r"\d+\s*/\s*\d+", normalized_desc))
    ratio_combo_collapse = False
    if combo_components and ratio_pattern and shared_combo_tokens:
        combo_components = []
        ratio_combo_collapse = True
    resolved = resolver.resolve(raw_desc) if isinstance(raw_desc, str) else None
    if resolved and not _accept_resolved_match(resolved, as_index_full):
        resolved = None
    if resolved and combo_hint and not _name_looks_combo(resolved.entry.canonical):
        if _has_unmatched_substances(normalized_desc, resolved):
            resolved = None
    if resolved and _is_placeholder_name(resolved.entry.canonical):
        resolved = None
    if resolved and combo_components:
        resolved_norm = normalize_text(resolved.entry.canonical)
        combo_missing = False
        for component in combo_components:
            comp_norm = normalize_text(component)
            if comp_norm and comp_norm not in resolved_norm:
                combo_missing = True
                break
        if combo_missing:
            resolved = None
    match_source = "fallback"
    if resolved:
        name = resolved.entry.canonical
        match_source = resolved.entry.source
        resolved_norm_tokens = normalize_text(name).split()
        filtered_tokens = [tok for tok in resolved_norm_tokens if tok not in {"as"}]
        if base_name_from_text and filtered_tokens and all(tok in SALT_TOKEN_WORDS for tok in filtered_tokens):
            name = base_name_from_text
            match_source = "fallback_preferred"
        combo_name = any(sep in name for sep in ("+", "/", " with "))
        if combo_name:
            generic_name = name
            salt_tokens = salts_from_text
        else:
            base_name, salts = extract_base_and_salts(name)
            generic_name = base_name or name
            salt_tokens = salts or salts_from_text
    elif combo_components:
        salt_tokens = combo_salts.copy()
        for tok in salts_from_text:
            if tok not in salt_tokens:
                salt_tokens.append(tok)
        generic_name = " + ".join(combo_components)
        match_source = "combo_fallback"
    else:
        name = _fallback_generic_name(raw_desc, norm_for_matching, as_index_trimmed)
        base_name, salts = extract_base_and_salts(name)
        generic_name = base_name or name
        if ratio_combo_collapse and "+" in generic_name:
            generic_name = generic_name.split("+", 1)[0].strip()
        salt_tokens = salts or salts_from_text
    generic_name, salt_tokens = _maybe_merge_primary_salt(generic_name, salt_tokens)
    generic_name = _refine_generic_display(raw_desc, normalized_desc, generic_name, match_source)
    if custom_descriptor:
        descriptor_upper = custom_descriptor.upper()
        lower_name = generic_name.lower()
        if "vitamins intravenous" in normalized_desc and "vitamins intravenous" not in lower_name:
            generic_name = f"VITAMINS INTRAVENOUS ({descriptor_upper})"
        elif descriptor_upper not in generic_name:
            generic_name = f"{generic_name} ({descriptor_upper})"
    generic_name = _restore_possessives(generic_name)
    generic_name = _clean_repeated_words(generic_name)
    sanitized_name = _strip_generic_name_extras(generic_name)
    if salt_tokens:
        salt_set = {tok.upper() for tok in salt_tokens if tok}
        if salt_set:
            tokens: List[str] = []
            for tok in sanitized_name.split():
                cleaned = tok.strip(".,;:-)")
                if cleaned and cleaned.upper() in salt_set:
                    continue
                tokens.append(tok)
            if tokens:
                sanitized_name = " ".join(tokens)
    derived_name, generic_variant, generic_synonyms = _split_generic_components(sanitized_name)
    if normalized_desc and RINGER_SOLUTION_RX.search(normalized_desc):
        if derived_name and not derived_name.endswith("SOLUTION"):
            derived_name = f"{derived_name} SOLUTION"
    final_name = derived_name or sanitized_name
    final_name, generic_synonyms = _apply_special_name_overrides(
        final_name,
        normalized_desc,
        raw_desc,
        generic_synonyms,
    )
    return final_name, match_source, salt_tokens, generic_variant, generic_synonyms


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

        (
            generic_name,
            generic_source,
            salt_tokens,
            generic_variant,
            generic_synonyms,
        ) = _derive_generic_name(desc, norm, resolver)
        generic_id = slug_id(generic_name) if generic_name else ""
        salt_form = serialize_salt_list(salt_tokens)
        generic_synonyms_text = "; ".join(generic_synonyms) if generic_synonyms else ""

        records.append(
            {
                "drug_code": str(raw_code).strip(),
                "generic_id": generic_id,
                "generic_name": generic_name,
                "generic_variant": generic_variant,
                "generic_synonyms": generic_synonyms_text,
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
