#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared text normalization helpers used across preparation and matching."""

import re
import unicodedata
from typing import Iterable, Optional, List, Tuple

from .unified_constants import (
    SALT_TOKENS,
    SALT_TOKENS_LOWER,
    STOPWORDS_LOWER,
    UNIT_TOKENS_LOWER,
    SALT_CATIONS,
    SALT_TAIL_BREAK_TOKENS,
)

# Use unified constants - lowercase versions for text matching
BASE_GENERIC_IGNORE = STOPWORDS_LOWER | UNIT_TOKENS_LOWER
MEASUREMENT_TOKENS = UNIT_TOKENS_LOWER
SPECIAL_SALT_TOKENS = {s.lower() for s in SALT_CATIONS}

_GM_TOKEN_RX = re.compile(r"(?<![a-z])gms?(?![a-z])")


def _normalize_unit_tokens(text: str) -> str:
    """Restrict gm/gms collapsing to standalone unit tokens (avoids STIGMINE -> STIGINE)."""
    return _GM_TOKEN_RX.sub("g", text)


def _token_core(token: str) -> str:
    """Return a normalized token key for comparisons."""
    return token.lower().strip(".,;:'\"()[]{}")

PAREN_CONTENT_RX = re.compile(r"\(([^)]+)\)")

def _normalize_text_basic(s: str) -> str:
    """Lowercase and collapse whitespace, leaving only alphanumeric tokens."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _base_name(name: str) -> str:
    """Strip trailing qualifiers so only the base molecule name remains."""
    name = str(name).lower().strip()
    name = re.split(r",| incl\.| including ", name, maxsplit=1)[0]
    return re.sub(r"\s+", " ", name).strip()

def normalize_text(s: str) -> str:
    """Produce the canonical normalized text used for matching and parsing routines."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\biv\b", "intravenous", s)
    s = re.sub(r"[^\w%/+\.\- ]+", " ", s)
    s = s.replace("microgram", "mcg").replace("μg", "mcg").replace("µg", "mcg")
    s = re.sub(r"(?<![a-z])cc(?![a-z])", "ml", s).replace("milli litre", "ml").replace("milliliter", "ml")
    s = _normalize_unit_tokens(s)
    s = s.replace("milligram", "mg")
    s = s.replace("polymixin", "polymyxin")
    s = s.replace("hydrochlorde", "hydrochloride")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Use lowercase version of unified constant
_SALT_TAIL_BREAK_TOKENS = {t.lower() for t in SALT_TAIL_BREAK_TOKENS}


def _looks_like_salt_tail(tokens: List[str], start_idx: int) -> bool:
    """Heuristic to ensure 'as' only signals salts, not new molecules."""
    seen_salt = False
    for tok in tokens[start_idx:]:
        if tok.lower() in _SALT_TAIL_BREAK_TOKENS:
            break
        if not tok:
            continue
        if any(ch.isdigit() for ch in tok) or tok in {"%", "per"}:
            break
        tok_lower = tok.lower()
        if tok_lower == "and/or":
            continue
        if tok_lower in SALT_TOKEN_WORDS:
            seen_salt = True
            continue
        return False
    return seen_salt


def detect_as_boundary(norm_text: str) -> Optional[int]:
    """Return the index of the first 'as' token that introduces salt descriptors."""
    if not isinstance(norm_text, str):
        return None
    tokens = norm_text.split()
    for idx, tok in enumerate(tokens):
        if tok != "as":
            continue
        if _looks_like_salt_tail(tokens, idx + 1):
            return idx
    return None


def strip_after_as(norm_text: str) -> str:
    """Remove tokens occurring after the first standalone 'as' token, preserving prefixes."""
    if not isinstance(norm_text, str):
        return ""
    boundary = detect_as_boundary(norm_text)
    if boundary is None or boundary <= 0:
        return norm_text
    tokens = norm_text.split()
    if boundary >= len(tokens):
        return norm_text
    stripped = " ".join(tokens[:boundary]).strip()
    return stripped or norm_text

def normalize_compact(s: str) -> str:
    """Compact the normalized text by removing whitespace and hyphens."""
    return re.sub(r"[ \-]", "", normalize_text(s))

def slug_id(name: str) -> str:
    """Turn arbitrary text into a lowercase slug suitable for identifiers."""
    base = normalize_text(str(name))
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_")

def clean_atc(s: Optional[str]) -> str:
    """Normalize ATC codes by trimming whitespace and non-breaking spaces."""
    if not isinstance(s, str):
        return ""
    return s.replace("\u00a0", " ").strip()

def safe_to_float(x):
    """Convert to float when possible, returning None on failures."""
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        return float(x)
    except Exception:
        return None

def extract_parenthetical_phrases(raw_text: str) -> List[str]:
    """Extract probable brand/details from the ORIGINAL text."""
    if not isinstance(raw_text, str) or "(" not in raw_text:
        return []
    items = [m.group(1).strip() for m in PAREN_CONTENT_RX.finditer(raw_text) if m.group(1).strip()]
    cleaned = []
    for it in items:
        if len(it) > 60:
            continue
        if re.fullmatch(r"[-/+\s]+", it):
            continue
        # Normalize whitespace within each parenthetical snippet.
        cleaned.append(re.sub(r"\s+", " ", it))
    seen = set()
    uniq = []
    for c in cleaned:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        # Preserve original casing for display purposes.
        uniq.append(c)
    return uniq

from .unified_constants import SALT_TOKENS
from .routes_forms import FORM_TO_ROUTE, ROUTE_ALIASES

STOPWORD_TOKENS = (
    set(SALT_TOKENS)
    | set(FORM_TO_ROUTE.keys())
    | set(ROUTE_ALIASES.keys())
    | {
        "ml","l","mg","g","mcg","ug","iu","lsu",
        "dose","dosing","unit","units","strength",
        "solution","suspension","syrup",
        "bottle","bottles","box","boxes","sachet","sachets","container","containers"
    }
)

def _build_salt_token_words() -> set:
    tokens: set[str] = set()
    for token in SALT_TOKENS:
        if not token:
            continue
        tokens.add(token.lower())
        norm = normalize_text(token)
        for part in norm.split():
            tokens.add(part)
    tokens.update({"salt", "salts"})
    return tokens


SALT_TOKEN_WORDS = _build_salt_token_words()

def _is_measurement_token(tok: str) -> bool:
    tok = tok.lower()
    if tok in MEASUREMENT_TOKENS or tok in {"%", "ratio", "per"}:
        return True
    if tok.endswith("ml") or tok.endswith("mg"):
        return True
    return False


def serialize_salt_list(salts: Iterable[str]) -> str:
    """Join unique salt labels using a consistent delimiter for CSV output."""
    ordered = []
    seen = set()
    for salt in salts:
        clean = str(salt).strip().upper()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return " + ".join(ordered)


def extract_base_and_salts(raw_text: str) -> Tuple[str, List[str]]:
    """Return the base molecule name (sans salts) plus a list of salt descriptors."""
    if not isinstance(raw_text, str):
        return "", []
    norm = normalize_text(raw_text)
    tokens = norm.split()
    boundary = detect_as_boundary(norm)
    base_candidates = tokens if boundary is None else tokens[:boundary]
    salt_candidates = [] if boundary is None else tokens[boundary + 1 :]
    salt_tokens: list[str] = []
    base_tokens: list[str] = []
    pending_leading_salts: list[str] = []
    for tok in salt_candidates:
        tok_lower = tok.lower()
        if tok_lower in {"and", "with", "plus", "+", "/"}:
            continue
        if not tok_lower:
            continue
        if not re.search(r"[a-z]", tok_lower):
            continue
        if tok_lower not in SALT_TOKEN_WORDS:
            continue
        if tok_lower in {"salt", "salts"}:
            continue
        salt_tokens.append(tok.upper())

    def _should_treat_as_salt(tok_lower: str, idx: int, candidates: List[str]) -> bool:
        if tok_lower not in SALT_TOKEN_WORDS:
            return False
        if tok_lower in {"salt", "salts"}:
            return False
        prev = candidates[idx - 1].lower() if idx > 0 else ""
        if prev == "as":
            return True
        if tok_lower in SPECIAL_SALT_TOKENS:
            return False
        return True

    def _is_candidate(tok: str, idx: int, candidates: List[str]) -> bool:
        tok_lower = tok.lower()
        tok_key = _token_core(tok)
        if tok_key in BASE_GENERIC_IGNORE:
            return False
        if _is_measurement_token(tok_key):
            return False
        if tok_lower == "%":
            return False
        if not re.search(r"[a-z]", tok_lower):
            return False
        if tok_lower[0].isdigit():
            return False
        if any(ch.isdigit() for ch in tok_lower):
            if not re.fullmatch(r"[a-z]+[0-9]+[a-z0-9]*", tok_lower):
                return False
        return True

    def _truncate_tokens(tokens: List[str]) -> List[str]:
        truncated: List[str] = []
        for idx, tok in enumerate(tokens):
            tok_lower = tok.lower()
            tok_key = _token_core(tok)
            if tok in {"+", "/", "&"}:
                if truncated:
                    truncated.append(tok.upper())
                continue
            if tok_lower in {"as"}:
                break
            if _should_treat_as_salt(tok_lower, idx, tokens):
                continue
            if _is_measurement_token(tok_key):
                continue
            if tok_key in BASE_GENERIC_IGNORE and tok_lower not in SALT_TOKEN_WORDS:
                continue
            if not re.search(r"[a-z]", tok_lower):
                continue
            if any(ch.isdigit() for ch in tok_lower):
                if re.fullmatch(r"[a-z]+[0-9]+[a-z0-9]*", tok_lower):
                    truncated.append(tok.upper())
                continue
            truncated.append(tok.upper())
        return truncated

    for idx, tok in enumerate(base_candidates):
        tok_lower = tok.lower()
        if tok in {"+", "/", "&"}:
            if base_tokens and any(_is_candidate(t, j, base_candidates) for j, t in enumerate(base_candidates[idx + 1 :], start=idx + 1)):
                base_tokens.append(tok)
            continue
        if _should_treat_as_salt(tok_lower, idx, base_candidates):
            if base_tokens:
                salt_tokens.append(tok.upper())
            else:
                pending_leading_salts.append(tok.upper())
            continue
        if not _is_candidate(tok, idx, base_candidates):
            continue
        base_tokens.append(tok.upper())

    if not base_tokens:
        base_tokens = _truncate_tokens(list(base_candidates))
    if not base_tokens and pending_leading_salts:
        base_tokens = pending_leading_salts

    def _trim_trailing_salts(seq: List[str]) -> List[str]:
        if not seq:
            return []
        if not any(tok.lower() not in SALT_TOKEN_WORDS for tok in seq):
            return []
        trimmed: List[str] = []
        while seq and seq[-1].lower() in SALT_TOKEN_WORDS:
            token = seq.pop()
            if token.lower() in {"salt", "salts"}:
                continue
            trimmed.append(token.upper())
        return list(reversed(trimmed))

    salt_tokens.extend(_trim_trailing_salts(base_tokens))
    if base_tokens:
        base = " ".join(base_tokens).strip().upper()
    else:
        base = ""
    unique_salts: List[str] = []
    seen = set()
    for tok in salt_tokens:
        if tok and tok not in seen:
            seen.add(tok)
            unique_salts.append(tok)
    if not base and unique_salts:
        base = " ".join(unique_salts)
        unique_salts = []
    if not base and raw_text:
        base = raw_text.strip().upper()
    return base, unique_salts
