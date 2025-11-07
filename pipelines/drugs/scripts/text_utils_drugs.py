
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared text normalization helpers used across preparation and matching."""

import re
import unicodedata
from typing import Iterable, Optional, List, Tuple

from .combos_drugs import SALT_TOKENS

BASE_GENERIC_IGNORE = {
    "and",
    "with",
    "plus",
    "in",
    "solution",
    "suspension",
    "syrup",
    "powder",
    "cream",
    "ointment",
    "gel",
    "lotion",
    "drops",
    "drop",
    "tablet",
    "capsule",
    "ampule",
    "ampoule",
    "vial",
    "bottle",
    "bag",
    "sachet",
    "nebule",
    "spray",
    "patch",
    "pre-filled",
    "syringe",
    "oral",
    "intravenous",
    "intramuscular",
    "subcutaneous",
    "ophthalmic",
    "nasal",
    "topical",
    "unit",
    "units",
}

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
    s = re.sub(r"[^\w%/+\.\- ]+", " ", s)
    s = s.replace("microgram", "mcg").replace("μg", "mcg").replace("µg", "mcg")
    s = s.replace("cc", "ml").replace("milli litre", "ml").replace("milliliter", "ml")
    s = s.replace("gm", "g").replace("gms", "g").replace("milligram", "mg")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_as_boundary(norm_text: str) -> Optional[int]:
    """Return the index of the first standalone 'as' token in already-normalized text."""
    if not isinstance(norm_text, str):
        return None
    tokens = norm_text.split()
    for idx, tok in enumerate(tokens):
        if tok == "as":
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

from .combos_drugs import SALT_TOKENS
from .routes_forms_drugs import FORM_TO_ROUTE, ROUTE_ALIASES

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

BASE_GENERIC_IGNORE |= STOPWORD_TOKENS


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
    scan_tokens = tokens[:boundary] if boundary else tokens
    salt_tokens: list[str] = []
    base_tokens: list[str] = []
    for tok in scan_tokens:
        tok_lower = tok.lower()
        if tok_lower in SALT_TOKEN_WORDS:
            salt_tokens.append(tok_upper := tok.upper())
            continue
        if tok_lower in BASE_GENERIC_IGNORE:
            continue
        if any(ch.isdigit() for ch in tok_lower) or tok_lower == "%":
            continue
        if not re.search(r"[a-z]", tok_lower):
            continue
        base_tokens.append(tok.upper())
    if base_tokens:
        base = " ".join(base_tokens).strip().upper()
    else:
        base = ""
    unique_salts = []
    seen = set()
    for tok in salt_tokens:
        if tok and tok not in seen:
            seen.add(tok)
            unique_salts.append(tok)
    if not base and unique_salts:
        base = " ".join(unique_salts)
    if not base and raw_text:
        base = raw_text.strip().upper()
    return base, unique_salts
