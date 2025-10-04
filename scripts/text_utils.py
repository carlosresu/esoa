
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared text normalization helpers used across preparation and matching."""

import re
import unicodedata
from typing import Optional, List

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

from .combos import SALT_TOKENS
from .routes_forms import FORM_TO_ROUTE, ROUTE_ALIASES

STOPWORD_TOKENS = (
    set(SALT_TOKENS)
    | set(FORM_TO_ROUTE.keys())
    | set(ROUTE_ALIASES.keys())
    | {
        "ml","l","mg","g","mcg","ug","iu","lsu",
        "dose","dosing","unit","units","strength",
        "solution","suspension","syrup"
    }
)
