#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Helpers for expanding PNF generic name aliases and special abbreviations."""
from __future__ import annotations

import re
from typing import Iterable, Set

from .text_utils import _base_name

_CONNECTOR_PATTERN = re.compile(r"\s*(?:\+|/|&|,| and | plus )\s*", re.IGNORECASE)
_PAREN_RX = re.compile(r"\(([^)]+)\)")

SPECIAL_GENERIC_ALIASES: dict[str, Set[str]] = {
    "aluminum_magnesium": {"almg", "almag", "aloh mgoh", "aloh mgo h", "alohmgoh", "aloh + mgoh"},
    "anti_tetanus_serum": {"ats"},
    "penicillin_g_benzylpenicillin_crystalline": {"pen g", "pen-g", "peng"},
    "isosorbide": {"ismn", "isdn"},
}


def _strip_parentheses(text: str) -> str:
    """Remove parenthetical segments while retaining surrounding spacing."""
    return _PAREN_RX.sub(" ", text)


def expand_generic_aliases(name: str) -> Set[str]:
    """Generate a set of lowercase alias strings for a PNF generic name."""
    variants: Set[str] = set()
    if not isinstance(name, str):
        return variants
    raw = name.strip()
    if not raw:
        return variants

    # Start with raw name and base form.
    variants.add(raw)
    base = _base_name(raw)
    variants.add(base)

    # Capture individual parenthetical entries (e.g., Co-amoxiclav).
    for par in _PAREN_RX.findall(raw):
        par = par.strip()
        if par:
            variants.add(par)

    # Remove parentheses content entirely.
    no_paren = _strip_parentheses(raw)
    variants.add(no_paren)

    # Normalize hyphen spacing variations.
    variants.add(raw.replace("-", " "))

    # Split on common connectors and register combinations + individual parts.
    expanded: Set[str] = set()
    for variant in list(variants):
        if not variant:
            continue
        pieces = [p.strip() for p in _CONNECTOR_PATTERN.split(variant) if p.strip()]
        if len(pieces) > 1:
            expanded.add(" ".join(pieces))
            expanded.add("".join(pieces))
            for piece in pieces:
                expanded.add(piece)
    variants |= expanded

    # Final pass: collapse whitespace and lowercase.
    cleaned: Set[str] = set()
    for variant in variants:
        norm = re.sub(r"\s+", " ", variant).strip().lower()
        if norm:
            cleaned.add(norm)
            # Provide ultra-compact form to help match tokens without spaces.
            compact = re.sub(r"\s+", "", norm)
            cleaned.add(compact)
    return cleaned

