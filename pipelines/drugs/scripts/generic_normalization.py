#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drug normalization helpers encoding the explicit raw_description -> generic_name rules,
built for Polars/Parquet-first pipelines (expression-friendly, no pandas usage).

This implements the precise algorithm described in the provided ruleset:
- Tokenization that respects molecule-internal hyphens.
- Classification of each token (uppercase words, numerics, connectors, etc.).
- Molecule boundary detection, salt pair preservation, formulation stripping, and combination formatting.

All helper functions referenced within the spec are provided explicitly and remain pure so they can
be invoked from Polars via `pl.col(...).map_elements(...)` where needed.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from pipelines.drugs.scripts.text_utils_drugs import extract_base_and_salts

# Token classification constants
UPPERCASE_WORD = "UPPERCASE_WORD"
NUMERIC_OR_STRENGTH = "NUMERIC_OR_STRENGTH"
FORMULATION_WORD = "FORMULATION_WORD"
ROUTE_WORD = "ROUTE_WORD"
CONNECTIVE = "CONNECTIVE"
OTHER = "OTHER"

# Constants derived from the rulebook
FORMULATION_BOUNDARY_WORDS = {
    "SOLUTION",
    "SOLUTIONS",
    "INJECTION",
    "INJECTIONS",
    "SYRUP",
    "SYRUPS",
    "TABLET",
    "TABLETS",
    "CAPSULE",
    "CAPSULES",
    "AMPULE",
    "AMPULES",
    "VIAL",
    "VIALS",
    "OINTMENT",
    "CREAM",
    "USP",
    "BP",
    "IP",
    "DROPS",
    "DROP",
    "OPHTHALMIC",
    "TOPICAL",
    "PARENTERAL",
}

FORMULATION_STRIP_WORDS = FORMULATION_BOUNDARY_WORDS | {
    "FAT-SOLUBLE",
    "WATER-SOLUBLE",
    "TRACE",
    "ELEMENTS",
    "WATER",
    "INTRAVENOUS",
    "PARENTERAL",
    "SOLUBLE",
}

ROUTE_WORDS = {
    "INTRAVENOUS",
    "ORAL",
    "IM",
    "IV",
    "SUBLINGUAL",
    "SUBCUTANEOUS",
}

CONNECTIVE_WORDS = {"IN", "WITH", "AND", "PLUS"}

# Known salts that must stay paired even when connectors try to split them.
SALT_UNIT_SET = {
    "SODIUM CHLORIDE",
    "POTASSIUM CHLORIDE",
    "MAGNESIUM SULFATE",
    "CALCIUM GLUCONATE",
}

DELIMITERS = {" ", "\t", "\n", "\r", ",", "(", ")", "/"}
PUNCTUATION_BREAKS = {":", ";", "!", "?"}

GENERIC_BAD_SINGLE_TOKENS = {
    "ACID",
    "SOLUTION",
    "SOLUTIONS",
    "VITAMIN",
    "VITAMINS",
    "ELEMENT",
    "ELEMENTS",
    "TRACE",
    "MIXTURE",
    "PREPARATION",
    "FORMULA",
    "FORMULATION",
}

_PREFIX_UNIT_TOKENS = {
    "%",
    "MG",
    "MCG",
    "G",
    "IU",
    "ML",
    "L",
    "MMOL",
}

_PREFIX_PACKAGING_TOKENS = {
    "BOTTLE",
    "BOTTLES",
    "VIAL",
    "VIALS",
    "AMPULE",
    "AMPULLA",
    "DRUM",
    "TUBE",
    "TUBES",
    "SACHET",
    "SACHETS",
    "BAG",
    "BAGS",
    "GLASS",
    "TABLET",
    "TABLETS",
    "CAPSULE",
    "CAPSULES",
    "PACK",
    "BOX",
    "CARTRIDGE",
}


def tokenize_raw(raw_description: str) -> List[str]:
    """Produce the working token stream according to section 1.1 of the rules."""
    if not raw_description:
        return []
    normalized = unicodedata.normalize("NFKD", raw_description)
    normalized = normalized.replace("\u2013", "-")
    normalized = normalized.replace("\u2014", "-")
    normalized = normalized.replace("’", "'")
    normalized = normalized.replace("‘", "'")
    normalized = normalized.replace(".", "")

    tokens: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        if not current:
            return
        fragment = "".join(current).strip()
        fragment = fragment.strip(".,;:!?\"'")
        if fragment:
            tokens.append(fragment)
        current.clear()

    idx = 0
    while idx < len(normalized):
        char = normalized[idx]
        if char in DELIMITERS:
            flush_current()
            idx += 1
            continue
        if char in PUNCTUATION_BREAKS:
            flush_current()
            idx += 1
            continue
        if char == "-":
            prev_char = current[-1] if current else ""
            next_char = normalized[idx + 1] if idx + 1 < len(normalized) else ""
            if prev_char.isupper() and next_char.isupper():
                current.append(char)
            else:
                flush_current()
            idx += 1
            continue
        if char == "+":
            flush_current()
            tokens.append("PLUS")
            idx += 1
            continue
        if char == "&":
            flush_current()
            tokens.append("AND")
            idx += 1
            continue
        if char == "%":
            flush_current()
            tokens.append("%")
            idx += 1
            continue
        current.append(char)
        idx += 1
    flush_current()
    return [token for token in tokens if token]


def classify_token(token: str) -> str:
    """Label each token so the downstream stages know how to behave (Section 1.2)."""
    if not token:
        return OTHER
    upper_token = token.upper()
    lower_token = token.lower()
    if any(ch.isdigit() for ch in token) or "%" in token:
        return NUMERIC_OR_STRENGTH
    if "mg" in lower_token or "ml" in lower_token:
        return NUMERIC_OR_STRENGTH
    if upper_token in FORMULATION_BOUNDARY_WORDS:
        return FORMULATION_WORD
    if upper_token in ROUTE_WORDS:
        return ROUTE_WORD
    if upper_token in CONNECTIVE_WORDS:
        return CONNECTIVE
    cleaned = upper_token.replace("-", "").replace("'", "")
    if cleaned and all("A" <= ch <= "Z" for ch in cleaned):
        return UPPERCASE_WORD
    return OTHER


def extract_molecule_block(tokens: List[str], classifications: List[str]) -> Tuple[List[str], List[str]]:
    """Capture the contiguous molecule block before strengths/formulations/routes (Section 2)."""
    block_tokens: List[str] = []
    block_classes: List[str] = []
    for token, classification in zip(tokens, classifications):
        if classification in {NUMERIC_OR_STRENGTH, FORMULATION_WORD, ROUTE_WORD}:
            break
        block_tokens.append(token)
        block_classes.append(classification)
    return block_tokens, block_classes


def split_molecules(tokens: List[str], classifications: List[str]) -> List[List[str]]:
    """Split the molecule block into contiguous uppercase phrases separated by connectors."""
    molecules: List[List[str]] = []
    current: List[str] = []
    for token, classification in zip(tokens, classifications):
        if classification == CONNECTIVE:
            if current:
                molecules.append(current.copy())
                current = []
            continue
        if classification == UPPERCASE_WORD:
            current.append(token)
        else:
            continue
    if current:
        molecules.append(current.copy())
    return detect_salt_units(molecules)


def filter_formulation_tokens(molecules: List[List[str]]) -> List[List[str]]:
    """Remove formulation-only tokens and unwanted qualifiers (Section 4 & 8)."""
    filtered: List[List[str]] = []
    for molecule in molecules:
        refined: List[str] = []
        for token in molecule:
            normalized = token.upper()
            if normalized in FORMULATION_STRIP_WORDS:
                continue
            if "%" in token:
                continue
            if any(ch.isdigit() for ch in token):
                continue
            refined.append(token)
        if refined:
            filtered.append(refined)
    return filtered


def detect_salt_units(molecules: List[List[str]]) -> List[List[str]]:
    """Merge salt pairs that would otherwise launch singleton outputs (Section 7)."""
    if not molecules:
        return []
    merged: List[List[str]] = []
    idx = 0
    while idx < len(molecules):
        current = molecules[idx]
        next_idx = idx + 1
        if next_idx < len(molecules):
            next_molecule = molecules[next_idx]
            if len(current) == 1 and len(next_molecule) == 1:
                combined = f"{current[0].upper()} {next_molecule[0].upper()}"
                if combined in SALT_UNIT_SET:
                    merged.append([combined])
                    idx += 2
                    continue
        merged.append(current)
        idx += 1
    return merged


def join_molecules(molecules: List[List[str]]) -> str:
    """Deduplicate and join the final molecules in presentation order (Section 9)."""
    output: List[str] = []
    seen: set[str] = set()
    for molecule in molecules:
        fragment = " ".join(molecule).strip()
        if not fragment or fragment in seen:
            continue
        seen.add(fragment)
        output.append(fragment)
    if not output:
        return ""
    return " + ".join(output)


def _extract_descriptive_prefix(raw_description: str) -> str:
    if not raw_description:
        return ""
    tokens = raw_description.upper().split()
    kept: List[str] = []
    for token in tokens:
        clean = token.strip(",.")
        if not clean:
            continue
        if any(ch.isdigit() for ch in clean):
            break
        if clean in _PREFIX_PACKAGING_TOKENS or clean in _PREFIX_UNIT_TOKENS:
            break
        kept.append(clean)
    return " ".join(kept).strip()


def _prefer_descriptive_generic(raw_description: str, candidate: str) -> str:
    if not candidate:
        return ""
    candidate_norm = candidate.strip().upper()
    if not candidate_norm:
        return ""
    candidate_tokens = [tok for tok in re.split(r"\s+", candidate_norm) if tok and tok != "+"]
    prefix = _extract_descriptive_prefix(raw_description)
    has_bad_token = any(tok in GENERIC_BAD_SINGLE_TOKENS for tok in candidate_tokens)
    should_expand = prefix and prefix != candidate_norm and (
        candidate_norm in GENERIC_BAD_SINGLE_TOKENS or (len(candidate_tokens) <= 2 and has_bad_token)
    )
    if should_expand and (prefix.startswith(candidate_norm) or candidate_norm in prefix.split()):
        return prefix
    if candidate_norm in GENERIC_BAD_SINGLE_TOKENS:
        fallback, _ = extract_base_and_salts(raw_description)
        fallback_clean = fallback.strip().upper()
        if fallback_clean and fallback_clean != candidate_norm:
            return fallback_clean
        fallback, _ = extract_base_and_salts(raw_description)
        fallback_clean = fallback.strip().upper()
        if fallback_clean and fallback_clean != candidate_norm:
            return fallback_clean
    return candidate_norm


def normalize_generic(raw_description: str) -> str:
    """Normalize the raw_description field into the canonical generic_name (Section 10)."""
    tokens = tokenize_raw(raw_description)
    if not tokens:
        return ""
    classifications = [classify_token(token) for token in tokens]
    first_token = tokens[0].upper()
    if first_token == "VITAMINS":
        return "VITAMINS"
    block_tokens, block_classes = extract_molecule_block(tokens, classifications)
    molecules = split_molecules(block_tokens, block_classes)
    filtered = filter_formulation_tokens(molecules)
    candidate = join_molecules(filtered)
    return _prefer_descriptive_generic(raw_description, candidate)

__all__ = [
    "tokenize_raw",
    "classify_token",
    "extract_molecule_block",
    "split_molecules",
    "filter_formulation_tokens",
    "detect_salt_units",
    "join_molecules",
    "normalize_generic",
]
