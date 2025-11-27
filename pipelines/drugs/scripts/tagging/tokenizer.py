"""
Tokenization and normalization functions for drug descriptions.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from .constants import (
    CATEGORY_DOSE, CATEGORY_FORM, CATEGORY_GENERIC, CATEGORY_OTHER,
    CATEGORY_ROUTE, CATEGORY_SALT, ELEMENT_DRUGS, FORM_CANON, GENERIC_JUNK_TOKENS,
    NATURAL_STOPWORDS, PURE_SALT_COMPOUNDS, ROUTE_CANON, SALT_TOKENS,
    UNIT_TOKENS,
)


# Regex patterns
_DOSE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|ug|ml|l|iu|unit|units|pct|%|mg/ml|mcg/ml|iu/ml|mg/5ml)",
    re.IGNORECASE,
)
_PARENTHESES_PATTERN = re.compile(r"\([^)]*\)")


def split_with_parentheses(text: str) -> List[str]:
    """
    Split text into tokens, preserving parenthetical content as single tokens.
    """
    if not text:
        return []
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    # Extract parenthetical content
    parens = _PARENTHESES_PATTERN.findall(text)
    
    # Replace parentheses with placeholder
    temp = _PARENTHESES_PATTERN.sub(" __PAREN__ ", text)
    
    # Split on whitespace and common delimiters
    tokens = re.split(r"[\s,;]+", temp)
    
    # Restore parenthetical content
    result = []
    paren_idx = 0
    for tok in tokens:
        if tok == "__PAREN__" and paren_idx < len(parens):
            result.append(parens[paren_idx])
            paren_idx += 1
        elif tok:
            result.append(tok)
    
    return result


def detect_compound_salts(tokens: List[str], original_text: str) -> List[str]:
    """
    Detect and preserve compound salt names (e.g., SODIUM CHLORIDE).
    """
    text_upper = original_text.upper()
    result = []
    skip_next = False
    
    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        
        tok_upper = tok.upper()
        
        # Check if this token starts a compound salt
        if i + 1 < len(tokens):
            compound = f"{tok_upper} {tokens[i + 1].upper()}"
            if compound in PURE_SALT_COMPOUNDS:
                result.append(compound)
                skip_next = True
                continue
        
        result.append(tok)
    
    return result


def normalize_tokens(
    tokens: List[str],
    drop_stopwords: bool = True,
    multiword_generics: Optional[Set[str]] = None,
) -> List[str]:
    """
    Normalize tokens: uppercase, strip punctuation, handle multi-word generics.
    """
    if multiword_generics is None:
        multiword_generics = set()
    
    result = []
    text = " ".join(tokens).upper()
    
    # First, extract multi-word generics
    for mwg in sorted(multiword_generics, key=len, reverse=True):
        if mwg in text:
            result.append(mwg)
            text = text.replace(mwg, " ")
    
    # Then split remaining text
    remaining = re.split(r"[\s,;]+", text)
    
    for tok in remaining:
        if not tok:
            continue
        
        # Strip punctuation from ends
        tok = tok.strip(".,;:!?\"'()[]{}").upper()
        
        if not tok:
            continue
        
        if drop_stopwords and tok in NATURAL_STOPWORDS:
            continue
        
        result.append(tok)
    
    return result


def categorize_tokens(tokens: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Categorize tokens into GENERIC, SALT, DOSE, FORM, ROUTE, OTHER.
    
    Returns dict of {category: {token: count}}.
    """
    categories: Dict[str, Dict[str, int]] = {
        CATEGORY_GENERIC: {},
        CATEGORY_SALT: {},
        CATEGORY_DOSE: {},
        CATEGORY_FORM: {},
        CATEGORY_ROUTE: {},
        CATEGORY_OTHER: {},
    }
    
    for tok in tokens:
        tok_upper = tok.upper()
        
        # Check dose pattern
        if _DOSE_PATTERN.match(tok_upper) or tok_upper in UNIT_TOKENS:
            categories[CATEGORY_DOSE][tok_upper] = categories[CATEGORY_DOSE].get(tok_upper, 0) + 1
            continue
        
        # Check form
        if tok_upper in FORM_CANON:
            canon = FORM_CANON[tok_upper]
            categories[CATEGORY_FORM][canon] = categories[CATEGORY_FORM].get(canon, 0) + 1
            continue
        
        # Check route
        if tok_upper in ROUTE_CANON:
            canon = ROUTE_CANON[tok_upper]
            categories[CATEGORY_ROUTE][canon] = categories[CATEGORY_ROUTE].get(canon, 0) + 1
            continue
        
        # Check salt (but element drugs can be generics too)
        if tok_upper in SALT_TOKENS:
            # If it's an element drug and appears to be the main drug (first token),
            # treat it as generic instead of salt
            if tok_upper in ELEMENT_DRUGS:
                # Check if this is likely the main drug (not a salt modifier)
                # It's a main drug ONLY if it's the first token in the list
                tok_idx = tokens.index(tok) if tok in tokens else -1
                is_main_drug = tok_idx == 0
                if is_main_drug:
                    categories[CATEGORY_GENERIC][tok_upper] = categories[CATEGORY_GENERIC].get(tok_upper, 0) + 1
                    continue
            categories[CATEGORY_SALT][tok_upper] = categories[CATEGORY_SALT].get(tok_upper, 0) + 1
            continue
        
        # Check if it's a pure number (dose without unit)
        if tok_upper.replace(".", "").isdigit():
            categories[CATEGORY_DOSE][tok_upper] = categories[CATEGORY_DOSE].get(tok_upper, 0) + 1
            continue
        
        # Check junk tokens
        if tok_upper in GENERIC_JUNK_TOKENS:
            categories[CATEGORY_OTHER][tok_upper] = categories[CATEGORY_OTHER].get(tok_upper, 0) + 1
            continue
        
        # Default to generic
        categories[CATEGORY_GENERIC][tok_upper] = categories[CATEGORY_GENERIC].get(tok_upper, 0) + 1
    
    return categories


def extract_generic_tokens(
    text: str,
    multiword_generics: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Extract generic drug tokens from text.
    
    Returns (all_tokens, generic_tokens).
    """
    if multiword_generics is None:
        multiword_generics = set()
    
    # Check for multiword generics with commas BEFORE tokenizing
    text_upper = text.upper()
    matched_multiword = []
    for mw in multiword_generics:
        if mw in text_upper:
            # Store with position for ordering
            pos = text_upper.find(mw)
            matched_multiword.append((pos, mw))
    # Sort by position in text
    matched_multiword.sort(key=lambda x: x[0])
    
    # Tokenize
    raw_tokens = split_with_parentheses(text)
    raw_tokens = detect_compound_salts(raw_tokens, text)
    tokens = normalize_tokens(raw_tokens, drop_stopwords=True, multiword_generics=multiword_generics)
    
    # Categorize
    categories = categorize_tokens(tokens)
    generic_tokens = list(categories.get(CATEGORY_GENERIC, {}).keys())
    
    # Add matched multiword generics that may have been split (in order of appearance)
    for pos, mw in matched_multiword:
        if mw not in generic_tokens:
            # Insert at appropriate position based on text order
            inserted = False
            for i, gt in enumerate(generic_tokens):
                gt_pos = text_upper.find(gt)
                if gt_pos > pos:
                    generic_tokens.insert(i, mw)
                    inserted = True
                    break
            if not inserted:
                generic_tokens.append(mw)
    
    # Add pure salt compounds that appear in text
    text_upper = text.upper()
    for psc in PURE_SALT_COMPOUNDS:
        if psc in text_upper and psc not in generic_tokens:
            generic_tokens.append(psc)
    
    # Handle combination drugs with "+" separator
    if "+" in text_upper:
        parts = text_upper.split("+")
        for part in parts:
            part = part.strip()
            words = []
            for word in part.split():
                if word and not any(c.isdigit() for c in word) and word not in UNIT_TOKENS:
                    words.append(word)
                else:
                    break
            if words:
                combo_part = " ".join(words)
                if combo_part and combo_part not in generic_tokens:
                    generic_tokens.append(combo_part)
    
    # Handle " IN " separator for IV solutions
    # Reorder generics so active ingredient (before IN) comes first
    if " IN " in text_upper and "+" not in text_upper:
        parts = text_upper.split(" IN ", 1)
        if len(parts) == 2:
            skip_words = {"SOLUTION", "BOTTLE", "BAG", "VIAL", "AMPULE", "L", "ML", "WATER"}
            
            # Extract active ingredient (before IN)
            # Skip leading dose tokens (e.g., "5%")
            active_words = []
            for word in parts[0].strip().split():
                # Skip dose tokens at the start
                if any(c.isdigit() for c in word) or word in UNIT_TOKENS:
                    continue
                if word in skip_words:
                    continue
                if word:
                    active_words.append(word)
            
            active_name = " ".join(active_words) if active_words else None
            
            # Extract solution base (after IN)
            base_words = []
            for word in parts[1].strip().split():
                if word and not any(c.isdigit() for c in word) and word not in UNIT_TOKENS and word not in skip_words:
                    base_words.append(word)
                else:
                    break
            
            base_name = " ".join(base_words) if base_words else None
            
            # Reorder: active first, then base, then others
            if active_name or base_name:
                new_order = []
                if active_name:
                    # Move active to front if it exists in list
                    if active_name in generic_tokens:
                        generic_tokens.remove(active_name)
                    new_order.append(active_name)
                if base_name:
                    # Move base to second position if it exists in list
                    if base_name in generic_tokens:
                        generic_tokens.remove(base_name)
                    new_order.append(base_name)
                # Add remaining generics
                new_order.extend(generic_tokens)
                generic_tokens = new_order
    
    return tokens, generic_tokens


def strip_salt_suffix(
    generic: str,
    salt_suffixes: Optional[Set[str]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Strip salt suffix from a generic name.
    
    Returns (base_name, salt_suffix or None).
    """
    if salt_suffixes is None:
        salt_suffixes = SALT_TOKENS
    
    generic_upper = generic.upper()
    
    # Don't strip from pure salt compounds
    if generic_upper in PURE_SALT_COMPOUNDS:
        return generic_upper, None
    
    # Check each salt suffix
    for suffix in sorted(salt_suffixes, key=len, reverse=True):
        if generic_upper.endswith(" " + suffix):
            base = generic_upper[:-len(suffix) - 1].strip()
            # Also strip trailing "AS" (e.g., "AMLODIPINE AS BESILATE" -> "AMLODIPINE")
            if base.endswith(" AS"):
                base = base[:-3].strip()
            return base, suffix
    
    # Handle "X AS Y" pattern where Y is a salt
    if " AS " in generic_upper:
        parts = generic_upper.split(" AS ", 1)
        if len(parts) == 2:
            potential_salt = parts[1].strip()
            if potential_salt in salt_suffixes:
                return parts[0].strip(), potential_salt
    
    return generic_upper, None
