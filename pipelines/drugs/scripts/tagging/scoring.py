"""
Rule-based candidate selection for drug tagging.

Rules (in order of priority):
1. GENERIC MUST MATCH - Base generic name must match (after synonym normalization)
2. SALT IS FLEXIBLE - Different salt forms of same drug are acceptable
3. DOSE IS FLEXIBLE - Different doses are acceptable if ATC code is the same
4. FORM IS FLEXIBLE - Certain forms are interchangeable (tablet/capsule, solution/suspension)
5. ROUTE IS FLEXIBLE - If not specified in input, any route is acceptable
6. BRANDS -> GENERICS - Brand names are resolved to generics before matching
7. SYNONYMS -> CANONICAL - Synonyms are resolved to canonical names before matching
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    ATC_COMBINATION_PATTERNS,
    CATEGORY_DOSE, CATEGORY_FORM, CATEGORY_GENERIC, CATEGORY_ROUTE, CATEGORY_SALT,
    FORM_CANON,
)


# ============================================================================
# FORM EQUIVALENCE GROUPS
# Forms within the same group are considered interchangeable
# ============================================================================

FORM_EQUIVALENCE_GROUPS = [
    # Solid oral forms - generally interchangeable
    {"TABLET", "CAPSULE", "CAPLET"},
    # Liquid oral forms
    {"SOLUTION", "SYRUP", "ELIXIR", "ORAL SOLUTION"},
    # Suspensions
    {"SUSPENSION", "ORAL SUSPENSION"},
    # Topical semi-solids
    {"CREAM", "OINTMENT", "GEL"},
    # Injectable forms
    {"INJECTION", "SOLUTION FOR INJECTION", "POWDER FOR INJECTION"},
    # Inhalation forms
    {"INHALER", "AEROSOL", "MDI", "DPI", "NEBULE"},
    # Eye preparations
    {"EYE DROPS", "OPHTHALMIC SOLUTION", "DROPS"},
]

# Build lookup: form -> set of equivalent forms
FORM_EQUIVALENTS: Dict[str, Set[str]] = {}
for group in FORM_EQUIVALENCE_GROUPS:
    for form in group:
        FORM_EQUIVALENTS[form] = group


def is_combination_atc(atc_code: str) -> bool:
    """Check if an ATC code represents a combination product."""
    if not atc_code:
        return False
    
    atc_upper = atc_code.upper()
    for pattern in ATC_COMBINATION_PATTERNS:
        if atc_upper.startswith(pattern):
            return True
    
    return False


def sort_atc_codes(
    atc_codes: List[str],
    prefer_single: bool = True,
) -> List[str]:
    """
    Sort ATC codes by preference.
    
    If prefer_single=True, single-agent codes come before combinations.
    """
    def sort_key(atc: str) -> Tuple[bool, int, str]:
        is_combo = is_combination_atc(atc)
        if prefer_single:
            return (is_combo, len(atc), atc)
        return (not is_combo, len(atc), atc)
    
    return sorted([a for a in atc_codes if a], key=sort_key)


def forms_are_equivalent(form1: str, form2: str) -> bool:
    """Check if two forms are considered equivalent/interchangeable."""
    if not form1 or not form2:
        return True  # If either is missing, consider flexible
    
    f1 = form1.upper()
    f2 = form2.upper()
    
    if f1 == f2:
        return True
    
    # Check if in same equivalence group
    equiv1 = FORM_EQUIVALENTS.get(f1, {f1})
    return f2 in equiv1


def generics_match(
    input_generics: Set[str],
    candidate_generic: str,
    apply_synonyms_fn,
) -> Tuple[bool, str]:
    """
    Check if input generics match candidate generic.
    
    Returns (matches, reason).
    
    Rules:
    - Synonyms are normalized to canonical form
    - Salt forms are stripped for comparison
    - For combinations, all components must match
    """
    if not input_generics or not candidate_generic:
        return False, "missing_generic"
    
    cand_upper = candidate_generic.upper()
    cand_normalized = apply_synonyms_fn(cand_upper)
    
    # Check if candidate is a combination
    cand_is_combo = " + " in cand_upper or " AND " in cand_upper
    
    if cand_is_combo:
        # Split candidate into parts
        cand_parts = set(
            p.strip() for p in re.split(r'\s*\+\s*|\s+AND\s+|,\s*', cand_upper) 
            if p.strip()
        )
        cand_parts_normalized = {apply_synonyms_fn(p) for p in cand_parts}
        
        # Normalize input generics
        input_normalized = {apply_synonyms_fn(g) for g in input_generics}
        
        # Check overlap
        overlap = input_normalized & cand_parts_normalized
        if overlap:
            return True, "combo_match"
        
        # Try substring matching for partial names
        for inp in input_normalized:
            for cp in cand_parts_normalized:
                if inp in cp or cp in inp:
                    return True, "combo_partial"
        
        return False, "combo_no_match"
    else:
        # Single drug - check exact or substring match
        for inp in input_generics:
            inp_normalized = apply_synonyms_fn(inp)
            
            # Exact match
            if inp_normalized == cand_normalized or inp_normalized == cand_upper:
                return True, "exact"
            
            # Substring match (for multi-word generics)
            if inp_normalized in cand_normalized or cand_normalized in inp_normalized:
                return True, "substring"
            
            if inp in cand_upper or cand_upper in inp:
                return True, "substring"
        
        return False, "no_match"


def select_best_candidate(
    candidates: List[Dict[str, Any]],
    input_tokens: List[str],
    input_categories: Dict[str, Dict[str, int]],
    input_generics_normalized: Set[str],
    is_single_drug: bool,
    is_combination: bool,
    is_iv_solution: bool,
    stripped_generics: List[str],
    apply_synonyms_fn,
) -> Optional[Dict[str, Any]]:
    """
    Select the best candidate using rule-based matching.
    
    Priority rules:
    1. Generic MUST match (required)
    2. Prefer single ATC for single drugs, combo ATC for combinations
    3. Form equivalence (tablet/capsule are interchangeable)
    4. Dose is flexible (same ATC regardless of dose)
    5. Salt is flexible (same base drug regardless of salt form)
    
    Returns the best candidate or None.
    """
    # Extract input characteristics
    input_forms = set(input_categories.get(CATEGORY_FORM, {}).keys())
    input_routes = set(input_categories.get(CATEGORY_ROUTE, {}).keys())
    input_doses = set(input_categories.get(CATEGORY_DOSE, {}).keys())
    
    # Filter candidates by generic match (REQUIRED)
    valid_candidates = []
    
    for cand in candidates:
        cand_generic = str(cand.get("generic_name", "")).upper()
        cand_atc = str(cand.get("atc_code", ""))
        
        # Rule 1: Generic must match
        matches, match_reason = generics_match(
            input_generics_normalized, cand_generic, apply_synonyms_fn
        )
        
        if not matches:
            continue
        
        # For IV solutions: prefer active ingredient over vehicle
        if is_iv_solution and stripped_generics and len(stripped_generics) > 1:
            active = stripped_generics[0].upper()
            vehicle = stripped_generics[1].upper()
            active_normalized = apply_synonyms_fn(active)
            vehicle_normalized = apply_synonyms_fn(vehicle)
            cand_normalized = apply_synonyms_fn(cand_generic)
            
            # Skip if matches vehicle but not active
            is_vehicle_match = (vehicle_normalized in cand_generic or 
                               cand_generic in vehicle_normalized or
                               vehicle_normalized == cand_normalized)
            is_active_match = (active_normalized in cand_generic or 
                              cand_generic in active_normalized or
                              active_normalized == cand_normalized)
            
            if is_vehicle_match and not is_active_match:
                continue
        
        # For combinations: prefer combo candidates
        cand_is_combo = " + " in cand_generic or " AND " in cand_generic
        if is_combination and not cand_is_combo:
            continue
        
        valid_candidates.append((cand, match_reason))
    
    if not valid_candidates:
        return None
    
    # If only one valid candidate, return it
    if len(valid_candidates) == 1:
        return valid_candidates[0][0]
    
    # Rank valid candidates by preference
    def rank_candidate(item: Tuple[Dict[str, Any], str]) -> Tuple[int, int, int, int, str]:
        cand, match_reason = item
        cand_atc = str(cand.get("atc_code", ""))
        cand_form = str(cand.get("form", "")).upper()
        cand_route = str(cand.get("route", "")).upper()
        cand_source = str(cand.get("source", "")).lower()
        
        # Priority 1: Match type (exact > substring > combo)
        match_priority = {
            "exact": 0,
            "combo_match": 1,
            "substring": 2,
            "combo_partial": 3,
        }.get(match_reason, 4)
        
        # Priority 2: ATC type preference
        is_combo_atc = is_combination_atc(cand_atc)
        if is_single_drug:
            atc_priority = 0 if not is_combo_atc else 1
        elif is_combination:
            atc_priority = 0 if is_combo_atc else 1
        else:
            atc_priority = 0
        
        # Priority 3: Form match
        form_priority = 0
        if input_forms:
            if cand_form in input_forms:
                form_priority = 0  # Exact match
            elif any(forms_are_equivalent(cand_form, f) for f in input_forms):
                form_priority = 1  # Equivalent form
            else:
                form_priority = 2  # Different form
        
        # Priority 4: Source preference (pnf > who > drugbank)
        source_priority = {
            "pnf": 0,
            "who": 1,
            "drugbank": 2,
        }.get(cand_source.split("|")[0] if cand_source else "", 3)
        
        return (match_priority, atc_priority, form_priority, source_priority, cand_atc)
    
    # Sort by rank and return best
    valid_candidates.sort(key=rank_candidate)
    return valid_candidates[0][0]
