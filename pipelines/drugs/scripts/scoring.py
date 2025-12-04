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

from .unified_constants import (
    # Categories
    CATEGORY_DOSE, CATEGORY_FORM, CATEGORY_GENERIC, CATEGORY_ROUTE, CATEGORY_SALT,
    # ATC patterns
    ATC_COMBINATION_PATTERNS,
    COMBINATION_ATC_SUFFIXES,
    # Form equivalence
    FORM_EQUIVALENCE_GROUPS,
    FORM_EQUIVALENTS,
    # Helper functions
    is_combination_atc as _is_combination_atc,
    forms_are_equivalent,
)
from .form_route_mapping import (
    FORM_ALIASES,
    FORM_TO_ROUTE,
    ROUTE_ALIASES,
    normalize_form,
    normalize_route,
    infer_route_from_form,
)


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


def parse_generic_with_subtype(generic: str) -> Tuple[str, Optional[str]]:
    """
    Parse a generic name that may have a subtype after a comma.
    
    Examples:
    - "VITAMIN INTRAVENOUS, FAT-SOLUBLE" -> ("VITAMIN INTRAVENOUS", "FAT-SOLUBLE")
    - "AMINO ACIDS, CRYSTALLINE STANDARD" -> ("AMINO ACIDS", "CRYSTALLINE STANDARD")
    - "PARACETAMOL" -> ("PARACETAMOL", None)
    
    Returns (base_name, subtype or None).
    """
    if "," in generic and " + " not in generic and " AND " not in generic:
        parts = generic.split(",", 1)
        base = parts[0].strip()
        subtype = parts[1].strip() if len(parts) > 1 else None
        return base, subtype
    return generic, None


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
    - Comma separates base name from subtype (e.g., "VITAMIN, FAT-SOLUBLE")
      - First filter by base name, then by subtype
    """
    if not input_generics or not candidate_generic:
        return False, "missing_generic"
    
    cand_upper = candidate_generic.upper()
    cand_normalized = apply_synonyms_fn(cand_upper)
    
    # Parse candidate for base name and subtype
    cand_base, cand_subtype = parse_generic_with_subtype(cand_upper)
    cand_base_normalized = apply_synonyms_fn(cand_base)
    
    # Check if candidate is a combination (using + or AND, not comma)
    cand_is_combo = " + " in cand_upper or " AND " in cand_upper
    
    if cand_is_combo:
        # Split candidate into parts (don't split on comma for combos)
        cand_parts = set(
            p.strip() for p in re.split(r'\s*\+\s*|\s+AND\s+', cand_upper) 
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
            
            # Parse input for base name and subtype
            inp_base, inp_subtype = parse_generic_with_subtype(inp)
            inp_base_normalized = apply_synonyms_fn(inp_base)
            
            # Step 1: Base name must match
            base_matches = (
                inp_base_normalized == cand_base_normalized or
                inp_base_normalized == cand_base or
                inp_base_normalized in cand_base_normalized or
                cand_base_normalized in inp_base_normalized or
                inp_base in cand_base or
                cand_base in inp_base
            )
            
            if not base_matches:
                continue
            
            # Step 2: If input has subtype, candidate must also have matching subtype
            if inp_subtype:
                if not cand_subtype:
                    # Input has subtype but candidate doesn't - partial match
                    continue
                # Check if subtypes match
                if inp_subtype.upper() in cand_subtype.upper() or cand_subtype.upper() in inp_subtype.upper():
                    return True, "exact_with_subtype"
                else:
                    continue  # Subtypes don't match
            
            # No subtype in input, or subtypes match
            if inp_normalized == cand_normalized or inp_normalized == cand_upper:
                return True, "exact"
            
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
    input_details: Optional[Dict[str, Optional[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Select the best candidate using rule-based matching.
    
    Priority rules:
    1. Generic MUST match (required)
    2. Prefer single ATC for single drugs, combo ATC for combinations
    3. Form equivalence (tablet/capsule are interchangeable)
    4. Dose is flexible (same ATC regardless of dose)
    5. Salt is flexible (same base drug regardless of salt form)
    6. Tie-break using _details (type, release, form, indication, etc.)
    
    Returns the best candidate or None.
    """
    input_details = input_details or {}
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
    
    # Extract input details for tie-breaking
    input_type = str(input_details.get("type_details") or "").upper()
    input_release = str(input_details.get("release_details") or "").upper()
    input_form_det = str(input_details.get("form_details") or "").upper()
    input_indication = str(input_details.get("indication_details") or "").upper()
    input_salt = str(input_details.get("salt_details") or "").upper()
    input_brand = str(input_details.get("brand_details") or "").upper()
    input_alias = str(input_details.get("alias_details") or "").upper()
    input_diluent = str(input_details.get("diluent_details") or "").upper()
    input_iv_type = str(input_details.get("iv_diluent_type") or "").upper()
    
    # Rank valid candidates by preference
    def rank_candidate(item: Tuple[Dict[str, Any], str]) -> Tuple[int, int, int, int, int, str]:
        cand, match_reason = item
        cand_atc = str(cand.get("atc_code", ""))
        cand_form = str(cand.get("form", "")).upper()
        cand_route = str(cand.get("route", "")).upper()
        cand_source = str(cand.get("source", "")).lower()
        cand_generic = str(cand.get("generic_name", "")).upper()
        cand_ref = str(cand.get("reference_text", "")).upper()
        
        # Priority 1: Match type (exact > substring > combo)
        match_priority = {
            "exact": 0,
            "exact_with_subtype": 0,
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
        
        # Priority 4: Details match (tie-breaker)
        # Count how many detail fields match between input and candidate
        details_score = 0
        
        # Release details match (e.g., MR, SR, XR, ER) - highest priority
        if input_release:
            if input_release in cand_ref or input_release in cand_generic:
                details_score -= 10  # Lower is better
        
        # Type details match (e.g., HUMAN, ANHYDROUS)
        if input_type:
            if input_type in cand_ref or input_type in cand_generic:
                details_score -= 5
        
        # Form details match (e.g., FILM COATED, CHEWABLE)
        if input_form_det:
            if input_form_det in cand_ref or input_form_det in cand_generic:
                details_score -= 5
        
        # Indication details match (e.g., FOR HEPATIC FAILURE)
        if input_indication:
            if input_indication in cand_ref or input_indication in cand_generic:
                details_score -= 5
        
        # Salt details match - bonus if reference mentions same salt
        if input_salt:
            if input_salt in cand_ref or input_salt in cand_generic:
                details_score -= 3
        
        # Brand details - used for resolution only (brand→generic), NOT for preference
        # Two different brands of the same generic are equivalent
        # Brand match only helps when candidate reference mentions the brand
        # (This helps match brand-only entries to their generic equivalents)
        if input_brand:
            if input_brand in cand_ref:
                details_score -= 1  # Minimal bonus - just for resolution
        
        # Alias details match (e.g., VIT. D3 = CHOLECALCIFEROL)
        if input_alias:
            if input_alias in cand_ref or input_alias in cand_generic:
                details_score -= 2
        
        # IV diluent type match (e.g., WATER, SODIUM CHLORIDE, LACTATED RINGER'S)
        if input_iv_type:
            if input_iv_type in cand_ref or input_iv_type in cand_generic:
                details_score -= 5
        
        # Priority 5: Prefer longer/more specific generic names
        # This prevents IODINE from being preferred over IODAMIDE when both match
        # Negative length means longer = better (lower sort value)
        length_priority = -len(cand_generic)
        
        return (match_priority, atc_priority, form_priority, details_score, length_priority, cand_atc)
    
    # Sort by rank and return best
    valid_candidates.sort(key=rank_candidate)
    return valid_candidates[0][0]
