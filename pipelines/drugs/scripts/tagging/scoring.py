"""
Candidate scoring functions for drug tagging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    ATC_COMBINATION_PATTERNS,
    CATEGORY_DOSE, CATEGORY_FORM, CATEGORY_GENERIC, CATEGORY_ROUTE, CATEGORY_SALT,
    PRIMARY_WEIGHTS, SECONDARY_WEIGHTS,
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


def score_candidate(
    input_tokens: List[str],
    input_categories: Dict[str, Dict[str, int]],
    candidate: Dict[str, Any],
) -> Tuple[float, float, str]:
    """
    Score a candidate match against input tokens.
    
    Returns (primary_score, secondary_score, reason).
    """
    primary = 0.0
    secondary = 0.0
    reasons = []
    
    cand_generic = str(candidate.get("generic_name", "")).upper()
    cand_form = str(candidate.get("form", "")).upper()
    cand_route = str(candidate.get("route", "")).upper()
    cand_doses = str(candidate.get("doses", "")).upper()
    
    # Score generic match
    input_generics = set(input_categories.get(CATEGORY_GENERIC, {}).keys())
    if cand_generic:
        cand_generic_parts = set(cand_generic.split())
        overlap = input_generics & cand_generic_parts
        if overlap:
            primary += PRIMARY_WEIGHTS[CATEGORY_GENERIC] * len(overlap)
            secondary += SECONDARY_WEIGHTS[CATEGORY_GENERIC] * len(overlap)
            reasons.append(f"generic:{len(overlap)}")
    
    # Score form match
    input_forms = set(input_categories.get(CATEGORY_FORM, {}).keys())
    if cand_form and cand_form in input_forms:
        primary += PRIMARY_WEIGHTS[CATEGORY_FORM]
        secondary += SECONDARY_WEIGHTS[CATEGORY_FORM]
        reasons.append("form")
    
    # Score route match
    input_routes = set(input_categories.get(CATEGORY_ROUTE, {}).keys())
    if cand_route and cand_route in input_routes:
        primary += PRIMARY_WEIGHTS[CATEGORY_ROUTE]
        secondary += SECONDARY_WEIGHTS[CATEGORY_ROUTE]
        reasons.append("route")
    
    # Score dose match
    input_doses = set(input_categories.get(CATEGORY_DOSE, {}).keys())
    if cand_doses:
        cand_dose_parts = set(cand_doses.replace("|", " ").split())
        dose_overlap = input_doses & cand_dose_parts
        if dose_overlap:
            primary += PRIMARY_WEIGHTS[CATEGORY_DOSE] * len(dose_overlap)
            secondary += SECONDARY_WEIGHTS[CATEGORY_DOSE] * len(dose_overlap)
            reasons.append(f"dose:{len(dose_overlap)}")
    
    reason = ",".join(reasons) if reasons else "no_match"
    return primary, secondary, reason


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
    Select the best candidate from a list.
    
    Returns the best candidate or None.
    """
    best_score = -float("inf")
    best_candidate = None
    
    for cand in candidates:
        cand_generic = str(cand.get("generic_name", "")).upper()
        cand_generic_normalized = apply_synonyms_fn(cand_generic)
        cand_is_combo = " + " in cand_generic
        
        # For combinations: check if candidate matches all input generics
        if is_combination:
            if cand_is_combo:
                cand_parts = set(p.strip() for p in cand_generic.split(" + "))
                input_base_parts = set()
                for ig in input_generics_normalized:
                    base = ig
                    for suffix in ["HYDROXIDE", "CHLORIDE", "SULFATE", "SULPHATE", "CARBONATE", "PHOSPHATE", "ACETATE", "CITRATE"]:
                        if base.endswith(" " + suffix):
                            base = base[:-len(suffix)-1].strip()
                            break
                    if base:
                        input_base_parts.add(base)
                
                if not input_base_parts.issubset(cand_parts) and not cand_parts.issubset(input_base_parts):
                    if not all(any(cp in ig or ig in cp for ig in input_base_parts) for cp in cand_parts):
                        continue
            else:
                continue
        
        # For single drugs: require exact match
        if is_single_drug and input_generics_normalized:
            input_generic = list(input_generics_normalized)[0]
            if input_generic != cand_generic_normalized and input_generic != cand_generic:
                if input_generic not in cand_generic and cand_generic not in input_generic:
                    continue
        
        # For IV solutions: match on active ingredient
        if is_iv_solution and stripped_generics:
            active = stripped_generics[0].upper() if stripped_generics else None
            if active:
                active_normalized = apply_synonyms_fn(active)
                if (active_normalized not in cand_generic and 
                    cand_generic not in active_normalized and
                    active_normalized != cand_generic_normalized and
                    active not in cand_generic):
                    continue
        
        # Score
        primary, secondary, reason = score_candidate(input_tokens, input_categories, cand)
        total_score = primary + secondary * 0.1
        
        # Bonus for exact generic match
        for ig in input_generics_normalized:
            if ig == cand_generic_normalized or ig == cand_generic:
                total_score += 10
                break
        
        # Bonus for IV solution matching base
        if is_iv_solution and len(input_generics_normalized) > 1:
            match_count = sum(1 for ig in input_generics_normalized 
                             if ig in cand_generic or cand_generic in ig)
            if match_count > 1:
                total_score += 5
        
        # ATC preference
        is_combo_atc = is_combination_atc(str(cand.get("atc_code", "")))
        if is_single_drug and is_combo_atc:
            total_score -= 5
        elif is_combination and not is_combo_atc:
            total_score -= 5
        
        if total_score > best_score:
            best_score = total_score
            best_candidate = cand
    
    return best_candidate
