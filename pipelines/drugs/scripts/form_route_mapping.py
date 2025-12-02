"""
Unified form-route mapping derived from DrugBank, PNF, and WHO reference datasets.

This module provides:
1. FORM_ALIASES: Spelling variations -> canonical form (from unified_constants.FORM_CANON)
2. FORM_TO_ROUTE: Form -> most common route (from unified_constants.FORM_TO_ROUTE)
3. ROUTE_ALIASES: Route abbreviations (from unified_constants.ROUTE_CANON)
4. Helper functions for normalization and inference
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .unified_constants import (
    FORM_CANON as _FORM_CANON,
    FORM_TO_ROUTE as _FORM_TO_ROUTE,
    FORM_TO_ROUTES as _FORM_TO_ROUTES,
    ROUTE_CANON as _ROUTE_CANON,
    get_valid_routes_for_form,
    is_valid_form_route_pair,
)

# Re-export from unified_constants (uppercase for this module's API)
FORM_ALIASES: Dict[str, str] = _FORM_CANON
FORM_TO_ROUTE: Dict[str, str] = _FORM_TO_ROUTE
FORM_TO_ROUTES: Dict[str, List[str]] = _FORM_TO_ROUTES
ROUTE_ALIASES: Dict[str, str] = _ROUTE_CANON


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_form(form: str) -> str:
    """Normalize a dosage form to its canonical name."""
    if not form:
        return ""
    form_upper = form.upper().strip()
    return FORM_ALIASES.get(form_upper, form_upper)


def normalize_route(route: str) -> str:
    """Normalize a route to its canonical name."""
    if not route:
        return ""
    route_upper = route.upper().strip()
    return ROUTE_ALIASES.get(route_upper, route_upper)


def infer_route_from_form(form: str) -> Optional[str]:
    """
    Infer the administration route from the dosage form.
    
    Returns the most common route for the given form, or None if unknown.
    """
    if not form:
        return None
    
    # Normalize form first
    form_normalized = normalize_form(form)
    
    # Direct lookup
    if form_normalized in FORM_TO_ROUTE:
        return FORM_TO_ROUTE[form_normalized]
    
    # Try base form (strip everything after comma except release info)
    if "," in form_normalized:
        parts = form_normalized.split(",", 1)
        base = parts[0].strip()
        modifier = parts[1].strip() if len(parts) > 1 else ""
        
        # Keep release modifiers
        if "RELEASE" in modifier:
            combined = f"{base}, {modifier}"
            if combined in FORM_TO_ROUTE:
                return FORM_TO_ROUTE[combined]
        
        # Try just base form
        if base in FORM_TO_ROUTE:
            return FORM_TO_ROUTE[base]
    
    return None


def explode_kit_forms(form: str) -> list[str]:
    """
    Explode kit forms into individual components.
    
    Example: "KIT; TABLET" -> ["KIT", "TABLET"]
    """
    if not form:
        return []
    
    form_upper = form.upper().strip()
    
    if ";" in form_upper:
        parts = [p.strip() for p in form_upper.split(";") if p.strip()]
        return [normalize_form(p) for p in parts]
    
    return [normalize_form(form_upper)]
