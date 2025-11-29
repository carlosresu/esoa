"""
Unified form-route mapping derived from DrugBank, PNF, and WHO reference datasets.

This module provides:
1. FORM_ALIASES: Spelling variations -> canonical form
2. FORM_TO_ROUTE: Form -> most common route (for inference when route not specified)
3. VALID_FORM_ROUTE_PAIRS: Set of valid (form, route) combinations from reference data
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple


# ============================================================================
# FORM ALIASES (spelling variations -> canonical form)
# Derived from DrugBank, PNF, WHO reference datasets
# ============================================================================

FORM_ALIASES: Dict[str, str] = {
    # Tablet variations
    "TAB": "TABLET",
    "TABS": "TABLET",
    "TABLETS": "TABLET",
    "TABLET, FILM COATED": "TABLET",
    "TABLET, COATED": "TABLET",
    "TABLET, SUGAR COATED": "TABLET",
    "TABLET, CHEWABLE": "TABLET",
    "TABLET, ORALLY DISINTEGRATING": "TABLET",
    "TABLET, EFFERVESCENT": "TABLET",
    "TABLET, SOLUBLE": "TABLET",
    "TABLET, MULTILAYER": "TABLET",
    "TABLET, FOR SUSPENSION": "TABLET",
    # Extended/modified release tablets (keep release info)
    "TABLET, EXTENDED RELEASE": "TABLET, EXTENDED RELEASE",
    "TABLET, FILM COATED, EXTENDED RELEASE": "TABLET, EXTENDED RELEASE",
    "TABLET, DELAYED RELEASE": "TABLET, DELAYED RELEASE",
    "TABLET, MULTILAYER, EXTENDED RELEASE": "TABLET, EXTENDED RELEASE",
    "TABLET, CHEWABLE, EXTENDED RELEASE": "TABLET, EXTENDED RELEASE",
    # Capsule variations
    "CAP": "CAPSULE",
    "CAPS": "CAPSULE",
    "CAPSULES": "CAPSULE",
    "CAPSULE, LIQUID FILLED": "CAPSULE",
    "CAPSULE, GELATIN COATED": "CAPSULE",
    "CAPSULE, COATED": "CAPSULE",
    "CAPSULE, COATED PELLETS": "CAPSULE",
    # Extended/modified release capsules (keep release info)
    "CAPSULE, EXTENDED RELEASE": "CAPSULE, EXTENDED RELEASE",
    "CAPSULE, DELAYED RELEASE": "CAPSULE, DELAYED RELEASE",
    "CAPSULE, COATED, EXTENDED RELEASE": "CAPSULE, EXTENDED RELEASE",
    "CAPSULE, DELAYED RELEASE PELLETS": "CAPSULE, DELAYED RELEASE",
    # Solution variations
    "SOLN": "SOLUTION",
    "SOL": "SOLUTION",
    "SOLUTIONS": "SOLUTION",
    "SOLUTION, CONCENTRATE": "SOLUTION",
    "ORAL SOLUTION": "SOLUTION",
    # Suspension variations
    "SUSP": "SUSPENSION",
    "SUSPENSIONS": "SUSPENSION",
    "ORAL SUSPENSION": "SUSPENSION",
    "SUSPENSION, EXTENDED RELEASE": "SUSPENSION, EXTENDED RELEASE",
    # Syrup variations
    "SYR": "SYRUP",
    "SYRUPS": "SYRUP",
    # Injection variations
    "INJ": "INJECTION",
    "INJECTABLE": "INJECTION",
    "INJECTION, SOLUTION": "INJECTION",
    "INJECTION, EMULSION": "INJECTION",
    "INJECTION, SUSPENSION": "INJECTION",
    "INJECTION, POWDER, FOR SOLUTION": "INJECTION",
    "INJECTION, POWDER, FOR SUSPENSION": "INJECTION",
    "INJECTION, POWDER, LYOPHILIZED, FOR SOLUTION": "INJECTION",
    "INJECTION, POWDER, LYOPHILIZED, FOR SUSPENSION": "INJECTION",
    "INJECTION, SOLUTION, CONCENTRATE": "INJECTION",
    "INJECTION, SUSPENSION, EXTENDED RELEASE": "INJECTION, EXTENDED RELEASE",
    "INJECTION, POWDER, FOR SUSPENSION, EXTENDED RELEASE": "INJECTION, EXTENDED RELEASE",
    "INJECTABLE, LIPOSOMAL": "INJECTION",
    # Ampule/Vial variations
    "AMP": "AMPULE",
    "AMPUL": "AMPULE",
    "AMPULS": "AMPULE",
    "AMPULES": "AMPULE",
    "AMPOULE": "AMPULE",
    "VIALS": "VIAL",
    # Cream variations
    "CREAMS": "CREAM",
    "CREAM, AUGMENTED": "CREAM",
    # Ointment variations
    "OINTMENTS": "OINTMENT",
    # Gel variations
    "GELS": "GEL",
    "GEL, METERED": "GEL",
    # Lotion variations
    "LOTIONS": "LOTION",
    "LOTION, AUGMENTED": "LOTION",
    # Powder variations
    "PWDR": "POWDER",
    "POWDERS": "POWDER",
    "POWDER, FOR SOLUTION": "POWDER",
    "POWDER, FOR SUSPENSION": "POWDER",
    "POWDER, METERED": "POWDER, METERED",  # Keep for inhalation
    # Granule variations
    "GRAN": "GRANULE",
    "GRANULES": "GRANULE",
    "GRANULE, FOR SOLUTION": "GRANULE",
    "GRANULE, FOR SUSPENSION": "GRANULE",
    "GRANULE, EFFERVESCENT": "GRANULE",
    "GRANULE, DELAYED RELEASE": "GRANULE, DELAYED RELEASE",
    # Spray variations
    "SPRAYS": "SPRAY",
    "SPRAY, METERED": "SPRAY, METERED",  # Keep for nasal
    "SPRAY, SUSPENSION": "SPRAY",
    # Aerosol variations
    "AEROSOLS": "AEROSOL",
    "AEROSOL, SPRAY": "AEROSOL",
    "AEROSOL, FOAM": "AEROSOL",
    "AEROSOL, POWDER": "AEROSOL",
    "AEROSOL, METERED": "AEROSOL, METERED",  # Keep for inhalation
    # Inhaler variations
    "INHALERS": "INHALER",
    "MDI": "INHALER",
    "DPI": "INHALER",
    "METERED DOSE INHALER": "INHALER",
    "DRY POWDER INHALER": "INHALER",
    "INHALANT": "INHALER",
    # Nebule variations
    "NEB": "NEBULE",
    "NEBULES": "NEBULE",
    "NEBULIZER SOLUTION": "NEBULE",
    # Drops variations
    "DROP": "DROPS",
    "EYE DROPS": "DROPS",
    "EAR DROPS": "DROPS",
    "NASAL DROPS": "DROPS",
    "ORAL DROPS": "DROPS",
    "SOLUTION / DROPS": "DROPS",
    "SUSPENSION / DROPS": "DROPS",
    "SOLUTION, GEL FORMING / DROPS": "DROPS",
    # Suppository variations
    "SUPP": "SUPPOSITORY",
    "SUPPOSITORIES": "SUPPOSITORY",
    # Patch variations
    "PATCHES": "PATCH",
    "PATCH, EXTENDED RELEASE": "PATCH",
    # Film variations
    "FILMS": "FILM",
    "FILM, SOLUBLE": "FILM",
    "FILM, EXTENDED RELEASE": "FILM",
    # Sachet variations
    "SACHETS": "SACHET",
    # Lozenge variations
    "LOZENGES": "LOZENGE",
    # Other variations
    "EMULSIONS": "EMULSION",
    "FOAMS": "FOAM",
    "PASTES": "PASTE",
    "SHAMPOOS": "SHAMPOO",
    "SOAPS": "SOAP",
    "ENEMAS": "ENEMA",
    "IMPLANTS": "IMPLANT",
    "INSERTS": "INSERT",
    "RINGS": "RING",
    "WAFERS": "WAFER",
    "STRIPS": "STRIP",
    "SWABS": "SWAB",
    "CLOTHS": "CLOTH",
    "SPONGES": "SPONGE",
    "DRESSINGS": "DRESSING",
    "BOTTLES": "BOTTLE",
    "BOT": "BOTTLE",
    "BOTT": "BOTTLE",
    "BAGS": "BAG",
    "KITS": "KIT",
    # WHO specific
    "INHAL.AEROSOL": "AEROSOL, METERED",
    "INHAL.POWDER": "POWDER, METERED",
    "INHAL.SOLUTION": "NEBULE",
    "ORAL AEROSOL": "AEROSOL, METERED",
    "INSTILL.SOLUTION": "DROPS",
    "LAMELLA": "FILM",
    "S.C. IMPLANT": "IMPLANT",
    "CHEWING GUM": "GUM",
    "GUM, CHEWING": "GUM",
}


# ============================================================================
# FORM TO ROUTE MAPPING
# Most common route for each form (for inference when route not specified)
# Derived from DrugBank products dataset frequency analysis
# ============================================================================

FORM_TO_ROUTE: Dict[str, str] = {
    # Oral forms (high confidence from DrugBank)
    "TABLET": "ORAL",
    "TABLET, EXTENDED RELEASE": "ORAL",
    "TABLET, DELAYED RELEASE": "ORAL",
    "CAPSULE": "ORAL",
    "CAPSULE, EXTENDED RELEASE": "ORAL",
    "CAPSULE, DELAYED RELEASE": "ORAL",
    "SYRUP": "ORAL",
    "ELIXIR": "ORAL",
    "SOLUTION": "ORAL",  # Most common, but can be other routes
    "SUSPENSION": "ORAL",
    "SUSPENSION, EXTENDED RELEASE": "ORAL",
    "GRANULE": "ORAL",
    "GRANULE, DELAYED RELEASE": "ORAL",
    "POWDER": "ORAL",  # When not metered
    "LOZENGE": "ORAL",
    "GUM": "ORAL",
    "WAFER": "ORAL",
    "STRIP": "ORAL",
    "SACHET": "ORAL",
    # Topical forms (high confidence from DrugBank)
    "CREAM": "TOPICAL",
    "LOTION": "TOPICAL",
    "GEL": "TOPICAL",
    "OINTMENT": "TOPICAL",
    "PASTE": "TOPICAL",
    "SPRAY": "TOPICAL",
    "AEROSOL": "TOPICAL",
    "SHAMPOO": "TOPICAL",
    "SOAP": "TOPICAL",
    "LIQUID": "TOPICAL",
    "EMULSION": "TOPICAL",
    "FOAM": "TOPICAL",
    "SWAB": "TOPICAL",
    "CLOTH": "TOPICAL",
    "STICK": "TOPICAL",
    "SPONGE": "TOPICAL",
    "DRESSING": "TOPICAL",
    # Transdermal
    "PATCH": "TRANSDERMAL",
    # Injectable forms (from DrugBank)
    "INJECTION": "INTRAVENOUS",  # Most common
    "INJECTION, EXTENDED RELEASE": "INTRAMUSCULAR",
    "AMPULE": "PARENTERAL",
    "VIAL": "PARENTERAL",
    "IMPLANT": "SUBCUTANEOUS",
    # Ophthalmic forms
    "DROPS": "OPHTHALMIC",  # Most common for drops
    # Inhalation forms (from DrugBank/WHO)
    "INHALER": "INHALATION",
    "AEROSOL, METERED": "INHALATION",
    "POWDER, METERED": "INHALATION",
    "NEBULE": "INHALATION",
    "GAS": "INHALATION",
    # Nasal forms
    "SPRAY, METERED": "NASAL",
    # Rectal forms
    "SUPPOSITORY": "RECTAL",
    "ENEMA": "RECTAL",
    # Vaginal forms
    "INSERT": "VAGINAL",
    "RING": "VAGINAL",
    "PESSARY": "VAGINAL",
    # Buccal/Sublingual
    "FILM": "BUCCAL",
    # IV fluids
    "BOTTLE": "INTRAVENOUS",
    "BAG": "INTRAVENOUS",
}


# ============================================================================
# ROUTE ALIASES (normalize route names)
# ============================================================================

ROUTE_ALIASES: Dict[str, str] = {
    # Oral
    "PO": "ORAL",
    "OR": "ORAL",
    "O": "ORAL",
    "BY MOUTH": "ORAL",
    "PER OREM": "ORAL",
    "PER OS": "ORAL",
    # Intravenous
    "IV": "INTRAVENOUS",
    # Intramuscular
    "IM": "INTRAMUSCULAR",
    # Subcutaneous
    "SC": "SUBCUTANEOUS",
    "SQ": "SUBCUTANEOUS",
    "SUBCUT": "SUBCUTANEOUS",
    "SUBDERMAL": "SUBCUTANEOUS",
    # Parenteral (generic injectable)
    "P": "PARENTERAL",
    "INJ": "PARENTERAL",
    "INJECTION": "PARENTERAL",
    # Sublingual
    "SL": "SUBLINGUAL",
    # Transdermal
    "TD": "TRANSDERMAL",
    "DERMAL": "TRANSDERMAL",
    # Rectal
    "R": "RECTAL",
    "PR": "RECTAL",
    "PER RECTUM": "RECTAL",
    # Vaginal
    "V": "VAGINAL",
    "PV": "VAGINAL",
    "PER VAGINAM": "VAGINAL",
    # Nasal
    "N": "NASAL",
    "INTRANASAL": "NASAL",
    "PER NASAL": "NASAL",
    # Inhalation
    "INH": "INHALATION",
    "INHAL": "INHALATION",
    "RESPIRATORY (INHALATION)": "INHALATION",
    # Ophthalmic
    "OPH": "OPHTHALMIC",
    "EYE": "OPHTHALMIC",
    # Otic
    "EAR": "OTIC",
    # Topical
    "CUTANEOUS": "TOPICAL",
    # Buccal
    "BUCC": "BUCCAL",
    # Intradermal
    "ID": "INTRADERMAL",
}


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
