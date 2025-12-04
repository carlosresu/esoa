"""
Tokenization and normalization functions for drug descriptions.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from .unified_constants import (
    CATEGORY_DOSE, CATEGORY_FORM, CATEGORY_GENERIC, CATEGORY_OTHER,
    CATEGORY_ROUTE, CATEGORY_SALT, ELEMENT_DRUGS, FORM_CANON,
    FORM_MODIFIER_IGNORE, PURE_SALT_COMPOUNDS, ROUTE_CANON, SALT_TOKENS,
    STOPWORDS, UNIT_TOKENS,
)

# Legacy aliases for backward compatibility
NATURAL_STOPWORDS = STOPWORDS
GENERIC_JUNK_TOKENS = STOPWORDS

# Pre-sorted salt tokens for faster lookup (sorted by length descending)
_SALT_TOKENS_SORTED: List[str] = sorted(SALT_TOKENS, key=len, reverse=True)


# Regex patterns
_DOSE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|ug|ml|l|iu|unit|units|pct|%|mg/ml|mcg/ml|iu/ml|mg/5ml)",
    re.IGNORECASE,
)
_PARENTHESES_PATTERN = re.compile(r"\([^)]*\)")

# Patterns for extracting details from drug names
_SALT_PARENTHETICAL = re.compile(r"\(\s*as\s+([^)]+)\)", re.IGNORECASE)
_BRAND_PARENTHETICAL = re.compile(r"\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\)")  # Title case = brand
_QUALIFIER_PATTERN = re.compile(r"\b(for|in)\s+(hepatic|renal|infants?|pediatric|adults?|immunonutrition|immunoenhancement)\b", re.IGNORECASE)
_INDICATION_PATTERN = re.compile(r"\bfor\s+(\w+(?:\s+\w+){0,3}?)(?:\s+(?:failure|conditions?|patients?))?", re.IGNORECASE)

# Release detail keywords (defined early for use in extract_drug_details)
_RELEASE_KEYWORDS = {
    "EXTENDED RELEASE", "EXTENDED-RELEASE",
    "SUSTAINED RELEASE", "SUSTAINED-RELEASE",
    "MODIFIED RELEASE", "MODIFIED-RELEASE",
    "CONTROLLED RELEASE", "CONTROLLED-RELEASE",
    "DELAYED RELEASE", "DELAYED-RELEASE",
    "IMMEDIATE RELEASE", "IMMEDIATE-RELEASE",
    "LONG ACTING", "LONG-ACTING",
    "RETARD", "SLOW RELEASE",
}
_RELEASE_ABBREVS = {"ER", "XR", "XL", "SR", "CR", "DR", "IR", "MR", "LA"}

# Form detail keywords (non-release modifiers)
_FORM_DETAIL_KEYWORDS = {
    "FILM COATED", "FILM-COATED",
    "ENTERIC COATED", "ENTERIC-COATED",
    "SUGAR COATED", "SUGAR-COATED",
    "CHEWABLE", "DISPERSIBLE", "EFFERVESCENT",
    "SUBLINGUAL", "BUCCAL", "ORALLY DISINTEGRATING",
    "RECTAL", "VAGINAL",
}
_FORM_DETAIL_ABBREVS = {"FC", "EC", "ODT"}


def _extract_type_detail_impl(text: str) -> Tuple[str, Optional[str]]:
    """Internal: Extract type detail from comma-separated text."""
    if "," not in text:
        return text, None
    if " + " in text.upper() or " AND " in text.upper():
        return text, None
    parts = text.split(",", 1)
    base = parts[0].strip()
    after_comma = parts[1].strip() if len(parts) > 1 else ""
    if not after_comma:
        return base, None
    after_upper = after_comma.upper()
    after_words = set(after_upper.split())
    for kw in _RELEASE_KEYWORDS:
        if kw in after_upper:
            return text, None
    for kw in _FORM_DETAIL_KEYWORDS:
        if kw in after_upper:
            return text, None
    if after_words & (_FORM_DETAIL_ABBREVS | _RELEASE_ABBREVS):
        return text, None
    form_words = {"TABLET", "CAPSULE", "SOLUTION", "SUSPENSION", "INJECTION", "CREAM", "OINTMENT"}
    if any(fw in after_upper for fw in form_words):
        return text, None
    return base, after_comma


def _extract_release_detail_impl(form_text: str) -> Tuple[str, Optional[str]]:
    """Internal: Extract release modifier from form text."""
    form_upper = form_text.upper()
    form_words = form_upper.split()
    if "," in form_text:
        parts = form_text.split(",", 1)
        base = parts[0].strip()
        after_comma = parts[1].strip() if len(parts) > 1 else ""
        after_upper = after_comma.upper()
        after_words = set(after_upper.split())
        for kw in _RELEASE_KEYWORDS:
            if kw in after_upper:
                return base, after_comma
        if after_words & _RELEASE_ABBREVS:
            return base, after_comma
    for kw in _RELEASE_KEYWORDS:
        if f" {kw}" in form_upper or form_upper.endswith(f" {kw}"):
            idx = form_upper.find(kw)
            base = form_text[:idx].strip()
            release = form_text[idx:].strip()
            if base:
                return base, release
    if len(form_words) >= 2 and form_words[-1] in _RELEASE_ABBREVS:
        base = " ".join(form_text.split()[:-1])
        return base, form_words[-1]
    for word in form_words:
        if word in _RELEASE_ABBREVS:
            return form_text, word
    return form_text, None


def _extract_form_detail_impl(form_text: str) -> Tuple[str, Optional[str]]:
    """Internal: Extract form modifier (non-release) from form text."""
    form_upper = form_text.upper()
    form_words = form_upper.split()
    if "," in form_text:
        parts = form_text.split(",", 1)
        base = parts[0].strip()
        after_comma = parts[1].strip() if len(parts) > 1 else ""
        after_upper = after_comma.upper()
        after_words = set(after_upper.split())
        for kw in _FORM_DETAIL_KEYWORDS:
            if kw in after_upper:
                return base, after_comma
        if after_words & _FORM_DETAIL_ABBREVS:
            return base, after_comma
    for kw in _FORM_DETAIL_KEYWORDS:
        if f" {kw}" in form_upper or form_upper.endswith(f" {kw}"):
            idx = form_upper.find(kw)
            base = form_text[:idx].strip()
            detail = form_text[idx:].strip()
            if base:
                return base, detail
    if len(form_words) >= 2 and form_words[-1] in _FORM_DETAIL_ABBREVS:
        base = " ".join(form_text.split()[:-1])
        return base, form_words[-1]
    for word in form_words:
        if word in _FORM_DETAIL_ABBREVS:
            return form_text, word
    return form_text, None


# ============================================================================
# DOSE PARSING - Structured extraction of dose values and units
# ============================================================================

# Unit conversion factors to milligrams (for mass units)
_MASS_TO_MG = {
    "MG": 1.0,
    "G": 1000.0,
    "GM": 1000.0,
    "GR": 1000.0,  # gram
    "MCG": 0.001,
    "UG": 0.001,
    "ΜG": 0.001,  # micro symbol
    "KG": 1_000_000.0,
}

# Volume conversion factors to milliliters
_VOLUME_TO_ML = {
    "ML": 1.0,
    "L": 1000.0,
    "CC": 1.0,  # cubic centimeter = mL
    "DL": 100.0,  # deciliter
}

# Dose pattern with named groups for structured extraction
_STRUCTURED_DOSE_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>mg|g|gm|gr|mcg|ug|μg|kg|ml|l|cc|dl|iu|unit|units|%|pct)"
    r"(?:\s*/\s*(?P<per_value>\d+(?:[.,]\d+)?)\s*(?P<per_unit>ml|l|cc|dl|tab|tablet|cap|capsule|dose|unit|5ml))?",
    re.IGNORECASE
)


def parse_dose_components(text: str) -> Dict[str, any]:
    """
    Parse dose components from text into structured format.
    
    Returns dict with:
        - doses: List of parsed dose dicts, each with:
            - value: Numeric value
            - unit: Original unit (uppercase)
            - unit_type: 'mass', 'volume', 'percentage', 'iu', 'concentration'
            - value_mg: Value converted to mg (for mass units)
            - value_ml: Value converted to mL (for volume units)
            - concentration_mg_per_ml: For concentration units (mg/mL)
        - total_volume_ml: Total solution volume in mL (if found)
        - percentages: List of percentage values found
    
    Examples:
        "500 mg TABLET" -> doses: [{value: 500, unit: 'MG', unit_type: 'mass', value_mg: 500}]
        "5% DEXTROSE 250 mL" -> doses: [...], total_volume_ml: 250, percentages: [5.0]
        "10 mg/5 mL SYRUP" -> doses: [{..., concentration_mg_per_ml: 2.0}]
    """
    result = {
        "doses": [],
        "total_volume_ml": None,
        "percentages": [],
    }
    
    text_upper = text.upper()
    
    # Find all dose patterns
    for match in _STRUCTURED_DOSE_PATTERN.finditer(text_upper):
        value_str = match.group("value").replace(",", ".")
        value = float(value_str)
        unit = match.group("unit").upper()
        per_value_str = match.group("per_value")
        per_unit = match.group("per_unit").upper() if match.group("per_unit") else None
        
        dose = {
            "value": value,
            "unit": unit,
            "unit_type": None,
            "value_mg": None,
            "value_ml": None,
            "concentration_mg_per_ml": None,
        }
        
        # Classify unit type
        if unit in ("%", "PCT"):
            dose["unit_type"] = "percentage"
            dose["unit"] = "%"
            result["percentages"].append(value)
        elif unit in _MASS_TO_MG:
            dose["unit_type"] = "mass"
            dose["value_mg"] = value * _MASS_TO_MG[unit]
        elif unit in _VOLUME_TO_ML:
            dose["unit_type"] = "volume"
            dose["value_ml"] = value * _VOLUME_TO_ML[unit]
            # Track as potential total volume
            if result["total_volume_ml"] is None or dose["value_ml"] > result["total_volume_ml"]:
                result["total_volume_ml"] = dose["value_ml"]
        elif unit in ("IU", "UNIT", "UNITS"):
            dose["unit_type"] = "iu"
        
        # Handle concentration (X per Y)
        if per_value_str and per_unit:
            per_value = float(per_value_str.replace(",", "."))
            
            # Special case: mg/5mL (common pediatric dosing)
            if per_unit == "5ML":
                per_value = 5.0
                per_unit = "ML"
            
            if per_unit in _VOLUME_TO_ML and dose["value_mg"] is not None:
                per_ml = per_value * _VOLUME_TO_ML.get(per_unit, 1.0)
                if per_ml > 0:
                    dose["concentration_mg_per_ml"] = dose["value_mg"] / per_ml
                    dose["unit_type"] = "concentration"
        
        result["doses"].append(dose)
    
    return result


def calculate_iv_amounts(
    text: str,
    drug_percentages: List[float],
    diluent_type: Optional[str],
    diluent_percentage: Optional[float],
    total_volume_ml: Optional[float],
) -> Dict[str, any]:
    """
    Calculate actual amounts for IV solution components.
    
    For IV solutions, percentage = w/v (weight/volume) = grams per 100mL.
    
    Args:
        text: Original drug description
        drug_percentages: List of drug concentration percentages (e.g., [5.0] for 5% dextrose)
        diluent_type: Type of diluent (WATER, SODIUM CHLORIDE, LACTATED RINGER'S, etc.)
        diluent_percentage: Concentration of diluent if applicable (e.g., 0.9 for 0.9% NaCl)
        total_volume_ml: Total solution volume in mL
    
    Returns dict with:
        - drug_amount_mg: Calculated drug amount in milligrams
        - drug_amount_g: Calculated drug amount in grams
        - diluent_amount_mg: Calculated diluent amount in mg (for saline)
        - diluent_amount_g: Calculated diluent amount in g
        - diluent_volume_ml: Diluent volume (≈ total volume for dissolved solids)
        - concentration_mg_per_ml: Drug concentration in mg/mL
    """
    result = {
        "drug_amount_mg": None,
        "drug_amount_g": None,
        "diluent_amount_mg": None,
        "diluent_amount_g": None,
        "diluent_volume_ml": None,
        "concentration_mg_per_ml": None,
    }
    
    if total_volume_ml is None or not drug_percentages:
        return result
    
    # Calculate drug amount from percentage (w/v: grams per 100mL)
    # 5% = 5g/100mL = 50mg/mL
    primary_pct = drug_percentages[0]
    drug_g = (primary_pct / 100.0) * total_volume_ml  # grams
    drug_mg = drug_g * 1000  # milligrams
    
    result["drug_amount_g"] = round(drug_g, 3)
    result["drug_amount_mg"] = round(drug_mg, 3)
    result["concentration_mg_per_ml"] = round((primary_pct / 100.0) * 1000, 3)  # mg/mL
    
    # For dissolved solids, the volume they occupy is negligible
    # So diluent volume ≈ total volume
    result["diluent_volume_ml"] = total_volume_ml
    
    # Calculate diluent amount if it has a concentration (e.g., 0.9% NaCl)
    if diluent_percentage is not None:
        diluent_g = (diluent_percentage / 100.0) * total_volume_ml
        diluent_mg = diluent_g * 1000
        result["diluent_amount_g"] = round(diluent_g, 3)
        result["diluent_amount_mg"] = round(diluent_mg, 3)
    
    return result


def extract_drug_details(drug_name: str) -> Dict[str, Optional[str]]:
    """
    Extract parentheticals and qualifiers from a drug name into separate fields.
    
    Returns dict with:
        - generic_name: Base drug name without qualifiers
        - salt_details: Salt form (e.g., "SODIUM SALT", "SULFATE")
        - brand_details: Brand names in parentheses
        - indication_details: Indication qualifiers (e.g., "FOR HEPATIC FAILURE")
        - alias_details: Other aliases (e.g., "VIT. D3")
        - type_details: Type qualifier (e.g., "DRY POWDER", "HUMAN")
        - release_details: Release modifier (e.g., "SR", "MR", "EXTENDED RELEASE")
        - form_details: Form qualifier (e.g., "FILM COATED", "CHEWABLE")
    
    Examples:
        "AMINO ACID SOLUTIONS FOR HEPATIC FAILURE" 
            -> generic: "AMINO ACIDS", indication: "FOR HEPATIC FAILURE"
        "ALENDRONATE + CHOLECALCIFEROL (VIT. D3) ( as SODIUM SALT)"
            -> generic: "ALENDRONATE + CHOLECALCIFEROL", salt: "SODIUM SALT", alias: "VIT. D3"
        "NIFEDIPINE 30 mg MR TABLET"
            -> generic: "NIFEDIPINE", release: "MR"
    """
    result = {
        "generic_name": drug_name.strip().upper(),
        "salt_details": None,
        "brand_details": None,
        "indication_details": None,
        "alias_details": None,
        "type_details": None,
        "release_details": None,
        "form_details": None,
        "diluent_details": None,  # Volume of diluent/solvent (base solution volume)
        "iv_diluent_type": None,  # IV solution base: WATER, SODIUM CHLORIDE, LACTATED RINGER'S, etc.
        "iv_diluent_amount": None,  # IV diluent concentration: 0.9%, 0.3%, etc.
        # Structured dose information
        "dose_values": None,  # List of dose values (e.g., [5.0, 0.9])
        "dose_units": None,  # List of units (e.g., ["%", "%"])
        "dose_types": None,  # List of types (e.g., ["percentage", "percentage"])
        "total_volume_ml": None,  # Total solution volume in mL
        # Computed amounts for IV solutions (w/v calculation)
        "drug_amount_mg": None,  # Computed drug amount in mg
        "diluent_amount_mg": None,  # Computed diluent amount in mg (for saline)
        "concentration_mg_per_ml": None,  # Drug concentration in mg/mL
    }
    
    working = drug_name.strip()
    
    # Extract IV solution diluent from "X% DRUG IN Y% DILUENT" patterns
    # Common diluents: WATER, SODIUM CHLORIDE (saline), LACTATED/ACETATED RINGER'S
    iv_diluent_pattern = re.compile(
        r'\bIN\s+'
        r'(?:(\d+(?:\.\d+)?\s*%)\s+)?'  # Optional concentration (e.g., "0.9%")
        r'(WATER|SODIUM\s+CHLORIDE|LACTATED\s+RINGER[\'\'`]?S?(?:\s+SOLUTION)?|'
        r'ACETATED\s+RINGER[\'\'`]?S?(?:\s+SOLUTION)?|RINGER[\'\'`]?S?\s+(?:SOLUTION|LACTATE))'
        r'(?:\s+SOLUTION)?',
        re.IGNORECASE
    )
    iv_match = iv_diluent_pattern.search(working)
    if iv_match:
        diluent_amount = iv_match.group(1)  # e.g., "0.9%" or None
        diluent_type = iv_match.group(2).upper()  # e.g., "SODIUM CHLORIDE"
        
        # Normalize apostrophe variants in RINGER'S
        diluent_type = re.sub(r"RINGER[\'\'`]?S?", "RINGER'S", diluent_type)
        # Ensure SOLUTION suffix is included if present
        if 'SOLUTION' not in diluent_type and ('RINGER' in diluent_type or iv_match.group(0).upper().endswith('SOLUTION')):
            if 'LACTATED' in diluent_type or 'ACETATED' in diluent_type:
                if not diluent_type.endswith('SOLUTION'):
                    diluent_type = diluent_type.rstrip() + ' SOLUTION'
        
        result["iv_diluent_type"] = diluent_type.strip()
        result["iv_diluent_amount"] = diluent_amount.strip() if diluent_amount else None
    
    # Handle drug names starting with percentage (e.g., "0.9% SODIUM CHLORIDE")
    # Move the percentage to dose position and keep the drug name
    pct_start_match = re.match(r'^(\d+(?:\.\d+)?)\s*%\s+(.+)$', working)
    if pct_start_match:
        pct_value = pct_start_match.group(1)
        rest = pct_start_match.group(2)
        # Move percentage to end as dose info (will be extracted later)
        working = f"{rest} {pct_value}%"
    
    # Normalize: remove whitespace after opening parenthesis and before closing
    working = re.sub(r"\(\s+", "(", working)
    working = re.sub(r"\s+\)", ")", working)
    
    # Extract diluent/solvent volume BEFORE stripping
    # Diluent volume = base solution volume for concentration calculation
    # e.g., "250 mg + 5 mL diluent" → dose=250mg, diluent=5mL → concentration=50mg/mL
    diluent_keywords = (
        r"diluent|solvent|reconstitution\s+fluid|sterile\s+water|"
        r"water\s+for\s+injection|w\.?f\.?i\.?"
    )
    
    # Extract diluent volume patterns
    diluent_volumes = []
    
    # Pattern 1: "+ X mL diluent" (explicit diluent after +)
    diluent_vol_pattern1 = re.compile(
        r"\+\s*(\d+(?:[.,]\d+)?)\s*(m?L)\s*(?:" + diluent_keywords + r")",
        re.IGNORECASE
    )
    for match in diluent_vol_pattern1.finditer(working):
        vol = match.group(1).replace(",", ".")
        unit = match.group(2).upper()
        if unit == "L":
            diluent_volumes.append(f"{vol} L")
        else:
            diluent_volumes.append(f"{vol} mL")
    
    # Pattern 2: "+ X mL LYOPHILIZED/FREEZE-DRIED POWDER + DILUENT" - volume before form word
    diluent_vol_pattern2 = re.compile(
        r"\+\s*(\d+(?:[.,]\d+)?)\s*(m?L)\s+(?:LYOPHILIZED|FREEZE-?DRIED)\s+POWDER\s*\+\s*(?:" + diluent_keywords + r")",
        re.IGNORECASE
    )
    for match in diluent_vol_pattern2.finditer(working):
        vol = match.group(1).replace(",", ".")
        unit = match.group(2).upper()
        if unit == "L":
            diluent_volumes.append(f"{vol} L")
        else:
            diluent_volumes.append(f"{vol} mL")
    
    # Pattern 3: "X mg/Y mL + Diluent" - dose/volume ratio before diluent
    # e.g., "METHYLPREDNISOLONE 1 g/16 mL + Diluent" → diluent is 16 mL
    diluent_vol_pattern3 = re.compile(
        r"(\d+(?:[.,]\d+)?)\s*(?:mg|g|mcg|iu)\s*/\s*(\d+(?:[.,]\d+)?)\s*(m?L)\s*\+\s*(?:" + diluent_keywords + r")",
        re.IGNORECASE
    )
    for match in diluent_vol_pattern3.finditer(working):
        vol = match.group(2).replace(",", ".")
        unit = match.group(3).upper()
        if unit == "L":
            diluent_volumes.append(f"{vol} L")
        else:
            diluent_volumes.append(f"{vol} mL")
    
    # Pattern 4: "+ Diluent" without volume (just note presence)
    if re.search(r"\+\s*(?:" + diluent_keywords + r")", working, re.IGNORECASE):
        if not diluent_volumes:
            diluent_volumes.append("with diluent")
    
    # Pattern 5: "LYOPHILIZED POWDER + DILUENT/SOLVENT" without volume
    if re.search(r"(?:LYOPHILIZED|FREEZE-?DRIED)\s+POWDER\s*\+\s*(?:" + diluent_keywords + r")", working, re.IGNORECASE):
        if not diluent_volumes:
            diluent_volumes.append("with diluent")
    
    if diluent_volumes:
        result["diluent_details"] = "|".join(diluent_volumes)
    
    # Now strip diluent patterns from working string
    # Pattern 0: Strip "monodose vial + X mL diluent" and "multidose vial + X mL diluent" entirely
    monodose_diluent = re.compile(
        r"\s+(?:mono|multi)?dose\s+vial\s*\+\s*\d+(?:[.,]\d+)?\s*m?L?\s*" + diluent_keywords + r".*$",
        re.IGNORECASE
    )
    working = monodose_diluent.sub("", working)
    
    # Pattern 0b: Strip trailing "LYOPHILIZED POWDER + DILUENT VIAL" etc.
    lyoph_diluent = re.compile(
        r"\s+(?:LYOPHILIZED|FREEZE-?DRIED)\s+POWDER\s*\+\s*(?:" + diluent_keywords + r").*$",
        re.IGNORECASE
    )
    working = lyoph_diluent.sub("", working)
    
    # Pattern 0c: Strip "+ X mL LYOPHILIZED POWDER + DILUENT" patterns
    ml_lyoph_pattern = re.compile(
        r"\s*\+\s*\d+(?:[.,]\d+)?\s*m?L?\s+(?:LYOPHILIZED|FREEZE-?DRIED)\s+POWDER\s*\+\s*(?:" + diluent_keywords + r").*$",
        re.IGNORECASE
    )
    working = ml_lyoph_pattern.sub("", working)
    
    # Pattern 1a: "+ X mL diluent" - explicit volume before keyword  
    diluent_pattern1a = re.compile(
        r"\s*\+\s*\d+(?:[.,]\d+)?\s*m?L?\s+" + diluent_keywords,
        re.IGNORECASE
    )
    working = diluent_pattern1a.sub("", working)
    
    # Pattern 1a2: "X mg + Y mL diluent" - dose before +, volume after
    # This catches "250 mg + 5 mL diluent SOLUTION VIAL"
    diluent_pattern1a2 = re.compile(
        r"(\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|iu|units?))\s*\+\s*\d+(?:[.,]\d+)?\s*m?L?\s*" + diluent_keywords + r".*$",
        re.IGNORECASE
    )
    working = diluent_pattern1a2.sub(r"\1", working)
    
    # Pattern 1a3: "X IU + Y mL + Z mL" multi-part patterns with trailing packaging
    # This catches "100 IU/g + diluent SOLUTION VIAL"
    diluent_pattern1a3 = re.compile(
        r"\s*\+\s*" + diluent_keywords + r"\s+(?:SOLUTION|SUSPENSION|POWDER)?\s*(?:VIAL|AMPULE?|BOTTLE)?.*$",
        re.IGNORECASE
    )
    working = diluent_pattern1a3.sub("", working)
    
    # Pattern 1b: "+ diluent" without volume (also handle "+ diluent vial")
    diluent_pattern1b = re.compile(
        r"\s*\+\s*" + diluent_keywords + r"(?:\s+(?:VIAL|AMPULE?|BOTTLE))?\s*",
        re.IGNORECASE
    )
    working = diluent_pattern1b.sub("", working)
    
    # Also strip leftover "+ X mL" patterns (orphaned after diluent stripped)
    leftover_ml_pattern = re.compile(r"\s*\+\s*\d+(?:[.,]\d+)?\s*m?L?\s*(?=\s|$)", re.IGNORECASE)
    working = leftover_ml_pattern.sub("", working)
    
    # Strip vaccine-specific potency info: "1000 DL 50 mouse min", "X PFU" (not regular mg/mcg doses)
    # Only strip DL/LD (lethal dose) and PFU (plaque-forming units) patterns
    vaccine_potency_pattern = re.compile(
        r"\s+\d+(?:[.,]\d+)?\s*(?:DL|LD)(?:\s+\d+)?(?:\s+(?:mouse|mice))?\s*(?:min|minimum)?\s*",
        re.IGNORECASE
    )
    working = vaccine_potency_pattern.sub(" ", working)
    
    # Strip "not less than X PFU" patterns from vaccines
    potency_qualifier_pattern = re.compile(
        r"\s+not\s+less\s+than(?:\s+\d+(?:[.,]\d+)?\s*(?:PFU)?)?\s*",
        re.IGNORECASE
    )
    working = potency_qualifier_pattern.sub(" ", working)
    
    # Strip "freeze-dried powder monodose vial" patterns
    freeze_dried_pattern = re.compile(
        r"\s+freeze-?dried\s+powder\s+(?:mono|multi)?dose\s+vial.*$",
        re.IGNORECASE
    )
    working = freeze_dried_pattern.sub("", working)
    
    # Pattern 2: "POWDER + DILUENT", "SOLUTION + DILUENT" 
    diluent_pattern2 = re.compile(
        r"\s*\+\s*(?:\d+(?:[.,]\d+)?\s*(?:mL|g)\s+)?" + diluent_keywords,
        re.IGNORECASE
    )
    working = diluent_pattern2.sub("", working)
    
    # Pattern 3: "dose + X mL diluent"
    diluent_pattern3 = re.compile(
        r"\b(?:\d+\s+)?dose\s*\+\s*(?:\d+(?:[.,]\d+)?\s*m?L?\s+)?" + diluent_keywords,
        re.IGNORECASE
    )
    working = diluent_pattern3.sub("", working)
    
    # Pattern 4: Standalone diluent references like "VIAL + PRE-FILLED SYRINGE DILUENT"
    diluent_pattern4 = re.compile(
        r"\s+(?:PRE-?FILLED\s+)?(?:SYRINGE\s+)?DILUENT\b",
        re.IGNORECASE
    )
    working = diluent_pattern4.sub("", working)
    
    # Pattern 5: Strip trailing packaging/form words like "monodose vial", "multidose vial", "SOLUTION VIAL"
    packaging_pattern = re.compile(
        r"\s+(?:mono|multi)?dose\s+(?:vial|ampoule?|syringe)(?:\s+SOLUTION\s+(?:VIAL|AMPOULE?|BOTTLE))?\s*$",
        re.IGNORECASE
    )
    working = packaging_pattern.sub("", working)
    # Also strip trailing form words: "SOLUTION VIAL", "SOLUTION BOTTLE", etc.
    trailing_form_pattern = re.compile(
        r"\s+(?:SOLUTION|SUSPENSION|POWDER|FREEZE-?DRIED(?:\s+POWDER)?|LYOPHILIZED(?:\s+POWDER)?)"
        r"(?:\s+(?:VIAL|AMPOULE?|BOTTLE|DRUM|BAG))?\s*$",
        re.IGNORECASE
    )
    working = trailing_form_pattern.sub("", working)
    
    # Extract salt forms: ( as SODIUM SALT), ( as SULFATE), etc.
    salt_matches = _SALT_PARENTHETICAL.findall(working)
    if salt_matches:
        result["salt_details"] = "|".join(s.strip().upper() for s in salt_matches)
        working = _SALT_PARENTHETICAL.sub("", working)
    
    # Extract indication qualifiers: FOR HEPATIC FAILURE, FOR INFANTS, etc.
    indication_match = _INDICATION_PATTERN.search(working)
    if indication_match:
        indication = indication_match.group(0).strip().upper()
        # Common indication patterns
        if any(x in indication.upper() for x in ["HEPATIC", "RENAL", "INFANT", "PEDIATRIC", "IMMUNONUTRITION", "IMMUNOENHANCEMENT"]):
            result["indication_details"] = indication
            # Remove from generic name
            working = working[:indication_match.start()] + working[indication_match.end():]
    
    # Also check for "SOLUTIONS FOR X" pattern
    solutions_match = re.search(r"\bSOLUTIONS?\s+FOR\s+(\w+(?:\s+\w+){0,3})", working, re.IGNORECASE)
    if solutions_match and not result["indication_details"]:
        result["indication_details"] = solutions_match.group(0).strip().upper()
        working = working[:solutions_match.start()] + "SOLUTIONS" + working[solutions_match.end():]
    
    # Extract remaining parentheticals as aliases (but not doses)
    remaining_parens = re.findall(r"\(([^)]+)\)", working)
    aliases = []
    for paren in remaining_parens:
        paren_upper = paren.strip().upper()
        # Skip if it looks like a dose
        if re.match(r"^\d+", paren_upper) or any(u in paren_upper for u in ["MG", "ML", "MCG", "IU", "%"]):
            continue
        # Skip if it's a salt we already captured
        if paren_upper.startswith("AS "):
            continue
        aliases.append(paren_upper)
    
    if aliases:
        result["alias_details"] = "|".join(aliases)
        # Remove alias parentheticals from working string
        for alias in aliases:
            working = re.sub(r"\(\s*" + re.escape(alias) + r"\s*\)", "", working, flags=re.IGNORECASE)
    
    # Handle comma-separated details
    # e.g., "VITAMIN A, RETINOL" → generic: "VITAMIN A", alias: "RETINOL"
    # But NOT "A, B AND C" patterns (multi-ingredient) or if + appears after comma
    if "," in working and " + " not in working:
        parts = working.split(",")
        first_part = parts[0].strip()
        
        # Check if this looks like a multi-ingredient comma list (A, B AND C)
        remaining = ",".join(parts[1:]).strip()
        is_multi_ingredient = bool(re.search(r"\bAND\b", remaining, re.IGNORECASE)) or "+" in remaining
        
        if not is_multi_ingredient and len(parts) > 1:
            # Everything after first comma is detail
            comma_details = [p.strip().upper() for p in parts[1:] if p.strip()]
            # Filter out dose-like details
            comma_details = [d for d in comma_details if not re.match(r"^\d+", d)]
            
            if comma_details:
                if result["alias_details"]:
                    result["alias_details"] += "|" + "|".join(comma_details)
                else:
                    result["alias_details"] = "|".join(comma_details)
                working = first_part
    
    # Clean up generic name
    working = re.sub(r"\s+", " ", working).strip().upper()
    
    # Strip dose/form info from the end
    # Match patterns like "70 MG + 2800 IU TABLET" or "400 UNITS + 5 MG + 5000 UNITS OINTMENT"
    # Find where numeric dose info starts (number followed by unit)
    dose_start = re.search(r"\s+\d+(?:\.\d+)?\s*(?:MG|G|MCG|UG|IU|ML|L|UNITS?|%)", working, re.IGNORECASE)
    if dose_start:
        working = working[:dose_start.start()].strip()
    
    # Remove trailing "SOLUTIONS" if we extracted indication
    if result["indication_details"] and working.endswith(" SOLUTIONS"):
        working = working[:-10].strip()
    elif result["indication_details"] and working.endswith(" SOLUTION"):
        working = working[:-9].strip()
    
    # Strip trailing salt suffixes (SODIUM PHOSPHATE, SODIUM SUCCINATE, etc.)
    # This handles ESOA-style "DEXAMETHASONE SODIUM PHOSPHATE" pattern
    trailing_salt_suffixes = [
        "SODIUM PHOSPHATE", "DISODIUM PHOSPHATE", "SODIUM SUCCINATE",
        "SODIUM SULFATE", "SODIUM CHLORIDE", "POTASSIUM PHOSPHATE",
        "CALCIUM PHOSPHATE", "MAGNESIUM SULFATE",
    ]
    for suffix in trailing_salt_suffixes:
        if working.endswith(" " + suffix):
            base = working[:-len(suffix)-1].strip()
            # Only strip if there's still a meaningful base name
            if base and len(base) > 2:
                if result["salt_details"]:
                    result["salt_details"] += "|" + suffix
                else:
                    result["salt_details"] = suffix
                working = base
                break
    
    # Normalize "DRUG+DRUG" to "DRUG + DRUG" (add spaces around +)
    if "+" in working and " + " not in working:
        working = re.sub(r"\+", " + ", working)
        working = re.sub(r"\s+", " ", working).strip()
    
    result["generic_name"] = working if working else drug_name.strip().upper()
    
    # Extract type/release/form details from original text
    # These functions are defined later in this module
    _, type_det = _extract_type_detail_impl(drug_name)
    _, release_det = _extract_release_detail_impl(drug_name)
    _, form_det = _extract_form_detail_impl(drug_name) if not release_det else (None, None)
    
    result["type_details"] = type_det
    result["release_details"] = release_det
    result["form_details"] = form_det
    
    # Parse structured dose information from original text
    dose_info = parse_dose_components(drug_name)
    
    if dose_info["doses"]:
        result["dose_values"] = [d["value"] for d in dose_info["doses"]]
        result["dose_units"] = [d["unit"] for d in dose_info["doses"]]
        result["dose_types"] = [d["unit_type"] for d in dose_info["doses"]]
    
    if dose_info["total_volume_ml"]:
        result["total_volume_ml"] = dose_info["total_volume_ml"]
    
    # Calculate IV solution amounts if we have percentage + volume
    # Pharmaceutical % = w/v (weight/volume) = grams per 100mL
    if dose_info["percentages"] and dose_info["total_volume_ml"]:
        # Parse diluent percentage if present (e.g., "0.9%" from "0.9% SODIUM CHLORIDE")
        diluent_pct = None
        if result["iv_diluent_amount"]:
            try:
                diluent_pct = float(result["iv_diluent_amount"].replace("%", "").strip())
            except (ValueError, AttributeError):
                pass
        
        iv_amounts = calculate_iv_amounts(
            text=drug_name,
            drug_percentages=dose_info["percentages"],
            diluent_type=result["iv_diluent_type"],
            diluent_percentage=diluent_pct,
            total_volume_ml=dose_info["total_volume_ml"],
        )
        
        result["drug_amount_mg"] = iv_amounts["drug_amount_mg"]
        result["diluent_amount_mg"] = iv_amounts["diluent_amount_mg"]
        result["concentration_mg_per_ml"] = iv_amounts["concentration_mg_per_ml"]
    
    return result

# Dose with denominator pattern: 500MG/5ML, 10MG/ML, etc.
_DOSE_RATIO_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|ug|iu)\s*/\s*(\d+(?:\.\d+)?)\s*(ml|l)",
    re.IGNORECASE,
)

# Weight unit conversion factors to mg (for normalize_weight_to_mg)
# Note: _MASS_TO_MG is more complete, defined earlier with DOSE PARSING section
_WEIGHT_TO_MG: Dict[str, float] = _MASS_TO_MG.copy()
_WEIGHT_TO_MG["IU"] = 1.0  # Keep IU as-is (no standard conversion)


def normalize_dose_ratio(dose_str: str) -> Tuple[str, bool]:
    """
    Normalize a dose ratio to per-1-unit format.
    
    Examples:
    - "500MG/5ML" -> ("100MG/ML", True)
    - "10MG/ML" -> ("10MG/ML", True)
    - "1G/100ML" -> ("10MG/ML", True)
    - "500MG" -> ("500MG", False)  # Not a ratio
    
    Returns (normalized_dose, was_normalized).
    """
    match = _DOSE_RATIO_PATTERN.match(dose_str.strip())
    if not match:
        return dose_str, False
    
    numerator = float(match.group(1))
    num_unit = match.group(2).upper()
    denominator = float(match.group(3))
    denom_unit = match.group(4).upper()
    
    # Convert numerator to mg
    mg_factor = _WEIGHT_TO_MG.get(num_unit, 1.0)
    mg_value = numerator * mg_factor
    
    # Convert denominator to ml
    ml_factor = _VOLUME_TO_ML.get(denom_unit, 1.0)
    ml_value = denominator * ml_factor
    
    # Calculate per-1-ml concentration
    if ml_value == 0:
        return dose_str, False
    
    per_ml = mg_value / ml_value
    
    # Format result
    # Use integer if whole number, otherwise 2 decimal places
    if per_ml == int(per_ml):
        normalized = f"{int(per_ml)}MG/ML"
    else:
        normalized = f"{per_ml:.2f}MG/ML"
    
    # Remove trailing zeros
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
        if not normalized.endswith("MG/ML"):
            normalized += "MG/ML"
    
    return normalized, True


def normalize_weight_to_mg(dose_str: str) -> Tuple[str, bool]:
    """
    Normalize a weight dose to mg.
    
    Examples:
    - "1G" -> ("1000MG", True)
    - "500MCG" -> ("0.5MG", True)
    - "500MG" -> ("500MG", False)  # Already in mg
    
    Returns (normalized_dose, was_normalized).
    """
    # Pattern for simple weight doses
    pattern = re.compile(r"^(\d+(?:\.\d+)?)\s*(g|mcg|ug)$", re.IGNORECASE)
    match = pattern.match(dose_str.strip())
    
    if not match:
        return dose_str, False
    
    value = float(match.group(1))
    unit = match.group(2).upper()
    
    factor = _WEIGHT_TO_MG.get(unit, 1.0)
    if factor == 1.0:
        return dose_str, False
    
    mg_value = value * factor
    
    # Format result
    if mg_value == int(mg_value):
        normalized = f"{int(mg_value)}MG"
    elif mg_value < 1:
        # For small values, keep more precision
        normalized = f"{mg_value}MG"
        # Clean up: 0.5000 -> 0.5
        parts = normalized.split("MG")
        clean_num = parts[0].rstrip("0").rstrip(".")
        if clean_num.startswith("."):
            clean_num = "0" + clean_num
        normalized = clean_num + "MG"
    else:
        normalized = f"{mg_value:.2f}MG".rstrip("0").rstrip(".")
        if not normalized.endswith("MG"):
            normalized += "MG"
    
    return normalized, True

def extract_type_detail(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract type detail from comma-separated text.
    
    Examples:
    - "ALBUMIN, HUMAN" -> ("ALBUMIN", "HUMAN")
    - "ALCOHOL, ETHYL" -> ("ALCOHOL", "ETHYL")
    - "PARACETAMOL" -> ("PARACETAMOL", None)
    
    Returns (base_text, type_detail or None).
    """
    return _extract_type_detail_impl(text)


def extract_release_detail(form_text: str) -> Tuple[str, Optional[str]]:
    """
    Extract release modifier from form text.
    
    Examples:
    - "TABLET, EXTENDED RELEASE" -> ("TABLET", "EXTENDED RELEASE")
    - "CAPSULE SR" -> ("CAPSULE", "SR")
    - "TABLET" -> ("TABLET", None)
    
    Returns (base_form, release_detail or None).
    """
    return _extract_release_detail_impl(form_text)


def extract_form_detail(form_text: str) -> Tuple[str, Optional[str]]:
    """
    Extract form modifier (non-release) from form text.
    
    Examples:
    - "TABLET, FILM COATED" -> ("TABLET", "FILM COATED")
    - "CAPSULE EC" -> ("CAPSULE", "EC")
    - "TABLET" -> ("TABLET", None)
    
    Returns (base_form, form_detail or None).
    """
    return _extract_form_detail_impl(form_text)


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
    original_text: Optional[str] = None,
) -> List[str]:
    """
    Normalize tokens: uppercase, strip punctuation, handle multi-word generics.
    
    Args:
        original_text: If provided, used to detect "( as ...)" salt patterns
                      so we can exclude multiword matches inside them.
    """
    if multiword_generics is None:
        multiword_generics = set()
    
    result = []
    text = " ".join(tokens).upper()
    
    # Find "( as ...)" salt pattern ranges if original_text provided
    salt_pattern_content: Set[str] = set()
    if original_text:
        for match in re.finditer(r"\(\s*as\s+([^)]+)\)", original_text, re.IGNORECASE):
            salt_pattern_content.add(match.group(1).strip().upper())
    
    # Also identify trailing salt suffixes (DRUG SALT pattern)
    trailing_salt_words = {
        "SODIUM PHOSPHATE", "DISODIUM PHOSPHATE", "SODIUM SUCCINATE",
        "SODIUM SULFATE", "POTASSIUM PHOSPHATE", "CALCIUM PHOSPHATE",
        "MAGNESIUM SULFATE", "SODIUM CHLORIDE",
    }
    
    def is_trailing_salt(mwg: str, orig_text: str) -> bool:
        """Check if multiword is a trailing salt suffix in the original text."""
        if not orig_text or mwg not in trailing_salt_words:
            return False
        orig_upper = orig_text.upper()
        pos = orig_upper.find(mwg)
        if pos < 0:
            return False
        before = orig_upper[:pos].strip()
        if before and len(before.split()) >= 1:
            last_word = before.split()[-1]
            if last_word not in {"SODIUM", "DISODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM"}:
                return True
        return False
    
    # First, extract multi-word generics (but exclude those inside salt patterns)
    for mwg in sorted(multiword_generics, key=len, reverse=True):
        if mwg in text:
            # Skip if this multiword is inside a salt pattern
            if any(mwg in sc or sc in mwg for sc in salt_pattern_content):
                continue
            # Skip if this multiword is a trailing salt suffix
            if is_trailing_salt(mwg, original_text):
                continue
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
        
        # Strict validation for generic tokens
        # Only allow alphanumeric tokens with reasonable length
        if (not tok_upper.strip() or 
            len(tok_upper.strip()) < 2 or
            not any(c.isalpha() for c in tok_upper) or
            tok_upper.count('*') > 0 or  # No asterisks in generic names
            tok_upper in {'GENERIC', 'OP', 'GRAM', '100S'}):  # Descriptors, not drug names
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
    
    # Check for multiword generics BEFORE tokenizing
    # Sort by length descending to prefer longer matches and avoid substrings
    text_upper = text.upper()
    
    # Find "( as ...)" salt pattern ranges to exclude matches inside them
    salt_pattern_ranges = []
    for match in re.finditer(r"\(\s*as\s+[^)]+\)", text_upper, re.IGNORECASE):
        salt_pattern_ranges.append((match.start(), match.end()))
    
    # Also identify trailing salt suffixes (DRUG SALT pattern like "DEXAMETHASONE SODIUM PHOSPHATE")
    trailing_salt_words = {
        "SODIUM PHOSPHATE", "DISODIUM PHOSPHATE", "SODIUM SUCCINATE",
        "SODIUM SULFATE", "POTASSIUM PHOSPHATE", "CALCIUM PHOSPHATE",
        "MAGNESIUM SULFATE", "SODIUM CHLORIDE",
    }
    
    def is_trailing_salt_suffix(mw: str) -> bool:
        """Check if multiword is a trailing salt suffix in the text."""
        if mw not in trailing_salt_words:
            return False
        # Check if it appears at end of a word sequence (after a drug name)
        # Pattern: DRUGNAME + SALT at start of text
        pos = text_upper.find(mw)
        if pos < 0:
            return False
        # If there's at least one word before the salt, it's likely a trailing suffix
        before = text_upper[:pos].strip()
        if before and len(before.split()) >= 1:
            # EXCEPTION: If preceded by " IN " pattern, it's an IV solution component, NOT a salt suffix
            # e.g., "5% DEXTROSE IN 0.9% SODIUM CHLORIDE" - SODIUM CHLORIDE is the base solution
            if " IN " in before.upper():
                return False
            # Check if the word before isn't another salt word
            last_word = before.split()[-1]
            if last_word not in {"SODIUM", "DISODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM"}:
                return True
        return False
    
    def is_inside_salt_pattern(pos: int, length: int) -> bool:
        """Check if position range overlaps with any salt pattern."""
        end = pos + length
        for start, stop in salt_pattern_ranges:
            if pos >= start and end <= stop:
                return True
        return False
    
    matched_multiword = []
    for mw in sorted(multiword_generics, key=len, reverse=True):
        if mw in text_upper:
            pos = text_upper.find(mw)
            # Skip if this multiword is inside a "( as ...)" salt pattern
            if is_inside_salt_pattern(pos, len(mw)):
                continue
            # Skip if this multiword is a trailing salt suffix (DRUG SALT pattern)
            if is_trailing_salt_suffix(mw):
                continue
            # Check if this is a substring of an already-matched multiword
            is_substring = False
            for _, existing_mw in matched_multiword:
                if mw in existing_mw:
                    is_substring = True
                    break
            if not is_substring:
                matched_multiword.append((pos, mw))
    # Sort by position in text
    matched_multiword.sort(key=lambda x: x[0])
    
    # Tokenize
    raw_tokens = split_with_parentheses(text)
    raw_tokens = detect_compound_salts(raw_tokens, text)
    tokens = normalize_tokens(raw_tokens, drop_stopwords=True, multiword_generics=multiword_generics, original_text=text)
    
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
    # BUT exclude those that appear inside "( as ...)" salt patterns or as trailing salts
    text_upper = text.upper()
    # Find salt pattern content to exclude
    salt_pattern_content = set()
    for match in re.finditer(r"\(\s*as\s+([^)]+)\)", text_upper, re.IGNORECASE):
        salt_pattern_content.add(match.group(1).strip())
    
    for psc in PURE_SALT_COMPOUNDS:
        if psc in text_upper and psc not in generic_tokens:
            # Skip if this pure salt is inside a "( as ...)" pattern
            if any(psc in salt_content for salt_content in salt_pattern_content):
                continue
            # Skip if this is a trailing salt suffix (DRUG SALT pattern)
            if is_trailing_salt_suffix(psc):
                continue
            generic_tokens.append(psc)
    
    # Handle combination drugs with "+" separator
    # BUT skip if "+" is preceded by "diluent", "solvent", "dose", etc. (packaging info)
    if "+" in text_upper:
        parts = text_upper.split("+")
        added_parts = []
        skip_combo_words = {"DILUENT", "SOLVENT", "DOSE", "DOSES", "VIAL", "AMPULE", "SYRINGE"}
        form_words = {"TABLET", "CAPSULE", "SOLUTION", "INJECTION", "SYRUP", "OINTMENT", "CREAM"}
        for part in parts:
            part = part.strip()
            # Skip parts that are packaging/diluent info
            part_words = part.split()
            if part_words and part_words[0] in skip_combo_words:
                continue
            # Remove salt parentheticals from the part before extracting words
            part_clean = re.sub(r"\(\s*as\s+[^)]+\)", "", part, flags=re.IGNORECASE)
            # Also remove any remaining empty or near-empty parentheticals
            part_clean = re.sub(r"\(\s*\)", "", part_clean)
            
            # First pass: collect all non-dose words
            all_words = []
            for word in part_clean.split():
                if word and not any(c.isdigit() for c in word) and word not in UNIT_TOKENS:
                    if word not in form_words:
                        all_words.append(word)
                else:
                    break
            
            # Check if the full combo part is a known multiword generic
            full_combo = " ".join(all_words)
            if full_combo in multiword_generics:
                # Use the full multiword generic as-is
                if full_combo and full_combo not in generic_tokens:
                    generic_tokens.append(full_combo)
                    added_parts.append(full_combo)
                continue
            
            # Second pass: filter stopwords/salt tokens, but keep them if they're standalone
            # (e.g., ZINC alone should be kept, not filtered as a salt token)
            words = []
            for word in all_words:
                if word in STOPWORDS or word in SALT_TOKENS:
                    # Keep if it's the only word (likely the actual drug)
                    if len(all_words) == 1:
                        words.append(word)
                    else:
                        continue
                else:
                    words.append(word)
            
            if words:
                combo_part = " ".join(words)
                if combo_part and combo_part not in generic_tokens:
                    generic_tokens.append(combo_part)
                    added_parts.append(combo_part)
        
        # Remove combined tokens that contain + if we've added individual parts
        # e.g., remove "IBUPROFEN+PARACETAMOL" if we added "IBUPROFEN" and "PARACETAMOL"
        # Also remove tokens that start with + (e.g., "+ZINC" from "ACID+ZINC")
        if len(added_parts) >= 2:
            generic_tokens = [g for g in generic_tokens if "+" not in g and not g.startswith("+")]
    
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
            # Skip leading dose tokens (e.g., "0.9%" in "0.9% SODIUM CHLORIDE")
            base_words = []
            started = False
            for word in parts[1].strip().split():
                # Skip leading dose tokens (digits, units)
                if not started:
                    if any(c.isdigit() for c in word) or word in UNIT_TOKENS:
                        continue
                    started = True
                
                if word and word not in skip_words:
                    # Stop at form/packaging words or subsequent dose tokens
                    if any(c.isdigit() for c in word) and started:
                        break
                    if word in UNIT_TOKENS:
                        break
                    base_words.append(word)
                else:
                    break
            
            # Check if base_words form a known pure salt compound (e.g., SODIUM CHLORIDE)
            base_name = " ".join(base_words) if base_words else None
            # Also check for known IV solution bases like LACTATED RINGER'S
            if base_name:
                # Normalize RINGER'S variations
                if "RINGER" in base_name:
                    # Keep the full phrase including LACTATED/ACETATED
                    pass
            
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
    
    # Filter out form modifier words that appear after form words (e.g., GELATIN after CAPSULE)
    # This prevents matching GELATIN as a drug in "CAPSULE SOFT GELATIN"
    form_words = {"CAPSULE", "CAPSULES", "TABLET", "TABLETS", "SOLUTION", "SOLUTIONS",
                  "SUSPENSION", "CREAM", "OINTMENT", "GEL", "LOTION", "POWDER"}
    text_upper = text.upper()
    
    # Check if any form word appears in text
    form_pos = -1
    for fw in form_words:
        pos = text_upper.find(fw)
        if pos >= 0:
            if form_pos < 0 or pos < form_pos:
                form_pos = pos
    
    if form_pos >= 0:
        # Remove generic tokens that are form modifiers AND appear after the form word
        filtered_generics = []
        for g in generic_tokens:
            g_upper = g.upper()
            if g_upper in FORM_MODIFIER_IGNORE:
                g_pos = text_upper.find(g_upper)
                # Only filter if this token appears AFTER the first form word
                if g_pos > form_pos:
                    continue  # Skip this token
            filtered_generics.append(g)
        
        # Only apply filter if we still have at least one generic token
        if filtered_generics:
            generic_tokens = filtered_generics
    
    return tokens, generic_tokens


def strip_salt_suffix(
    generic: str,
    salt_suffixes: Optional[Set[str]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Strip salt suffix from a generic name.
    
    Returns (base_name, salt_suffix or None).
    """
    generic_upper = generic.upper()
    
    # Don't strip from pure salt compounds
    if generic_upper in PURE_SALT_COMPOUNDS:
        return generic_upper, None
    
    # Use pre-sorted list for default case (fast path)
    if salt_suffixes is None:
        suffixes_to_check = _SALT_TOKENS_SORTED
        salt_lookup = SALT_TOKENS
    else:
        suffixes_to_check = sorted(salt_suffixes, key=len, reverse=True)
        salt_lookup = salt_suffixes
    
    # Check each salt suffix (longest first)
    for suffix in suffixes_to_check:
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
            if potential_salt in salt_lookup:
                return parts[0].strip(), potential_salt
    
    return generic_upper, None


def parse_combo_doses(text: str, generics: List[str]) -> Dict[str, str]:
    """
    Parse combination drug doses and associate each dose with its generic.
    
    Doses separated by +, /, or | correspond to generics in order.
    
    Examples:
        "AMPICILLIN + SULBACTAM 500MG/250MG" with generics ["AMPICILLIN", "SULBACTAM"]
            -> {"AMPICILLIN": "500MG", "SULBACTAM": "250MG"}
        
        "HRZE 1250MG+75MG+400MG+276MG" with generics ["H", "R", "Z", "E"]
            -> {"H": "1250MG", "R": "75MG", "Z": "400MG", "E": "276MG"}
    
    Returns dict of generic -> dose, or empty dict if parsing fails.
    """
    if not generics:
        return {}
    
    text_upper = text.upper()
    
    # Find dose patterns: "500MG/250MG" or "500MG+250MG" or "500 MG + 250 MG"
    # Pattern for individual doses
    dose_pattern = r'(\d+(?:[.,]\d+)?)\s*(MG|G|MCG|UG|IU|ML|%)'
    
    # Try to find dose sequences separated by +, /, or |
    # First look for "XMGYYMG" or "XMG/YMG" or "XMG+YMG" patterns
    combo_dose_pattern = re.compile(
        r'(\d+(?:[.,]\d+)?)\s*(MG|G|MCG|UG|IU|ML|%)\s*[/+|]\s*' +
        r'(\d+(?:[.,]\d+)?)\s*(MG|G|MCG|UG|IU|ML|%)',
        re.IGNORECASE
    )
    
    # Find all dose values in order
    all_doses = []
    dose_matches = list(re.finditer(dose_pattern, text_upper))
    
    if not dose_matches:
        return {}
    
    # Check if doses are in a combo pattern (separated by +, /, |)
    # by looking at the text between matches
    prev_end = 0
    for match in dose_matches:
        between = text_upper[prev_end:match.start()]
        # Skip if this looks like a concentration (X/Y mL pattern)
        if prev_end > 0 and '/' in between and 'ML' in text_upper[match.end():match.end()+5]:
            continue
        
        dose_val = match.group(1).replace(",", ".")
        dose_unit = match.group(2).upper()
        all_doses.append(f"{dose_val}{dose_unit}")
        prev_end = match.end()
    
    # If we have the same number of doses as generics, map them in order
    if len(all_doses) == len(generics):
        return dict(zip([g.upper() for g in generics], all_doses))
    
    # If more doses than generics (e.g., "500MG/5ML" concentration pattern), 
    # try to match first N doses
    if len(all_doses) > len(generics) and len(generics) > 0:
        # Check if this is a concentration pattern (dose/volume)
        # by looking for ML or L at the end
        if all_doses[-1].endswith('ML') or all_doses[-1].endswith('L'):
            # Last dose is volume, use previous doses for generics
            return dict(zip([g.upper() for g in generics], all_doses[:len(generics)]))
    
    # Fallback: try to pair first dose with first generic
    if len(all_doses) >= 1 and len(generics) >= 1:
        result = {}
        for i, g in enumerate(generics):
            if i < len(all_doses):
                result[g.upper()] = all_doses[i]
        return result
    
    return {}


def format_combo_doses(generics: List[str], dose_map: Dict[str, str]) -> str:
    """
    Format combo doses as a pipe-separated string.
    
    Example: {"AMPICILLIN": "500MG", "SULBACTAM": "250MG"}
        -> "AMPICILLIN 500MG|SULBACTAM 250MG"
    """
    if not dose_map:
        return ""
    
    parts = []
    for g in generics:
        g_upper = g.upper()
        if g_upper in dose_map:
            parts.append(f"{g_upper} {dose_map[g_upper]}")
    
    return "|".join(parts) if parts else ""
