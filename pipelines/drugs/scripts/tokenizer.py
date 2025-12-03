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
    }
    
    working = drug_name.strip()
    # Normalize: remove whitespace after opening parenthesis and before closing
    working = re.sub(r"\(\s+", "(", working)
    working = re.sub(r"\s+\)", ")", working)
    
    # Strip diluent/solvent patterns - these are packaging info, not drug components
    # Patterns to remove:
    # - "+ diluent", "+ 2 mL diluent", "+ solvent", "+ reconstitution fluid"
    # - "LYOPHILIZED POWDER + DILUENT", "FREEZE-DRIED POWDER + DILUENT"
    # - "monodose vial + 0.5 mL diluent", "1 dose + 1 mL diluent"
    diluent_keywords = (
        r"diluent|solvent|reconstitution\s+fluid|sterile\s+water|"
        r"water\s+for\s+injection|w\.?f\.?i\.?"
    )
    # Pattern 1a: "+ X mL diluent" - explicit volume before keyword
    diluent_pattern1a = re.compile(
        r"\s*\+\s*\d+(?:[.,]\d+)?\s*m?L?\s+" + diluent_keywords,
        re.IGNORECASE
    )
    working = diluent_pattern1a.sub("", working)
    
    # Pattern 1b: "+ diluent" without volume
    diluent_pattern1b = re.compile(
        r"\s*\+\s*" + diluent_keywords,
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
    
    return result

# Dose with denominator pattern: 500MG/5ML, 10MG/ML, etc.
_DOSE_RATIO_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|ug|iu)\s*/\s*(\d+(?:\.\d+)?)\s*(ml|l)",
    re.IGNORECASE,
)

# Weight unit conversion factors to mg
_WEIGHT_TO_MG: Dict[str, float] = {
    "MG": 1.0,
    "G": 1000.0,
    "MCG": 0.001,
    "UG": 0.001,
    "IU": 1.0,  # Keep IU as-is (no standard conversion)
}

# Volume unit conversion factors to ml
_VOLUME_TO_ML: Dict[str, float] = {
    "ML": 1.0,
    "L": 1000.0,
}


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
