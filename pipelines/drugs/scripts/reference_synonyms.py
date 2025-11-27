#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference synonyms and normalization for drug matching.

This module provides:
1. Generic name synonyms loaded from DrugBank (grouped by drugbank_id)
2. Brand/generic flip detection for FDA data
3. Common salt form normalization

Synonyms are derived from DrugBank data where multiple lexemes share the same
drugbank_id (e.g., PARACETAMOL and ACETAMINOPHEN both map to DB00316).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Set, Dict, Tuple, Optional

# ============================================================================
# DRUGBANK-BASED SYNONYMS
# Loaded dynamically from drugbank_generics_master.csv
# ============================================================================

_DRUGBANK_SYNONYMS_CACHE: Optional[Dict[str, str]] = None
_DRUGBANK_GENERICS_CACHE: Optional[Set[str]] = None


def _find_drugbank_path() -> Optional[Path]:
    """Find the DrugBank generics master file."""
    # Try various possible locations
    possible_paths = [
        Path(__file__).resolve().parents[3] / "inputs" / "drugs" / "drugbank_generics_master.csv",
        Path.cwd() / "inputs" / "drugs" / "drugbank_generics_master.csv",
    ]
    for path in possible_paths:
        if path.is_file():
            return path
    return None


def load_drugbank_synonyms() -> Dict[str, str]:
    """
    Load synonym mapping from DrugBank data.
    
    Groups lexemes by drugbank_id and maps each non-canonical name to its
    canonical form. For example: PARACETAMOL -> ACETAMINOPHEN
    
    Returns:
        Dict mapping synonym names to their canonical DrugBank names
    """
    global _DRUGBANK_SYNONYMS_CACHE
    if _DRUGBANK_SYNONYMS_CACHE is not None:
        return _DRUGBANK_SYNONYMS_CACHE
    
    synonyms: Dict[str, str] = {}
    
    path = _find_drugbank_path()
    if not path:
        print("[synonyms] Warning: DrugBank file not found, using empty synonym map")
        _DRUGBANK_SYNONYMS_CACHE = synonyms
        return synonyms
    
    # Build synonym groups by drugbank_id
    synonym_groups: Dict[str, Set[str]] = defaultdict(set)
    canonical_by_id: Dict[str, str] = {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                db_id = (row.get("drugbank_id") or "").strip()
                canonical = (row.get("canonical_generic_name") or "").strip().upper()
                lexeme = (row.get("lexeme") or "").strip().upper()
                
                if not db_id:
                    continue
                
                # Track canonical name for this drugbank_id
                if canonical and canonical != "NAN":
                    canonical_by_id[db_id] = canonical
                    synonym_groups[db_id].add(canonical)
                
                # Add lexeme as synonym
                if lexeme and lexeme != "NAN":
                    synonym_groups[db_id].add(lexeme)
        
        # Build synonym mapping: each name maps to its canonical form
        for db_id, names in synonym_groups.items():
            canonical = canonical_by_id.get(db_id)
            if canonical:
                for name in names:
                    if name != canonical:
                        synonyms[name] = canonical
        
        print(f"[synonyms] Loaded {len(synonyms)} synonyms from DrugBank")
    except Exception as e:
        print(f"[synonyms] Warning: Could not load DrugBank synonyms: {e}")
    
    # Add additional synonyms not in DrugBank (spelling variants, regional names)
    additional_synonyms = {
        # Spelling variants
        "BECLOMETASONE": "BECLOMETHASONE DIPROPIONATE",
        "BECLOMETHASONE": "BECLOMETHASONE DIPROPIONATE",
        "PHYTOMENADIONE": "PHYTONADIONE",
        "ISOSORBIDE-5-MONONITRATE": "ISOSORBIDE MONONITRATE",
        "ISOSORBIDE 5 MONONITRATE": "ISOSORBIDE MONONITRATE",
        "PHENOXYMETHYL PENICILLIN": "PHENOXYMETHYLPENICILLIN",
        "PENICILLIN V": "PHENOXYMETHYLPENICILLIN",
        "COLESTYRAMINE": "CHOLESTYRAMINE",
        "LEUPRORELINE": "LEUPROLIDE",
        "MEDROXYPROGESTERONE": "MEDROXYPROGESTERONE ACETATE",
        "FERROUS SULFATE": "IRON",
        "FERROUS SULPHATE": "IRON",
        "FERROUS SALT": "IRON",
        # Regional names
        "ADRENALINE": "EPINEPHRINE",
        "NORADRENALINE": "NOREPINEPHRINE",
        "FRUSEMIDE": "FUROSEMIDE",
        "LIGNOCAINE": "LIDOCAINE",
        "GLYCERYL TRINITRATE": "NITROGLYCERIN",
        "GTN": "NITROGLYCERIN",
        "CICLOSPORIN": "CYCLOSPORINE",
        "CICLOSPORINE": "CYCLOSPORINE",
        "RIFAMPICIN": "RIFAMPIN",
        "PHENOBARBITONE": "PHENOBARBITAL",
        "CHLORPHENIRAMINE": "CHLORPHENAMINE",
        "PETHIDINE": "MEPERIDINE",
    }
    for name, canonical in additional_synonyms.items():
        if name not in synonyms:
            synonyms[name] = canonical
    
    _DRUGBANK_SYNONYMS_CACHE = synonyms
    return synonyms


def load_drugbank_generics() -> Set[str]:
    """
    Load all known generic names from DrugBank (canonical + lexemes).
    
    Returns:
        Set of all known generic drug names (uppercase)
    """
    global _DRUGBANK_GENERICS_CACHE
    if _DRUGBANK_GENERICS_CACHE is not None:
        return _DRUGBANK_GENERICS_CACHE
    
    generics: Set[str] = set()
    
    path = _find_drugbank_path()
    if not path:
        print("[synonyms] Warning: DrugBank file not found, using empty generics set")
        _DRUGBANK_GENERICS_CACHE = generics
        return generics
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                canonical = (row.get("canonical_generic_name") or "").strip().upper()
                lexeme = (row.get("lexeme") or "").strip().upper()
                
                if canonical and canonical != "NAN":
                    generics.add(canonical)
                if lexeme and lexeme != "NAN":
                    generics.add(lexeme)
        
        print(f"[synonyms] Loaded {len(generics)} generic names from DrugBank")
    except Exception as e:
        print(f"[synonyms] Warning: Could not load DrugBank generics: {e}")
    
    _DRUGBANK_GENERICS_CACHE = generics
    return generics


def normalize_generic_name(name: str) -> str:
    """
    Normalize a generic name to its canonical DrugBank form.
    
    Args:
        name: Generic drug name to normalize
        
    Returns:
        Canonical form if a synonym exists, otherwise the original name (uppercase)
    """
    if not name:
        return ""
    upper = name.upper().strip()
    synonyms = load_drugbank_synonyms()
    return synonyms.get(upper, upper)


def get_all_synonyms(name: str) -> Set[str]:
    """
    Get all known synonyms for a generic name (including itself).
    
    Args:
        name: Generic drug name
        
    Returns:
        Set of all names that map to the same DrugBank ID
    """
    if not name:
        return set()
    
    upper = name.upper().strip()
    synonyms = load_drugbank_synonyms()
    
    # Find canonical form
    canonical = synonyms.get(upper, upper)
    
    # Find all names that map to this canonical form
    result = {canonical}
    for syn_name, syn_canonical in synonyms.items():
        if syn_canonical == canonical:
            result.add(syn_name)
    
    # Add the original name
    result.add(upper)
    
    return result


# ============================================================================
# SALT FORM NORMALIZATION
# ============================================================================

SALT_FORMS: Set[str] = {
    "HYDROCHLORIDE", "HCL", "SODIUM", "POTASSIUM", "CALCIUM", "MAGNESIUM",
    "SULFATE", "SULPHATE", "ACETATE", "MALEATE", "FUMARATE", "TARTRATE",
    "CITRATE", "PHOSPHATE", "CHLORIDE", "BESILATE", "BESYLATE", "MESYLATE",
    "NITRATE", "SUCCINATE", "LACTATE", "GLUCONATE", "CARBONATE", "BICARBONATE",
    "TRIHYDRATE", "DIHYDRATE", "MONOHYDRATE", "ANHYDROUS",
}


def strip_salt_form(name: str) -> str:
    """Remove salt form suffix from a generic name."""
    if not name:
        return ""
    upper = name.upper().strip()
    
    # Remove "AS <SALT>" pattern
    upper = re.sub(r'\s+AS\s+\w+$', '', upper)
    
    # Remove "(AS <SALT>)" pattern
    upper = re.sub(r'\s*\(AS\s+[^)]+\)\s*$', '', upper)
    
    # Remove trailing salt forms
    for salt in SALT_FORMS:
        pattern = rf'\s+{re.escape(salt)}\s*$'
        upper = re.sub(pattern, '', upper, flags=re.IGNORECASE)
    
    return upper.strip()


def extract_base_generic(name: str) -> str:
    """Extract the base generic name without salt forms, normalized to canonical."""
    stripped = strip_salt_form(name)
    normalized = normalize_generic_name(stripped)
    return normalized


# ============================================================================
# BRAND/GENERIC FLIP DETECTION
# Uses DrugBank generics for comprehensive detection
# ============================================================================

def is_likely_generic(name: str, drugbank_generics: Optional[Set[str]] = None) -> bool:
    """
    Check if a name is likely a generic drug name.
    
    Args:
        name: The name to check
        drugbank_generics: Optional set of known DrugBank generics (loaded if not provided)
        
    Returns:
        True if the name matches a known generic (including salt forms)
    """
    if not name:
        return False
    upper = name.upper().strip()
    
    # Load DrugBank generics if not provided
    if drugbank_generics is None:
        drugbank_generics = load_drugbank_generics()
    
    # Direct match
    if upper in drugbank_generics:
        return True
    
    # Check base name without salt form
    base = strip_salt_form(upper)
    if base and base != upper and base in drugbank_generics:
        return True
    
    # Check for combination patterns (e.g., "IBUPROFEN + PARACETAMOL")
    if "+" in upper:
        parts = [strip_salt_form(p.strip()) for p in upper.split("+")]
        if any(p in drugbank_generics for p in parts if p):
            return True
    
    return False


def detect_brand_generic_flip(brand: str, generic: str, drugbank_generics: Optional[Set[str]] = None) -> bool:
    """
    Detect if brand and generic columns are likely swapped.
    
    Args:
        brand: The brand name from FDA data
        generic: The generic name from FDA data
        drugbank_generics: Optional set of known DrugBank generics
        
    Returns:
        True if the columns appear to be flipped
    """
    if not brand or not generic:
        return False
    
    # Load DrugBank generics if not provided
    if drugbank_generics is None:
        drugbank_generics = load_drugbank_generics()
    
    brand_is_generic = is_likely_generic(brand, drugbank_generics)
    generic_is_generic = is_likely_generic(generic, drugbank_generics)
    
    # Clear flip: brand column has a known generic, generic column doesn't
    if brand_is_generic and not generic_is_generic:
        return True
    
    return False


def fix_brand_generic_flip(brand: str, generic: str, drugbank_generics: Optional[Set[str]] = None) -> Tuple[str, str]:
    """
    Fix brand/generic flip if detected.
    
    Returns:
        (corrected_brand, corrected_generic)
    """
    if detect_brand_generic_flip(brand, generic, drugbank_generics):
        return generic, brand
    return brand, generic
