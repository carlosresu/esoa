#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference synonyms and normalization for drug matching.

This module provides:
1. Generic name synonyms (paracetamol <-> acetaminophen, etc.)
2. Brand/generic flip detection for FDA data
3. Common salt form normalization
"""

from __future__ import annotations

import re
from typing import Set, Dict, Tuple, Optional

# ============================================================================
# GENERIC SYNONYMS
# Maps alternative names to canonical DrugBank names
# ============================================================================
GENERIC_SYNONYMS: Dict[str, str] = {
    # Paracetamol/Acetaminophen (most common)
    "PARACETAMOL": "ACETAMINOPHEN",
    "PANADOL": "ACETAMINOPHEN",  # Common brand used as generic
    
    # Spelling variants
    "BECLOMETASONE": "BECLOMETHASONE",
    "AMIDOTRIZOATE": "DIATRIZOATE",
    "DIATRIZOIC": "DIATRIZOATE",
    "DIATRIZOIC ACID": "DIATRIZOATE",
    "DIATRIZOIC ACID DIHYDRATE": "DIATRIZOATE",
    "SULPHATE": "SULFATE",
    "SULPHASALAZINE": "SULFASALAZINE",
    "SULPHAMETHOXAZOLE": "SULFAMETHOXAZOLE",
    "SULPHADIAZINE": "SULFADIAZINE",
    "ALUMINIUM": "ALUMINUM",
    "ADRENALINE": "EPINEPHRINE",
    "NORADRENALINE": "NOREPINEPHRINE",
    "FRUSEMIDE": "FUROSEMIDE",
    "LIGNOCAINE": "LIDOCAINE",
    "GLYCERYL TRINITRATE": "NITROGLYCERIN",
    "GTN": "NITROGLYCERIN",
    "SALBUTAMOL": "ALBUTEROL",
    "CICLOSPORIN": "CYCLOSPORINE",
    "CICLOSPORINE": "CYCLOSPORINE",
    "CEPHALEXIN": "CEFALEXIN",
    "CEFACLOR": "CEFACLOR",
    "CEFTRIAXONE": "CEFTRIAXONE",
    "RIFAMPICIN": "RIFAMPIN",
    "PHENOBARBITONE": "PHENOBARBITAL",
    "CHLORPHENIRAMINE": "CHLORPHENAMINE",
    "CHLORPHENAMINE": "CHLORPHENIRAMINE",  # Bidirectional
    "PETHIDINE": "MEPERIDINE",
    "TRAMADOL HCL": "TRAMADOL",
    "TRAMADOL HYDROCHLORIDE": "TRAMADOL",
    
    # Vitamin synonyms
    "VITAMIN B1": "THIAMINE",
    "VITAMIN B2": "RIBOFLAVIN",
    "VITAMIN B3": "NIACIN",
    "VITAMIN B5": "PANTOTHENIC ACID",
    "VITAMIN B6": "PYRIDOXINE",
    "VITAMIN B7": "BIOTIN",
    "VITAMIN B9": "FOLIC ACID",
    "VITAMIN B12": "CYANOCOBALAMIN",
    "VITAMIN C": "ASCORBIC ACID",
    "VITAMIN D": "CHOLECALCIFEROL",
    "VITAMIN D3": "CHOLECALCIFEROL",
    "VITAMIN E": "TOCOPHEROL",
    "VITAMIN K": "PHYTONADIONE",
    "VITAMIN K1": "PHYTONADIONE",
    
    # Common abbreviations
    "VIT": "VITAMIN",
    "HCL": "HYDROCHLORIDE",
    "NA": "SODIUM",
    "K": "POTASSIUM",
    "CA": "CALCIUM",
    "MG": "MAGNESIUM",
}

# Reverse mapping for bidirectional lookup
GENERIC_SYNONYMS_REVERSE: Dict[str, str] = {v: k for k, v in GENERIC_SYNONYMS.items()}


def normalize_generic_name(name: str) -> str:
    """Normalize a generic name using synonyms."""
    if not name:
        return ""
    upper = name.upper().strip()
    return GENERIC_SYNONYMS.get(upper, upper)


def get_all_synonyms(name: str) -> Set[str]:
    """Get all known synonyms for a generic name."""
    if not name:
        return set()
    upper = name.upper().strip()
    synonyms = {upper}
    
    # Add canonical form
    if upper in GENERIC_SYNONYMS:
        synonyms.add(GENERIC_SYNONYMS[upper])
    
    # Add reverse mappings
    if upper in GENERIC_SYNONYMS_REVERSE:
        synonyms.add(GENERIC_SYNONYMS_REVERSE[upper])
    
    # Add all names that map to the same canonical form
    canonical = GENERIC_SYNONYMS.get(upper, upper)
    for k, v in GENERIC_SYNONYMS.items():
        if v == canonical:
            synonyms.add(k)
    
    return synonyms


# ============================================================================
# BRAND/GENERIC FLIP DETECTION
# Detects when FDA data has brand and generic columns swapped
# ============================================================================

# Known generic names that should never be in the brand column
KNOWN_GENERICS: Set[str] = {
    "ACETAMINOPHEN", "ACETYLCYSTEINE", "ACYCLOVIR", "ALBENDAZOLE", "ALBUTEROL",
    "ALENDRONATE", "ALLOPURINOL", "ALPRAZOLAM", "AMBROXOL", "AMIKACIN",
    "AMILORIDE", "AMINOPHYLLINE", "AMIODARONE", "AMITRIPTYLINE", "AMLODIPINE",
    "AMOXICILLIN", "AMPICILLIN", "ANASTROZOLE", "ASCORBIC ACID", "ASPIRIN",
    "ATENOLOL", "ATORVASTATIN", "ATROPINE", "AZITHROMYCIN", "BACLOFEN",
    "BECLOMETHASONE", "BENZOYL PEROXIDE", "BETAMETHASONE", "BISACODYL",
    "BISOPROLOL", "BUDESONIDE", "BUPIVACAINE", "CAFFEINE", "CALCIUM",
    "CAPTOPRIL", "CARBAMAZEPINE", "CARBIDOPA", "CARVEDILOL", "CEFACLOR",
    "CEFIXIME", "CEFTRIAXONE", "CEFUROXIME", "CELECOXIB", "CETIRIZINE",
    "CHLORAMPHENICOL", "CHLORHEXIDINE", "CHLORPHENIRAMINE", "CIPROFLOXACIN",
    "CLARITHROMYCIN", "CLINDAMYCIN", "CLOBETASOL", "CLONAZEPAM", "CLONIDINE",
    "CLOPIDOGREL", "CLOTRIMAZOLE", "CODEINE", "COLCHICINE", "DEXAMETHASONE",
    "DEXTROMETHORPHAN", "DIAZEPAM", "DICLOFENAC", "DIGOXIN", "DILTIAZEM",
    "DIPHENHYDRAMINE", "DOMPERIDONE", "DOXYCYCLINE", "ENALAPRIL", "EPINEPHRINE",
    "ERYTHROMYCIN", "ESOMEPRAZOLE", "ETHAMBUTOL", "FAMOTIDINE", "FELODIPINE",
    "FENOFIBRATE", "FENTANYL", "FERROUS", "FLUCONAZOLE", "FLUOXETINE",
    "FLUTICASONE", "FOLIC ACID", "FUROSEMIDE", "GABAPENTIN", "GENTAMICIN",
    "GLIBENCLAMIDE", "GLICLAZIDE", "GLIMEPIRIDE", "GUAIFENESIN", "HALOPERIDOL",
    "HEPARIN", "HYDRALAZINE", "HYDROCHLOROTHIAZIDE", "HYDROCORTISONE",
    "HYDROXYCHLOROQUINE", "HYOSCINE", "IBUPROFEN", "IMIPENEM", "INSULIN",
    "IPRATROPIUM", "IRBESARTAN", "ISONIAZID", "ISOSORBIDE", "ITRACONAZOLE",
    "KETOCONAZOLE", "KETOPROFEN", "KETOROLAC", "LABETALOL", "LACTULOSE",
    "LAMIVUDINE", "LAMOTRIGINE", "LANSOPRAZOLE", "LEVETIRACETAM", "LEVOCETIRIZINE",
    "LEVOFLOXACIN", "LEVOTHYROXINE", "LIDOCAINE", "LISINOPRIL", "LITHIUM",
    "LOPERAMIDE", "LORATADINE", "LORAZEPAM", "LOSARTAN", "LOVASTATIN",
    "MAGNESIUM", "MANNITOL", "MEBENDAZOLE", "MECLIZINE", "MEDROXYPROGESTERONE",
    "MEFENAMIC ACID", "MELOXICAM", "METFORMIN", "METHOTREXATE", "METHYLDOPA",
    "METHYLPREDNISOLONE", "METOCLOPRAMIDE", "METOPROLOL", "METRONIDAZOLE",
    "MICONAZOLE", "MIDAZOLAM", "MONTELUKAST", "MORPHINE", "MOXIFLOXACIN",
    "MULTIVITAMINS", "MUPIROCIN", "NAPROXEN", "NEBIVOLOL", "NIFEDIPINE",
    "NITROFURANTOIN", "NITROGLYCERIN", "NORFLOXACIN", "NYSTATIN", "OFLOXACIN",
    "OMEPRAZOLE", "ONDANSETRON", "ORLISTAT", "OSELTAMIVIR", "OXYTOCIN",
    "PANTOPRAZOLE", "PARACETAMOL", "PENICILLIN", "PHENOBARBITAL", "PHENYLEPHRINE",
    "PHENYTOIN", "PIOGLITAZONE", "PIPERACILLIN", "PIROXICAM", "POTASSIUM",
    "PRAVASTATIN", "PREDNISOLONE", "PREDNISONE", "PREGABALIN", "PRIMAQUINE",
    "PROPRANOLOL", "PYRAZINAMIDE", "PYRIDOXINE", "QUETIAPINE", "QUINAPRIL",
    "RABEPRAZOLE", "RAMIPRIL", "RANITIDINE", "RIFAMPICIN", "RISPERIDONE",
    "RIVAROXABAN", "ROSUVASTATIN", "SALBUTAMOL", "SERTRALINE", "SILDENAFIL",
    "SIMVASTATIN", "SODIUM", "SPIRONOLACTONE", "STREPTOMYCIN", "SULFASALAZINE",
    "TADALAFIL", "TAMOXIFEN", "TAMSULOSIN", "TELMISARTAN", "TERAZOSIN",
    "TERBINAFINE", "TERBUTALINE", "TETRACYCLINE", "THEOPHYLLINE", "THIAMINE",
    "TICAGRELOR", "TIMOLOL", "TOBRAMYCIN", "TRAMADOL", "TRANEXAMIC ACID",
    "TRIAMCINOLONE", "TRIMETHOPRIM", "VALPROIC ACID", "VALSARTAN", "VANCOMYCIN",
    "VERAPAMIL", "WARFARIN", "ZINC",
}

# Known brand name patterns (typically capitalized single words, often ending in specific suffixes)
BRAND_SUFFIXES: Tuple[str, ...] = (
    "OL", "IL", "IN", "AN", "ON", "EN", "UM", "AL", "AR", "ER", "OR",
    "IX", "AX", "EX", "OX", "UX", "ID", "AD", "ED", "OD", "UD",
)

# Patterns that strongly suggest a brand name
BRAND_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^[A-Z][a-z]+$"),  # CamelCase single word
    re.compile(r"^\d+$"),  # Pure numbers (unlikely generic)
    re.compile(r"^[A-Z]{2,}$"),  # All caps short word
)


def is_likely_generic(name: str) -> bool:
    """Check if a name is likely a generic drug name."""
    if not name:
        return False
    upper = name.upper().strip()
    
    # Check against known generics
    if upper in KNOWN_GENERICS:
        return True
    
    # Check synonyms
    normalized = normalize_generic_name(upper)
    if normalized in KNOWN_GENERICS:
        return True
    
    # Check for salt forms (e.g., "METFORMIN HYDROCHLORIDE")
    base = re.sub(r'\s+(HYDROCHLORIDE|HCL|SODIUM|POTASSIUM|CALCIUM|SULFATE|ACETATE|MALEATE|FUMARATE|TARTRATE|CITRATE|PHOSPHATE|CHLORIDE|BESILATE|BESYLATE|MESYLATE)\s*$', '', upper, flags=re.IGNORECASE)
    if base in KNOWN_GENERICS:
        return True
    
    # Check for "AS" salt forms (e.g., "AMLODIPINE AS BESILATE")
    as_match = re.match(r'^(.+?)\s+AS\s+', upper)
    if as_match:
        base = as_match.group(1).strip()
        if base in KNOWN_GENERICS:
            return True
    
    return False


def is_likely_brand(name: str) -> bool:
    """Check if a name is likely a brand name."""
    if not name:
        return False
    
    # If it's a known generic, it's not a brand
    if is_likely_generic(name):
        return False
    
    # Short single words with brand-like patterns
    if len(name.split()) == 1 and len(name) <= 15:
        # Check for brand patterns
        for pattern in BRAND_PATTERNS:
            if pattern.match(name):
                return True
    
    return False


def detect_brand_generic_flip(brand: str, generic: str) -> bool:
    """
    Detect if brand and generic columns are likely swapped.
    
    Returns True if the columns appear to be flipped.
    """
    if not brand or not generic:
        return False
    
    brand_is_generic = is_likely_generic(brand)
    generic_is_brand = is_likely_brand(generic)
    
    # Clear flip: brand column has a known generic, generic column has brand-like name
    if brand_is_generic and not is_likely_generic(generic):
        return True
    
    return False


def fix_brand_generic_flip(brand: str, generic: str) -> Tuple[str, str]:
    """
    Fix brand/generic flip if detected.
    
    Returns (corrected_brand, corrected_generic).
    """
    if detect_brand_generic_flip(brand, generic):
        return generic, brand
    return brand, generic


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
    
    # Remove trailing salt forms
    for salt in SALT_FORMS:
        pattern = rf'\s+{re.escape(salt)}\s*$'
        upper = re.sub(pattern, '', upper, flags=re.IGNORECASE)
    
    return upper.strip()


def extract_base_generic(name: str) -> str:
    """Extract the base generic name without salt forms."""
    stripped = strip_salt_form(name)
    normalized = normalize_generic_name(stripped)
    return normalized
