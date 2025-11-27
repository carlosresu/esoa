"""
Constants for drug tagging.

Token categories, scoring weights, and normalization mappings.
"""

# ============================================================================
# TOKEN CATEGORIES
# ============================================================================

CATEGORY_GENERIC = "generic"
CATEGORY_SALT = "salt"
CATEGORY_DOSE = "dose"
CATEGORY_FORM = "form"
CATEGORY_ROUTE = "route"
CATEGORY_OTHER = "other"

# ============================================================================
# MATCHING RULES (Rule-based, not weight-based)
# ============================================================================

# Generic match is REQUIRED - no match without it
GENERIC_MATCH_REQUIRED = True

# Salt forms are flexible - different salts of same drug are acceptable
SALT_FLEXIBLE = True

# Dose is flexible - different doses are acceptable (same ATC)
DOSE_FLEXIBLE = True

# Form equivalence is defined in scoring.py FORM_EQUIVALENCE_GROUPS

# ============================================================================
# STOPWORDS AND JUNK TOKENS
# ============================================================================

NATURAL_STOPWORDS = {
    "AS", "IN", "FOR", "TO", "WITH", "EQUIV", "EQUIV.", "AND", "OF", "OR",
    "NOT", "THAN", "HAS", "DURING", "THIS", "W/", "W", "PLUS", "APPROX",
    "APPROXIMATELY", "PRE", "FILLED", "PRE-FILLED",
}

GENERIC_JUNK_TOKENS = {
    "SOLUTION", "SOLUTIONS", "SOLN", "IRRIGATION", "IRRIGATING",
    "INJECTION", "INJECTIONS", "INJECTABLE", "INFUSION", "INFUSIONS",
    "DILUENT", "DILUTION", "POWDER", "POWDERS", "MICRONUTRIENT",
    "FORMULA", "FORMULATION", "WATER", "VEHICLE",
}

# ============================================================================
# FORM CANONICALIZATION
# ============================================================================

FORM_CANON = {
    "TAB": "TABLET", "TABS": "TABLET", "TABLET": "TABLET", "TABLETS": "TABLET",
    "CAP": "CAPSULE", "CAPS": "CAPSULE", "CAPSULE": "CAPSULE", "CAPSULES": "CAPSULE",
    "BOT": "BOTTLE", "BOTT": "BOTTLE", "BOTTLE": "BOTTLE", "BOTTLES": "BOTTLE",
    "VIAL": "VIAL", "VIALS": "VIAL",
    "INJ": "INJECTION", "INJECTABLE": "INJECTION",
    "SYR": "SYRUP", "SYRUP": "SYRUP",
    "LOTION": "LOTION", "CREAM": "CREAM", "GEL": "GEL", "OINTMENT": "OINTMENT",
    "PASTE": "PASTE", "FOAM": "FOAM", "EMULSION": "EMULSION", "SHAMPOO": "SHAMPOO",
    "SOLUTION": "SOLUTION", "SOLN": "SOLUTION", "SOL": "SOLUTION",
    "SUSPENSION": "SUSPENSION", "SUSP": "SUSPENSION",
    "DROPS": "DROPS", "DROP": "DROPS", "GTTS": "DROPS",
    "POWDER": "POWDER", "PWDR": "POWDER",
    "GRANULES": "GRANULES", "GRANULE": "GRANULES", "GRAN": "GRANULES",
    "SACHET": "SACHET", "SACHETS": "SACHET",
    "INHALER": "INHALER", "INH": "INHALER",
    "NEBULE": "NEBULE", "NEBULES": "NEBULE", "NEB": "NEBULE",
    "DPI": "DPI", "MDI": "MDI", "AEROSOL": "AEROSOL", "SPRAY": "SPRAY",
    "SUPPOSITORY": "SUPPOSITORY", "SUPP": "SUPPOSITORY",
    "PATCH": "PATCH", "FILM": "FILM",
    "AMPULE": "AMPULE", "AMPUL": "AMPULE", "AMP": "AMPULE", "AMPOULE": "AMPULE",
}

# ============================================================================
# ROUTE CANONICALIZATION
# ============================================================================

ROUTE_CANON = {
    "PO": "ORAL", "OR": "ORAL", "ORAL": "ORAL",
    "IV": "INTRAVENOUS", "INTRAVENOUS": "INTRAVENOUS",
    "IM": "INTRAMUSCULAR", "INTRAMUSCULAR": "INTRAMUSCULAR",
    "SC": "SUBCUTANEOUS", "SUBCUTANEOUS": "SUBCUTANEOUS", "SUBCUT": "SUBCUTANEOUS",
    "NASAL": "NASAL", "TOPICAL": "TOPICAL", "RECTAL": "RECTAL",
    "OPHTHALMIC": "OPHTHALMIC", "BUCCAL": "BUCCAL",
}

# ============================================================================
# SALT TOKENS
# ============================================================================

SALT_TOKENS = {
    "CALCIUM", "SODIUM", "POTASSIUM", "MAGNESIUM", "ZINC", "AMMONIUM",
    "MEGLUMINE", "ALUMINUM", "HYDROCHLORIDE", "NITRATE", "NITRITE",
    "SULFATE", "SULPHATE", "PHOSPHATE", "HYDROXIDE", "DIPROPIONATE",
    "ACETATE", "TARTRATE", "FUMARATE", "OXALATE", "MALEATE", "MESYLATE",
    "TOSYLATE", "BESYLATE", "BESILATE", "BITARTRATE", "SUCCINATE",
    "CITRATE", "LACTATE", "GLUCONATE", "BICARBONATE", "CARBONATE",
    "BROMIDE", "CHLORIDE", "IODIDE", "SELENITE", "THIOSULFATE",
    "DIHYDRATE", "TRIHYDRATE", "MONOHYDRATE", "HYDRATE", "HEMIHYDRATE",
    "ANHYDROUS", "DECANOATE", "PALMITATE", "STEARATE", "PAMOATE",
    "BENZOATE", "VALERATE", "PROPIONATE", "HYDROBROMIDE", "DOCUSATE",
}

# Elements that can be standalone drugs (not just salt modifiers)
# These should be treated as generics when they appear as the main drug
ELEMENT_DRUGS = {
    "ZINC", "CALCIUM", "IRON", "MAGNESIUM", "POTASSIUM", "SODIUM",
    "COPPER", "MANGANESE", "SELENIUM", "CHROMIUM", "IODINE",
}

# ============================================================================
# PURE SALT COMPOUNDS - should NOT have salt stripped
# ============================================================================

PURE_SALT_COMPOUNDS = {
    "SODIUM CHLORIDE", "POTASSIUM CHLORIDE", "CALCIUM CHLORIDE",
    "MAGNESIUM CHLORIDE", "ZINC CHLORIDE", "AMMONIUM CHLORIDE",
    "SODIUM BICARBONATE", "POTASSIUM BICARBONATE", "CALCIUM CARBONATE",
    "MAGNESIUM CARBONATE", "SODIUM CARBONATE", "MAGNESIUM SULFATE",
    "SODIUM SULFATE", "POTASSIUM SULFATE", "CALCIUM SULFATE",
    "SODIUM PHOSPHATE", "POTASSIUM PHOSPHATE", "CALCIUM PHOSPHATE",
    "MAGNESIUM PHOSPHATE", "SODIUM CITRATE", "POTASSIUM CITRATE",
    "CALCIUM CITRATE", "MAGNESIUM CITRATE", "SODIUM LACTATE",
    "CALCIUM LACTATE", "SODIUM ACETATE", "POTASSIUM ACETATE",
    "CALCIUM ACETATE", "MAGNESIUM ACETATE", "SODIUM GLUCONATE",
    "CALCIUM GLUCONATE", "MAGNESIUM GLUCONATE", "POTASSIUM GLUCONATE",
    "ZINC GLUCONATE", "FERROUS SULFATE", "FERROUS FUMARATE",
    "FERROUS GLUCONATE", "ZINC SULFATE", "COPPER SULFATE",
    "MANGANESE SULFATE", "SODIUM HYDROXIDE", "POTASSIUM HYDROXIDE",
    "CALCIUM HYDROXIDE", "MAGNESIUM HYDROXIDE", "ALUMINUM HYDROXIDE",
    "SODIUM NITRATE", "POTASSIUM NITRATE", "SILVER NITRATE",
    "SODIUM IODIDE", "POTASSIUM IODIDE", "SODIUM BROMIDE",
    "POTASSIUM BROMIDE", "SODIUM FLUORIDE", "CALCIUM FLUORIDE",
    "SODIUM SELENITE", "SODIUM THIOSULFATE",
}

# ============================================================================
# ATC COMBINATION PATTERNS
# ============================================================================

ATC_COMBINATION_PATTERNS = [
    "C09DA", "C09DB", "C09DX",  # ARBs + diuretics/CCBs
    "C09BA", "C09BB", "C09BX",  # ACE inhibitors + combos
    "C07FB", "C07BB", "C07CB",  # Beta-blockers + combos
    "C10BA", "C10BX",           # Statins + combos
    "N02AA55", "N02AA59",       # Opioid combinations
    "N02AJ",                    # Opioid + non-opioid combos
    "N02BE51", "N02BE71",       # Paracetamol combinations
    "J01CR", "J01RA",           # Antibiotic combinations
    "R03AL",                    # Respiratory combinations
    "A02BD",                    # H. pylori eradication combos
]

# ============================================================================
# UNIT TOKENS (for filtering)
# ============================================================================

UNIT_TOKENS = {
    "MG", "G", "MCG", "UG", "ML", "L", "IU", "UNIT", "UNITS",
    "PCT", "%", "MG/ML", "MCG/ML", "IU/ML", "MG/5ML",
}
