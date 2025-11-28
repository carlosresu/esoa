"""
Constants for drug tagging.

DEPRECATED: This file re-exports from unified_constants.py for backward compatibility.
New code should import directly from unified_constants.py.

Token categories, scoring weights, and normalization mappings.
"""

# Re-export everything from unified_constants for backward compatibility
from .unified_constants import (
    # Categories
    CATEGORY_GENERIC,
    CATEGORY_SALT,
    CATEGORY_DOSE,
    CATEGORY_FORM,
    CATEGORY_ROUTE,
    CATEGORY_OTHER,
    
    # Rules
    GENERIC_MATCH_REQUIRED,
    SALT_FLEXIBLE,
    DOSE_FLEXIBLE,
    
    # Token sets
    STOPWORDS,
    SALT_TOKENS,
    PURE_SALT_COMPOUNDS,
    ELEMENT_DRUGS,
    UNIT_TOKENS,
    
    # Mappings
    FORM_CANON,
    ROUTE_CANON,
    
    # ATC patterns
    ATC_COMBINATION_PATTERNS,
    COMBINATION_ATC_SUFFIXES,
    
    # Helper functions
    is_stopword,
    is_salt_token,
    is_pure_salt_compound,
    is_element_drug,
    is_combination_atc,
)

# Legacy aliases for backward compatibility
NATURAL_STOPWORDS = STOPWORDS
GENERIC_JUNK_TOKENS = STOPWORDS  # Merged into STOPWORDS
