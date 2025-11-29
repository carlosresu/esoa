#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Route and form parsing utilities.

All mappings are sourced from unified_constants.py (converted to lowercase).
This file provides parsing functions that use those constants.
"""

import re
from typing import List, Optional, Tuple

from .unified_constants import (
    FORM_TO_ROUTE as _UNIFIED_FORM_TO_ROUTE,
    ROUTE_CANON as _UNIFIED_ROUTE_CANON,
)

# Convert unified constants to lowercase for text matching
FORM_TO_ROUTE = {k.lower(): v.lower() for k, v in _UNIFIED_FORM_TO_ROUTE.items()}
ROUTE_ALIASES = {k.lower(): v.lower() for k, v in _UNIFIED_ROUTE_CANON.items()}
FORM_WORDS = sorted(set(FORM_TO_ROUTE.keys()), key=len, reverse=True)


def map_route_token(r) -> List[str]:
    """Translate PNF route descriptors into canonical route token lists."""
    if not isinstance(r, str):
        return []
    r = r.strip()
    table = {
        "Oral:": ["oral"],
        "Oral/Tube feed:": ["oral"],
        "Inj.:": ["intravenous", "intramuscular", "subcutaneous"],
        "IV:": ["intravenous"],
        "IV/SC:": ["intravenous", "subcutaneous"],
        "SC:": ["subcutaneous"],
        "Subdermal:": ["subcutaneous"],
        "Inhalation:": ["inhalation"],
        "Topical:": ["topical"],
        "Patch:": ["transdermal"],
        "Ophthalmic:": ["ophthalmic"],
        "Intraocular:": ["ophthalmic"],
        "Otic:": ["otic"],
        "Nasal:": ["nasal"],
        "Rectal:": ["rectal"],
        "Vaginal:": ["vaginal"],
        "Sublingual:": ["sublingual"],
        "Oral antiseptic:": ["oral"],
        "Oral/Inj.:": ["oral", "intravenous", "intramuscular", "subcutaneous"],
    }
    return table.get(r, [])

def parse_form_from_text(s_norm: str) -> Optional[str]:
    """Extract a recognized dosage form keyword from normalized text."""
    for fw in FORM_WORDS:
        if re.search(rf"\b{re.escape(fw)}\b", s_norm):
            # Return the first matching form keyword encountered.
            return fw
    return None

def extract_route_and_form(s_norm: str) -> Tuple[Optional[str], Optional[str], str]:
    """Simultaneously infer route, form, and evidence strings from normalized text, honoring the alias/whitelist logic described in README (route evidences plus imputed route from form when allowed)."""
    route_found = None
    form_found = None
    evidence = []
    for fw in FORM_WORDS:
        if re.search(rf"\b{re.escape(fw)}\b", s_norm):
            form_found = fw
            evidence.append(f"form:{fw}")
            break
    for alias, route in ROUTE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", s_norm):
            route_found = route
            evidence.append(f"route:{alias}->{route}")
            break
    if not route_found and form_found in FORM_TO_ROUTE:
        # Infer the route from the form when no explicit alias appears in the text.
        route_found = FORM_TO_ROUTE[form_found]
        evidence.append(f"impute_route:{form_found}->{route_found}")
    return route_found, form_found, ";".join(evidence)
