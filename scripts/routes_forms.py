#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import List, Optional, Tuple

FORM_TO_ROUTE = {
    "tablet": "oral", "tab": "oral", "tabs": "oral", "chewing gum": "oral",
    "capsule": "oral", "cap": "oral", "caps": "oral",
    "syrup": "oral", "syrups": "oral",
    "suspension": "oral", "suspensions": "oral",
    "solution": "oral", "solutions": "oral",
    "sachet": "oral",
    "granule": "oral", "granules": "oral",
    "lozenge": "oral",
    "mouthwash": "oral",
    "drops": "oral", "oral drops": "oral",
    "drop": "ophthalmic", "eye drop": "ophthalmic", "ear drop": "otic",
    "eye drops": "ophthalmic", "ear drops": "otic", "nasal drops": "nasal",
    "cream": "topical", "ointment": "topical", "gel": "topical", "lotion": "topical",
    "soap": "topical", "shampoo": "topical", "wash": "topical",
    "patch": "transdermal",
    "inhaler": "inhalation", "nebule": "inhalation", "neb": "inhalation",
    "inhal.aerosol": "inhalation", "inhal.powder": "inhalation", "inhal.solution": "inhalation", "oral aerosol": "inhalation",
    "ampoule": "intravenous", "amp": "intravenous", "ampul": "intravenous", "ampule": "intravenous",
    "vial": "intravenous", "vl": "intravenous", "inj": "intravenous", "injection": "intravenous",
    "suppository": "rectal", "ovule": "vaginal", "ovules": "vaginal",
    "mdi": "inhalation",
    "dpi": "inhalation",
    "metered dose inhaler": "inhalation",
    "dry powder inhaler": "inhalation",
    "spray": "nasal",
    "nasal spray": "nasal",
    "susp": "oral",
    "soln": "oral",
    "syr": "oral",
    "td": "transdermal",
    "supp": "rectal",
    "instill.solution": "ophthalmic", "lamella": "ophthalmic",
    "implant": "subcutaneous", "s.c. implant": "subcutaneous"
}
FORM_WORDS = sorted(set(FORM_TO_ROUTE.keys()), key=len, reverse=True)

ROUTE_ALIASES = {
    "po": "oral", "per orem": "oral", "by mouth": "oral",
    "iv": "intravenous", "intravenous": "intravenous",
    "im": "intramuscular", "intramuscular": "intramuscular",
    "sc": "subcutaneous", "subcut": "subcutaneous", "subcutaneous": "subcutaneous",
    "sl": "sublingual", "sublingual": "sublingual", "bucc": "buccal", "buccal": "buccal",
    "topical": "topical", "cutaneous": "topical", "dermal": "transdermal",
    "oph": "ophthalmic", "eye": "ophthalmic", "ophthalmic": "ophthalmic",
    "otic": "otic", "ear": "otic",
    "inh": "inhalation", "neb": "inhalation", "inhalation": "inhalation",
    "rectal": "rectal", "vaginal": "vaginal",
    "intrathecal": "intrathecal", "nasal": "nasal",
    "per os": "oral",
    "td": "transdermal",
    "transdermal": "transdermal",
    "intradermal": "intradermal",
    "id": "intradermal",
    "subdermal": "subcutaneous",
    "per rectum": "rectal",
    "pr": "rectal",
    "per vaginam": "vaginal",
    "pv": "vaginal",
    "per nasal": "nasal",
    "intranasal": "nasal",
    "inhaler": "inhalation"
}

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
