# ===============================
# File: scripts/routes_forms.py
# ===============================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import List, Optional, Tuple

FORM_TO_ROUTE = {
    "tablet": "oral", "tab": "oral", "capsule": "oral", "cap": "oral",
    "syrup": "oral", "suspension": "oral", "solution": "oral",
    "sachet": "oral",
    "drop": "ophthalmic", "eye drop": "ophthalmic", "ear drop": "otic",
    "cream": "topical", "ointment": "topical", "gel": "topical", "lotion": "topical",
    "patch": "transdermal", "inhaler": "inhalation", "nebule": "inhalation", "neb": "inhalation",
    "ampoule": "intravenous", "amp": "intravenous", "ampul": "intravenous",
    "vial": "intravenous", "vl": "intravenous", "inj": "intravenous",
    "suppository": "rectal"
}
FORM_WORDS = sorted(set(FORM_TO_ROUTE.keys()), key=len, reverse=True)

ROUTE_ALIASES = {
    "po": "oral", "per orem": "oral", "by mouth": "oral",
    "iv": "intravenous", "intravenous": "intravenous",
    "im": "intramuscular", "intramuscular": "intramuscular",
    "sc": "subcutaneous", "subcut": "subcutaneous",
    "sl": "sublingual", "bucc": "buccal",
    "topical": "topical", "cut": "topical", "dermal": "transdermal",
    "oph": "ophthalmic", "eye": "ophthalmic",
    "otic": "otic", "ear": "otic",
    "inh": "inhalation", "neb": "inhalation",
    "rectal": "rectal", "vaginal": "vaginal",
    "intrathecal": "intrathecal", "nasal": "nasal",
}


def map_route_token(r) -> List[str]:
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
    for fw in FORM_WORDS:
        if re.search(rf"\\b{re.escape(fw)}\\b", s_norm):
            return fw
    return None


def extract_route_and_form(s_norm: str) -> Tuple[Optional[str], Optional[str], str]:
    route_found = None
    form_found = None
    evidence = []
    for fw in FORM_WORDS:
        if re.search(rf"\\b{re.escape(fw)}\\b", s_norm):
            form_found = fw
            evidence.append(f"form:{fw}")
            break
    for alias, route in ROUTE_ALIASES.items():
        if re.search(rf"\\b{re.escape(alias)}\\b", s_norm):
            route_found = route
            evidence.append(f"route:{alias}->{route}")
            break
    if not route_found and form_found in FORM_TO_ROUTE:
        route_found = FORM_TO_ROUTE[form_found]
        evidence.append(f"impute_route:{form_found}->{route_found}")
    return route_found, form_found, ";".join(evidence)