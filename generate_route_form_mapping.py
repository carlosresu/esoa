#!/usr/bin/env python3
"""
Generate route-to-form mappings from reference data and identify
unencountered forms from Annex F.
"""

import pandas as pd
from collections import defaultdict
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "inputs" / "drugs"
OUTPUTS_DIR = BASE_DIR / "outputs" / "drugs"

# Canonical form mappings (normalize variations)
FORM_CANON = {
    "tab": "tablet",
    "tabs": "tablet",
    "tablets": "tablet",
    "cap": "capsule",
    "caps": "capsule",
    "capsules": "capsule",
    "bot": "bottle",
    "bott": "bottle",
    "bottles": "bottle",
    "vials": "vial",
    "inj": "injection",
    "injectable": "injection",
    "syr": "syrup",
    "ampul": "ampule",
    "ampuls": "ampule",
    "ampules": "ampule",
    "solutions": "solution",
    "granules": "granule",
    "suppositories": "suppository",
    "eye drops": "drops",
    "ear drops": "drops",
    "dpi": "dry powder inhaler",
    "mdi": "metered dose inhaler",
}

# Canonical route mappings
ROUTE_CANON = {
    "po": "oral",
    "or": "oral",
    "iv": "intravenous",
    "im": "intramuscular",
    "sc": "subcutaneous",
    "subcut": "subcutaneous",
    "sq": "subcutaneous",
}


def normalize_form(form: str) -> str:
    """Normalize a form string."""
    if not form or pd.isna(form):
        return ""
    form = str(form).strip().lower()
    return FORM_CANON.get(form, form)


def normalize_route(route: str) -> str:
    """Normalize a route string."""
    if not route or pd.isna(route):
        return ""
    route = str(route).strip().lower()
    return ROUTE_CANON.get(route, route)


def extract_forms_from_description(desc: str) -> set[str]:
    """Extract form tokens from an Annex F drug description."""
    if not desc or pd.isna(desc):
        return set()
    
    desc_upper = str(desc).upper()
    forms = set()
    
    # Common form patterns in Annex F
    form_patterns = [
        r'\b(TABLET|TABLETS|TAB|TABS)\b',
        r'\b(CAPSULE|CAPSULES|CAP|CAPS)\b',
        r'\b(BOTTLE|BOTTLES|BOT|BOTT)\b',
        r'\b(VIAL|VIALS)\b',
        r'\b(AMPULE|AMPULES|AMPUL|AMPULS)\b',
        r'\b(SOLUTION|SOLUTIONS|SOLN)\b',
        r'\b(SUSPENSION|SUSPENSIONS|SUSP)\b',
        r'\b(SYRUP|SYRUPS|SYR)\b',
        r'\b(INJECTION|INJECTABLE|INJ)\b',
        r'\b(CREAM|CREAMS)\b',
        r'\b(OINTMENT|OINTMENTS)\b',
        r'\b(GEL|GELS)\b',
        r'\b(LOTION|LOTIONS)\b',
        r'\b(DROPS|DROP)\b',
        r'\b(SPRAY|SPRAYS)\b',
        r'\b(POWDER|POWDERS)\b',
        r'\b(GRANULE|GRANULES)\b',
        r'\b(SUPPOSITORY|SUPPOSITORIES)\b',
        r'\b(PATCH|PATCHES)\b',
        r'\b(ENEMA|ENEMAS)\b',
        r'\b(INHALER|INHALERS)\b',
        r'\b(NEBULE|NEBULES)\b',
        r'\b(SACHET|SACHETS)\b',
        r'\b(LOZENGE|LOZENGES)\b',
        r'\b(ELIXIR|ELIXIRS)\b',
        r'\b(EMULSION|EMULSIONS)\b',
        r'\b(IMPLANT|IMPLANTS)\b',
        r'\b(FILM|FILMS)\b',
        r'\b(PASTE|PASTES)\b',
        r'\b(SHAMPOO|SHAMPOOS)\b',
        r'\b(BAR|BARS)\b',
        r'\b(AEROSOL|AEROSOLS)\b',
        r'\b(FOAM|FOAMS)\b',
        r'\b(STRIP|STRIPS)\b',
        r'\b(WAFER|WAFERS)\b',
        r'\b(PELLET|PELLETS)\b',
        r'\b(RING|RINGS)\b',
        r'\b(INSERT|INSERTS)\b',
        r'\b(SWAB|SWABS)\b',
        r'\b(CLOTH|CLOTHS)\b',
        r'\b(SPONGE|SPONGES)\b',
        r'\b(DRESSING|DRESSINGS)\b',
        r'\b(GLASS BOTTLE)\b',
        r'\b(PLASTIC BOTTLE)\b',
        r'\b(BAG|BAGS)\b',
    ]
    
    for pattern in form_patterns:
        match = re.search(pattern, desc_upper)
        if match:
            form = match.group(1).lower()
            forms.add(normalize_form(form))
    
    return forms


def extract_route_from_description(desc: str) -> str | None:
    """Extract route from an Annex F drug description."""
    if not desc or pd.isna(desc):
        return None
    
    desc_upper = str(desc).upper()
    
    # Route patterns in Annex F
    route_patterns = [
        (r'\bINTRAVENOUS\b', 'intravenous'),
        (r'\bINTRAMUSCULAR\b', 'intramuscular'),
        (r'\bSUBCUTANEOUS\b', 'subcutaneous'),
        (r'\bORAL\b', 'oral'),
        (r'\bTOPICAL\b', 'topical'),
        (r'\bRECTAL\b', 'rectal'),
        (r'\bVAGINAL\b', 'vaginal'),
        (r'\bOPHTHALMIC\b', 'ophthalmic'),
        (r'\bOTIC\b', 'otic'),
        (r'\bNASAL\b', 'nasal'),
        (r'\bINHALATION\b', 'inhalation'),
        (r'\bSUBLINGUAL\b', 'sublingual'),
        (r'\bTRANSDERMAL\b', 'transdermal'),
        (r'\bBUCCAL\b', 'buccal'),
    ]
    
    for pattern, route in route_patterns:
        if re.search(pattern, desc_upper):
            return route
    
    # Infer route from form
    if any(term in desc_upper for term in ['TABLET', 'CAPSULE', 'SYRUP', 'ELIXIR', 'SACHET', 'GRANULE']):
        return 'oral'
    if any(term in desc_upper for term in ['AMPULE', 'VIAL', 'BOTTLE']) and 'SOLUTION' in desc_upper:
        return 'intravenous'  # Most IV solutions
    if any(term in desc_upper for term in ['CREAM', 'OINTMENT', 'LOTION', 'GEL']) and 'OPHTHALMIC' not in desc_upper:
        return 'topical'
    if 'SUPPOSITORY' in desc_upper:
        return 'rectal'
    if 'EYE' in desc_upper or 'OPHTHALMIC' in desc_upper:
        return 'ophthalmic'
    if 'EAR' in desc_upper or 'OTIC' in desc_upper:
        return 'otic'
    if 'INHALER' in desc_upper or 'NEBULE' in desc_upper:
        return 'inhalation'
    if 'PATCH' in desc_upper:
        return 'transdermal'
    
    return None


def build_route_form_mapping() -> dict[str, set[str]]:
    """Build route -> forms mapping from reference data."""
    route_forms = defaultdict(set)
    
    # Load DrugBank
    db_path = INPUTS_DIR / "drugbank_generics_master.csv"
    if db_path.exists():
        db = pd.read_csv(db_path)
        for _, row in db.iterrows():
            route = normalize_route(row.get('route_norm', ''))
            form = normalize_form(row.get('form_norm', ''))
            if route and form:
                route_forms[route].add(form)
    
    # Load PNF
    pnf_path = INPUTS_DIR / "pnf_lexicon.csv"
    if pnf_path.exists():
        pnf = pd.read_csv(pnf_path)
        for _, row in pnf.iterrows():
            route = normalize_route(row.get('route_allowed', ''))
            form = normalize_form(row.get('form_token', ''))
            if route and form:
                route_forms[route].add(form)
    
    return dict(route_forms)


def find_unencountered_forms(route_forms: dict[str, set[str]]) -> dict[str, set[str]]:
    """Find forms in Annex F that haven't been encountered for each route."""
    unencountered = defaultdict(set)
    
    # Load Annex F
    annex_path = INPUTS_DIR / "annex_f.csv"
    if not annex_path.exists():
        return dict(unencountered)
    
    annex = pd.read_csv(annex_path)
    
    # All known forms across all routes
    all_known_forms = set()
    for forms in route_forms.values():
        all_known_forms.update(forms)
    
    for _, row in annex.iterrows():
        desc = row.get('Drug Description', '')
        forms = extract_forms_from_description(desc)
        route = extract_route_from_description(desc)
        
        for form in forms:
            if not form:
                continue
            
            # Check if this form is known for this route
            if route:
                known_forms_for_route = route_forms.get(route, set())
                if form not in known_forms_for_route:
                    unencountered[route].add(form)
            
            # Also check if form is completely unknown
            if form not in all_known_forms:
                unencountered['_unknown_'].add(form)
    
    return dict(unencountered)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build route -> forms mapping
    print("Building route-form mapping from reference data...")
    route_forms = build_route_form_mapping()
    
    # Save route-form mapping
    mapping_rows = []
    for route in sorted(route_forms.keys()):
        for form in sorted(route_forms[route]):
            mapping_rows.append({
                'route': route,
                'form': form,
                'source': 'reference'
            })
    
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = OUTPUTS_DIR / "route_form_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Route-form mapping saved to {mapping_path}")
    print(f"  Total routes: {len(route_forms)}")
    print(f"  Total route-form pairs: {len(mapping_rows)}")
    
    # Find unencountered forms
    print("\nFinding unencountered forms in Annex F...")
    unencountered = find_unencountered_forms(route_forms)
    
    # Save unencountered forms
    unencountered_rows = []
    for route in sorted(unencountered.keys()):
        for form in sorted(unencountered[route]):
            unencountered_rows.append({
                'route': route,
                'form': form,
                'status': 'unencountered'
            })
    
    if unencountered_rows:
        unencountered_df = pd.DataFrame(unencountered_rows)
        unencountered_path = OUTPUTS_DIR / "route_form_unencountered.csv"
        unencountered_df.to_csv(unencountered_path, index=False)
        print(f"Unencountered forms saved to {unencountered_path}")
        print(f"  Total unencountered route-form pairs: {len(unencountered_rows)}")
        
        # Print summary
        print("\n=== UNENCOUNTERED FORMS BY ROUTE ===")
        for route in sorted(unencountered.keys()):
            forms = sorted(unencountered[route])
            print(f"{route.upper()}: {', '.join(forms)}")
    else:
        print("No unencountered forms found.")
    
    # Print full mapping summary
    print("\n=== ROUTE-FORM MAPPING SUMMARY ===")
    for route in sorted(route_forms.keys()):
        forms = sorted(route_forms[route])
        print(f"{route.upper()} ({len(forms)} forms): {', '.join(forms[:10])}{'...' if len(forms) > 10 else ''}")


if __name__ == "__main__":
    main()
