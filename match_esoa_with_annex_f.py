#!/usr/bin/env python3
"""
Match ESOA rows to Annex F Drug Codes via ATC/DrugBank ID.

Pipeline:
1. Parse ESOA descriptions to extract generic name, brand, dose, form, route
2. Match ESOA to ATC/DrugBank ID using the same logic as annex_f matching
3. Use ATC/DrugBank ID to filter candidate Annex F rows
4. Score candidates by dose, form, route similarity to pick best Drug Code
"""

import re
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import List

BASE_DIR = Path(__file__).resolve().parent
INPUTS_DIR = BASE_DIR / "inputs" / "drugs"
OUTPUTS_DIR = BASE_DIR / "outputs" / "drugs"

# Import utilities from match_annex_f_with_atc
from match_annex_f_with_atc import (
    _normalize_tokens,
    split_with_parentheses,
    _categorize_tokens,
    _extract_generic_tokens,
    _extract_high_value_generics,
    _as_str_or_empty,
    FORM_CANON,
    ROUTE_CANON,
    GENERIC_SYNONYMS,
    SALT_TOKENS,
    CATEGORY_GENERIC,
    CATEGORY_DOSE,
    CATEGORY_FORM,
    CATEGORY_ROUTE,
)


# Brand name patterns to strip from ESOA descriptions
BRAND_PATTERN = re.compile(r'\s*\([^)]+\)\s*')

# Common form abbreviations in ESOA
ESOA_FORM_CANON = {
    **FORM_CANON,
    "AMP": "AMPULE",
    "AMPS": "AMPULE",
    "NEB": "NEBULE",
    "NEBS": "NEBULE",
    "NEBULE": "NEBULE",
    "NEBULES": "NEBULE",
    "SUPP": "SUPPOSITORY",
    "SUPPS": "SUPPOSITORY",
    "SUSP": "SUSPENSION",
    "SOLN": "SOLUTION",
    "SOL": "SOLUTION",
    "SACHET": "SACHET",
    "SACHETS": "SACHET",
    "DROP": "DROPS",
    "GTTS": "DROPS",
    "CREAM": "CREAM",
    "OINT": "OINTMENT",
    "GEL": "GEL",
    "LOT": "LOTION",
    "PATCH": "PATCH",
    "SPRAY": "SPRAY",
    "MDI": "MDI",
    "DPI": "DPI",
    "INHALER": "INHALER",
    "LOZENGE": "LOZENGE",
    "LOZENGES": "LOZENGE",
    "GRANULE": "GRANULE",
    "GRANULES": "GRANULE",
    "POWDER": "POWDER",
    "PWDR": "POWDER",
}

# Dose unit patterns
DOSE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(MG|MCG|G|ML|L|IU|UNIT|UNITS|%|PCT)',
    re.IGNORECASE
)


def parse_esoa_description(desc: str) -> dict:
    """Parse an ESOA description to extract components."""
    if not desc or pd.isna(desc):
        return {
            "original": "",
            "generic": "",
            "brand": "",
            "dose_value": None,
            "dose_unit": "",
            "form": "",
            "route": "",
            "tokens": [],
        }
    
    desc = str(desc).strip().upper()
    original = desc
    
    # Extract brand name (in parentheses)
    brand_match = BRAND_PATTERN.search(desc)
    brand = brand_match.group(0).strip("() ") if brand_match else ""
    
    # Remove brand name for further parsing
    desc_no_brand = BRAND_PATTERN.sub(" ", desc).strip()
    
    # Tokenize
    tokens = split_with_parentheses(desc_no_brand)
    normalized = _normalize_tokens(tokens, drop_stopwords=True)
    
    # Extract dose
    dose_value = None
    dose_unit = ""
    dose_match = DOSE_PATTERN.search(desc_no_brand)
    if dose_match:
        dose_value = float(dose_match.group(1))
        dose_unit = dose_match.group(2).upper()
        # Normalize to MG
        if dose_unit == "MCG":
            dose_value = dose_value / 1000
            dose_unit = "MG"
        elif dose_unit == "G":
            dose_value = dose_value * 1000
            dose_unit = "MG"
    
    # Extract form
    form = ""
    for tok in reversed(normalized):
        tok_upper = tok.upper()
        if tok_upper in ESOA_FORM_CANON:
            form = ESOA_FORM_CANON[tok_upper]
            break
        if tok_upper in FORM_CANON.values():
            form = tok_upper
            break
    
    # Extract route (if present)
    route = ""
    for tok in normalized:
        tok_upper = tok.upper()
        if tok_upper in ROUTE_CANON:
            route = ROUTE_CANON[tok_upper]
            break
        if tok_upper in ROUTE_CANON.values():
            route = tok_upper
            break
    
    # Infer route from form if not explicit
    if not route:
        if form in ("TABLET", "CAPSULE", "SYRUP", "SACHET", "GRANULE", "ELIXIR"):
            route = "ORAL"
        elif form in ("AMPULE", "VIAL", "INJECTION"):
            route = "INTRAVENOUS"
        elif form in ("CREAM", "OINTMENT", "LOTION", "GEL"):
            route = "TOPICAL"
        elif form in ("SUPPOSITORY",):
            route = "RECTAL"
        elif form in ("DROPS",):
            # Could be ophthalmic, otic, or nasal
            if "EYE" in desc or "OPHTHALMIC" in desc:
                route = "OPHTHALMIC"
            elif "EAR" in desc or "OTIC" in desc:
                route = "OTIC"
            else:
                route = "OPHTHALMIC"  # Default
        elif form in ("MDI", "DPI", "INHALER", "NEBULE"):
            route = "INHALATION"
        elif form in ("PATCH",):
            route = "TRANSDERMAL"
    
    # Extract generic name (first significant tokens before dose/form)
    generic_tokens = _extract_high_value_generics(normalized)
    generic = " ".join(sorted(generic_tokens)) if generic_tokens else ""
    
    return {
        "original": original,
        "generic": generic,
        "brand": brand,
        "dose_value": dose_value,
        "dose_unit": dose_unit,
        "form": form,
        "route": route,
        "tokens": normalized,
    }


def parse_annex_f_description(desc: str) -> dict:
    """Parse an Annex F description to extract components."""
    if not desc or pd.isna(desc):
        return {
            "original": "",
            "generic": "",
            "dose_value": None,
            "dose_unit": "",
            "form": "",
            "route": "",
            "volume": None,
            "volume_unit": "",
            "tokens": [],
        }
    
    desc = str(desc).strip().upper()
    original = desc
    
    # Tokenize
    tokens = split_with_parentheses(desc)
    normalized = _normalize_tokens(tokens, drop_stopwords=True)
    
    # Extract dose (strength)
    dose_value = None
    dose_unit = ""
    # Look for patterns like "500 mg" or "10 mg/mL"
    dose_match = re.search(r'(\d+(?:\.\d+)?)\s*(MG|MCG|G|IU|UNIT|%)', desc, re.IGNORECASE)
    if dose_match:
        dose_value = float(dose_match.group(1))
        dose_unit = dose_match.group(2).upper()
        # Normalize to MG
        if dose_unit == "MCG":
            dose_value = dose_value / 1000
            dose_unit = "MG"
        elif dose_unit == "G":
            dose_value = dose_value * 1000
            dose_unit = "MG"
    
    # Extract volume (for solutions)
    volume = None
    volume_unit = ""
    vol_match = re.search(r'(\d+(?:\.\d+)?)\s*(ML|L)\s+(BOTTLE|AMPULE|VIAL|BAG)', desc, re.IGNORECASE)
    if vol_match:
        volume = float(vol_match.group(1))
        volume_unit = vol_match.group(2).upper()
        if volume_unit == "L":
            volume = volume * 1000
            volume_unit = "ML"
    
    # Extract form (last token usually)
    form = ""
    for tok in reversed(normalized):
        tok_upper = tok.upper()
        if tok_upper in ESOA_FORM_CANON:
            form = ESOA_FORM_CANON[tok_upper]
            break
        if tok_upper in FORM_CANON.values():
            form = tok_upper
            break
    
    # Extract route
    route = ""
    for tok in normalized:
        tok_upper = tok.upper()
        if tok_upper in ROUTE_CANON:
            route = ROUTE_CANON[tok_upper]
            break
        if tok_upper in ROUTE_CANON.values():
            route = tok_upper
            break
    
    # Infer route from form/context
    if not route:
        if form in ("TABLET", "CAPSULE", "SYRUP", "SACHET", "GRANULE"):
            route = "ORAL"
        elif form in ("AMPULE", "VIAL", "BOTTLE") and "SOLUTION" in desc:
            route = "INTRAVENOUS"
        elif form in ("CREAM", "OINTMENT", "LOTION", "GEL"):
            route = "TOPICAL"
        elif form in ("SUPPOSITORY",):
            route = "RECTAL"
        elif "OPHTHALMIC" in desc or "EYE" in desc:
            route = "OPHTHALMIC"
        elif "OTIC" in desc or "EAR" in desc:
            route = "OTIC"
        elif "NASAL" in desc:
            route = "NASAL"
        elif "INHALATION" in desc:
            route = "INHALATION"
    
    # Extract generic
    generic_tokens = _extract_high_value_generics(normalized)
    generic = " ".join(sorted(generic_tokens)) if generic_tokens else ""
    
    return {
        "original": original,
        "generic": generic,
        "dose_value": dose_value,
        "dose_unit": dose_unit,
        "form": form,
        "route": route,
        "volume": volume,
        "volume_unit": volume_unit,
        "tokens": normalized,
    }


def score_annex_f_candidate(esoa_parsed: dict, annex_parsed: dict) -> float:
    """Score how well an Annex F row matches an ESOA row."""
    score = 0.0
    
    # Generic match (most important)
    esoa_generics = set(esoa_parsed.get("generic", "").split())
    annex_generics = set(annex_parsed.get("generic", "").split())
    if esoa_generics and annex_generics:
        overlap = len(esoa_generics & annex_generics)
        if overlap > 0:
            score += overlap * 10
        else:
            return -100  # No generic overlap = bad match
    
    # Dose match
    esoa_dose = esoa_parsed.get("dose_value")
    annex_dose = annex_parsed.get("dose_value")
    if esoa_dose and annex_dose:
        if abs(esoa_dose - annex_dose) < 0.01:
            score += 5  # Exact match
        elif abs(esoa_dose - annex_dose) / max(esoa_dose, annex_dose) < 0.1:
            score += 2  # Close match
        else:
            score -= 3  # Dose mismatch penalty
    
    # Form match
    esoa_form = esoa_parsed.get("form", "")
    annex_form = annex_parsed.get("form", "")
    if esoa_form and annex_form:
        if esoa_form == annex_form:
            score += 4
        else:
            # Check for compatible forms
            compatible_forms = {
                ("AMPULE", "VIAL"): 2,
                ("VIAL", "AMPULE"): 2,
                ("TABLET", "CAPSULE"): 1,
                ("CAPSULE", "TABLET"): 1,
                ("BOTTLE", "VIAL"): 1,
                ("VIAL", "BOTTLE"): 1,
            }
            score += compatible_forms.get((esoa_form, annex_form), -1)
    
    # Route match
    esoa_route = esoa_parsed.get("route", "")
    annex_route = annex_parsed.get("route", "")
    if esoa_route and annex_route:
        if esoa_route == annex_route:
            score += 3
        else:
            # Check for compatible routes
            compatible_routes = {
                ("INTRAVENOUS", "INTRAMUSCULAR"): 1,
                ("INTRAMUSCULAR", "INTRAVENOUS"): 1,
                ("INTRAMUSCULAR", "SUBCUTANEOUS"): 1,
                ("SUBCUTANEOUS", "INTRAMUSCULAR"): 1,
            }
            score += compatible_routes.get((esoa_route, annex_route), -2)
    
    # Volume match (for solutions)
    esoa_vol = esoa_parsed.get("volume")
    annex_vol = annex_parsed.get("volume")
    if esoa_vol and annex_vol:
        if abs(esoa_vol - annex_vol) < 1:
            score += 2
    
    return score


def prepare_esoa_data(esoa_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare ESOA data with parsed components."""
    records = []
    for _, row in esoa_df.iterrows():
        desc = row.get("DESCRIPTION", "")
        parsed = parse_esoa_description(desc)
        records.append({
            "ITEM_NUMBER": row.get("ITEM_NUMBER"),
            "ITEM_REF_CODE": row.get("ITEM_REF_CODE"),
            "DESCRIPTION": desc,
            "IS_OFFICIAL": row.get("IS_OFFICIAL"),
            "parsed_generic": parsed["generic"],
            "parsed_brand": parsed["brand"],
            "parsed_dose_value": parsed["dose_value"],
            "parsed_dose_unit": parsed["dose_unit"],
            "parsed_form": parsed["form"],
            "parsed_route": parsed["route"],
            "parsed_tokens": "|".join(parsed["tokens"]),
        })
    return pd.DataFrame(records)


def prepare_annex_f_with_parsed(annex_df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed components to Annex F data."""
    records = []
    for _, row in annex_df.iterrows():
        desc = row.get("Drug Description", "")
        parsed = parse_annex_f_description(desc)
        records.append({
            **row.to_dict(),
            "parsed_generic": parsed["generic"],
            "parsed_dose_value": parsed["dose_value"],
            "parsed_dose_unit": parsed["dose_unit"],
            "parsed_form": parsed["form"],
            "parsed_route": parsed["route"],
            "parsed_volume": parsed["volume"],
            "parsed_volume_unit": parsed["volume_unit"],
            "parsed_tokens": "|".join(parsed["tokens"]),
        })
    return pd.DataFrame(records)


def match_esoa_to_annex_f(
    esoa_df: pd.DataFrame,
    annex_atc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match ESOA rows to Annex F Drug Codes.
    
    Strategy:
    1. Parse ESOA description
    2. Find candidate Annex F rows with matching ATC/DrugBank ID
    3. Score candidates by dose, form, route similarity
    4. Pick best match
    """
    # Build ATC -> Annex F lookup
    atc_to_annex = defaultdict(list)
    drugbank_to_annex = defaultdict(list)
    
    for _, row in annex_atc_df.iterrows():
        atc = _as_str_or_empty(row.get("atc_code"))
        db_id = _as_str_or_empty(row.get("drugbank_id"))
        if atc:
            for code in atc.split("|"):
                atc_to_annex[code.strip()].append(row.to_dict())
        if db_id:
            drugbank_to_annex[db_id].append(row.to_dict())
    
    # Parse Annex F descriptions
    annex_parsed_cache = {}
    for _, row in annex_atc_df.iterrows():
        drug_code = row.get("Drug Code")
        desc = row.get("Drug Description", "")
        annex_parsed_cache[drug_code] = parse_annex_f_description(desc)
    
    # Match each ESOA row
    results = []
    for _, esoa_row in esoa_df.iterrows():
        desc = esoa_row.get("DESCRIPTION", "")
        esoa_parsed = parse_esoa_description(desc)
        
        # Find candidates by generic name matching
        candidates = []
        esoa_generics = set(esoa_parsed.get("generic", "").split())
        
        # Search through all Annex F rows for generic overlap
        for _, annex_row in annex_atc_df.iterrows():
            annex_parsed = annex_parsed_cache.get(annex_row.get("Drug Code"), {})
            annex_generics = set(annex_parsed.get("generic", "").split())
            
            if esoa_generics & annex_generics:
                score = score_annex_f_candidate(esoa_parsed, annex_parsed)
                if score > 0:
                    candidates.append((score, annex_row.to_dict(), annex_parsed))
        
        # Sort by score and pick best
        candidates.sort(key=lambda x: -x[0])
        
        result = {
            "ITEM_NUMBER": esoa_row.get("ITEM_NUMBER"),
            "ITEM_REF_CODE": esoa_row.get("ITEM_REF_CODE"),
            "DESCRIPTION": desc,
            "IS_OFFICIAL": esoa_row.get("IS_OFFICIAL"),
            "parsed_generic": esoa_parsed.get("generic"),
            "parsed_dose_mg": esoa_parsed.get("dose_value"),
            "parsed_form": esoa_parsed.get("form"),
            "parsed_route": esoa_parsed.get("route"),
            "matched_drug_code": None,
            "matched_drug_description": None,
            "matched_atc_code": None,
            "matched_drugbank_id": None,
            "match_score": None,
            "candidate_count": len(candidates),
        }
        
        if candidates:
            best_score, best_annex, best_parsed = candidates[0]
            result["matched_drug_code"] = best_annex.get("Drug Code")
            result["matched_drug_description"] = best_annex.get("Drug Description")
            result["matched_atc_code"] = best_annex.get("atc_code")
            result["matched_drugbank_id"] = best_annex.get("drugbank_id")
            result["match_score"] = best_score
        
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    esoa_df = pd.read_csv(INPUTS_DIR / "esoa_combined.csv")
    annex_atc_df = pd.read_csv(OUTPUTS_DIR / "annex_f_with_atc.csv")
    
    print(f"ESOA rows: {len(esoa_df)}")
    print(f"Annex F rows: {len(annex_atc_df)}")
    
    # Prepare ESOA data
    print("\nPreparing ESOA data...")
    esoa_prepared = prepare_esoa_data(esoa_df)
    esoa_prepared.to_csv(OUTPUTS_DIR / "esoa_prepared.csv", index=False)
    print(f"Saved to {OUTPUTS_DIR / 'esoa_prepared.csv'}")
    
    # Prepare Annex F with parsed data
    print("\nPreparing Annex F with parsed components...")
    annex_parsed = prepare_annex_f_with_parsed(annex_atc_df)
    annex_parsed.to_csv(OUTPUTS_DIR / "annex_f_parsed.csv", index=False)
    print(f"Saved to {OUTPUTS_DIR / 'annex_f_parsed.csv'}")
    
    # Match ESOA to Annex F
    print("\nMatching ESOA to Annex F...")
    matched = match_esoa_to_annex_f(esoa_df, annex_atc_df)
    matched.to_csv(OUTPUTS_DIR / "esoa_matched.csv", index=False)
    print(f"Saved to {OUTPUTS_DIR / 'esoa_matched.csv'}")
    
    # Summary
    total = len(matched)
    with_match = matched["matched_drug_code"].notna().sum()
    print(f"\n=== RESULTS ===")
    print(f"Total ESOA rows: {total}")
    print(f"Matched to Drug Code: {with_match} ({100*with_match/total:.1f}%)")
    print(f"Unmatched: {total - with_match}")
    
    # Sample matches
    print("\n=== SAMPLE MATCHES ===")
    sample = matched[matched["matched_drug_code"].notna()].head(10)
    for _, row in sample.iterrows():
        print(f"ESOA: {row['DESCRIPTION'][:50]}")
        print(f"  -> {row['matched_drug_code']}: {row['matched_drug_description'][:50] if row['matched_drug_description'] else 'N/A'}")
        print(f"  Score: {row['match_score']}, ATC: {row['matched_atc_code']}")
        print()


if __name__ == "__main__":
    main()
