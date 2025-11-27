#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 4: Bridge ESOA rows to Annex F Drug Codes via ATC/DrugBank ID.

This script:
- Loads ESOA rows with ATC/DrugBank IDs (from Part 3)
- Loads Annex F rows with ATC/DrugBank IDs (from Part 2)
- For each ESOA row, finds Annex F candidates with matching ATC/DrugBank ID
- Scores candidates by: exact dose, exact route, flexible form, all generics match
- Selects the best Drug Code for each ESOA row

Matching Rules:
- DOSE: Must be exact match for a Drug Code match
- ROUTE: Must be exact match
- FORM: Flexible (tablet/capsule both oral), but prefer exact matches
- GENERICS: All ESOA generics must match all Annex F generics (no more, no less)

Prerequisites:
- Run Part 2 (annex_f_atc) to have Annex F tagged with ATC/DrugBank IDs
- Run Part 3 (esoa_atc) to have ESOA tagged with ATC/DrugBank IDs
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from pipelines.drugs.scripts.spinner import run_with_spinner

PROJECT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"

# Forms that are considered equivalent (both oral solid dosage forms)
EQUIVALENT_FORMS = {
    frozenset({"tablet", "capsule"}),
    frozenset({"ampule", "vial"}),
    frozenset({"solution", "suspension"}),
}

# Form abbreviation normalization
FORM_NORMALIZE = {
    "tab": "tablet",
    "tabs": "tablet",
    "cap": "capsule",
    "caps": "capsule",
    "amp": "ampule",
    "amps": "ampule",
    "ampoule": "ampule",
    "ampul": "ampule",
    "vl": "vial",
    "inj": "injection",
    "soln": "solution",
    "sol": "solution",
    "susp": "suspension",
    "supp": "suppository",
    "supps": "suppository",
    "crm": "cream",
    "oint": "ointment",
    "neb": "nebule",
    "nebs": "nebule",
    "nebules": "nebule",
    "sach": "sachet",
    "sachet": "sachet",
    "sachets": "sachet",
    "gtts": "drops",
    "drop": "drops",
    "pwdr": "powder",
    "gran": "granule",
    "granules": "granule",
    "loz": "lozenge",
    "lozenges": "lozenge",
}

# Route inference from form
FORM_TO_ROUTE = {
    "tablet": "oral",
    "capsule": "oral",
    "syrup": "oral",
    "sachet": "oral",
    "granule": "oral",
    "elixir": "oral",
    "suspension": "oral",
    "ampule": "intravenous",
    "vial": "intravenous",
    "injection": "intravenous",
    "cream": "topical",
    "ointment": "topical",
    "lotion": "topical",
    "gel": "topical",
    "suppository": "rectal",
    "mdi": "inhalation",
    "dpi": "inhalation",
    "inhaler": "inhalation",
    "nebule": "inhalation",
    "patch": "transdermal",
}


def _safe_str(val) -> str:
    """Convert value to string, handling NaN/None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _safe_float(val) -> Optional[float]:
    """Convert value to float, handling NaN/None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_list(val) -> list[str]:
    """Parse a string representation of a list or pipe-separated values."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    # Try ast.literal_eval for Python list format
    if s.startswith("["):
        try:
            result = ast.literal_eval(s)
            if isinstance(result, list):
                return [str(x).strip().lower() for x in result if x]
        except (ValueError, SyntaxError):
            pass
    # Fall back to pipe-separated
    return [x.strip().lower() for x in s.split("|") if x.strip()]


# Generic name synonyms for normalization
GENERIC_SYNONYMS = {
    "paracetamol": "acetaminophen",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "frusemide": "furosemide",
    "lignocaine": "lidocaine",
    "salbutamol": "albuterol",
    "ciclosporin": "cyclosporine",
    "ciclosporine": "cyclosporine",
    "rifampicin": "rifampin",
    "phenobarbitone": "phenobarbital",
    "chlorpheniramine": "chlorphenamine",
    "pethidine": "meperidine",
    "beclometasone": "beclomethasone",
    "aluminium": "aluminum",
    "sulphate": "sulfate",
}


def _normalize_generics(generics: list[str]) -> set[str]:
    """Normalize generic names for comparison."""
    normalized = set()
    for g in generics:
        # Remove salt forms and normalize
        g = g.lower().strip()
        
        # Skip non-generic tokens
        if g in ("+", "-", "/", "&", "and", "with"):
            continue
        
        # Remove common suffixes
        for suffix in (" hydrochloride", " hcl", " sodium", " potassium", " sulfate", " acetate", 
                       " maleate", " fumarate", " tartrate", " citrate", " phosphate", " chloride"):
            if g.endswith(suffix):
                g = g[:-len(suffix)].strip()
        
        # Apply synonym normalization
        g = GENERIC_SYNONYMS.get(g, g)
        
        # Handle multi-word generics (e.g., "tranexamic acid" -> "tranexamic acid")
        # Don't split these, keep as single token
        if g:
            normalized.add(g)
    return normalized


def _normalize_annex_generics(generic_str: str) -> set[str]:
    """Normalize Annex F generic names which may have + separators."""
    if not generic_str:
        return set()
    
    # Split by common separators
    parts = re.split(r'\s*[+/&]\s*|\s+and\s+|\s+with\s+', generic_str.lower())
    
    normalized = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Remove common suffixes
        for suffix in (" hydrochloride", " hcl", " sodium", " potassium", " sulfate", " acetate",
                       " maleate", " fumarate", " tartrate", " citrate", " phosphate", " chloride"):
            if part.endswith(suffix):
                part = part[:-len(suffix)].strip()
        
        # Apply synonym normalization
        part = GENERIC_SYNONYMS.get(part, part)
        
        if part:
            normalized.add(part)
    
    return normalized


def _normalize_dose(strength: Optional[float], unit: str) -> Optional[float]:
    """Normalize dose to mg."""
    if strength is None:
        return None
    unit = unit.lower().strip()
    if unit == "mg":
        return strength
    elif unit == "g":
        return strength * 1000
    elif unit in ("mcg", "ug"):
        return strength / 1000
    elif unit == "%":
        return strength  # Keep as percentage
    return strength


def _normalize_form(form: str) -> str:
    """Normalize form abbreviations to canonical form."""
    f = form.lower().strip()
    return FORM_NORMALIZE.get(f, f)


def _forms_compatible(form1: str, form2: str) -> bool:
    """Check if two forms are compatible (equivalent)."""
    f1 = _normalize_form(form1)
    f2 = _normalize_form(form2)
    if f1 == f2:
        return True
    for equiv_set in EQUIVALENT_FORMS:
        if f1 in equiv_set and f2 in equiv_set:
            return True
    return False


def _infer_route(form: str) -> str:
    """Infer route from form."""
    return FORM_TO_ROUTE.get(form.lower().strip(), "")


def build_annex_f_index(annex_df: pd.DataFrame) -> tuple[dict, dict]:
    """Build ATC → Annex F and DrugBank ID → Annex F lookup indexes."""
    atc_to_annex = defaultdict(list)
    drugbank_to_annex = defaultdict(list)

    for _, row in annex_df.iterrows():
        row_dict = row.to_dict()
        atc = _safe_str(row.get("atc_code"))
        db_id = _safe_str(row.get("drugbank_id"))

        if atc:
            # Handle pipe-separated ATC codes
            for code in atc.split("|"):
                code = code.strip()
                if code:
                    atc_to_annex[code].append(row_dict)

        if db_id:
            drugbank_to_annex[db_id].append(row_dict)

    return dict(atc_to_annex), dict(drugbank_to_annex)


def find_annex_f_candidates(
    esoa_row: dict,
    atc_to_annex: dict,
    drugbank_to_annex: dict,
) -> list[dict]:
    """Find Annex F candidates for an ESOA row based on ATC/DrugBank ID."""
    candidates = []
    seen_drug_codes = set()

    # Try ATC code (check multiple possible column names)
    atc = _safe_str(esoa_row.get("atc_code_final") or esoa_row.get("probable_atc") or esoa_row.get("who_atc_codes"))
    if atc:
        for code in atc.split("|"):
            code = code.strip()
            if code:
                for annex_row in atc_to_annex.get(code, []):
                    drug_code = annex_row.get("Drug Code")
                    if drug_code and drug_code not in seen_drug_codes:
                        seen_drug_codes.add(drug_code)
                        candidates.append(annex_row)

    # Also try DrugBank ID (check multiple possible column names)
    db_id = _safe_str(esoa_row.get("drugbank_id") or esoa_row.get("drugbank_generics_list"))
    if db_id:
        # Handle if it's a list
        db_ids = _parse_list(db_id) if "[" in db_id else [db_id]
        for did in db_ids:
            did = did.strip()
            if did.startswith("DB"):
                for annex_row in drugbank_to_annex.get(did, []):
                    drug_code = annex_row.get("Drug Code")
                    if drug_code and drug_code not in seen_drug_codes:
                        seen_drug_codes.add(drug_code)
                        candidates.append(annex_row)

    return candidates


def score_annex_f_candidate(esoa_row: dict, annex_row: dict) -> tuple[float, str]:
    """
    Score how well an Annex F row matches an ESOA row.
    
    Returns (score, reason) tuple.
    
    Scoring rules:
    - GENERICS: Must match exactly (all ESOA generics = all Annex F generics)
    - DOSE: Must be exact for a match
    - ROUTE: Must be exact
    - FORM: Flexible, but prefer exact matches
    """
    reasons = []
    
    # === GENERICS CHECK (REQUIRED: all must match, no more no less) ===
    esoa_generics_raw = _parse_list(esoa_row.get("molecules_recognized_list") or esoa_row.get("generic_final"))
    annex_generic_str = _safe_str(annex_row.get("matched_generic_name"))
    
    esoa_generics = _normalize_generics(esoa_generics_raw)
    annex_generics = _normalize_annex_generics(annex_generic_str)
    
    if not esoa_generics or not annex_generics:
        return -1000, "missing_generics"
    
    # All generics must match exactly (no more, no less)
    if esoa_generics != annex_generics:
        missing_in_annex = esoa_generics - annex_generics
        extra_in_annex = annex_generics - esoa_generics
        if missing_in_annex or extra_in_annex:
            return -1000, f"generic_mismatch:esoa={esoa_generics},annex={annex_generics}"
    
    score = 100.0  # Base score for generic match
    
    # === DOSE CHECK (REQUIRED: exact match) ===
    esoa_strength = _safe_float(esoa_row.get("selected_strength") or esoa_row.get("selected_strength_mg"))
    esoa_unit = _safe_str(esoa_row.get("selected_unit") or "mg")
    esoa_dose_mg = _normalize_dose(esoa_strength, esoa_unit)
    
    # Parse Annex F dose from description or matched_reference_raw
    annex_desc = _safe_str(annex_row.get("Drug Description"))
    annex_dose_mg = None
    dose_match = re.search(r'(\d+(?:\.\d+)?)\s*(MG|MCG|G|%)', annex_desc, re.IGNORECASE)
    if dose_match:
        annex_strength = float(dose_match.group(1))
        annex_unit = dose_match.group(2).upper()
        annex_dose_mg = _normalize_dose(annex_strength, annex_unit)
    
    if esoa_dose_mg is not None and annex_dose_mg is not None:
        # Exact dose match required
        if abs(esoa_dose_mg - annex_dose_mg) < 0.001:
            score += 50  # Exact dose match bonus
            reasons.append("dose_exact")
        else:
            return -1000, f"dose_mismatch:esoa={esoa_dose_mg}mg,annex={annex_dose_mg}mg"
    elif esoa_dose_mg is not None and annex_dose_mg is None:
        reasons.append("annex_dose_unknown")
    elif esoa_dose_mg is None and annex_dose_mg is not None:
        reasons.append("esoa_dose_unknown")
    
    # === ROUTE CHECK (REQUIRED: exact match) ===
    esoa_route = _safe_str(esoa_row.get("route") or esoa_row.get("selected_route_allowed")).lower()
    
    # Infer Annex F route from description or form
    annex_route = ""
    annex_desc_upper = annex_desc.upper()
    if "ORAL" in annex_desc_upper:
        annex_route = "oral"
    elif "INTRAVENOUS" in annex_desc_upper or "IV" in annex_desc_upper:
        annex_route = "intravenous"
    elif "INTRAMUSCULAR" in annex_desc_upper or "IM" in annex_desc_upper:
        annex_route = "intramuscular"
    elif "SUBCUTANEOUS" in annex_desc_upper or "SC" in annex_desc_upper:
        annex_route = "subcutaneous"
    elif "TOPICAL" in annex_desc_upper:
        annex_route = "topical"
    elif "OPHTHALMIC" in annex_desc_upper or "EYE" in annex_desc_upper:
        annex_route = "ophthalmic"
    elif "OTIC" in annex_desc_upper or "EAR" in annex_desc_upper:
        annex_route = "otic"
    elif "NASAL" in annex_desc_upper:
        annex_route = "nasal"
    elif "INHALATION" in annex_desc_upper:
        annex_route = "inhalation"
    elif "RECTAL" in annex_desc_upper:
        annex_route = "rectal"
    elif "VAGINAL" in annex_desc_upper:
        annex_route = "vaginal"
    
    # Infer from form if not explicit
    if not annex_route:
        for form_kw, route in FORM_TO_ROUTE.items():
            if form_kw.upper() in annex_desc_upper:
                annex_route = route
                break
    
    if esoa_route and annex_route:
        if esoa_route == annex_route:
            score += 30  # Route match bonus
            reasons.append("route_exact")
        else:
            return -1000, f"route_mismatch:esoa={esoa_route},annex={annex_route}"
    elif esoa_route and not annex_route:
        reasons.append("annex_route_unknown")
    elif not esoa_route and annex_route:
        reasons.append("esoa_route_unknown")
    
    # === FORM CHECK (FLEXIBLE: prefer exact, accept compatible) ===
    esoa_form_raw = _safe_str(esoa_row.get("form") or esoa_row.get("selected_form")).lower()
    esoa_form = _normalize_form(esoa_form_raw)
    
    # Extract form from Annex F description
    annex_form = ""
    form_keywords = ["TABLET", "CAPSULE", "AMPULE", "VIAL", "SYRUP", "SUSPENSION", 
                     "SOLUTION", "CREAM", "OINTMENT", "GEL", "SUPPOSITORY", "SACHET",
                     "INJECTION", "DROPS", "SPRAY", "PATCH", "POWDER", "GRANULE"]
    for fk in form_keywords:
        if fk in annex_desc_upper:
            annex_form = fk.lower()
            break
    
    if esoa_form and annex_form:
        if esoa_form == annex_form:
            score += 20  # Exact form match bonus
            reasons.append("form_exact")
        elif _forms_compatible(esoa_form, annex_form):
            score += 10  # Compatible form bonus
            reasons.append(f"form_compatible:{esoa_form}~{annex_form}")
        else:
            score -= 10  # Form mismatch penalty (but not disqualifying)
            reasons.append(f"form_mismatch:{esoa_form}!={annex_form}")
    
    return score, "|".join(reasons) if reasons else "matched"


def match_esoa_to_annex_f(
    esoa_df: pd.DataFrame,
    atc_to_annex: dict,
    drugbank_to_annex: dict,
) -> pd.DataFrame:
    """Match each ESOA row to the best Annex F Drug Code."""
    results = []

    for idx, esoa_row in esoa_df.iterrows():
        esoa_dict = esoa_row.to_dict()

        # Find candidates via ATC/DrugBank ID
        candidates = find_annex_f_candidates(esoa_dict, atc_to_annex, drugbank_to_annex)

        # Score and select best
        best_score = -float("inf")
        best_annex = None
        best_reason = "no_candidates"

        for annex_row in candidates:
            score, reason = score_annex_f_candidate(esoa_dict, annex_row)
            if score > best_score:
                best_score = score
                best_annex = annex_row
                best_reason = reason

        # Build result row - preserve key ESOA columns
        result = {
            "esoa_idx": esoa_dict.get("esoa_idx", idx),
            "raw_text": esoa_dict.get("raw_text"),
            "molecules_recognized": esoa_dict.get("molecules_recognized"),
            "selected_strength_mg": esoa_dict.get("selected_strength_mg"),
            "selected_form": esoa_dict.get("selected_form"),
            "route": esoa_dict.get("route"),
            "atc_code_final": esoa_dict.get("atc_code_final"),
            "candidate_count": len(candidates),
            "matched_drug_code": best_annex.get("Drug Code") if best_annex and best_score > 0 else None,
            "matched_drug_description": best_annex.get("Drug Description") if best_annex and best_score > 0 else None,
            "matched_atc_code": best_annex.get("atc_code") if best_annex and best_score > 0 else None,
            "matched_drugbank_id": best_annex.get("drugbank_id") if best_annex and best_score > 0 else None,
            "match_score": best_score if best_score > 0 else None,
            "match_reason": best_reason,
        }
        results.append(result)

    return pd.DataFrame(results)


def run_part_4(
    esoa_atc_filename: str = "esoa_with_atc.csv",
    annex_atc_filename: str = "annex_f_with_atc.csv",
    out_filename: str = "esoa_matched_drug_codes.csv",
    standalone: bool = True,
) -> dict:
    """
    Run Part 4: Bridge ESOA to Annex F Drug Codes.
    
    Returns dict with results summary.
    """
    if standalone:
        print("=" * 60)
        print("Part 4: Bridge ESOA to Annex F Drug Codes")
        print("=" * 60)

    # Resolve paths
    esoa_atc_path = OUTPUTS_DIR / esoa_atc_filename
    annex_atc_path = OUTPUTS_DIR / annex_atc_filename
    out_path = OUTPUTS_DIR / out_filename

    if not esoa_atc_path.exists():
        raise FileNotFoundError(f"ESOA with ATC not found: {esoa_atc_path}. Run Part 3 first.")

    if not annex_atc_path.exists():
        raise FileNotFoundError(f"Annex F with ATC not found: {annex_atc_path}. Run Part 2 first.")

    # Load data
    esoa_df = run_with_spinner("Load ESOA with ATC", lambda: pd.read_csv(esoa_atc_path))
    annex_df = run_with_spinner("Load Annex F with ATC", lambda: pd.read_csv(annex_atc_path))

    # Build index
    atc_to_annex, drugbank_to_annex = run_with_spinner(
        "Build Annex F index", lambda: build_annex_f_index(annex_df)
    )

    # Match
    matched_df = run_with_spinner(
        "Match ESOA to Annex F Drug Codes",
        lambda: match_esoa_to_annex_f(esoa_df, atc_to_annex, drugbank_to_annex),
    )

    # Save output
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    run_with_spinner("Write output CSV", lambda: matched_df.to_csv(out_path, index=False))

    # Summary
    total = len(matched_df)
    matched = matched_df["matched_drug_code"].notna().sum()

    results = {
        "total": total,
        "matched": matched,
        "matched_pct": 100 * matched / total if total else 0,
        "unmatched": total - matched,
        "atc_codes_indexed": len(atc_to_annex),
        "drugbank_ids_indexed": len(drugbank_to_annex),
        "output_path": out_path,
    }

    if standalone:
        print(f"\n  ATC codes indexed: {len(atc_to_annex)}")
        print(f"  DrugBank IDs indexed: {len(drugbank_to_annex)}")
        print(f"  Total ESOA rows: {total}")
        print(f"  Matched to Drug Code: {matched} ({results['matched_pct']:.1f}%)")
        print(f"  Unmatched: {total - matched}")
        print(f"  Output: {out_path}")

    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Part 4: Bridge ESOA rows to Annex F Drug Codes via ATC/DrugBank ID."
    )
    parser.add_argument(
        "--esoa-atc",
        metavar="PATH",
        default="esoa_with_atc.csv",
        help="ESOA with ATC/DrugBank IDs (default: outputs/drugs/esoa_with_atc.csv).",
    )
    parser.add_argument(
        "--annex-atc",
        metavar="PATH",
        default="annex_f_with_atc.csv",
        help="Annex F with ATC/DrugBank IDs (default: outputs/drugs/annex_f_with_atc.csv).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default="esoa_matched_drug_codes.csv",
        help="Output filename (default: esoa_matched_drug_codes.csv).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_part_4(
        esoa_atc_filename=args.esoa_atc,
        annex_atc_filename=args.annex_atc,
        out_filename=args.out,
        standalone=True,
    )
    
    print("\nPipeline complete!")


if __name__ == "__main__":
    main(sys.argv[1:])
