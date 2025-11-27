#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Drug Tagger - Single algorithm for both Annex F and ESOA tagging.

This module provides a consistent tagging algorithm that:
1. Tokenizes drug descriptions
2. Categorizes tokens (GENERIC, SALT, DOSE, FORM, ROUTE, OTHER)
3. Queries DuckDB for matches against unified reference
4. Scores candidates using weighted algorithm
5. Assigns ATC codes and DrugBank IDs

Usage:
    from pipelines.drugs.scripts.unified_tagger import UnifiedTagger
    
    tagger = UnifiedTagger()
    results = tagger.tag_descriptions(df, text_column="Drug Description")
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"

PIPELINE_INPUTS_DIR = Path(os.environ.get("PIPELINE_INPUTS_DIR", INPUTS_DIR))
PIPELINE_OUTPUTS_DIR = Path(os.environ.get("PIPELINE_OUTPUTS_DIR", OUTPUTS_DIR))

# ============================================================================
# CONSTANTS - Token Categories
# ============================================================================

CATEGORY_GENERIC = "generic"
CATEGORY_SALT = "salt"
CATEGORY_DOSE = "dose"
CATEGORY_FORM = "form"
CATEGORY_ROUTE = "route"
CATEGORY_OTHER = "other"

# ============================================================================
# CONSTANTS - Scoring Weights (from Part 2)
# ============================================================================

PRIMARY_WEIGHTS = {
    CATEGORY_GENERIC: 5,
    CATEGORY_SALT: 4,
    CATEGORY_DOSE: 4,
    CATEGORY_FORM: 3,
    CATEGORY_ROUTE: 3,
    CATEGORY_OTHER: 1,
}

SECONDARY_WEIGHTS = {
    CATEGORY_GENERIC: 3,
    CATEGORY_SALT: 3,
    CATEGORY_DOSE: 3,
    CATEGORY_FORM: 4,
    CATEGORY_ROUTE: 4,
    CATEGORY_OTHER: 1,
}

GENERIC_MISS_PENALTY_PRIMARY = 6
GENERIC_MISS_PENALTY_SECONDARY = 4
GENERIC_MATCH_REQUIRED = True
GENERIC_REF_MISMATCH_TOLERANCE_PRIMARY = 1
GENERIC_REF_MISMATCH_TOLERANCE_SECONDARY = 1
GENERIC_REF_EXTRA_PENALTY_PRIMARY = 4
GENERIC_REF_EXTRA_PENALTY_SECONDARY = 3
DOSE_MISMATCH_PENALTY = 20
REQUIRE_DOSE_MATCH = True

# ============================================================================
# CONSTANTS - Token Normalization
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

ROUTE_CANON = {
    "PO": "ORAL", "OR": "ORAL", "ORAL": "ORAL",
    "IV": "INTRAVENOUS", "INTRAVENOUS": "INTRAVENOUS",
    "IM": "INTRAMUSCULAR", "INTRAMUSCULAR": "INTRAMUSCULAR",
    "SC": "SUBCUTANEOUS", "SUBCUTANEOUS": "SUBCUTANEOUS", "SUBCUT": "SUBCUTANEOUS",
    "NASAL": "NASAL", "TOPICAL": "TOPICAL", "RECTAL": "RECTAL",
    "OPHTHALMIC": "OPHTHALMIC", "BUCCAL": "BUCCAL",
}

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

# Pure salt compounds - should NOT have salt stripped
PURE_SALT_COMPOUNDS = {
    "SODIUM CHLORIDE", "POTASSIUM CHLORIDE", "CALCIUM CHLORIDE",
    "MAGNESIUM SULFATE", "MAGNESIUM SULPHATE", "SODIUM BICARBONATE",
    "POTASSIUM BICARBONATE", "CALCIUM CARBONATE", "CALCIUM GLUCONATE",
    "POTASSIUM PHOSPHATE", "SODIUM PHOSPHATE", "ALUMINUM HYDROXIDE",
    "MAGNESIUM HYDROXIDE", "FERROUS SULFATE", "FERROUS SULPHATE",
    "ZINC SULFATE", "ZINC SULPHATE", "SODIUM LACTATE", "CALCIUM LACTATE",
    "SODIUM CITRATE", "POTASSIUM CITRATE", "SODIUM ACETATE", "POTASSIUM ACETATE",
}

# ATC combination patterns
COMBINATION_ATC_PATTERNS = {
    "A10BD", "C09BA", "C09BB", "C09BX", "C09DA", "C09DB", "C09DX",
    "C10BA", "C10BX", "M05BB", "R03AL", "R03AK", "R03DA20", "R03DA55", "R03DB",
}
COMBINATION_ATC_SUFFIXES = {"20", "30", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59"}

# ============================================================================
# REGEX PATTERNS
# ============================================================================

_NUMERIC_RX = re.compile(r"^-?\d+(?:\.\d+)?$")
_COMBINED_WEIGHT_RX = re.compile(r"^(-?[\d,.]+)\s*(MG|G|MCG|UG|KG)$", re.IGNORECASE)
_WEIGHT_UNIT_FACTORS = {"MG": 1.0, "G": 1000.0, "MCG": 0.001, "UG": 0.001, "KG": 1_000_000.0}
_UNIT_TOKENS = {"MG", "G", "MCG", "UG", "KG", "ML", "L"}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _maybe_none(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_str_or_empty(value) -> str:
    val = _maybe_none(value)
    if val is None:
        return ""
    return str(val)


def _is_numeric_token(token: str) -> bool:
    if not isinstance(token, str):
        return False
    return bool(_NUMERIC_RX.match(token.strip()))


def _format_number(value) -> str | None:
    if value is None:
        return None
    try:
        num = float(str(value).replace(",", ""))
        if num == int(num):
            return str(int(num))
        return f"{num:.2f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return None


def _convert_to_mg(number_text: str, unit: str) -> str | None:
    try:
        num = float(str(number_text).replace(",", ""))
    except Exception:
        return None
    factor = _WEIGHT_UNIT_FACTORS.get(unit.upper())
    if factor is None:
        return None
    mg_val = num * factor
    return _format_number(mg_val)


def _is_combination_atc(atc_code: str) -> bool:
    """Check if an ATC code indicates a combination product."""
    if not atc_code:
        return False
    atc = atc_code.upper()
    for pattern in COMBINATION_ATC_PATTERNS:
        if atc.startswith(pattern):
            return True
    if len(atc) >= 7 and atc[-2:] in COMBINATION_ATC_SUFFIXES:
        return True
    return False


# ============================================================================
# TOKENIZATION
# ============================================================================

def split_with_parentheses(text: str) -> List[str]:
    """Split on spaces while keeping parentheses content together; drop commas and parens."""
    if text is None:
        return []
    chars = str(text)
    tokens: List[str] = []
    current: List[str] = []
    depth = 0

    for ch in chars:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            if depth:
                depth -= 1
            continue
        if ch.isspace() and depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(ch)

    if current:
        tokens.append("".join(current))

    cleaned = [tok.rstrip(",").strip() for tok in tokens]
    return [tok.upper() for tok in cleaned if tok]


def normalize_tokens(
    tokens: List[str],
    drop_stopwords: bool = False,
    multiword_generics: Optional[set] = None,
) -> List[str]:
    """Normalize tokens: expand, canonicalize forms/routes, handle doses."""
    multiword_generics = multiword_generics or set()
    
    expanded: List[str] = []
    for tok in tokens:
        if tok is None:
            continue
        tok_upper = str(tok).upper().strip()
        if tok_upper in multiword_generics:
            expanded.append(tok_upper)
        else:
            expanded.extend(str(tok).split())

    normalized: List[str] = []
    i = 0
    while i < len(expanded):
        raw_tok = expanded[i]
        tok_clean = raw_tok.replace(",", "").strip("()").strip()
        tok_upper = tok_clean.upper()

        # Handle percentage
        if tok_upper.endswith("%") and _is_numeric_token(tok_upper.rstrip("%")):
            pct_val = _format_number(tok_upper.rstrip("%"))
            if pct_val:
                normalized.append(pct_val)
                normalized.append("PCT")
            i += 1
            continue

        # Handle combined weight (e.g., "500MG")
        combined_match = _COMBINED_WEIGHT_RX.match(tok_upper)
        if combined_match:
            orig_val = _format_number(combined_match.group(1))
            orig_unit = combined_match.group(2).upper()
            mg_val = _convert_to_mg(combined_match.group(1), orig_unit)
            if mg_val and orig_val:
                normalized.append(orig_val)
                normalized.append(orig_unit)
                if mg_val != orig_val or orig_unit != "MG":
                    normalized.append(mg_val)
                    normalized.append("MG")
            i += 1
            continue

        # Handle separate number + unit
        if _is_numeric_token(tok_upper) and i + 1 < len(expanded):
            next_clean = expanded[i + 1].replace(",", "").strip("()").strip().upper()
            if next_clean in _WEIGHT_UNIT_FACTORS:
                orig_val = _format_number(tok_upper)
                mg_val = _convert_to_mg(tok_upper, next_clean)
                if mg_val and orig_val:
                    normalized.append(orig_val)
                    normalized.append(next_clean)
                    if mg_val != orig_val or next_clean != "MG":
                        normalized.append(mg_val)
                        normalized.append("MG")
                    i += 2
                    continue

        # Format standalone numbers
        if _is_numeric_token(tok_upper):
            tok_upper = _format_number(tok_upper) or tok_upper

        # Canonicalize forms and routes
        tok_upper = FORM_CANON.get(tok_upper, tok_upper)
        tok_upper = ROUTE_CANON.get(tok_upper, tok_upper)

        # Drop stopwords if requested
        if drop_stopwords and tok_upper in NATURAL_STOPWORDS:
            i += 1
            continue

        if tok_upper:
            normalized.append(tok_upper)
        i += 1

    return normalized


def categorize_tokens(tokens: List[str]) -> Dict[str, Counter]:
    """Categorize tokens into GENERIC, SALT, DOSE, FORM, ROUTE, OTHER."""
    categories: Dict[str, Counter] = {
        CATEGORY_GENERIC: Counter(),
        CATEGORY_SALT: Counter(),
        CATEGORY_DOSE: Counter(),
        CATEGORY_FORM: Counter(),
        CATEGORY_ROUTE: Counter(),
        CATEGORY_OTHER: Counter(),
    }
    
    for tok in tokens:
        if not tok:
            continue
        tok_upper = tok.upper()
        
        if tok_upper in FORM_CANON.values():
            categories[CATEGORY_FORM][tok_upper] += 1
        elif tok_upper in ROUTE_CANON.values():
            categories[CATEGORY_ROUTE][tok_upper] += 1
        elif tok_upper in SALT_TOKENS:
            categories[CATEGORY_SALT][tok_upper] += 1
        elif _is_numeric_token(tok_upper) or tok_upper in _UNIT_TOKENS or tok_upper == "PCT":
            categories[CATEGORY_DOSE][tok_upper] += 1
        elif tok_upper in NATURAL_STOPWORDS or tok_upper in GENERIC_JUNK_TOKENS:
            categories[CATEGORY_OTHER][tok_upper] += 1
        else:
            # Assume it's a generic name
            categories[CATEGORY_GENERIC][tok_upper] += 1
    
    return categories


# ============================================================================
# UNIFIED TAGGER CLASS
# ============================================================================

class UnifiedTagger:
    """
    Unified drug tagger using DuckDB for reference lookups.
    
    This provides a single, consistent algorithm for tagging both
    Annex F and ESOA drug descriptions.
    """
    
    def __init__(
        self,
        outputs_dir: Optional[Path] = None,
        inputs_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.outputs_dir = outputs_dir or PIPELINE_OUTPUTS_DIR
        self.inputs_dir = inputs_dir or PIPELINE_INPUTS_DIR
        self.verbose = verbose
        self.con: Optional[duckdb.DuckDBPyConnection] = None
        self.multiword_generics: set = set()
        self.generic_synonyms: Dict[str, str] = {}
        self._loaded = False
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[UnifiedTagger] {msg}")
    
    def load(self):
        """Load reference data into DuckDB."""
        if self._loaded:
            return
        
        self._log("Loading reference data...")
        self.con = duckdb.connect(":memory:")
        
        # Load unified reference
        unified_path = self.outputs_dir / "unified_drug_reference.parquet"
        if unified_path.exists():
            self.con.execute(f"""
                CREATE TABLE unified AS 
                SELECT * FROM read_parquet('{unified_path}')
            """)
            count = self.con.execute("SELECT COUNT(*) FROM unified").fetchone()[0]
            self._log(f"  - unified: {count:,} rows")
        
        # Load generics lookup
        generics_path = self.outputs_dir / "generics_lookup.parquet"
        if generics_path.exists():
            self.con.execute(f"""
                CREATE TABLE generics AS 
                SELECT * FROM read_parquet('{generics_path}')
            """)
            count = self.con.execute("SELECT COUNT(*) FROM generics").fetchone()[0]
            self._log(f"  - generics: {count:,} rows")
            
            # Build multiword generics set
            names = self.con.execute("SELECT DISTINCT generic_name FROM generics").fetchdf()
            for name in names["generic_name"]:
                if name and " " in str(name):
                    self.multiword_generics.add(str(name).upper())
        
        # Load brands lookup
        brands_path = self.outputs_dir / "brands_lookup.parquet"
        if brands_path.exists():
            self.con.execute(f"""
                CREATE TABLE brands AS 
                SELECT * FROM read_parquet('{brands_path}')
            """)
            count = self.con.execute("SELECT COUNT(*) FROM brands").fetchone()[0]
            self._log(f"  - brands: {count:,} rows")
        
        # Load mixtures lookup
        mixtures_path = self.outputs_dir / "mixtures_lookup.parquet"
        if mixtures_path.exists():
            self.con.execute(f"""
                CREATE TABLE mixtures AS 
                SELECT * FROM read_parquet('{mixtures_path}')
            """)
            count = self.con.execute("SELECT COUNT(*) FROM mixtures").fetchone()[0]
            self._log(f"  - mixtures: {count:,} rows")
        
        # Load salt suffixes
        salts_path = self.inputs_dir / "drugbank_salt_suffixes.csv"
        if salts_path.exists():
            self.con.execute(f"""
                CREATE TABLE salt_suffixes AS 
                SELECT * FROM read_csv_auto('{salts_path}')
            """)
        
        # Load pure salts
        pure_salts_path = self.inputs_dir / "drugbank_pure_salts.csv"
        if pure_salts_path.exists():
            self.con.execute(f"""
                CREATE TABLE pure_salts AS 
                SELECT * FROM read_csv_auto('{pure_salts_path}')
            """)
        
        # Load synonyms
        try:
            from .reference_synonyms import load_drugbank_synonyms
            self.generic_synonyms = load_drugbank_synonyms()
        except ImportError:
            self.generic_synonyms = {
                "PARACETAMOL": "ACETAMINOPHEN",
                "SALBUTAMOL": "ALBUTEROL",
                "ALUMINIUM": "ALUMINUM",
            }
        
        # Add multi-word generics from synonyms
        for canonical in self.generic_synonyms.values():
            if " " in canonical:
                self.multiword_generics.add(canonical.upper())
        
        # Add common multi-word generics
        self.multiword_generics.update({
            "ISOSORBIDE MONONITRATE", "ISOSORBIDE DINITRATE",
            "TRANEXAMIC ACID", "FOLIC ACID", "ASCORBIC ACID", "VALPROIC ACID",
            "ALUMINUM HYDROXIDE", "MAGNESIUM HYDROXIDE", "CALCIUM CARBONATE",
            "SODIUM CHLORIDE", "POTASSIUM CHLORIDE", "SODIUM BICARBONATE",
            "MEDROXYPROGESTERONE ACETATE", "BECLOMETHASONE DIPROPIONATE",
            "INSULIN GLARGINE", "INSULIN LISPRO", "INSULIN ASPART",
        })
        
        self._loaded = True
        self._log("Reference data loaded.")
    
    def _apply_synonyms(self, token: str) -> str:
        """Apply synonym normalization to a token."""
        return self.generic_synonyms.get(token.upper(), token.upper())
    
    def _strip_salt(self, text: str) -> Tuple[str, List[str]]:
        """
        Strip salt suffixes from a drug name.
        
        Returns (base_name, list_of_salts_stripped).
        Does NOT strip if the compound is a pure salt.
        """
        text_upper = text.upper().strip()
        
        # Check if it's a pure salt compound
        if text_upper in PURE_SALT_COMPOUNDS:
            return text_upper, []
        
        # Try to strip salt suffixes
        salts_found = []
        base = text_upper
        
        for salt in sorted(SALT_TOKENS, key=len, reverse=True):
            if base.endswith(" " + salt):
                base = base[: -(len(salt) + 1)].strip()
                salts_found.append(salt)
            elif base == salt:
                # The entire name is a salt - don't strip
                return text_upper, []
        
        # If stripping would leave nothing, don't strip
        if not base.strip():
            return text_upper, []
        
        return base, salts_found
    
    def _find_generic_matches(self, tokens: List[str]) -> List[Dict]:
        """Find matching generics from the reference."""
        if not self.con:
            return []
        
        # Apply synonyms and get unique tokens
        normalized = [self._apply_synonyms(t) for t in tokens if t]
        unique_tokens = list(set(t.upper() for t in normalized if t))
        
        if not unique_tokens:
            return []
        
        # Query generics table - use LIKE for partial matching too
        results = []
        for token in unique_tokens:
            query = """
                SELECT DISTINCT generic_name, drugbank_id, atc_code, source
                FROM generics
                WHERE UPPER(generic_name) = ?
                   OR UPPER(generic_name) LIKE ?
            """
            try:
                df = self.con.execute(query, [token, f"%{token}%"]).fetchdf()
                results.extend(df.to_dict("records"))
            except Exception:
                pass
        
        # Deduplicate by generic_name
        seen = set()
        unique_results = []
        for r in results:
            key = r.get("generic_name", "")
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results
    
    def _find_brand_matches(self, tokens: List[str]) -> List[Dict]:
        """Find matching brands and return their generic mappings."""
        if not self.con:
            return []
        
        unique_tokens = list(set(t.upper() for t in tokens if t))
        
        if not unique_tokens:
            return []
        
        placeholders = ", ".join(["?" for _ in unique_tokens])
        query = f"""
            SELECT DISTINCT brand_name, generic_name, drugbank_id, source
            FROM brands
            WHERE UPPER(brand_name) IN ({placeholders})
        """
        
        try:
            result = self.con.execute(query, unique_tokens).fetchdf()
            return result.to_dict("records")
        except Exception:
            return []
    
    def _find_mixture_matches(self, generic_tokens: List[str]) -> List[Dict]:
        """Find matching mixtures based on component generics."""
        if not self.con or len(generic_tokens) < 2:
            return []
        
        # Build component key
        component_key = "||".join(sorted(set(t.upper() for t in generic_tokens)))
        
        query = """
            SELECT *
            FROM mixtures
            WHERE component_key = ?
        """
        
        try:
            result = self.con.execute(query, [component_key]).fetchdf()
            return result.to_dict("records")
        except Exception:
            return []
    
    def _score_candidate(
        self,
        input_tokens: List[str],
        input_categories: Dict[str, Counter],
        ref_row: Dict,
    ) -> Tuple[float, float, str]:
        """
        Score a reference candidate against input tokens.
        
        Returns (primary_score, secondary_score, reason).
        """
        # Get reference tokens
        ref_generic = _as_str_or_empty(ref_row.get("generic_name"))
        ref_form = _as_str_or_empty(ref_row.get("form"))
        ref_route = _as_str_or_empty(ref_row.get("route"))
        ref_doses = _as_str_or_empty(ref_row.get("doses"))
        
        ref_tokens = []
        if ref_generic:
            ref_tokens.extend(ref_generic.upper().split())
        if ref_form:
            ref_tokens.append(ref_form.upper())
        if ref_route:
            ref_tokens.append(ref_route.upper())
        
        ref_categories = categorize_tokens(ref_tokens)
        
        # Calculate primary score
        input_generic_count = sum(input_categories.get(CATEGORY_GENERIC, Counter()).values())
        ref_generic_count = sum(ref_categories.get(CATEGORY_GENERIC, Counter()).values())
        
        # Generic overlap - also check synonyms
        input_generics = set(input_categories.get(CATEGORY_GENERIC, {}).keys())
        ref_generics = set(ref_categories.get(CATEGORY_GENERIC, {}).keys())
        
        # Normalize both sides with synonyms for comparison
        input_generics_normalized = set()
        for g in input_generics:
            input_generics_normalized.add(self._apply_synonyms(g))
        ref_generics_normalized = set()
        for g in ref_generics:
            ref_generics_normalized.add(self._apply_synonyms(g))
        
        generic_overlap = len(input_generics_normalized & ref_generics_normalized)
        
        # If no direct overlap, check if input is contained in ref or vice versa
        if generic_overlap == 0:
            for ig in input_generics_normalized:
                for rg in ref_generics_normalized:
                    if ig in rg or rg in ig:
                        generic_overlap = 1
                        break
                if generic_overlap > 0:
                    break
        
        # Special case: if the ref_generic matches the full input text (for pure salts, multi-word)
        if generic_overlap == 0 and ref_generic:
            ref_generic_upper = ref_generic.upper()
            # Check if any input generic is part of the ref generic name
            for ig in input_generics:
                if ig in ref_generic_upper:
                    generic_overlap = 1
                    break
        
        if GENERIC_MATCH_REQUIRED and input_generic_count > 0 and generic_overlap == 0:
            return -1000, 0, "no_generic_overlap"
        
        # Calculate category scores
        primary_score = 0.0
        for cat, weight in PRIMARY_WEIGHTS.items():
            input_counts = input_categories.get(cat, Counter())
            ref_counts = ref_categories.get(cat, Counter())
            match_count = sum((input_counts & ref_counts).values())
            mismatch = max(0, sum(ref_counts.values()) - match_count)
            
            mismatch_penalty = mismatch
            if cat == CATEGORY_GENERIC:
                excess = max(0, mismatch - GENERIC_REF_MISMATCH_TOLERANCE_PRIMARY)
                mismatch_penalty += excess * GENERIC_REF_EXTRA_PENALTY_PRIMARY
            
            primary_score += weight * match_count - mismatch_penalty
        
        # Penalty for missing generics
        generic_missing = max(0, input_generic_count - generic_overlap)
        if input_generic_count > 0:
            primary_score -= GENERIC_MISS_PENALTY_PRIMARY * generic_missing
        
        # Secondary score (form/route matching)
        secondary_score = 0.0
        for cat, weight in SECONDARY_WEIGHTS.items():
            input_counts = input_categories.get(cat, Counter())
            ref_counts = ref_categories.get(cat, Counter())
            match_count = sum((input_counts & ref_counts).values())
            secondary_score += weight * match_count
        
        return primary_score, secondary_score, "scored"
    
    def tag_single(self, text: str) -> Dict[str, Any]:
        """
        Tag a single drug description.
        
        Returns dict with:
        - atc_code: Best matching ATC code
        - drugbank_id: Best matching DrugBank ID
        - generic_name: Matched generic name
        - match_score: Primary score
        - match_reason: Reason for match/no-match
        - sources: Pipe-delimited sources
        """
        if not self._loaded:
            self.load()
        
        # Tokenize
        raw_tokens = split_with_parentheses(text)
        tokens = normalize_tokens(raw_tokens, drop_stopwords=True, multiword_generics=self.multiword_generics)
        
        if not tokens:
            return {
                "atc_code": None,
                "drugbank_id": None,
                "generic_name": None,
                "match_score": 0,
                "match_reason": "no_tokens",
                "sources": "",
            }
        
        # Categorize tokens
        categories = categorize_tokens(tokens)
        generic_tokens = list(categories.get(CATEGORY_GENERIC, {}).keys())
        
        # Check for multi-word generics that might have been split
        # Reconstruct potential multi-word generics from consecutive tokens
        text_upper = text.upper()
        for mwg in self.multiword_generics:
            if mwg in text_upper and mwg not in generic_tokens:
                generic_tokens.append(mwg)
        
        # Also check for pure salt compounds
        for psc in PURE_SALT_COMPOUNDS:
            if psc in text_upper and psc not in generic_tokens:
                generic_tokens.append(psc)
        
        # Try brand→generic swap
        brand_matches = self._find_brand_matches(tokens)
        if brand_matches:
            for bm in brand_matches:
                generic_name = bm.get("generic_name")
                if generic_name:
                    # Add the generic to our tokens
                    generic_tokens.append(generic_name.upper())
        
        # Strip salts from generics (but not pure salt compounds)
        stripped_generics = []
        for g in generic_tokens:
            if g.upper() in PURE_SALT_COMPOUNDS:
                stripped_generics.append(g.upper())
            else:
                base, _ = self._strip_salt(g)
                stripped_generics.append(base)
        
        # Find generic matches
        generic_matches = self._find_generic_matches(stripped_generics)
        
        # Check for mixtures if multiple generics
        mixture_matches = []
        if len(set(stripped_generics)) >= 2:
            mixture_matches = self._find_mixture_matches(stripped_generics)
        
        # Build candidate list from unified reference
        candidates = []
        
        # Add generic matches
        for gm in generic_matches:
            atc_codes = _as_str_or_empty(gm.get("atc_code")).split("|")
            for atc in atc_codes:
                if atc:
                    candidates.append({
                        "atc_code": atc,
                        "drugbank_id": gm.get("drugbank_id"),
                        "generic_name": gm.get("generic_name"),
                        "source": gm.get("source"),
                        "form": "",
                        "route": "",
                        "doses": "",
                    })
        
        # Add mixture matches
        for mm in mixture_matches:
            candidates.append({
                "atc_code": None,
                "drugbank_id": mm.get("drugbank_id"),
                "generic_name": mm.get("components"),
                "source": "mixture",
                "form": "",
                "route": "",
                "doses": "",
            })
        
        if not candidates:
            return {
                "atc_code": None,
                "drugbank_id": None,
                "generic_name": "|".join(stripped_generics) if stripped_generics else None,
                "match_score": 0,
                "match_reason": "no_candidates",
                "sources": "",
            }
        
        # Score candidates
        best_score = -float("inf")
        best_candidate = None
        
        for cand in candidates:
            primary, secondary, reason = self._score_candidate(tokens, categories, cand)
            total_score = primary + secondary * 0.1  # Secondary is tie-breaker
            
            # ATC preference: prefer single ATCs for single drugs
            is_single_drug = len(set(stripped_generics)) == 1
            is_combo_atc = _is_combination_atc(_as_str_or_empty(cand.get("atc_code")))
            if is_single_drug and is_combo_atc:
                total_score -= 5  # Penalize combo ATCs for single drugs
            elif not is_single_drug and not is_combo_atc:
                total_score -= 5  # Penalize single ATCs for combo drugs
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = cand
        
        if best_candidate:
            return {
                "atc_code": best_candidate.get("atc_code"),
                "drugbank_id": best_candidate.get("drugbank_id"),
                "generic_name": best_candidate.get("generic_name"),
                "match_score": best_score,
                "match_reason": "matched",
                "sources": best_candidate.get("source", ""),
            }
        
        return {
            "atc_code": None,
            "drugbank_id": None,
            "generic_name": None,
            "match_score": 0,
            "match_reason": "no_match",
            "sources": "",
        }
    
    def tag_descriptions(
        self,
        df: pd.DataFrame,
        text_column: str = "Drug Description",
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Tag all drug descriptions in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing drug descriptions
            id_column: Optional ID column to preserve
        
        Returns:
            DataFrame with tagging results
        """
        if not self._loaded:
            self.load()
        
        results = []
        total = len(df)
        
        for idx, row in df.iterrows():
            text = _as_str_or_empty(row.get(text_column))
            result = self.tag_single(text)
            
            if id_column and id_column in row:
                result["id"] = row[id_column]
            result["input_text"] = text
            result["row_idx"] = idx
            
            results.append(result)
            
            if self.verbose and (idx + 1) % 500 == 0:
                self._log(f"  Processed {idx + 1:,}/{total:,} rows")
        
        return pd.DataFrame(results)
    
    def close(self):
        """Close DuckDB connection."""
        if self.con:
            self.con.close()
            self.con = None
            self._loaded = False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Test the tagger
    tagger = UnifiedTagger()
    tagger.load()
    
    test_descriptions = [
        "PARACETAMOL 500MG TABLET",
        "AMLODIPINE BESYLATE 10MG TAB",
        "LOSARTAN POTASSIUM 50MG TABLET",
        "METFORMIN HYDROCHLORIDE 500MG TABLET",
        "SODIUM CHLORIDE 0.9% SOLUTION 1L",
        "AMOXICILLIN + CLAVULANIC ACID 625MG TABLET",
    ]
    
    print("\n=== Testing Unified Tagger ===\n")
    for desc in test_descriptions:
        result = tagger.tag_single(desc)
        print(f"Input: {desc}")
        print(f"  ATC: {result['atc_code']}")
        print(f"  DrugBank: {result['drugbank_id']}")
        print(f"  Generic: {result['generic_name']}")
        print(f"  Score: {result['match_score']}")
        print(f"  Reason: {result['match_reason']}")
        print()
    
    tagger.close()
