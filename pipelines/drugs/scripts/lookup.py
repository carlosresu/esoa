"""
Reference data lookup functions for drug tagging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
import pandas as pd

from .unified_constants import PURE_SALT_COMPOUNDS
from .tokenizer import strip_salt_suffix

# Import rapidfuzz at module level for performance
try:
    from rapidfuzz import fuzz, process as rapidfuzz_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    fuzz = None
    rapidfuzz_process = None


# Default paths
PROJECT_DIR = Path(__file__).resolve().parents[4]
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def swap_brand_to_generic(
    token: str,
    brand_map: Dict[str, str],
) -> tuple:
    """
    Swap a brand name to its generic equivalent.
    
    Returns:
        (swapped_name, was_swapped): Tuple of (result, bool indicating if swap occurred)
    """
    token_upper = token.upper().strip()
    
    if token_upper in brand_map:
        return brand_map[token_upper], True
    
    return token_upper, False


def _singularize(word: str) -> str:
    """Convert a plural word to singular form."""
    word_upper = word.upper()
    
    # Common plural endings
    if word_upper.endswith("IES"):
        return word_upper[:-3] + "Y"
    elif word_upper.endswith("ES") and len(word_upper) > 3:
        # Handle -ES endings (e.g., BOXES -> BOX, but not DOSES)
        if word_upper[-3] in "SXZH":
            return word_upper[:-2]
        return word_upper[:-1]  # Just remove S
    elif word_upper.endswith("S") and not word_upper.endswith("SS"):
        return word_upper[:-1]
    
    return word_upper


def apply_synonym(
    generic: str,
    synonyms: Dict[str, str],
) -> str:
    """Apply synonym mapping to a generic name, including plural->singular."""
    generic_upper = generic.upper()
    
    # First check explicit synonyms
    if generic_upper in synonyms:
        return synonyms[generic_upper]
    
    # Try singularizing the first word if it looks plural
    words = generic_upper.split()
    if words and words[0].endswith("S") and not words[0].endswith("SS"):
        singular_first = _singularize(words[0])
        singular_name = " ".join([singular_first] + words[1:])
        
        # Check if singular form is in synonyms
        if singular_name in synonyms:
            return synonyms[singular_name]
        
        # Return singular form (it might match in lookup)
        return singular_name
    
    return generic_upper


def lookup_generic_exact(
    token: str,
    con: duckdb.DuckDBPyConnection,
) -> List[Dict[str, Any]]:
    """Exact match lookup for a generic token using unified + atc tables."""
    query = """
        SELECT DISTINCT u.generic_name, u.drugbank_id, a.atc_code, u.source,
               u.generic_name as reference_text
        FROM unified u
        LEFT JOIN atc a ON u.generic_name = a.generic_name
        WHERE UPPER(u.generic_name) = ?
    """
    try:
        rows = con.execute(query, [token.upper()]).fetchall()
        cols = ["generic_name", "drugbank_id", "atc_code", "source", "reference_text"]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


def lookup_generic_prefix(
    token: str,
    con: duckdb.DuckDBPyConnection,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Prefix match lookup for a generic token using unified + atc tables."""
    query = """
        SELECT DISTINCT u.generic_name, u.drugbank_id, a.atc_code, u.source,
               u.generic_name as reference_text
        FROM unified u
        LEFT JOIN atc a ON u.generic_name = a.generic_name
        WHERE UPPER(u.generic_name) LIKE ?
        ORDER BY LENGTH(u.generic_name) ASC
        LIMIT ?
    """
    try:
        rows = con.execute(query, [f"{token.upper()} %", limit]).fetchall()
        cols = ["generic_name", "drugbank_id", "atc_code", "source", "reference_text"]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


def lookup_generic_contains(
    token: str,
    con: duckdb.DuckDBPyConnection,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Contains match lookup for a generic token using unified + atc tables."""
    # Join unified_generics with unified_atc to get ATC codes
    query = """
        SELECT DISTINCT u.generic_name, u.drugbank_id, a.atc_code, u.source,
               u.generic_name as reference_text
        FROM unified u
        LEFT JOIN atc a ON u.generic_name = a.generic_name
        WHERE UPPER(u.generic_name) LIKE ?
        ORDER BY LENGTH(u.generic_name) ASC
        LIMIT ?
    """
    try:
        rows = con.execute(query, [f"%{token.upper()}%", limit]).fetchall()
        cols = ["generic_name", "drugbank_id", "atc_code", "source", "reference_text"]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        return []


def lookup_generic_fuzzy(
    token: str,
    con: duckdb.DuckDBPyConnection,
    threshold: int = 85,
    limit: int = 3,
    cached_generics: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fuzzy match lookup for a generic token using rapidfuzz.
    """
    if not RAPIDFUZZ_AVAILABLE:
        return []
    
    if len(token) < 4:
        return []
    
    # Use cached generics if provided
    if cached_generics is None or not cached_generics:
        return []  # Require pre-loaded cache for performance
    
    # Find best fuzzy matches
    token_upper = token.upper()
    matches = rapidfuzz_process.extract(
        token_upper,
        cached_generics,
        scorer=fuzz.ratio,
        limit=limit,
        score_cutoff=threshold,
    )
    
    if not matches:
        return []
    
    # Batch lookup matched generics
    match_names = [m[0] for m in matches]
    match_scores = {m[0]: m[1] for m in matches}
    
    placeholders = ",".join(["?" for _ in match_names])
    query = f"""
        SELECT DISTINCT u.generic_name, u.drugbank_id, a.atc_code, u.source,
               u.generic_name as reference_text
        FROM unified u
        LEFT JOIN atc a ON u.generic_name = a.generic_name
        WHERE u.generic_name IN ({placeholders})
    """
    try:
        rows = con.execute(query, match_names).fetchall()
        cols = ["generic_name", "drugbank_id", "atc_code", "source", "reference_text"]
        results = []
        for row in rows:
            rec = dict(zip(cols, row))
            rec["fuzzy_score"] = match_scores.get(rec.get("generic_name"), 0)
            rec["fuzzy_match"] = True
            results.append(rec)
        return results
    except Exception:
        return []


def batch_lookup_generics(
    tokens: Set[str],
    con: duckdb.DuckDBPyConnection,
    synonyms: Optional[Dict[str, str]] = None,
    enable_fuzzy: bool = True,
    cached_generics: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Batch lookup for multiple generic tokens using optimized SQL.
    
    Returns dict of {token: [matches]}.
    """
    if synonyms is None:
        synonyms = {}
    
    cache: Dict[str, List[Dict[str, Any]]] = {}
    if not tokens:
        return cache
    
    # Normalize tokens
    token_list = [t.upper() for t in tokens if t]
    if not token_list:
        return cache
    
    # Also add synonyms to lookup
    all_lookups = set(token_list)
    for t in token_list:
        syn = synonyms.get(t)
        if syn and syn != t:
            all_lookups.add(syn)
    
    # BATCH EXACT MATCH - single SQL query for all tokens
    if all_lookups:
        placeholders = ",".join(["?" for _ in all_lookups])
        query = f"""
            SELECT generic_name, drugbank_id, atc_code, source,
                   generic_name as reference_text
            FROM unified
            WHERE UPPER(generic_name) IN ({placeholders})
        """
        try:
            rows = con.execute(query, list(all_lookups)).fetchall()
            cols = ["generic_name", "drugbank_id", "atc_code", "source", "reference_text"]
            # Group by generic_name
            for row in rows:
                rec = dict(zip(cols, row))
                gn = rec.get("generic_name", "")
                gn_upper = gn.upper() if gn else ""
                if gn_upper not in cache:
                    cache[gn_upper] = []
                cache[gn_upper].append(rec)
        except Exception:
            pass
    
    # Map synonyms back to original tokens
    for t in token_list:
        if t in cache:
            continue
        syn = synonyms.get(t)
        if syn and syn in cache:
            cache[t] = cache[syn]
    
    # For tokens still missing, try prefix/fuzzy (slower path)
    missing = [t for t in token_list if t not in cache]
    
    for token in missing:
        # Try prefix match
        matches = lookup_generic_prefix(token, con, limit=3)
        if matches:
            cache[token] = matches
            continue
        
        # Try fuzzy match (last resort)
        if enable_fuzzy and len(token) >= 4:
            matches = lookup_generic_fuzzy(
                token, con, threshold=85, limit=1, cached_generics=cached_generics
            )
            cache[token] = matches
        else:
            cache[token] = []
    
    return cache


def build_combination_keys(
    generic_tokens: List[str],
) -> List[str]:
    """
    Build combination lookup keys from generic tokens.
    
    E.g., ["ALUMINUM HYDROXIDE", "MAGNESIUM HYDROXIDE"] -> ["ALUMINUM + MAGNESIUM"]
    """
    # Filter junk
    junk = {"+", "MG/5", "MG", "G", "MCG", "ML", "L", "PCT"}
    clean = []
    for g in generic_tokens:
        if not g:
            continue
        g_upper = g.upper()
        # Skip junk tokens
        if g_upper in junk:
            continue
        # Skip tokens with digits (doses)
        if any(c.isdigit() for c in g):
            continue
        # Skip tokens with parentheses (brand names)
        if "(" in g or ")" in g:
            continue
        # Strip trailing + from tokens like "SALBUTAMOL+"
        g_clean = g_upper.rstrip("+").strip()
        if not g_clean:
            continue
        # Split values that contain + (with or without spaces)
        # e.g., "SALBUTAMOL SULFATE + IPRATROPIUM BROMIDE" or "IBUPROFEN+PARACETAMOL"
        if "+" in g_clean:
            # Split on + with optional surrounding spaces
            import re
            parts = re.split(r'\s*\+\s*', g_clean)
            for part in parts:
                part = part.strip()
                if part and part not in junk:
                    clean.append(part)
        else:
            clean.append(g_clean)
    
    if len(clean) < 2:
        return []
    
    # Strip salt suffixes (including HYDROXIDE, CHLORIDE, etc.)
    salt_suffixes = {"HYDROXIDE", "CHLORIDE", "SULFATE", "SULPHATE", "CARBONATE", "PHOSPHATE", "ACETATE", "CITRATE"}
    base_parts = []
    for part in clean:
        base = part.upper()
        # First try standard salt stripping
        stripped, _ = strip_salt_suffix(base)
        # Also strip common compound suffixes
        for suffix in salt_suffixes:
            if stripped.endswith(" " + suffix):
                stripped = stripped[:-len(suffix)-1].strip()
                break
        if stripped:
            base_parts.append(stripped)
    
    if len(base_parts) < 2:
        return []
    
    # Deduplicate while preserving order
    seen = set()
    unique_parts = []
    for p in base_parts:
        if p not in seen:
            seen.add(p)
            unique_parts.append(p)
    
    if len(unique_parts) < 2:
        return []
    
    # Build keys in multiple formats
    keys = set()
    sorted_parts = sorted(unique_parts)
    
    # Format: "A + B" (sorted)
    keys.add(" + ".join(sorted_parts))
    # Format: "A AND B" (sorted, WHO style)
    keys.add(" AND ".join(sorted_parts))
    # Format: "B AND A" (reverse sorted, WHO style)
    keys.add(" AND ".join(sorted_parts[::-1]))
    # Format: "A, B AND C" (WHO style for 3+ components)
    if len(sorted_parts) > 2:
        keys.add(", ".join(sorted_parts[:-1]) + " AND " + sorted_parts[-1])
    
    return list(keys)
