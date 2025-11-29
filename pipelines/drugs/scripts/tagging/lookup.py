"""
Reference data lookup functions for drug tagging.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import duckdb
import pandas as pd

from .constants import PURE_SALT_COMPOUNDS
from .tokenizer import strip_salt_suffix


# Default paths
PROJECT_DIR = Path(__file__).resolve().parents[4]
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


def load_generics_lookup(
    outputs_dir: Optional[Path] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Load generics lookup table."""
    if outputs_dir is None:
        outputs_dir = Path(os.environ.get("PIPELINE_OUTPUTS_DIR", OUTPUTS_DIR))
    
    path = outputs_dir / "generics_lookup.parquet"
    if not path.exists():
        path = outputs_dir / "generics_lookup.csv"
    
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_brands_lookup(
    outputs_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load brands lookup table."""
    if outputs_dir is None:
        outputs_dir = Path(os.environ.get("PIPELINE_OUTPUTS_DIR", OUTPUTS_DIR))
    
    path = outputs_dir / "brands_lookup.parquet"
    if not path.exists():
        path = outputs_dir / "brands_lookup.csv"
    
    if not path.exists():
        return pd.DataFrame(columns=["brand_name", "generic_name", "drugbank_id", "source"])
    
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_brand_to_generic_map(
    brands_df: pd.DataFrame,
    generics_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    """
    Build a dictionary mapping brand names to generic names.
    
    Used to swap brand names to generics before matching.
    Excludes entries where the brand_name is actually a known generic.
    """
    # Build set of known generic names to exclude
    known_generics: Set[str] = set()
    if generics_df is not None and "generic_name" in generics_df.columns:
        known_generics = set(generics_df["generic_name"].str.upper().dropna())
    
    brand_map: Dict[str, str] = {}
    
    for _, row in brands_df.iterrows():
        brand = str(row.get("brand_name", "")).upper().strip()
        generic = str(row.get("generic_name", "")).upper().strip()
        
        if brand and generic and brand != generic:
            # Skip if brand_name is actually a known generic (data error)
            if brand in known_generics:
                continue
            # Don't overwrite existing entries (first wins)
            if brand not in brand_map:
                brand_map[brand] = generic
    
    return brand_map


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


def load_synonyms(
    generics_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Load synonyms from generics lookup.
    
    Synonyms are rows where generic_name != canonical_name.
    """
    from .unified_constants import SPELLING_SYNONYMS
    
    synonyms: Dict[str, str] = {}
    
    # Load spelling synonyms from unified_constants.py
    # NOTE: Regional name variants (ADRENALINE->EPINEPHRINE, PARACETAMOL->ACETAMINOPHEN, etc.)
    # should be in the unified reference dataset (generics_master.parquet), not here.
    synonyms.update(SPELLING_SYNONYMS)
    
    # Load from generics lookup
    if "canonical_name" in generics_df.columns:
        synonym_rows = generics_df[
            generics_df["generic_name"].str.upper() != generics_df["canonical_name"].str.upper()
        ]
        synonyms.update(dict(zip(
            synonym_rows["generic_name"].str.upper(),
            synonym_rows["canonical_name"].str.upper()
        )))
    
    return synonyms


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
    """Exact match lookup for a generic token."""
    query = """
        SELECT DISTINCT generic_name, drugbank_id, atc_code, source, reference_text
        FROM generics
        WHERE UPPER(generic_name) = ?
    """
    try:
        df = con.execute(query, [token.upper()]).fetchdf()
        return df.to_dict("records")
    except Exception:
        return []


def lookup_generic_prefix(
    token: str,
    con: duckdb.DuckDBPyConnection,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Prefix match lookup for a generic token."""
    query = """
        SELECT DISTINCT generic_name, drugbank_id, atc_code, source, reference_text
        FROM generics
        WHERE UPPER(generic_name) LIKE ?
        ORDER BY LENGTH(generic_name) ASC
        LIMIT ?
    """
    try:
        df = con.execute(query, [f"{token.upper()} %", limit]).fetchdf()
        return df.to_dict("records")
    except Exception:
        return []


def lookup_generic_contains(
    token: str,
    con: duckdb.DuckDBPyConnection,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Contains match lookup for a generic token."""
    query = """
        SELECT DISTINCT generic_name, drugbank_id, atc_code, source, reference_text
        FROM generics
        WHERE UPPER(generic_name) LIKE ?
        ORDER BY LENGTH(generic_name) ASC
        LIMIT ?
    """
    try:
        df = con.execute(query, [f"%{token.upper()}%", limit]).fetchdf()
        return df.to_dict("records")
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
    
    Args:
        token: The token to search for
        con: DuckDB connection
        threshold: Minimum similarity score (0-100), default 85
        limit: Maximum number of results
        cached_generics: Pre-loaded list of generic names (for performance)
    
    Returns:
        List of matching records with similarity scores
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return []  # rapidfuzz not installed
    
    if len(token) < 4:
        return []  # Too short for fuzzy matching
    
    # Use cached generics if provided, otherwise query
    if cached_generics is not None:
        all_generics = cached_generics
    else:
        try:
            all_generics = con.execute(
                "SELECT DISTINCT generic_name FROM generics WHERE generic_name IS NOT NULL"
            ).fetchdf()["generic_name"].tolist()
        except Exception:
            return []
    
    if not all_generics:
        return []
    
    # Find best fuzzy matches
    token_upper = token.upper()
    matches = process.extract(
        token_upper,
        all_generics,
        scorer=fuzz.ratio,
        limit=limit,
        score_cutoff=threshold,
    )
    
    if not matches:
        return []
    
    # Look up the matched generics
    results = []
    for match_name, score, _ in matches:
        records = lookup_generic_exact(match_name, con)
        for rec in records:
            rec["fuzzy_score"] = score
            rec["fuzzy_match"] = True
            results.append(rec)
    
    return results


def batch_lookup_generics(
    tokens: Set[str],
    con: duckdb.DuckDBPyConnection,
    synonyms: Optional[Dict[str, str]] = None,
    enable_fuzzy: bool = True,
    cached_generics: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Batch lookup for multiple generic tokens.
    
    Returns dict of {token: [matches]}.
    """
    if synonyms is None:
        synonyms = {}
    
    cache: Dict[str, List[Dict[str, Any]]] = {}
    
    for token in tokens:
        if not token or token in cache:
            continue
        
        token_upper = token.upper()
        
        # First try exact match
        matches = lookup_generic_exact(token_upper, con)
        if matches:
            cache[token_upper] = matches
            continue
        
        # Try synonym
        syn = synonyms.get(token_upper)
        if syn and syn != token_upper:
            matches = lookup_generic_exact(syn, con)
            if matches:
                cache[token_upper] = matches
                continue
        
        # Try prefix match
        matches = lookup_generic_prefix(token_upper, con)
        if matches:
            cache[token_upper] = matches
            continue
        
        # Try contains match (only for multi-word)
        if " " in token:
            matches = lookup_generic_contains(token_upper, con)
            if matches:
                cache[token_upper] = matches
                continue
        
        # Try fuzzy match as last resort (for potential misspellings)
        if enable_fuzzy and len(token) >= 4:
            matches = lookup_generic_fuzzy(
                token_upper, con, threshold=85, limit=1, cached_generics=cached_generics
            )
            cache[token_upper] = matches
        else:
            cache[token_upper] = []
    
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
    clean = [g for g in generic_tokens if g and g.upper() not in junk and not any(c.isdigit() for c in g)]
    
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
