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


def load_synonyms(
    generics_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Load synonyms from generics lookup.
    
    Synonyms are rows where generic_name != canonical_name.
    """
    synonyms: Dict[str, str] = {}
    
    # Add hardcoded synonyms for common variations
    hardcoded = {
        # Plural -> singular
        "VITAMINS": "VITAMIN",
        "VITAMINS FAT-SOLUBLE": "VITAMIN FAT-SOLUBLE",
        "VITAMINS WATER-SOLUBLE": "VITAMIN WATER-SOLUBLE",
        "VITAMINS INTRAVENOUS, FAT-SOLUBLE": "VITAMIN INTRAVENOUS, FAT-SOLUBLE",
        "VITAMINS INTRAVENOUS, WATER-SOLUBLE": "VITAMIN INTRAVENOUS, WATER-SOLUBLE",
        "VITAMINS INTRAVENOUS FAT-SOLUBLE": "VITAMIN INTRAVENOUS, FAT-SOLUBLE",
        "VITAMINS INTRAVENOUS WATER-SOLUBLE": "VITAMIN INTRAVENOUS, WATER-SOLUBLE",
        # Regional/spelling variants
        "ADRENALINE": "EPINEPHRINE",
        "FRUSEMIDE": "FUROSEMIDE",
        "LIGNOCAINE": "LIDOCAINE",
        "PARACETAMOL": "ACETAMINOPHEN",
        "SALBUTAMOL": "ALBUTEROL",
        # Common abbreviations
        "VIT": "VITAMIN",
        "VIT A": "VITAMIN A",
        "VIT B": "VITAMIN B",
        "VIT C": "VITAMIN C",
        "VIT D": "VITAMIN D",
        "VIT E": "VITAMIN E",
        "VIT K": "VITAMIN K",
    }
    synonyms.update(hardcoded)
    
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


def apply_synonym(
    generic: str,
    synonyms: Dict[str, str],
) -> str:
    """Apply synonym mapping to a generic name."""
    generic_upper = generic.upper()
    return synonyms.get(generic_upper, generic_upper)


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


def batch_lookup_generics(
    tokens: Set[str],
    con: duckdb.DuckDBPyConnection,
    synonyms: Optional[Dict[str, str]] = None,
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
