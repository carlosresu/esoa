"""
Main tagger interface for drug descriptions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import duckdb
import pandas as pd

from .constants import PURE_SALT_COMPOUNDS, UNIT_TOKENS
from .lookup import (
    apply_synonym, batch_lookup_generics, build_combination_keys,
    load_generics_lookup, load_synonyms,
)
from .scoring import select_best_candidate, sort_atc_codes
from .tokenizer import (
    categorize_tokens, detect_compound_salts, extract_generic_tokens,
    normalize_tokens, split_with_parentheses, strip_salt_suffix,
)


# Default paths
PROJECT_DIR = Path(__file__).resolve().parents[4]
INPUTS_DIR = PROJECT_DIR / "inputs" / "drugs"
OUTPUTS_DIR = PROJECT_DIR / "outputs" / "drugs"


class UnifiedTagger:
    """
    Unified drug tagger for both Annex F and ESOA.
    
    Usage:
        tagger = UnifiedTagger()
        tagger.load()
        results = tagger.tag_descriptions(df, text_column="Drug Description")
        tagger.close()
    """
    
    def __init__(
        self,
        outputs_dir: Optional[Path] = None,
        inputs_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        self.outputs_dir = Path(outputs_dir or os.environ.get("PIPELINE_OUTPUTS_DIR", OUTPUTS_DIR))
        self.inputs_dir = Path(inputs_dir or os.environ.get("PIPELINE_INPUTS_DIR", INPUTS_DIR))
        self.verbose = verbose
        
        self.con: Optional[duckdb.DuckDBPyConnection] = None
        self.synonyms: Dict[str, str] = {}
        self.multiword_generics: Set[str] = set()
        self._loaded = False
    
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[UnifiedTagger] {msg}")
    
    def load(self) -> None:
        """Load reference data into DuckDB."""
        if self._loaded:
            return
        
        self._log("Loading reference data...")
        
        # Create in-memory DuckDB
        self.con = duckdb.connect(":memory:")
        
        # Load generics
        generics_path = self.outputs_dir / "generics_lookup.parquet"
        if generics_path.exists():
            self.con.execute(f"CREATE TABLE generics AS SELECT * FROM read_parquet('{generics_path}')")
            count = self.con.execute("SELECT COUNT(*) FROM generics").fetchone()[0]
            self._log(f"  - generics: {count:,} rows")
        
        # Load synonyms
        generics_df = load_generics_lookup(self.outputs_dir)
        self.synonyms = load_synonyms(generics_df)
        self._log(f"  - synonyms: {len(self.synonyms):,} entries")
        
        # Build multiword generics set
        self.multiword_generics = set()
        for name in generics_df["generic_name"]:
            if " " in str(name):
                self.multiword_generics.add(str(name).upper())
        
        # Add common multi-word generics
        self.multiword_generics.update({
            "ISOSORBIDE MONONITRATE", "ISOSORBIDE DINITRATE",
            "TRANEXAMIC ACID", "FOLIC ACID", "ASCORBIC ACID", "VALPROIC ACID",
            "ACETYLSALICYLIC ACID", "HYALURONIC ACID", "RETINOIC ACID",
            "SODIUM CHLORIDE", "POTASSIUM CHLORIDE", "CALCIUM CHLORIDE",
            "MAGNESIUM SULFATE", "FERROUS SULFATE", "ZINC SULFATE",
            "INSULIN GLARGINE", "INSULIN LISPRO", "INSULIN ASPART",
        })
        
        self._loaded = True
        self._log("Reference data loaded.")
    
    def _apply_synonyms(self, generic: str) -> str:
        return apply_synonym(generic, self.synonyms)
    
    def _strip_salt(self, generic: str) -> tuple:
        return strip_salt_suffix(generic)
    
    def tag_single(self, text: str) -> Dict[str, Any]:
        """Tag a single drug description."""
        if not self._loaded:
            self.load()
        
        results = self._tag_batch([text], [0])
        return results[0] if results else {
            "atc_code": None,
            "drugbank_id": None,
            "generic_name": None,
            "reference_text": None,
            "match_score": 0,
            "match_reason": "error",
        }
    
    def tag_descriptions(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Tag all descriptions in a DataFrame.
        
        Returns DataFrame with tagging results.
        """
        if not self._loaded:
            self.load()
        
        texts = df[text_column].fillna("").astype(str).tolist()
        
        if id_column and id_column in df.columns:
            ids = df[id_column].tolist()
        else:
            ids = list(range(len(df)))
        
        results = self._tag_batch(texts, ids)
        return pd.DataFrame(results)
    
    def _tag_batch(
        self,
        texts: List[str],
        ids: List[Any],
    ) -> List[Dict[str, Any]]:
        """Tag a batch of texts."""
        total = len(texts)
        
        # Pre-process all texts
        all_tokens = []
        all_generic_tokens = []
        
        for text in texts:
            tokens, generic_tokens = extract_generic_tokens(text, self.multiword_generics)
            all_tokens.append(tokens)
            all_generic_tokens.append(generic_tokens)
        
        # Collect unique generics for batch lookup
        unique_generics: Set[str] = set()
        for gt in all_generic_tokens:
            for g in gt:
                if g.upper() in PURE_SALT_COMPOUNDS:
                    unique_generics.add(g.upper())
                else:
                    base, _ = self._strip_salt(g)
                    unique_generics.add(base)
                unique_generics.add(self._apply_synonyms(g))
            
            # Add combination keys
            combo_keys = build_combination_keys(gt)
            unique_generics.update(combo_keys)
        
        # Batch lookup
        generic_cache = batch_lookup_generics(unique_generics, self.con, self.synonyms)
        
        # Process each text
        results = []
        for i, text in enumerate(texts):
            tokens = all_tokens[i]
            generic_tokens = all_generic_tokens[i]
            
            # Get stripped generics
            stripped_generics = []
            for g in generic_tokens:
                if g.upper() in PURE_SALT_COMPOUNDS:
                    stripped_generics.append(g.upper())
                else:
                    base, _ = self._strip_salt(g)
                    stripped_generics.append(base)
            
            # Collect matches
            generic_matches = []
            for sg in stripped_generics:
                if sg in generic_cache:
                    generic_matches.extend(generic_cache[sg])
                syn = self._apply_synonyms(sg)
                if syn in generic_cache and syn != sg:
                    generic_matches.extend(generic_cache[syn])
            
            # Add combination matches
            combo_keys = build_combination_keys(stripped_generics)
            for ck in combo_keys:
                if ck in generic_cache:
                    generic_matches.extend(generic_cache[ck])
            
            # Deduplicate
            seen = set()
            unique_matches = []
            for m in generic_matches:
                key = m.get("generic_name", "")
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(m)
            
            if not unique_matches:
                results.append({
                    "id": ids[i],
                    "input_text": text,
                    "row_idx": i,
                    "atc_code": None,
                    "drugbank_id": None,
                    "generic_name": "|".join(stripped_generics) if stripped_generics else None,
                    "reference_text": None,
                    "match_score": 0,
                    "match_reason": "no_candidates",
                    "sources": "",
                })
                continue
            
            # Build candidates
            categories = categorize_tokens(tokens)
            candidates = []
            for gm in unique_matches:
                atc_codes = str(gm.get("atc_code", "")).split("|")
                atc_codes = sort_atc_codes(atc_codes)
                
                for atc in atc_codes:
                    if atc:
                        candidates.append({
                            "atc_code": atc,
                            "drugbank_id": gm.get("drugbank_id"),
                            "generic_name": gm.get("generic_name"),
                            "reference_text": gm.get("reference_text", ""),
                            "source": gm.get("source"),
                            "form": "",
                            "route": "",
                            "doses": "",
                        })
            
            if not candidates:
                results.append({
                    "id": ids[i],
                    "input_text": text,
                    "row_idx": i,
                    "atc_code": None,
                    "drugbank_id": None,
                    "generic_name": "|".join(stripped_generics) if stripped_generics else None,
                    "reference_text": None,
                    "match_score": 0,
                    "match_reason": "no_candidates",
                    "sources": "",
                })
                continue
            
            # Normalize input generics
            input_generics_normalized = set()
            for sg in stripped_generics:
                normalized = self._apply_synonyms(sg.upper())
                if normalized and normalized not in {"+", "MG/5"}:
                    input_generics_normalized.add(normalized)
            
            num_input = len(input_generics_normalized)
            has_plus = "+" in text
            has_in = " IN " in text.upper() and num_input > 1
            is_iv_solution = has_in and not has_plus
            is_combination = num_input > 1 and has_plus
            is_single_drug = num_input == 1
            
            # Select best candidate
            best = select_best_candidate(
                candidates=candidates,
                input_tokens=tokens,
                input_categories=categories,
                input_generics_normalized=input_generics_normalized,
                is_single_drug=is_single_drug,
                is_combination=is_combination,
                is_iv_solution=is_iv_solution,
                stripped_generics=stripped_generics,
                apply_synonyms_fn=self._apply_synonyms,
            )
            
            if best:
                results.append({
                    "id": ids[i],
                    "input_text": text,
                    "row_idx": i,
                    "atc_code": best.get("atc_code"),
                    "drugbank_id": best.get("drugbank_id"),
                    "generic_name": best.get("generic_name"),
                    "reference_text": best.get("reference_text", ""),
                    "match_score": 1,
                    "match_reason": "matched",
                    "sources": best.get("source", ""),
                })
            else:
                results.append({
                    "id": ids[i],
                    "input_text": text,
                    "row_idx": i,
                    "atc_code": None,
                    "drugbank_id": None,
                    "generic_name": None,
                    "reference_text": None,
                    "match_score": 0,
                    "match_reason": "no_match",
                    "sources": "",
                })
        
        return results
    
    def close(self) -> None:
        """Close DuckDB connection."""
        if self.con:
            self.con.close()
            self.con = None
            self._loaded = False


# Convenience functions
def tag_descriptions(
    df: pd.DataFrame,
    text_column: str,
    id_column: Optional[str] = None,
    outputs_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Tag drug descriptions in a DataFrame."""
    tagger = UnifiedTagger(outputs_dir=outputs_dir)
    tagger.load()
    results = tagger.tag_descriptions(df, text_column, id_column)
    tagger.close()
    return results


def tag_single(text: str, outputs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Tag a single drug description."""
    tagger = UnifiedTagger(outputs_dir=outputs_dir)
    tagger.load()
    result = tagger.tag_single(text)
    tagger.close()
    return result
