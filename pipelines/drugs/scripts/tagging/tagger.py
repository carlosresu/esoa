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
    build_brand_to_generic_map, load_brands_lookup, load_generics_lookup,
    load_synonyms, swap_brand_to_generic,
)
from .scoring import select_best_candidate, sort_atc_codes
from .tokenizer import (
    categorize_tokens, detect_compound_salts, extract_generic_tokens,
    extract_form_detail, extract_release_detail, extract_type_detail,
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
        self.brand_map: Dict[str, str] = {}
        self.multiword_generics: Set[str] = set()
        self.cached_generics_list: List[str] = []
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
        
        # Cache generics list for fuzzy matching performance
        self.cached_generics_list = generics_df["generic_name"].dropna().tolist()
        
        # Load brands for brand → generic swapping
        # Pass generics_df to exclude known generic names from brand map
        brands_df = load_brands_lookup(self.outputs_dir)
        self.brand_map = build_brand_to_generic_map(brands_df, generics_df)
        self._log(f"  - brands: {len(self.brand_map):,} entries")
        
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
        
        # Add plural forms of multiword generics for detection
        plural_forms = set()
        for mw in self.multiword_generics:
            words = mw.split()
            if words and not words[0].endswith("S"):
                # Add plural form (e.g., VITAMIN -> VITAMINS)
                plural_first = words[0] + "S"
                plural_forms.add(" ".join([plural_first] + words[1:]))
        self.multiword_generics.update(plural_forms)
        
        self._loaded = True
        self._log("Reference data loaded.")
    
    def _apply_synonyms(self, generic: str) -> str:
        return apply_synonym(generic, self.synonyms)
    
    def _swap_brand(self, token: str) -> tuple:
        """Swap brand to generic if found in brand_map."""
        return swap_brand_to_generic(token, self.brand_map)
    
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
    
    def tag_batch(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: Optional[str] = None,
        chunk_size: int = 10000,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Tag descriptions in a DataFrame using chunked processing.
        
        Processes data in chunks of `chunk_size` rows for better memory
        efficiency and progress reporting on large datasets.
        
        Args:
            df: Input DataFrame
            text_column: Column containing drug descriptions
            id_column: Optional column for row IDs
            chunk_size: Number of rows per chunk (default 10K)
            show_progress: Whether to print progress updates
        
        Returns:
            DataFrame with tagging results
        """
        import time
        
        if not self._loaded:
            self.load()
        
        total_rows = len(df)
        if total_rows == 0:
            return pd.DataFrame()
        
        texts = df[text_column].fillna("").astype(str).tolist()
        
        if id_column and id_column in df.columns:
            ids = df[id_column].tolist()
        else:
            ids = list(range(len(df)))
        
        # Process in chunks
        all_results = []
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        start_time = time.time()
        
        for i in range(0, total_rows, chunk_size):
            chunk_start = time.time()
            chunk_num = i // chunk_size + 1
            end_idx = min(i + chunk_size, total_rows)
            
            chunk_texts = texts[i:end_idx]
            chunk_ids = ids[i:end_idx]
            
            chunk_results = self._tag_batch(chunk_texts, chunk_ids)
            all_results.extend(chunk_results)
            
            if show_progress:
                chunk_time = time.time() - chunk_start
                elapsed = time.time() - start_time
                rows_done = end_idx
                rate = rows_done / elapsed if elapsed > 0 else 0
                eta = (total_rows - rows_done) / rate if rate > 0 else 0
                
                self._log(
                    f"  Chunk {chunk_num}/{num_chunks}: "
                    f"{rows_done:,}/{total_rows:,} rows "
                    f"({chunk_time:.1f}s, {rate:.0f} rows/s, ETA {eta:.0f}s)"
                )
        
        total_time = time.time() - start_time
        if show_progress:
            self._log(
                f"  Total: {total_rows:,} rows in {total_time:.1f}s "
                f"({total_rows/total_time:.0f} rows/s)"
            )
        
        return pd.DataFrame(all_results)
    
    def benchmark(
        self,
        df: pd.DataFrame,
        text_column: str,
        chunk_sizes: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark tagging performance with different chunk sizes.
        
        Args:
            df: Input DataFrame (will use first 50K rows)
            text_column: Column containing drug descriptions
            chunk_sizes: List of chunk sizes to test (default [5000, 10000, 15000])
        
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        if chunk_sizes is None:
            chunk_sizes = [5000, 10000, 15000]
        
        # Use subset for benchmark
        sample_size = min(50000, len(df))
        sample_df = df.head(sample_size).copy()
        
        results = {
            "sample_size": sample_size,
            "chunk_results": [],
        }
        
        for chunk_size in chunk_sizes:
            self._log(f"Benchmarking chunk_size={chunk_size}...")
            
            start = time.time()
            _ = self.tag_batch(
                sample_df,
                text_column,
                chunk_size=chunk_size,
                show_progress=False,
            )
            elapsed = time.time() - start
            
            rate = sample_size / elapsed
            results["chunk_results"].append({
                "chunk_size": chunk_size,
                "time_seconds": round(elapsed, 2),
                "rows_per_second": round(rate, 0),
            })
            
            self._log(f"  chunk_size={chunk_size}: {elapsed:.2f}s ({rate:.0f} rows/s)")
        
        # Find optimal
        best = max(results["chunk_results"], key=lambda x: x["rows_per_second"])
        results["optimal_chunk_size"] = best["chunk_size"]
        results["optimal_rate"] = best["rows_per_second"]
        
        self._log(f"Optimal: chunk_size={best['chunk_size']} ({best['rows_per_second']:.0f} rows/s)")
        
        return results
    
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
        all_brand_swaps = []  # Track which tokens were brand-swapped
        
        for text in texts:
            tokens, generic_tokens = extract_generic_tokens(text, self.multiword_generics)
            
            # Apply brand → generic swapping
            swapped_generics = []
            brand_swaps = []
            for g in generic_tokens:
                swapped, was_swapped = self._swap_brand(g)
                swapped_generics.append(swapped)
                if was_swapped:
                    brand_swaps.append((g, swapped))
            
            all_tokens.append(tokens)
            all_generic_tokens.append(swapped_generics)
            all_brand_swaps.append(brand_swaps)
        
        # Collect unique generics for batch lookup
        unique_generics: Set[str] = set()
        for gt in all_generic_tokens:
            # Normalize each component through synonyms
            normalized_components = []
            for g in gt:
                if g.upper() in PURE_SALT_COMPOUNDS:
                    unique_generics.add(g.upper())
                    normalized_components.append(g.upper())
                else:
                    base, _ = self._strip_salt(g)
                    unique_generics.add(base)
                    # Apply synonym to get canonical form
                    canonical = self._apply_synonyms(base)
                    unique_generics.add(canonical)
                    normalized_components.append(canonical)
            
            # Add combination keys (sorted for order-independent matching)
            # Build from both original and normalized components for #7 (synonym swapping in mixtures)
            combo_keys = build_combination_keys(gt)
            unique_generics.update(combo_keys)
            # Also build from normalized components (e.g., SALBUTAMOL -> ALBUTEROL)
            normalized_combo_keys = build_combination_keys(normalized_components)
            unique_generics.update(normalized_combo_keys)
        
        # Batch lookup with cached generics for faster fuzzy matching
        generic_cache = batch_lookup_generics(
            unique_generics, self.con, self.synonyms,
            enable_fuzzy=True, cached_generics=self.cached_generics_list
        )
        
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
            # Include fuzzy-matched names so scoring works with misspellings
            input_generics_normalized = set()
            fuzzy_corrections = {}  # misspelled -> corrected
            
            # First collect fuzzy corrections
            for gm in unique_matches:
                if gm.get("fuzzy_match"):
                    matched_name = str(gm.get("generic_name", "")).upper()
                    # Find which stripped generic this fuzzy match corresponds to
                    for sg in stripped_generics:
                        if sg.upper() not in fuzzy_corrections:
                            fuzzy_corrections[sg.upper()] = matched_name
                            break
            
            # Build normalized set using fuzzy corrections
            for sg in stripped_generics:
                sg_upper = sg.upper()
                # Use fuzzy-corrected name if available
                if sg_upper in fuzzy_corrections:
                    normalized = fuzzy_corrections[sg_upper]
                else:
                    normalized = self._apply_synonyms(sg_upper)
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
            
            # Extract categorized tokens for output
            from .constants import CATEGORY_DOSE, CATEGORY_FORM, CATEGORY_ROUTE
            input_doses = list(categories.get(CATEGORY_DOSE, {}).keys())
            input_forms = list(categories.get(CATEGORY_FORM, {}).keys())
            input_routes = list(categories.get(CATEGORY_ROUTE, {}).keys())
            
            # Extract type detail from input text (before tokenization)
            _, type_detail = extract_type_detail(text)
            
            # Extract release/form details from the full token list
            # Join tokens to reconstruct text for detail extraction
            token_text = " ".join(tokens)
            _, release_detail = extract_release_detail(token_text)
            _, form_detail = extract_form_detail(token_text) if not release_detail else (None, None)
            
            # Use normalized form from categories
            base_form = input_forms[0] if input_forms else None
            
            if best:
                # Use reference_text if available, otherwise use generic_name; always uppercase
                ref_text = best.get("reference_text") or best.get("generic_name") or ""
                if ref_text:
                    ref_text = str(ref_text).upper()
                
                results.append({
                    "id": ids[i],
                    "input_text": text,
                    "row_idx": i,
                    "atc_code": best.get("atc_code"),
                    "drugbank_id": best.get("drugbank_id"),
                    "generic_name": best.get("generic_name"),
                    "reference_text": ref_text,
                    "dose": "|".join(input_doses) if input_doses else None,
                    "form": base_form,
                    "route": "|".join(input_routes) if input_routes else None,
                    "type_detail": type_detail,
                    "release_detail": release_detail,
                    "form_detail": form_detail,
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
                    "dose": "|".join(input_doses) if input_doses else None,
                    "form": base_form,
                    "route": "|".join(input_routes) if input_routes else None,
                    "type_detail": type_detail,
                    "release_detail": release_detail,
                    "form_detail": form_detail,
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
