"""
Main tagger interface for drug descriptions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import duckdb
import pandas as pd

from .unified_constants import (
    PURE_SALT_COMPOUNDS, UNIT_TOKENS, get_regional_canonical,
    CATEGORY_DOSE, CATEGORY_FORM, CATEGORY_ROUTE,
    VACCINE_CANONICAL, normalize_vaccine_name,
)
from .lookup import (
    apply_synonym, batch_lookup_generics, build_combination_keys,
    swap_brand_to_generic,
)
from .scoring import select_best_candidate, sort_atc_codes
from .spinner import run_with_spinner
from .tokenizer import (
    categorize_tokens, detect_compound_salts, extract_drug_details,
    extract_generic_tokens, extract_form_detail, extract_release_detail,
    extract_type_detail, normalize_tokens, split_with_parentheses,
    strip_salt_suffix,
)


# Default paths
# tagger.py is at pipelines/drugs/scripts/tagger.py (3 levels from project root)
PROJECT_DIR = Path(__file__).resolve().parents[3]
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
        """Load unified_* reference tables into DuckDB."""
        if self._loaded:
            return
        
        self._log("Loading unified_* tables...")
        
        # Create in-memory DuckDB
        self.con = duckdb.connect(":memory:")
        
        # Load unified_generics (main reference) - CSV is canonical format
        generics_path = self.outputs_dir / "unified_generics.csv"
        if not generics_path.exists():
            raise FileNotFoundError(f"unified_generics.csv not found: {generics_path}")
        self.con.execute(f"CREATE TABLE unified AS SELECT * FROM read_csv_auto('{generics_path}')")
        # Create index for faster lookups
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_unified_generic ON unified(generic_name)")
        count = self.con.execute("SELECT COUNT(*) FROM unified").fetchone()[0]
        unique_generics = self.con.execute("SELECT COUNT(DISTINCT generic_name) FROM unified").fetchone()[0]
        self._log(f"  - unified_generics: {count:,} rows ({unique_generics:,} unique)")
        
        # Load unified_brands
        brands_path = self.outputs_dir / "unified_brands.csv"
        if brands_path.exists():
            self.con.execute(f"CREATE TABLE brands AS SELECT * FROM read_csv_auto('{brands_path}')")
            brand_count = self.con.execute("SELECT COUNT(*) FROM brands").fetchone()[0]
            self._log(f"  - unified_brands: {brand_count:,} rows")
        
        # Load unified_synonyms
        synonyms_path = self.outputs_dir / "unified_synonyms.csv"
        if synonyms_path.exists():
            self.con.execute(f"CREATE TABLE synonyms AS SELECT * FROM read_csv_auto('{synonyms_path}')")
            syn_count = self.con.execute("SELECT COUNT(*) FROM synonyms").fetchone()[0]
            self._log(f"  - unified_synonyms: {syn_count:,} rows")
        
        # Load unified_mixtures (queried on-demand for multi-generic inputs)
        mixtures_path = self.outputs_dir / "unified_mixtures.csv"
        self._mixtures_loaded = False
        if mixtures_path.exists():
            self.con.execute(f"CREATE TABLE mixtures AS SELECT * FROM read_csv_auto('{mixtures_path}')")
            mix_count = self.con.execute("SELECT COUNT(*) FROM mixtures").fetchone()[0]
            self._log(f"  - unified_mixtures: {mix_count:,} rows")
            self._mixtures_loaded = True
        
        # Load unified_atc (for ATC selection with form/route/dose)
        atc_path = self.outputs_dir / "unified_atc.csv"
        self._atc_loaded = False
        if atc_path.exists():
            self.con.execute(f"CREATE TABLE atc AS SELECT * FROM read_csv_auto('{atc_path}')")
            self.con.execute("CREATE INDEX IF NOT EXISTS idx_atc_generic ON atc(generic_name)")
            atc_count = self.con.execute("SELECT COUNT(*) FROM atc").fetchone()[0]
            self._log(f"  - unified_atc: {atc_count:,} rows")
            self._atc_loaded = True
        
        # Build synonyms dict from unified_synonyms table + spelling corrections + regional
        from .unified_constants import SPELLING_SYNONYMS, REGIONAL_TO_US
        self.synonyms = dict(SPELLING_SYNONYMS)
        
        # Add regional→US mappings (PARACETAMOL → ACETAMINOPHEN for lookups)
        for regional, us in REGIONAL_TO_US.items():
            self.synonyms[regional] = us
        
        # Parse unified_synonyms (format: drugbank_id, generic_name, synonyms pipe-separated)
        try:
            synonym_rows = self.con.execute("""
                SELECT generic_name, synonyms FROM synonyms
                WHERE synonyms IS NOT NULL AND synonyms != ''
            """).fetchall()
            for generic_name, synonyms_str in synonym_rows:
                if generic_name and synonyms_str:
                    generic_upper = generic_name.upper()
                    for syn in synonyms_str.split('|'):
                        syn = syn.strip().upper()
                        if syn and syn != generic_upper:
                            self.synonyms[syn] = generic_upper
        except Exception:
            pass
        self._log(f"  - synonym mappings: {len(self.synonyms):,}")
        
        # Build brand → generic map from unified_brands table
        self.brand_map = {}
        all_generics = set(row[0].upper() for row in self.con.execute(
            "SELECT DISTINCT generic_name FROM unified"
        ).fetchall())
        
        # Also check synonyms that map to generics (e.g., ASPIRIN -> ACETYLSALICYLIC ACID)
        from .unified_constants import SPELLING_SYNONYMS
        synonym_generics = set(k.upper() for k in SPELLING_SYNONYMS.keys())
        
        try:
            # Count rows per generic to prefer more common associations
            brand_rows = self.con.execute("""
                SELECT brand_name, generic_name, COUNT(*) as cnt
                FROM brands
                GROUP BY brand_name, generic_name
                ORDER BY cnt DESC
            """).fetchall()
            for brand, generic, _ in brand_rows:
                if brand and generic:
                    brand_upper = brand.upper()
                    generic_upper = generic.upper()
                    
                    # FDA often swaps brand/generic - if brand_name is a known generic,
                    # treat it as the generic and use generic_name as the brand
                    if brand_upper in all_generics or brand_upper in synonym_generics:
                        # Swap: the "brand" is actually a generic, "generic" is the brand
                        if generic_upper not in all_generics and generic_upper not in self.brand_map:
                            self.brand_map[generic_upper] = brand_upper
                    elif brand_upper not in self.brand_map:
                        self.brand_map[brand_upper] = generic_upper
        except Exception:
            pass
        self._log(f"  - brand mappings: {len(self.brand_map):,}")
        
        # Cache generics list for fuzzy matching
        self.cached_generics_list = [row[0] for row in self.con.execute(
            "SELECT DISTINCT generic_name FROM unified WHERE generic_name IS NOT NULL"
        ).fetchall()]
        
        # Build multiword generics set from data + constants
        from .unified_constants import MULTIWORD_GENERICS
        
        self.multiword_generics = set()
        for name in self.cached_generics_list:
            if " " in str(name):
                self.multiword_generics.add(str(name).upper())
        
        # Add multiword generics from unified_constants.py
        self.multiword_generics.update(MULTIWORD_GENERICS)
        
        # Add plural forms of multiword generics for detection
        plural_forms = set()
        for mw in self.multiword_generics:
            words = mw.split()
            if words and not words[0].endswith("S"):
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
        # Don't strip from known multiword generics (e.g., ISOSORBIDE DINITRATE)
        generic_upper = generic.upper()
        if generic_upper in self.multiword_generics:
            return generic_upper, None
        return strip_salt_suffix(generic)
    
    def _lookup_mixture(self, generics: List[str]) -> Optional[Dict[str, Any]]:
        """Look up a mixture by its component generics using component_key index."""
        if not self._mixtures_loaded:
            return None
        
        # Filter out junk tokens like "+" and tokens starting with "+"
        junk = {"+", "MG", "ML", "MCG", "G", "L", ""}
        generics = [g for g in generics if g.upper() not in junk and not g.startswith("+")]
        
        if len(generics) < 2:
            return None
        
        # Normalize generics (apply synonyms) and build lookup key
        # Use lowercase for key since mixtures table has lowercase keys
        normalized = [self._apply_synonyms(g.upper()).lower() for g in generics]
        
        # Deduplicate and remove substrings (e.g., "ascorbic" is substring of "ascorbic acid")
        unique = []
        for n in sorted(normalized, key=len, reverse=True):  # Longest first
            if not any(n in existing for existing in unique):
                unique.append(n)
        
        if len(unique) < 2:
            return None
        
        component_key = '|'.join(sorted(unique))
        
        # Fast lookup by component_key
        try:
            rows = self.con.execute("""
                SELECT drugbank_id, mixture_name, component_generics
                FROM mixtures
                WHERE component_key = ?
                LIMIT 1
            """, [component_key]).fetchall()
            
            if rows:
                drugbank_id, mixture_name, component_generics = rows[0]
                # Use uppercase for display name
                display_name = ' + '.join(sorted([n.upper() for n in unique]))
                return {
                    'drugbank_id': drugbank_id,
                    'generic_name': display_name,
                    'mixture_name': mixture_name,
                    'atc_code': None,
                    'source': 'drugbank_mixture',
                    'reference_text': component_generics,
                }
        except Exception:
            pass
        
        return None
    
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
        deduplicate: bool = True,
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
            deduplicate: If True, deduplicate by text_column before tagging (default True)
        
        Returns:
            DataFrame with tagging results
        """
        import time
        
        if not self._loaded:
            self.load()
        
        original_rows = len(df)
        if original_rows == 0:
            return pd.DataFrame()
        
        # Deduplicate by text column to avoid redundant work
        if deduplicate:
            unique_texts = df[[text_column]].drop_duplicates()
            total_rows = len(unique_texts)
            texts = unique_texts[text_column].fillna("").astype(str).tolist()
            ids = list(range(total_rows))
        else:
            total_rows = original_rows
            texts = df[text_column].fillna("").astype(str).tolist()
            if id_column and id_column in df.columns:
                ids = df[id_column].tolist()
            else:
                ids = list(range(total_rows))
        
        # Process in chunks
        all_results = []
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        start_time = time.time()
        last_rate: float = 0.0  # rows/s from previous chunk
        
        for i in range(0, total_rows, chunk_size):
            chunk_num = i // chunk_size + 1
            end_idx = min(i + chunk_size, total_rows)
            
            chunk_texts = texts[i:end_idx]
            chunk_ids = ids[i:end_idx]
            
            if show_progress:
                rows_in_chunk = len(chunk_texts)
                est_time = rows_in_chunk / last_rate if last_rate > 0 else 0.0
                
                def make_label(elapsed: float, n: int = rows_in_chunk, c: int = chunk_num, t: int = num_chunks, est: float = est_time) -> str:
                    if est > 0:
                        eta = est - elapsed
                        return f"Chunk {c:02d}/{t:02d} (ETA {eta:7.2f}s)"
                    return f"Chunk {c:02d}/{t:02d}"
                
                completion = lambda elapsed, n=rows_in_chunk, c=chunk_num, t=num_chunks: f"Chunk {c:02d}/{t:02d}: {n/elapsed:,.0f} rows/s"
                chunk_results = run_with_spinner(
                    make_label,
                    lambda t=chunk_texts, ids=chunk_ids: self._tag_batch(t, ids),
                    completion_label=completion,
                )
                # Update rate for next chunk's ETA
                chunk_time = time.time() - start_time - sum(r.get("_elapsed", 0) for r in all_results[:i] if isinstance(r, dict))
            else:
                chunk_results = self._tag_batch(chunk_texts, chunk_ids)
            all_results.extend(chunk_results)
            # Track rate after each chunk
            elapsed_so_far = time.time() - start_time
            rows_so_far = end_idx
            last_rate = rows_so_far / elapsed_so_far if elapsed_so_far > 0 else 0
        
        total_time = time.time() - start_time
        if show_progress:
            rate = total_rows / total_time if total_time > 0 else 0
            print(
                f"⣿ {total_time:7.2f}s "
                f"Total: {total_rows:,} rows ({rate:.0f} rows/s)"
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
        all_drug_details = []  # Store extracted details for later use
        
        for text in texts:
            # Pre-process: extract parentheticals and qualifiers into separate fields
            drug_details = extract_drug_details(text)
            
            # Check if this is a vaccine and normalize
            vaccine_name, vaccine_details = normalize_vaccine_name(text)
            if vaccine_name:
                drug_details["generic_name"] = vaccine_name
                if vaccine_details:
                    if drug_details.get("type_details"):
                        drug_details["type_details"] += "; " + vaccine_details
                    else:
                        drug_details["type_details"] = vaccine_details
            
            all_drug_details.append(drug_details)
            
            # Use cleaned generic name for tokenization
            clean_text = drug_details["generic_name"]
            # But also keep the original for dose/form extraction
            tokens, generic_tokens = extract_generic_tokens(text, self.multiword_generics)
            
            # If we extracted a cleaner generic name, use it
            clean_generic_tokens = []
            if drug_details["generic_name"] and drug_details["generic_name"] != text.upper():
                # Also extract from the cleaned version
                _, clean_generic_tokens = extract_generic_tokens(clean_text, self.multiword_generics)
                # Merge: prefer clean tokens but keep unique from original
                generic_tokens = list(dict.fromkeys(clean_generic_tokens + generic_tokens))
            
            # Store clean tokens for combo key building (without junk like "DRY", "POWDER", etc.)
            drug_details["_clean_tokens"] = clean_generic_tokens if clean_generic_tokens else generic_tokens[:2]
            
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
        for idx, gt in enumerate(all_generic_tokens):
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
            # Add synonyms of combo keys (e.g., "ETHYL ALCOHOL" -> "ETHANOL")
            for ck in combo_keys:
                ck_syn = self._apply_synonyms(ck)
                if ck_syn != ck:
                    unique_generics.add(ck_syn)
            # Also build from normalized components (e.g., SALBUTAMOL -> ALBUTEROL)
            normalized_combo_keys = build_combination_keys(normalized_components)
            unique_generics.update(normalized_combo_keys)
            for nck in normalized_combo_keys:
                nck_syn = self._apply_synonyms(nck)
                if nck_syn != nck:
                    unique_generics.add(nck_syn)
            
            # CRITICAL: Also build combo keys from CLEAN tokens (without junk like DRY, POWDER)
            # This ensures combinations like "BUDESONIDE + FORMOTEROL" are properly looked up
            clean_tokens = all_drug_details[idx].get("_clean_tokens", [])
            if clean_tokens and len(clean_tokens) >= 2:
                clean_combo_keys = build_combination_keys(clean_tokens)
                unique_generics.update(clean_combo_keys)
                for cck in clean_combo_keys:
                    cck_syn = self._apply_synonyms(cck)
                    if cck_syn != cck:
                        unique_generics.add(cck_syn)
        
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
            
            # Get stripped generics with defensive filtering
            stripped_generics = []
            for g in generic_tokens:
                if g.upper() in PURE_SALT_COMPOUNDS:
                    stripped_generics.append(g.upper())
                else:
                    base, _ = self._strip_salt(g)
                    # Defensive filtering: exclude known formulation markers and junk
                    if (base and 
                        base.upper() not in {"FC", "EC", "SR", "XR", "ER", "DR", 
                                           "NON-PNF", "NONPNF", "MG", "ML", 
                                           "TABLET", "CAPSULE", "SOLUTION"} and
                        len(base.strip()) > 1):
                        stripped_generics.append(base)
            
            # Collect matches - COMBO MATCHES FIRST for priority (e.g., ETHYL ALCOHOL -> ETHANOL)
            generic_matches = []
            
            # CRITICAL: First try combo keys from CLEAN tokens (e.g., "BUDESONIDE + FORMOTEROL")
            # These are extracted by extract_drug_details and don't contain junk like "DRY", "POWDER"
            clean_tokens = all_drug_details[i].get("_clean_tokens", [])
            if clean_tokens and len(clean_tokens) >= 2:
                clean_combo_keys = build_combination_keys(clean_tokens)
                for cck in clean_combo_keys:
                    if cck in generic_cache:
                        generic_matches.extend(generic_cache[cck])
                    cck_syn = self._apply_synonyms(cck)
                    if cck_syn != cck and cck_syn in generic_cache:
                        generic_matches.extend(generic_cache[cck_syn])
            
            # Add combination matches from stripped generics (both original and normalized)
            # This ensures combo matches like ETHANOL take priority over single-token fuzzy matches
            combo_keys = build_combination_keys(stripped_generics)
            for ck in combo_keys:
                if ck in generic_cache:
                    generic_matches.extend(generic_cache[ck])
                # Also apply synonym to the combo key itself (e.g., "ETHYL ALCOHOL" -> "ETHANOL")
                ck_syn = self._apply_synonyms(ck)
                if ck_syn != ck and ck_syn in generic_cache:
                    generic_matches.extend(generic_cache[ck_syn])
            
            # Also check normalized combo keys (e.g., PARACETAMOL -> ACETAMINOPHEN)
            normalized_components = [self._apply_synonyms(sg) for sg in stripped_generics]
            normalized_combo_keys = build_combination_keys(normalized_components)
            for nck in normalized_combo_keys:
                if nck in generic_cache and nck not in combo_keys:
                    generic_matches.extend(generic_cache[nck])
                # Apply synonym to normalized combo key too
                nck_syn = self._apply_synonyms(nck)
                if nck_syn != nck and nck_syn in generic_cache:
                    generic_matches.extend(generic_cache[nck_syn])
            
            # Then add individual token matches
            for sg in stripped_generics:
                if sg in generic_cache:
                    generic_matches.extend(generic_cache[sg])
                syn = self._apply_synonyms(sg)
                if syn in generic_cache and syn != sg:
                    generic_matches.extend(generic_cache[syn])
            
            # Deduplicate
            seen = set()
            unique_matches = []
            for m in generic_matches:
                key = m.get("generic_name", "")
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(m)
            
            if not unique_matches:
                # Check if any synonym maps to a mixture name (e.g., CO-AMOXICLAV -> AMOXICILLIN AND CLAVULANATE POTASSIUM)
                for sg in stripped_generics:
                    syn = self._apply_synonyms(sg)
                    if syn != sg and self._mixtures_loaded:
                        # Try to find the synonym in mixtures table by name
                        try:
                            mixture_result = self.con.execute("""
                                SELECT mixture_name, drugbank_id, component_key
                                FROM mixtures
                                WHERE UPPER(mixture_name) = ?
                                LIMIT 1
                            """, [syn.upper()]).fetchone()
                            if mixture_result:
                                unique_matches.append({
                                    "generic_name": mixture_result[0],
                                    "drugbank_id": mixture_result[1],
                                    "atc_code": None,  # Mixtures often don't have ATC
                                    "source": "mixtures",
                                    "reference_text": mixture_result[0],
                                })
                        except Exception:
                            pass
            
            if not unique_matches:
                # Try mixture lookup for multi-generic inputs
                if len(stripped_generics) >= 2:
                    mixture_match = self._lookup_mixture(stripped_generics)
                    if mixture_match:
                        results.append({
                            "id": ids[i],
                            "input_text": text,
                            "row_idx": i,
                            "atc_code": mixture_match.get("atc_code"),
                            "drugbank_id": mixture_match.get("drugbank_id"),
                            "generic_name": mixture_match.get("generic_name"),
                            "reference_text": mixture_match.get("reference_text"),
                            "match_score": 100,
                            "match_reason": "matched",
                            "sources": mixture_match.get("source", ""),
                            "dose": None,
                            "form": None,
                            "route": None,
                            "type_details": None,
                            "release_details": None,
                            "form_details": None,
                            "salt_details": all_drug_details[i].get("salt_details"),
                            "brand_details": all_drug_details[i].get("brand_details"),
                            "indication_details": all_drug_details[i].get("indication_details"),
                            "alias_details": all_drug_details[i].get("alias_details"),
                        })
                        continue
                
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
                    "sources": None,
                    "dose": None,
                    "form": None,
                    "route": None,
                    "type_details": None,
                    "release_details": None,
                    "form_details": None,
                    "salt_details": all_drug_details[i].get("salt_details"),
                    "brand_details": all_drug_details[i].get("brand_details"),
                    "indication_details": all_drug_details[i].get("indication_details"),
                    "alias_details": all_drug_details[i].get("alias_details"),
                })
                continue
            
            # Build candidates
            categories = categorize_tokens(tokens)
            candidates = []
            for gm in unique_matches:
                atc_codes = str(gm.get("atc_code", "")).split("|")
                atc_codes = sort_atc_codes(atc_codes)
                atc_codes = [a for a in atc_codes if a]  # Filter empty
                
                # For entries with ATC codes, add one candidate per ATC
                if atc_codes:
                    for atc in atc_codes:
                        candidates.append({
                            "atc_code": atc,
                            "drugbank_id": gm.get("drugbank_id"),
                            "generic_name": gm.get("generic_name"),
                            "reference_text": gm.get("reference_text"),
                            "source": gm.get("source"),
                            "form": None,
                            "route": None,
                            "doses": None,
                        })
                else:
                    # For mixtures without ATC codes, add candidate with drugbank_id only
                    # This allows matching combination drugs that don't have specific ATC codes
                    if gm.get("drugbank_id"):
                        candidates.append({
                            "atc_code": None,
                            "drugbank_id": gm.get("drugbank_id"),
                            "generic_name": gm.get("generic_name"),
                            "reference_text": gm.get("reference_text"),
                            "source": gm.get("source"),
                            "form": None,
                            "route": None,
                            "doses": None,
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
                    "sources": None,
                    "dose": None,
                    "form": None,
                    "route": None,
                    "type_details": None,
                    "release_details": None,
                    "form_details": None,
                    "salt_details": all_drug_details[i].get("salt_details"),
                    "brand_details": all_drug_details[i].get("brand_details"),
                    "indication_details": all_drug_details[i].get("indication_details"),
                    "alias_details": all_drug_details[i].get("alias_details"),
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
            
            # Also add combo synonyms to normalized set (e.g., ETHYL ALCOHOL -> ETHANOL)
            for ck in combo_keys:
                ck_syn = self._apply_synonyms(ck)
                if ck_syn != ck and ck_syn not in {"+", "MG/5"}:
                    input_generics_normalized.add(ck_syn)
            
            num_input = len(input_generics_normalized)
            has_plus = "+" in text
            has_in = " IN " in text.upper() and num_input > 1
            is_iv_solution = has_in and not has_plus
            is_combination = num_input > 1 and has_plus
            is_single_drug = num_input == 1
            
            # Select best candidate, using extracted details for tie-breaking
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
                input_details=all_drug_details[i],
            )
            
            # Extract categorized tokens for output
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
                
                # Apply regional canonical name (PH uses WHO names like PARACETAMOL)
                generic_name = best.get("generic_name")
                if generic_name:
                    generic_name = get_regional_canonical(generic_name)
                
                results.append({
                    "id": ids[i],
                    "input_text": text,
                    "row_idx": i,
                    "atc_code": best.get("atc_code"),
                    "drugbank_id": best.get("drugbank_id"),
                    "generic_name": generic_name,
                    "reference_text": ref_text,
                    "dose": "|".join(input_doses) if input_doses else None,
                    "form": base_form,
                    "route": "|".join(input_routes) if input_routes else None,
                    "type_details": type_detail,
                    "release_details": release_detail,
                    "form_details": form_detail,
                    "match_score": 1,
                    "match_reason": "matched",
                    "sources": best.get("source"),
                    "salt_details": all_drug_details[i].get("salt_details"),
                    "brand_details": all_drug_details[i].get("brand_details"),
                    "indication_details": all_drug_details[i].get("indication_details"),
                    "alias_details": all_drug_details[i].get("alias_details"),
                })
            else:
                # Try mixture lookup for multi-generic inputs when scoring fails
                if is_combination and len(stripped_generics) >= 2:
                    mixture_match = self._lookup_mixture(stripped_generics)
                    if mixture_match:
                        results.append({
                            "id": ids[i],
                            "input_text": text,
                            "row_idx": i,
                            "atc_code": mixture_match.get("atc_code"),
                            "drugbank_id": mixture_match.get("drugbank_id"),
                            "generic_name": mixture_match.get("generic_name"),
                            "reference_text": mixture_match.get("reference_text"),
                            "dose": "|".join(input_doses) if input_doses else None,
                            "form": base_form,
                            "route": "|".join(input_routes) if input_routes else None,
                            "type_details": type_detail,
                            "release_details": release_detail,
                            "form_details": form_detail,
                            "match_score": 100,
                            "match_reason": "matched",
                            "sources": mixture_match.get("source"),
                            "salt_details": all_drug_details[i].get("salt_details"),
                            "brand_details": all_drug_details[i].get("brand_details"),
                            "indication_details": all_drug_details[i].get("indication_details"),
                            "alias_details": all_drug_details[i].get("alias_details"),
                        })
                        continue
                
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
                    "type_details": type_detail,
                    "release_details": release_detail,
                    "form_details": form_detail,
                    "match_score": 0,
                    "match_reason": "no_match",
                    "sources": None,
                    "salt_details": all_drug_details[i].get("salt_details"),
                    "brand_details": all_drug_details[i].get("brand_details"),
                    "indication_details": all_drug_details[i].get("indication_details"),
                    "alias_details": all_drug_details[i].get("alias_details"),
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
