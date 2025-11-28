# Drug Pipeline Implementation Plan v2

**Created:** Nov 27, 2025  
**Updated:** Nov 28, 2025  
**Objective:** Unified drug tagging with consistent algorithms for Annex F and ESOA

> **IMPORTANT:** After every group of changes, update both `pipeline.md` (algorithmic logic) and this file (implementation status).

---

## Current State Summary (Nov 27, 2025)

| Metric | Current | Target |
|--------|---------|--------|
| Annex F tagging | 94.1% | Maximize taggable |
| ESOA ATC tagging | 55.6% | Maximize taggable |
| ESOA→Drug Code | 40.5% | 60%+ |

---

## All 34 TODOs

### Phase 1: Analysis

#### #9. Research Unknown Acronyms
**What:** Google/research: NEBS, GTTS, SOLN, SUSP, and compile complete form abbreviation list.

**Action:** Web search, then update `form_route_mapping.py` with findings.

---

#### #22. Compile and Externalize Hardcoded Data
**What:** Consolidate all hardcoded token lists into a single unified constants file. Deduplicate across active scripts and `debug/old_files/`.

**Target:** Create `pipelines/drugs/scripts/tagging/unified_constants.py` containing all token sets, then refactor all scripts to import from there.

**Conceptual categories (deduplicated):**

| Category | Variables to Merge | Est. Unique Items |
|----------|-------------------|-------------------|
| **STOPWORDS** | `NATURAL_STOPWORDS`, `GENERIC_JUNK_TOKENS`, `BASE_GENERIC_IGNORE`, `GENERIC_BAD_SINGLE_TOKENS`, `COMMON_UNKNOWN_STOPWORDS`, `_GENERIC_NOISE_PHRASES` | ~80 |
| **SALT_TOKENS** | `SALT_TOKENS` (3 copies), `SPECIAL_SALT_TOKENS`, `SALT_FORM_SUFFIXES`, `_SALT_MAP` | ~50 |
| **PURE_SALT_COMPOUNDS** | `PURE_SALT_COMPOUNDS` (2 copies), `SALT_UNIT_SET`, `COMPOUND_GENERICS` | ~45 |
| **FORM_CANON** | `FORM_CANON` (4 copies), `ESOA_FORM_CANON`, `_FORM_MAP` (2 copies), `FORMULATION_BOUNDARY_WORDS`, `FORMULATION_STRIP_WORDS`, `_PREFIX_PACKAGING_TOKENS`, `_TRAILING_DESCRIPTOR_WORDS` | ~60 |
| **ROUTE_CANON** | `ROUTE_CANON` (4 copies), `ROUTE_ALIASES`, `_ROUTE_MAP` (2 copies), `ROUTE_WORDS` | ~25 |
| **FORM_TO_ROUTE** | `FORM_TO_ROUTE` | ~35 |
| **FORM_EQUIVALENCE** | `FORM_EQUIVALENCE_GROUPS` | ~6 groups |
| **ELEMENT_DRUGS** | `ELEMENT_DRUGS` | ~11 |
| **UNIT_TOKENS** | `UNIT_TOKENS`, `MEASUREMENT_TOKENS` (2 copies), `_PREFIX_UNIT_TOKENS`, `_UNIT_TOKENS`, `_WEIGHT_UNIT_FACTORS` | ~20 |
| **ATC_COMBO_PATTERNS** | `ATC_COMBINATION_PATTERNS`, `COMBINATION_ATC_PATTERNS`, `COMBINATION_ATC_SUFFIXES` | ~25 |
| **CONNECTIVES** | `CONNECTIVE_WORDS`, `_SALT_TAIL_BREAK_TOKENS` | ~8 |
| **SCORING_WEIGHTS** | `PRIMARY_WEIGHTS` (2 copies), `SECONDARY_WEIGHTS` (2 copies) | keep in scoring.py |
| **SYNONYMS** | `hardcoded` in lookup.py | move to unified synonyms dataset |
| **DESCRIPTOR_VALUES** | `_DESCRIPTOR_VALUES` | ~10 |

**Sources (active + old_files):**

*Active scripts:*
- `combos_drugs.py`, `text_utils_drugs.py`, `generic_normalization.py`
- `routes_forms_drugs.py`, `dose_drugs.py`, `resolve_unknowns_drugs.py`
- `tagging/constants.py`, `tagging/scoring.py`, `tagging/lookup.py`

*Old files (recover useful terms):*
- `match_annex_f_with_atc.py` - `COMPOUND_GENERICS`, `COMBINATION_ATC_*`
- `unified_tagger.py` - duplicate constants
- `pnf_aliases_drugs.py` - `SALT_FORM_SUFFIXES`
- `run_drugs_pt_0_create_master_dataset.py` - `_FORM_MAP`, `_ROUTE_MAP`, `_SALT_MAP`, `_DESCRIPTOR_VALUES`
- `run_drugs_pt_1_detect_annex_f.py` - `_FORM_MAP`, `_ROUTE_MAP`, `_SALT_MAP`
- `run_drugs_pt_1_parse_annex_f.py` - `_GENERIC_NOISE_PHRASES`
- `generate_route_form_mapping.py` - `FORM_CANON`, `ROUTE_CANON`
- `match_outputs_drugs.py` - `FRIENDLY_*_LABELS`, `_ABBREVIATION_LOOKUP`
- `match_esoa_with_annex_f.py` - `ESOA_FORM_CANON`

**Action:**
1. Create `unified_constants.py` with deduplicated sets for each category
2. Add loader functions: `get_stopwords()`, `get_salt_tokens()`, `get_form_canon()`, etc.
3. Refactor all active scripts to `from .tagging.unified_constants import ...`
4. Delete `combos_drugs.py` (only kept for `SALT_TOKENS`)
5. Keep regex patterns and scoring weights in their respective files (logic, not data)

---

#### #24. Review and Classify Pipeline Scripts ✅ PARTIAL
**What:** Audit all files in `./pipelines/drugs/scripts/`:
- Classify into folders (tagging, reference, utils, deprecated)
- Identify unused/legacy code
- Flag important logic NOT being used (report to user)

**Status (Nov 28, 2025):** Script audit completed. Moved 5 unused scripts to `debug/old_files/`:
- `aho_drugs.py` - Deprecated (using DuckDB)
- `debug_drugs.py` - References non-existent module
- `pnf_aliases_drugs.py` - Only used by deprecated aho_drugs.py
- `pnf_partial_drugs.py` - Not imported anywhere
- `generate_route_form_mapping.py` - One-time script

**Remaining:** Folder reorganization (tagging, reference, utils).

---

#### #25. Find Unknown Synonyms in Raw Data
**What:** Extract all unique generic-like tokens from ESOA and Annex F, compare against unified synonyms, report gaps (potential synonyms we don't know about).

**Action:** Analysis script to find potential missing synonyms.

---

#### #26. DrugBank R Script Performance
**What:** Profile `drugbank_generics.R` and related scripts. Check if:
- Over-parallelizing (too much overhead)
- Under-parallelizing (not using all cores)
- Not vectorized (row-by-row operations)

**Action:** R script optimization.

---

### Phase 2: Data Foundation

#### #11. Expand Synonyms from DrugBank
**What:** Extract synonyms where:
- `language == 'english'`
- `coder` is not empty
- `coder` is not solely "iupac" (split by `/`, check if list is not empty and not solely "iupac")

**Action:** Update R script `drugbank_generics.R` to export synonyms properly, then integrate into unified reference.

---

#### #15. Data-Driven Route Inference
**What:** Build form-route validity from 4 sources:
- PNF: `Technical Specifications`
- WHO: canonical form/route per ATC
- DrugBank: `dosages$form`+`route`, `products$dosage_form`+`route`
- FDA: `dosage_form`+`route`

Create lookup: `{generic/atc/drugbank_id: [{form, route, source}]}` - functions as autocomplete when form or route is missing.

**Action:** Enhance `build_unified_reference.py` to extract all form-route pairings with provenance.

---

#### #16. Fix ESOA Row Binding
**What:** Investigate why 258K rows but only ~145K unique descriptions. Check for duplicates, fix concatenation logic.

**Action:** Analyze `esoa_combined.csv`, deduplicate properly, update `_resolve_esoa_source()`.

---

#### #17. Build Proper Tier 1 Unified Reference
**What:** Create explosion logic:
```
drugbank_id × atc × generic (single only) × dose × form × route
```
With aggregated columns:
- `salt_forms` (pipe-delimited if same ATC, exploded if different ATC)
- `mixtures_atc` (ATCs of mixtures containing this generic)
- `mixtures_drugbank` (DrugBank IDs of mixtures containing this generic)
- `brands_single` (brand names of this single generic)
- `brands_combination` (brand names of combinations containing this generic)

**Action:** Major refactor of `build_unified_reference.py`.

---

#### #18. Collect All Known Doses
**What:** Extract doses from:
- `drugbank$dosages$strength`
- `drugbank$products$strength`
- `pnf.csv$Technical Specifications`
- `who_atc_*$ddd` + `uom` (research what DDD columns mean)
- `fda_drug_*$dosage_strength`

Normalize all to standard units (mg, mg/mL, %). Propagate as pipe-delimited for corresponding form-route pairings.

**Action:** Add dose collection to unified reference builder.

---

#### #28. Use DuckDB as Primary Data Store
**What:** Utilize DuckDB as much as possible:
- Store unified reference dataset in DuckDB (not just parquet files)
- Create indexes on `generic_name`, `brand_name`, `atc_code`, `drugbank_id` for fast lookups
- Use DuckDB's SQL for all queries (generics, brands, mixtures, form-route lookups)
- Batch queries instead of row-by-row lookups

**Action:** Refactor `build_unified_reference.py` and `tagger.py` to use DuckDB as primary store.

---

#### #29. Enrich Unified Reference from DrugBank Products
**What:** Extract ALL dose/form/route combinations from `drugbank$products` per `drugbank_id`:
- **All rows** contribute dose/form/route info
- `generic=true` rows: `name` column is the generic compound name (useful for validation)
- `generic=false` rows: `name` column is a brand name → add to brands lookup for that `drugbank_id`
- Extract: `dosage_form`, `strength`, `route`

**Action:** Update R script to export products, then integrate into unified reference builder.

---

#### #32. Standardize Column Names Across Datasets
**What:** Ensure consistent column names:

| Concept | Standard Name |
|---------|---------------|
| DrugBank ID | `drugbank_id` |
| ATC Code | `atc_code` |
| Generic Name | `generic_name` |
| Brand Name | `brand_name` |
| Dose/Strength | `dose` |
| Form | `form` |
| Route | `route` |
| Salt Form | `salt` |
| Source | `source` |

**Action:** Audit all datasets, create mapping, standardize during unified reference build.

---

### Phase 3: Core Matching

#### #1. Brand → Generic Swapping
**What:** Create a comprehensive brands lookup from 4 sources:
- `fda_drug_*$brand_name` (only if it doesn't match any known generic exactly - detect swapped rows)
- `drugbank$drugs$mixtures$name` → maps to `ingredients` delimited by "+"
- `drugbank$drugs$international_brands$brand`
- `drugbank$products$name` where `generic=='false'`

**Action:** Update `build_unified_reference.py` to consolidate all brand sources, then use in tagger to swap brands to generics before matching.

---

#### #2. Order-Independent Combination Matching
**What:** Modify matching logic to sort components alphabetically before comparison:
```python
sorted(["PIPERACILLIN", "TAZOBACTAM"]) == sorted(["TAZOBACTAM", "PIPERACILLIN"])
```

**Action:** Update `run_drugs_pt_4_esoa_to_annex_f.py` and `scoring.py` to normalize combo order.

---

#### #5. Permutation-Independent Component Matching
**What:** Same as #2 but specifically for vitamin combos like `B12 + B1 + B6`. Normalize by sorting components.

**Action:** Combined with #2 implementation.

---

#### #7. Synonym Swapping in Mixtures
**What:** When matching combos, normalize each component through synonyms first:
- `IPRATROPIUM + SALBUTAMOL` → `IPRATROPIUM + ALBUTEROL` (via synonym)

**Action:** Apply synonym normalization to each component before combo matching.

---

#### #27. Create Unified Tagger with Pharmaceutical Scoring
**What:** Create `pipelines/drugs/scripts/unified_tagger.py` as single entry point for both Annex F and ESOA tagging.

**Scoring based on pharmaceutical principles (deterministic, not numeric weights):**

1. **Generic Match** (REQUIRED for a valid match)
   - Must match the active ingredient(s)
   - Synonyms are equivalent (SALBUTAMOL = ALBUTEROL)
   - Order doesn't matter for combinations

2. **Salt Form** (IGNORED unless pure salt)
   - Salts are delivery mechanisms, not active ingredients
   - LOSARTAN POTASSIUM ≈ LOSARTAN (same drug)
   - Exception: Pure salts like SODIUM CHLORIDE, POTASSIUM CHLORIDE are the active compound

3. **Dose** (FLEXIBLE for ATC tagging, EXACT for Drug Code matching)
   - Same drug at different doses = same ATC
   - Different doses = different Drug Codes

4. **Form** (FLEXIBLE with equivalence groups)
   - TABLET ≈ CAPSULE (both oral solid)
   - AMPULE ≈ VIAL (both injectable)
   - SOLUTION ≈ SUSPENSION (both liquid)

5. **Route** (INFERRED from form if missing)
   - TABLET → ORAL
   - AMPULE → PARENTERAL
   - CREAM → TOPICAL

6. **ATC Preference** (for tie-breaking)
   - Single drug → prefer single-drug ATC (not combo ATC)
   - Combination → prefer combo ATC

**Output columns:** `atc_code`, `drugbank_id`, `generic_name`, `dose`, `form`, `route`, `type_detail`, `release_detail`, `form_detail`, `match_score`, `match_reason`

**Action:** Create new module that consolidates tagging logic with pharmaceutical-principled scoring.

---

### Phase 4: Enhancements

#### #3. Fuzzy Matching for Misspellings
**What:** Implement Levenshtein distance matching (1-2 char tolerance) as fallback when exact match fails. Use `rapidfuzz` library for speed.

**Action:** Add fuzzy matching layer in `tagger.py` after exact match fails.

---

#### #4. Compound Salt Recognition
**What:** Use `drugbank$salts` to identify that "SODIUM CHLORIDE" shares the anion "CHLORIDE" with other chloride salts, and map to base cation "SODIUM". Build anion→cation mapping.

**Action:** Enhance salt handling in `build_unified_reference.py` using `drugbank_salts_master.csv`.

---

#### #6. Type Detail Detection via Comma
**What:** Before stripping commas, capture text after comma as `type_detail` column:
- `"ALBUMIN, HUMAN"` → generic=`ALBUMIN`, type_detail=`HUMAN`
- `"ALCOHOL, ETHYL"` → generic=`ALCOHOL`, type_detail=`ETHYL`

**Action:** Add type_detail extraction in tokenizer.

---

#### #12. Capture Type Detail Before Comma Normalization
**What:** Same as #6 - extract type_detail from comma-separated text before normalizing.

**Action:** Combined with #6.

---

#### #13. Release Detail Column
**What:** If form contains comma and text after comma contains "release", capture as `release_detail`:
- `"TABLET, EXTENDED RELEASE"` → form=`TABLET`, release_detail=`EXTENDED RELEASE`

**Action:** Add to tokenizer/form extraction logic.

---

#### #14. Form Detail Column
**What:** Capture non-release modifiers after comma:
- `"TABLET, FILM COATED"` → form=`TABLET`, form_detail=`FILM COATED`

**Action:** Add alongside #13.

---

### Phase 5: Normalization

#### #34. Stop Word Filtering
**What:** Filter out unnecessary/noise words during tokenization that don't contribute to drug identification:
- `AS`, `IN`, `FOR`, `TO`, `WITH`, `EQUIV.`, `AND`, `OF`, `OR`, `NOT`, `THAN`, `HAS`, `DURING`, `THIS`, `W/`
- Exception: `PER` should be kept when indicating ingredient ratio (e.g., "10MG PER ML")

**Action:** Add stop word list to tokenizer, filter during text processing.

---

#### #8. Dose Denominator Normalization
**What:** Parse doses like `500MG/5ML` and normalize to per-1-unit: `100MG/ML`. Apply to all units.

**Action:** Update `dose_drugs.py` or create new dose normalization function.

---

#### #20. Improve PNF Lexicon from PNF Prepared
**What:** Review what transformations `prepare_drugs.py` does to PNF and bake those improvements into `pnf_lexicon` itself.

**Action:** Update PNF preparation pipeline.

---

### Phase 6: Performance

#### #10. Batch Tagging Implementation
**What:** Add `tag_batch()` method to `UnifiedTagger` that processes 10K-15K rows at once. Benchmark both sizes.

**Action:** Modify `tagger.py` to add batch processing with DuckDB bulk queries.

---

### Phase 7: Fallbacks

#### #21. FDA Food Fallback for Untagged
**What:** For Annex F rows with no drug match, tokenize and search `fda_food_*` as last resort. Same for untaggable ESOA. This is the last resort match basis.

**Action:** Add fallback matching tier using FDA food data.

---

#### #23. Dose-Flexible Tagging, Exact Matching
**What:** Confirm that:
- Part 2/3 (ATC/DrugBank tagging): dose-flexible
- Part 4 (Drug Code matching): exact dose only (drug_code is unique down to dose)

**Action:** Review and document current behavior, fix if needed.

---

### Phase 8: Cleanup

#### #19. Move Learnings to Datasets, Not Scripts
**What:** Audit all hardcoded values and migrate to reference datasets.

**Action:** Combined with #22.

---

#### #31. Update Documentation
**What:** 
- Write `pipeline.md` with all algorithmic logic, decisions, rules, principles
- Update `AGENTS.md` to reference `pipeline.md` and `implementation_plan_v2.md`
- Keep both files updated with every group of changes

**Action:** Create and maintain documentation.

---

#### #33. Track Success Metrics
**What:** Create a metrics tracking system:
- After each pipeline run, log:
  - Annex F: % with ATC, % with DrugBank ID
  - ESOA: % with ATC, % with DrugBank ID
  - ESOA→Drug Code: % matched
  - Unique generics recognized
  - Unique brands recognized
- Store in `outputs/drugs/metrics_history.csv`

**Action:** Add metrics logging to pipeline runners.

---

## Execution Order

| Phase | Items | Rationale |
|-------|-------|-----------|
| **1. Analysis** | #9, #22, #24, #25, #26 | Understand current state, find hardcoded data, audit scripts |
| **2. Data Foundation** | #11, #15, #16, #17, #18, #28, #29, #32 | Build proper unified reference in DuckDB with all enrichments |
| **3. Core Matching** | #1, #2, #5, #7, #27 | Brand swapping, combo matching, unified tagger |
| **4. Enhancements** | #3, #4, #6, #12, #13, #14 | Fuzzy matching, salts, type_detail, form/release details |
| **5. Normalization** | #8, #20, #34 | Dose normalization, PNF improvements, stop words |
| **6. Performance** | #10 | Batch tagging |
| **7. Fallbacks** | #21, #23 | FDA food fallback, exact dose matching |
| **8. Cleanup** | #19, #31, #33 | Externalize hardcoded data, documentation, metrics |

---

## Files to Create/Modify

### New Files
- `debug/pipeline.md` - Algorithmic logic and pharmaceutical rules
- `pipelines/drugs/scripts/unified_tagger.py` - Single entry point for tagging

### Modified Files
- `AGENTS.md` - Reference pipeline.md and implementation_plan_v2.md
- `build_unified_reference.py` - Major refactor for unified dataset
- `run_drugs_pt_4_esoa_to_annex_f.py` - Order-independent matching
- `scoring.py` - Pharmaceutical-principled scoring
- `tagger.py` - Batch processing, DuckDB integration
- `drugbank_generics.R` - Synonym extraction, performance optimization
