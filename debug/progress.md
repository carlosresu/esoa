# Drug Pipeline Progress Tracker

**Started:** Nov 28, 2025  
**Last Updated:** Nov 28, 2025 (Phase 4 COMPLETE)

---

## Phase 1: Analysis ✅ COMPLETE

**Goal:** Understand current state, find hardcoded data, audit scripts

### Completed Work

#### 1.1 Unified Constants File
- **Created:** `pipelines/drugs/scripts/tagging/unified_constants.py`
- **Contents:**
  - 218 stopwords (deduplicated from 6 sources)
  - 88 salt tokens (deduplicated + 15 new)
  - 60 pure salt compounds
  - 120 form mappings
  - 46 route mappings
  - 72 form-to-route mappings
  - 7 form equivalence groups
  - 26 ATC combination patterns
- **Helper functions:** `is_stopword()`, `is_salt_token()`, `is_combination_atc()`, `forms_are_equivalent()`, etc.
- **Refactored imports:** constants.py, text_utils_drugs.py, routes_forms_drugs.py, scoring.py

#### 1.2 Script Audit
- **Moved to `debug/old_files/`:**
  - `aho_drugs.py` - Deprecated (using DuckDB instead)
  - `debug_drugs.py` - References non-existent module
  - `pnf_aliases_drugs.py` - Only used by deprecated aho_drugs.py
  - `pnf_partial_drugs.py` - Not imported anywhere
  - `generate_route_form_mapping.py` - One-time script
- **Folder reorganization:** Deferred to Phase 8

#### 1.3 Unknown Token Analysis
- **Method:** Compared Annex F + ESOA tokens against all reference sources
- **Finding:** 12,005 known generics across DrugBank/WHO/PNF/FDA
- **Correction:** Most "unknowns" were actually in data - initial analysis only checked 2 columns
- **Key insight:** Partial tokens (MEFENAMIC, TRANEXAMIC) are from multi-word drug names
- **Action:** Handle via multi-word generic preservation (AGENTS.md #15)

#### 1.4 R Script Performance
- **Finding:** Already well-optimized
- **Features:** `future`/`mclapply` parallelism, `data.table` threading, configurable workers
- **No changes needed**

---

## Phase 2: Data Foundation ✅ COMPLETE

**Goal:** Build proper unified reference in DuckDB with all enrichments

### Completed Work

#### DrugBank R Script Optimization ✅
- Created `_shared.R` with common setup (packages, parallel, utilities)
- Uses `min(8, cpu_count)` workers, cross-platform support
- Runtime: ~433s total for all DrugBank scripts

#### #0: Refresh All Base Datasets ✅
- `python run_drugs_pt_1_prepare_dependencies.py` - all artifacts generated

#### #28: DuckDB as Primary Data Store ✅
- `build_unified_reference.py` uses in-memory DuckDB for all queries
- SQL-based aggregation and joining across sources

#### #17: Build Tier 1 Unified Reference ✅
- **unified_drug_reference.parquet**: 52,002 rows
- Exploded by: drugbank_id × atc_code × form × route
- Aggregated doses per combination

#### #15: Form-Route Validity Mapping ✅
- **form_route_validity.parquet**: 53,039 combinations
- Sources: PNF, DrugBank products, FDA drug

#### #11: Synonyms from DrugBank ✅
- Already implemented in R script with proper filtering (language=english, not iupac-only)
- **generics_lookup.parquet**: 7,345 generics with synonyms

#### #29: Enrich from DrugBank Products ✅
- Extracted 455,970 product rows with dose/form/route

#### #16: Fix ESOA Row Binding ✅
- Added deduplication to `_concatenate_csv()` - removes 44% duplicates
- 258,878 → ~146,189 rows after dedup

#### #18: Collect All Known Doses ✅
- 28,230 rows have dose information (from products)
- Aggregated as pipe-delimited in unified reference

#### #32: Standardize Column Names ✅
- Consistent naming: `generic_name`, `atc_code`, `drugbank_id`, `form`, `route`, `doses`

### Deferred to Phase 8
- #35: Sync R/Python constants (lower priority cleanup)

---

## Phase 3: Core Matching ✅ COMPLETE

**Goal:** Brand swapping, combo matching, unified tagger

### Completed Work

#### #1: Brand → Generic Swapping ✅
- Added `load_brands_lookup()`, `build_brand_to_generic_map()` to lookup.py
- Excludes known generics from brand map (prevents AMOXICILLIN→combo bug)
- 126,413 brand entries loaded

#### #2/#5: Order-Independent Combination Matching ✅
- `build_combination_keys()` sorts components alphabetically
- PIPERACILLIN + TAZOBACTAM == TAZOBACTAM + PIPERACILLIN

#### #7: Synonym Swapping in Mixtures ✅
- Normalizes each component through synonyms before combo matching
- IPRATROPIUM + SALBUTAMOL matches via ALBUTEROL synonym

#### #27: Unified Tagger with Pharmaceutical Scoring ✅
- Generic must match (required)
- Salt forms flexible (except pure salts like NaCl)
- Single vs combo ATC preference based on input
- Output includes dose, form, route extracted from input

### Test Results
- BIOGESIC 500MG TAB → ACETAMINOPHEN, N02BE01 (brand swap)
- AMOXICILLIN 500MG CAP → AMOXICILLIN, J01CA04 (single ATC preferred)
- IPRATROPIUM + SALBUTAMOL → R03AL02 (combo matching)
- LOSARTAN POTASSIUM 50MG → LOSARTAN, C09CA01 (salt strip)

---

## Phase 4: Enhancements ✅ COMPLETE

**Goal:** Fuzzy matching, salts, type_detail, form/release details

### Completed Work

#### #3: Fuzzy Matching ✅
- Added `lookup_generic_fuzzy()` using rapidfuzz (threshold 85%)
- Integrated as fallback after exact/synonym/prefix matches fail
- Fixes: AMOXICILIN→AMOXICILLIN, PARACETMOL→PARACETAMOL, LOSATAN→LOSARTAN

#### #4: Compound Salt Recognition ✅
- `SALT_CATIONS`: SODIUM, POTASSIUM, CALCIUM, etc.
- `SALT_ANIONS`: CHLORIDE, SULFATE, ACETATE, etc.
- `parse_compound_salt()`: "SODIUM CHLORIDE" → (SODIUM, CHLORIDE)
- `get_related_salts()`: Find salts sharing same anion

#### #6/#12: Type Detail Extraction ✅
- `extract_type_detail()` parses comma-separated type info
- "ALBUMIN, HUMAN" → type_detail="HUMAN"

#### #13: Release Detail Column ✅
- `extract_release_detail()` detects EXTENDED RELEASE, XR, SR, ER, etc.
- Whole-word matching prevents false positives

#### #14: Form Detail Column ✅
- `extract_form_detail()` detects FILM COATED, FC, EC, CHEWABLE, etc.
- Whole-word matching (RECOMBINANT doesn't match EC)

---

## Phase 5: Normalization 🔲 PENDING

**Goal:** Dose normalization, PNF improvements, stop words

### TODOs
- [ ] #8: Dose normalization to mg
- [ ] #20: PNF lexicon improvements
- [ ] #34: Dynamic stopword loading

---

## Phase 6: Performance 🔲 PENDING

**Goal:** Batch tagging

### TODOs
- [ ] #10: Add `tag_batch()` method to UnifiedTagger

---

## Phase 7: Fallbacks 🔲 PENDING

**Goal:** FDA food fallback, exact dose matching

### TODOs
- [ ] #21: FDA food catalog fallback
- [ ] #23: Exact dose matching mode for Part 4

---

## Phase 8: Cleanup 🔲 PENDING

**Goal:** Externalize hardcoded data, documentation, metrics

### TODOs
- [ ] #19: Externalize remaining hardcoded data
- [ ] #24: Complete folder reorganization (tagging, reference, utils)
- [ ] #31: Update documentation
- [ ] #33: Metrics tracking system

---

## Current Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Annex F tagging | 96.3% ATC, 73.6% DrugBank ID | Maximize |
| ESOA ATC tagging | 87.5% | 95%+ |
| ESOA→Drug Code | 7.0% | 60%+ |

---

## Key Files

| File | Purpose |
|------|---------|
| `debug/implementation_plan_v2.md` | Full TODO list with details |
| `debug/pipeline.md` | Algorithmic logic and pharmaceutical rules |
| `debug/progress.md` | This file - phase-based progress tracking |
| `AGENTS.md` | Agent instructions and policies |
| `pipelines/drugs/scripts/tagging/unified_constants.py` | Consolidated token sets |
| `dependencies/drugbank_generics/_shared.R` | Common R setup (packages, parallel, utilities) |
| `run_drugs_all.py` | Main pipeline runner + DrugBank execution |

---

## Commits Log

### Phase 1
1. `Phase 1 #22: Create unified_constants.py` - Main constants file
2. `Phase 1 #9: Add NEBS abbreviation` - Form abbreviation
3. `Phase 1 #25: Add missing tokens from unknown analysis` - Salt tokens + stopwords
4. `Fix #25: Remove drug components incorrectly added as stopwords` - Refinement
5. `Complete Phase 1 Analysis` - Phase completion

### Phase 2
6. `Phase 2: DrugBank R script optimization` - _shared.R, native shell execution
7. `DrugBank: default to min(8, cores) workers` - Fixed long runtime issue
8. `Phase 2 #0: Refresh all base datasets` - Part 1 complete (~460s total)
9. `Phase 2: Build unified reference` - DuckDB, generics/brands/mixtures lookups
10. `Phase 2 #16: Fix ESOA deduplication` - Added drop_duplicates to _concatenate_csv
11. `Phase 2 Complete` - All data foundation items done

### Phase 3
12. `Phase 3 Complete: Core Matching` - Brand swapping, combo matching, unified tagger

### Phase 4
13. `Phase 4: Enhancements` - Fuzzy matching, type/release/form detail extraction
14. `Phase 4 Complete: #4 Compound salt recognition` - Cation/anion parsing
