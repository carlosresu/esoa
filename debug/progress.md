# Drug Pipeline Progress Tracker

**Started:** Nov 28, 2025  
**Last Updated:** Dec 4, 2025 (Phase 9: Dose normalization fixes)

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
- **unified_drug_reference.csv**: 52,002 rows
- Exploded by: drugbank_id × atc_code × form × route
- Aggregated doses per combination

#### #15: Form-Route Validity Mapping ✅
- **form_route_validity.csv**: 53,039 combinations
- Sources: PNF, DrugBank products, FDA drug

#### #11: Synonyms from DrugBank ✅
- Already implemented in R script with proper filtering (language=english, not iupac-only)
- **generics_lookup.csv**: 7,345 generics with synonyms

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

## Phase 5: Normalization ✅ COMPLETE

**Goal:** Dose normalization, PNF improvements, stop words

### Completed Work

#### #34: Stop Word Filtering ✅
- Already implemented via STOPWORDS in unified_constants.py
- Tokenizer filters stopwords (AS, IN, FOR, WITH, etc.)
- Multi-word generics preserved before filtering

#### #8: Dose Denominator Normalization ✅
- `normalize_dose_ratio()`: 500MG/5ML → 100MG/ML
- `normalize_weight_to_mg()`: 1G → 1000MG, 500MCG → 0.5MG
- Converts g/mcg/ug to mg, ratios to per-1-ML

#### #20: PNF Lexicon Improvements ✅
- prepare_drugs.py already applies all normalizations
- Includes: extract_base_and_salts, clean_atc, dose parsing
- pnf_lexicon.csv properly normalized

---

## Phase 6: Performance ✅ COMPLETE

**Goal:** Batch tagging with chunking and caching

### Completed Work

#### #10: Batch Tagging Implementation ✅
- `tag_batch()`: Chunked processing with progress reporting
- `benchmark()`: Test chunk sizes 5K/10K/15K to find optimal
- Cached generics list for faster fuzzy matching

### Performance Results
- Before: ~168 rows/s
- After: ~230 rows/s (37% faster)
- 258K rows: ~19 minutes (was ~25 minutes)
- Optimal chunk size: 10,000 rows

---

## Phase 7: Fallbacks ✅ COMPLETE

**Goal:** FDA food fallback, Part 4 implementation

### Completed Work

#### #21: FDA Food Fallback ✅
- `load_fda_food_lookup()`: Load 31K FDA food entries
- `check_fda_food_fallback()`: Match untagged items against food data
- Identifies food/supplement items vs unknown drugs

#### #23: Part 4 - ESOA to Drug Code ✅
- `run_esoa_to_drug_code()`: New runner function
- Exact matching: generic name + ATC code
- Match reasons: matched_generic_atc, matched_generic_only, generic_not_in_annex

---

## Phase 8: Cleanup ✅ COMPLETE

**Goal:** Externalize hardcoded data, documentation, metrics

### Completed Work

#### #19: Hardcoded Values Audit ✅
- unified_constants.py: 732 lines of consolidated constants
- All token lists, form mappings, salt handling centralized
- Scripts import from unified_constants

#### #31: Documentation ✅
- pipeline.md: 765 lines comprehensive documentation
- Pharmaceutical matching principles, scoring algorithm
- Data sources, decision log, file audit, glossary

#### #33: Metrics Tracking ✅
- `log_metrics()`: Append to metrics_history.csv
- `get_metrics_summary()`: Read metrics history
- `print_metrics_comparison()`: Show latest vs previous
- Auto-logs after each Part 2/3/4 run

---

## Phase 9: Part 4 Dose Matching Enhancement ✅ COMPLETE

**Goal:** Improve ESOA → Annex F Drug Code matching through better dose parsing and matching

### Completed Work

#### Dose String Parsing ✅
- Parse `40MG`, `1G`, `300MG/2ML`, `100MG/ML` formats
- IU handling: `1000IU/ML`, `10 I.U`, `10IU/ML|5ML`
- Percentage w/v conversion: `5%` → 50 mg/mL
- Bare numeric doses: `25` → 25mg (assume MG for tablet range)
- Bottle size extraction: `250MG/5ML|60ML` → conc=50mg/mL, vol=60mL

#### Combination Dose Handling ✅  
- Parse `500MG+125MG` → total 625mg
- Parse Annex F `250|MG|125` → total 375mg
- Parse `400|MG|57|ML|35` → 457mg combo (suspension)
- Match combo totals across formats

#### IV Solution Inference ✅
- NSS/PNSS with volume only → assume 0.9% = 9 mg/mL
- D5 with volume only → assume 5% = 50 mg/mL
- D10 with volume only → assume 10% = 100 mg/mL

#### Form Equivalence Matching ✅
- TABLET ↔ FILM COATED, CHEWABLE, SUBLINGUAL
- CAPSULE ↔ SOFTGEL, GELCAP
- SYRUP ↔ SUSPENSION ↔ SOLUTION
- AMPULE ↔ VIAL ↔ INJECTION
- NEBULE ↔ INHALATION

#### Dose Tolerance ✅
- MG matching: 1% relative or 0.5mg absolute tolerance
- Concentration matching: 1% relative or 0.1 mg/mL absolute tolerance
- Volume not required for concentration matching (same conc = same drug)

### Fuzzy Match Verification ✅
- Analyzed 66,940 `generic_not_in_annex` entries
- Only 288 (0.43%) are typos/synonyms
- 99.57% are genuinely different drugs not in Annex F

### Match Rate Progression
| Change | Match Rate |
|--------|------------|
| Initial (zero tolerance) | 1.5% |
| + Dose string parsing | 32.1% |
| + IU handling | 33.1% |
| + Volume-only inference | 33.6% |
| + NSS/D5/D10 inference | 34.0% |
| + Bare numeric doses | 34.0% |
| + Form equivalence | 34.3% |
| + Tolerance matching | 34.3% |
| + Fixed vial size parsing | 36.4% |
| + Strict dose requirement | **34.8%** |

#### Vial Size Parsing Fix
- `250|MG|1|G` was incorrectly parsed as combo (250mg + 1000mg = 1250mg)
- Fixed: Now correctly parses as 250mg (the `1|G` is vial size, not second dose)
- Pattern: If previous dose was in MG and current is in G with small value (≤10), treat as vial size

#### Strict Dose Matching Policy
- Dose matching is REQUIRED - no fallback when dose key is None
- Only allow unit conversions: 500mcg = 0.5mg, 1g = 1000mg
- Different doses never match: 400mg ≠ 600mg
- Bare numbers assumed to be MG: "275" → 275mg (for cases like "FLANAX 275")

---

## Current Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Annex F tagging | 86.6% ATC, 69.3% DrugBank ID | Maximize |
| ESOA ATC tagging | 67.0% | 95%+ |
| ESOA→Drug Code | **34.8%** | 60%+ |

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

### Phase 5
15. `Phase 5 Complete: Normalization` - Dose ratio normalization, stopwords, PNF improvements

### Phase 6
16. `Phase 6 Complete: Performance` - Batch tagging with chunking, cached fuzzy matching

### Phase 7
17. `Phase 7 Complete: Fallbacks` - Part 4 implementation, FDA food fallback

### Phase 8
18. `Phase 8 Complete: Cleanup` - Metrics tracking, documentation complete

### Post-Phase Improvements
19. `Vaccine Acronym Bidirectional Matching` - WHO/CDC standard vaccine abbreviations (DTP, MMR, PENTA, etc.) with bidirectional matching between acronyms and components
20. `Diluent Handling Confirmed` - Verified diluent presence doesn't affect generic detection (only dose, which is flexible at ATC stage)

### Phase 9
21. `Phase 9: Part 4 Dose Matching Enhancement` - Parse dose strings, handle combos, IV solution inference
22. `Form Equivalence Groups` - TABLET↔FILM COATED, SYRUP↔SUSPENSION, AMPULE↔VIAL
23. `Dose Tolerance Matching` - 1% relative tolerance for floating point precision
24. `Bare Numeric Dose Parsing` - "25" → 25mg for tablet-range values
25. `Vial Size Parsing Fix` - "250|MG|1|G" correctly parses as 250mg, not 1250mg combo
26. `Strict Dose Requirement` - Dose matching required, no fallback when dose missing
