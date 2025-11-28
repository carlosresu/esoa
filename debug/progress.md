# Drug Pipeline Progress Tracker

**Started:** Nov 28, 2025  
**Last Updated:** Nov 28, 2025

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

## Phase 2: Data Foundation 🔲 PENDING

**Goal:** Build proper unified reference in DuckDB with all enrichments

### TODOs
- [ ] #11: Expand synonyms from DrugBank
- [ ] #15: Build form-route validity mapping
- [ ] #16: Create unified reference dataset
- [ ] #17: Load all sources into DuckDB
- [ ] #18: Build reference indexes
- [ ] #28: Extract multi-word generics list
- [ ] #29: Build brand → generic mapping
- [ ] #32: Dose normalization reference

---

## Phase 3: Core Matching 🔲 PENDING

**Goal:** Brand swapping, combo matching, unified tagger

### TODOs
- [ ] #1: Improve brand → generic swapping
- [ ] #2: Fix combo drug ATC assignment (single vs combo)
- [ ] #5: Order-independent combo matching
- [ ] #7: Unified tagger for Annex F and ESOA
- [ ] #27: Molecule-based fallback matching

---

## Phase 4: Enhancements 🔲 PENDING

**Goal:** Fuzzy matching, salts, type_detail, form/release details

### TODOs
- [ ] #3: Fuzzy matching for near-misses
- [ ] #4: Salt stripping with pure salt detection
- [ ] #6: type_detail column extraction
- [ ] #12: Form modifier handling (FILM COATED, etc.)
- [ ] #13: Release modifier handling (SR, XR, ER)
- [ ] #14: Parenthetical brand extraction

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

---

## Commits Log (Phase 1)

1. `Phase 1 #22: Create unified_constants.py` - Main constants file
2. `Phase 1 #9: Add NEBS abbreviation` - Form abbreviation
3. `Phase 1 #25: Add missing tokens from unknown analysis` - Salt tokens + stopwords
4. `Fix #25: Remove drug components incorrectly added as stopwords` - Refinement
5. `Complete Phase 1 Analysis` - Phase completion
