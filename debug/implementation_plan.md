# Drug Pipeline Implementation Plan

**Created:** Nov 27, 2025  
**Updated:** Nov 27, 2025  
**Objective:** Unified drug tagging with consistent algorithms for Annex F and ESOA

---

## Current State Summary (Nov 27, 2025)

| Metric | Current | Target |
|--------|---------|--------|
| Annex F tagging | 97.0% | 99%+ |
| ESOA ATC tagging | 87.5% | 95%+ |
| ESOA→Drug Code | 7.0% | 60%+ |

### Architecture Decision
**Use Annex F tagging algorithm as base** (97% accuracy) and enhance with:
- DuckDB for all queries (faster than Aho-Corasick for our use case)
- Unified reference dataset
- Salt detection via drugbank$salts
- Consistent algorithm for both Annex F and ESOA

---

## NEW: Unified Architecture

### Core Principle
Both Annex F and ESOA are drug description texts. They should use **identical tagging algorithms**.

### Three Reference Structures (via DuckDB)
1. **Generics Table**: All single-entity drug names (base generics, normalized)
2. **Brands Table**: FDA + DrugBank brands → maps to generic(s)
3. **Mixtures Table**: Component combinations → maps to mixture info

### Unified Reference Dataset Schema
```
drugbank_id | atc_code | generic_name | form | route | salts | doses | brands | mixtures | sources
```
- **Exploded by**: `drugbank_id × atc_code × form × route` (only valid combos)
- **Aggregated**: `salts`, `doses`, `brands`, `mixtures` as pipe-delimited

### Dose Normalization
- Weight: normalize to `mg` (e.g., `1g` → `1000mg`)
- Combinations: `500mg+200mg` (fixed combo)
- Concentration: `10mg/mL`, `5%`
- Volume: `mL` is canonical for liquids

### Salt Handling
- Use `drugbank$salts` dataset for salt detection
- Strip salts from matching basis UNLESS pure salt compound (e.g., sodium chloride)
- Auto-detect pure salts: compounds where base would be empty after stripping

---

## Phase 1: Build Unified Reference Dataset

### 1.1 Load Source Datasets into DuckDB
Load all sources:
- `pnf_lexicon.csv` / `pnf_prepared.csv`
- `who_atc_*.parquet`
- `drugbank_generics_master.csv`
- `drugbank_mixtures_master.csv`
- `drugbank_brands_master.csv`
- `drugbank_products_export.csv`
- `fda_drug_*.csv`
- `annex_f.csv`

### 1.2 Extract Salts from DrugBank
**New R Script:** `dependencies/drugbank_generics/drugbank_salts.R`
- Extract `dbdataset::drugbank$salts`
- Output: `drugbank_salts_master.csv`
- Use for salt detection and stripping

### 1.3 Normalize Generics
- Strip salts using salts dataset (unless pure salt compound)
- Handle synonyms (PARACETAMOL ↔ ACETAMINOPHEN)
- Normalize to canonical DrugBank name

### 1.4 Extract Form-Route Validity Mapping
**Output:** `form_route_validity.parquet` (longform with provenance)
```
form | route | source | example_drugbank_id
tablet | oral | pnf | DB00316
solution | intravenous | drugbank_products | DB00316
```

### 1.5 Build Unified Dataset
Explosion logic: `drugbank_id × atc_code × form × route`
- Only valid form-route combinations
- Aggregate salts, doses, brands, mixtures as pipe-delimited
- Track sources

### 1.6 Export and Index
- Export as `unified_drug_reference.parquet`
- Create DuckDB indexes for fast queries

---

## Phase 2: Create Unified Tagger

### 2.1 Port Annex F Tokenization
Use Part 2's tokenization and categorization:
- GENERIC, DOSE, FORM, ROUTE, SALT, OTHER categories
- Handle parentheses, commas, special characters

### 2.2 Replace Tries with DuckDB Queries
```sql
-- Find generic matches
SELECT * FROM unified WHERE generic_name IN (unnest(?tokens))

-- Find brand matches and get generics
SELECT generic_name FROM brands WHERE brand_name IN (unnest(?tokens))

-- Find mixture matches
SELECT * FROM mixtures WHERE component_key = ?key
```

### 2.3 Port Scoring Algorithm
From Part 2:
- Primary score: GENERIC×5, SALT×4, DOSE×4, FORM×3, ROUTE×3
- Secondary score: form/route tie-breaking
- ATC preference: single vs combo

### 2.4 Add Salt Stripping
- Query salts table to detect salt tokens
- Strip unless pure salt compound
- Re-query with base generic

### 2.5 Unified Tagger Module
**New File:** `pipelines/drugs/scripts/unified_tagger.py`
- Single entry point for both Annex F and ESOA
- Uses DuckDB for all lookups
- Consistent scoring

---

## Phase 3: Refactor Pipeline Parts

### 3.1 Part 2 (Annex F)
- Call unified tagger
- Output format unchanged for Part 4 compatibility

### 3.2 Part 3 (ESOA)
- Call unified tagger (same as Part 2)
- Remove redundant multi-step matching

### 3.3 Part 4 (Bridging)
- Use unified reference for matching
- Simplified logic with DuckDB

---

## Phase 4: Verification and Documentation

### 4.1 Verify Coverage
- Annex F: should maintain 97%+
- ESOA: should improve from 87.5%
- Bridging: should improve from 7%

### 4.2 Update Documentation
- AGENTS.md with new policies
- Memories with new architecture
- README updates

---

## Legacy Phases (Superseded)

---

## Phase 1: DrugBank Products & Brands Pipeline

### 1A. Extract DrugBank Products Data (R Script)
**File:** `dependencies/drugbank_generics/drugbank_products.R`

Extract from `dbdataset::drugbank$products`:
- `drugbank_id` - links to parent drug
- `name` - product name (brand if generic=false, generic if generic=true)
- `generic` - boolean flag
- `dosage_form` - form info
- `strength` - dose info
- `route` - route info
- `country` - market info
- `approved` - approval status

**Output:** `output/drugbank_products_raw.csv`

### 1B. Enrich Generics with Product Info
**Goal:** Add novel salt forms, doses, forms, routes from products where `generic=true`

For each product where `generic=true`:
1. Match to `drugbank_id` in generics master
2. Extract dose/form/route from product
3. Add to generics master as additional variants (NOT new synonyms)

**Output:** Enhanced `drugbank_generics_master.csv` with additional dose/form/route columns

### 1C. Build DrugBank Brands Master
**File:** `dependencies/drugbank_generics/drugbank_brands.R`

For products where `generic=false`:
1. Extract `name` as brand name
2. Link to `drugbank_id` → get generic name(s)
3. Also include brand names from `drugbank_mixtures_master.csv` (`mixture_name` column)

**Output:** `output/drugbank_brands_master.csv`
- Columns: `brand_name`, `brand_name_normalized`, `drugbank_id`, `generic_name`, `is_mixture`

### 1D. Update drugbank_all.R
Add `drugbank_brands.R` to the subscripts list in `drugbank_all.R`

### 1E. Update run_drugs_all.py
- Copy `drugbank_brands_master.csv` to inputs/drugs
- Update `refresh_drugbank_generics_exports()` to handle brands

---

## Phase 2: FDA Data Preparation

### 2A. FDA Drug Brands Master
**Current:** Raw file `fda_drug_2025-11-12.csv` with columns:
- `brand_name`, `generic_name`, `dosage_form`, `route`, `dosage_strength`, `registration_number`

**Create:** `fda_drugs_master.csv`
- Normalize brand names (uppercase, trim)
- Normalize generic names (map to canonical)
- Parse dose/form/route
- Deduplicate

### 2B. FDA Food Master
**Current:** Raw file `fda_food_2025-11-23.csv` with columns:
- `brand_name`, `product_name`, `company_name`, `registration_number`

**Create:** `fda_food_master.csv`
- Extract product type from `product_name` (Raw Material, Low/Medium/High Risk Food)
- Normalize names
- This is for detecting non-drug items like AKAPULKO, YERBA BUENA

---

## Phase 3: PNF Preparation Improvements

### 3A. Analyze Current PNF Issues
- Check for dirty/inconsistent data
- Identify missing synonyms
- Check ATC code coverage

### 3B. PNF Cleaning
- Normalize generic names
- Add missing synonyms
- Validate ATC codes against WHO

---

## Phase 4: Unified Reference Tries

### 4A. Generics Trie
Build Aho-Corasick trie from:
- PNF generic names + synonyms
- WHO ATC names
- DrugBank canonical names + synonyms
- All normalized to lowercase

### 4B. Mixtures Trie
Build from:
- DrugBank mixtures (component combinations)
- PNF combination drugs

### 4C. Brands Trie
Build from:
- DrugBank brands (products where generic=false)
- FDA drug brands
- DrugBank mixture names (brand names of combos)

---

## Phase 5: Pipeline Part Improvements

### 5A. Part 1 - Prepare Dependencies
- [ ] Add DrugBank brands extraction
- [ ] Add FDA drugs master preparation
- [ ] Add FDA food master preparation
- [ ] Improve PNF preparation

### 5B. Part 2 - Annex F ATC Tagging
- [ ] Get to 99%+ tagging
- [ ] Manually analyze remaining untagged
- [ ] Parse untagged for fallback matching

### 5C. Part 3 - ESOA ATC Tagging (CRITICAL)
- [ ] Investigate why only 37.2% tagged
- [ ] Implement brand→generic swap before tagging
- [ ] Fix single vs combo ATC assignment
- [ ] Use unified generics trie for detection

### 5D. Part 4 - ESOA to Annex F Matching
- [ ] Allow dose-flexible matching
- [ ] Implement molecule-based fallback
- [ ] Use fallback index for untagged rows

---

## Phase 6: Brand Swapping Logic

### 6A. Brand Detection
Use brands trie to detect brand names in ESOA raw_text

### 6B. Brand→Generic Swap
For ESOA rows with brand but no generic:
1. Look up brand in brands trie
2. Get corresponding generic(s)
3. Replace brand with generic in text
4. Avoid duplication (e.g., "acetaminophen (tylenol)" → "acetaminophen", not "acetaminophen (acetaminophen)")

### 6C. Re-run Tagging
After brand swap, re-run Tier 1 tagging for newly converted rows

---

## Phase 7: Fallback Matching (Tier 3 & 4)

### 7A. FDA Food Matching
For rows not matching Tier 1 or 2:
- Run through FDA food trie
- Normalize/standardize food product names
- Bridge Annex F untagged ↔ ESOA untagged

### 7B. Unknown Category
For rows failing all tiers:
- Parse as best as possible
- Match Annex F untagged ↔ ESOA untagged by parsed components

---

## Implementation Order

1. **Phase 1A-1E:** DrugBank products/brands pipeline (foundation) ✅ DONE
   - Created drugbank_brands.R extracting 208,999 brand entries
   - Created drugbank_products_export.csv with 455,970 products
   - Integrated DrugBank brands into brand_map_drugs.py
   - Total brands now: 239,638 (DrugBank 208,999 + FDA 30,639)
2. **Phase 2A-2B:** FDA data preparation
3. **Phase 3:** PNF improvements
4. **Phase 4:** Build unified tries
5. **Phase 5C:** Fix Part 3 ESOA tagging (biggest impact)
6. **Phase 6:** Brand swapping - PARTIALLY DONE (integrated into Part 3)
7. **Phase 5B:** Maximize Annex F tagging
8. **Phase 5D:** Improve Part 4 matching
9. **Phase 7:** Fallback matching

---

## Files to Create/Modify

### New Files
- `dependencies/drugbank_generics/drugbank_products.R`
- `dependencies/drugbank_generics/drugbank_brands.R` (replace placeholder)
- `pipelines/drugs/scripts/prepare_fda_drugs.py`
- `pipelines/drugs/scripts/prepare_fda_food.py`
- `pipelines/drugs/scripts/build_unified_tries.py`
- `pipelines/drugs/scripts/brand_swap.py`

### Modified Files
- `dependencies/drugbank_generics/drugbank_all.R` - add brands script
- `run_drugs_all.py` - handle new outputs
- `run_drugs_pt_1_prepare_dependencies.py` - add new preparation steps
- `run_drugs_pt_3_esoa_atc.py` - add brand swap, fix ATC assignment
- `run_drugs_pt_4_esoa_to_annex_f.py` - dose-flexible matching, fallback

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | DrugBank brands extracted | 10K+ brands |
| Phase 2 | FDA drugs prepared | 30K rows cleaned |
| Phase 3 | PNF cleaned | 3K rows validated |
| Phase 5B | Annex F tagging | 99%+ |
| Phase 5C | ESOA ATC tagging | 70%+ |
| Phase 5D | ESOA→Drug Code | 60%+ |

---

## Notes

- Always use 8 workers when running pipeline scripts
- Update Cascade memories as progress is made
- Standardize column names across all datasets for multi-reference algorithms
- Keep datasets separate but ensure consistent key columns
