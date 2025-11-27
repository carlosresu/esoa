# Drug Pipeline Implementation Plan

**Created:** Nov 27, 2025  
**Objective:** Increase ESOA→Drug Code matching from 20.1% to 60%+

---

## Current State Summary

| Metric | Current | Target |
|--------|---------|--------|
| Annex F tagging | 97.0% | 99%+ |
| ESOA ATC tagging | 37.2% | 70%+ |
| ESOA→Drug Code | 20.1% | 60%+ |

### Key Bottlenecks
1. **Part 3 ESOA ATC tagging only 37.2%** - 162K rows have no ATC
2. **Generic mismatch** - 16K rows (single vs combo ATC codes)
3. **Dose mismatch** - 10K rows (same drug, different dose)
4. **Missing brand→generic swap** - Brand-only ESOA rows not converted

---

## Tier System for Matching

| Tier | Sources | Purpose |
|------|---------|---------|
| **1** | PNF + WHO + DrugBank Generics + DrugBank Mixtures | Known pharmaceuticals - tag with ATC/DrugBank ID |
| **2** | DrugBank Brands + FDA Drug Brands | Brand→Generic swap, then re-run Tier 1 |
| **3** | FDA Food | Food/supplement detection and normalization |
| **4** | Unknown | Parse and fallback match untagged rows |

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

1. **Phase 1A-1E:** DrugBank products/brands pipeline (foundation)
2. **Phase 2A-2B:** FDA data preparation
3. **Phase 3:** PNF improvements
4. **Phase 4:** Build unified tries
5. **Phase 5C:** Fix Part 3 ESOA tagging (biggest impact)
6. **Phase 6:** Brand swapping
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
