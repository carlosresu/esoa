# Drug Pipeline - Algorithmic Logic & Pharmaceutical Rules

**Created:** Nov 27, 2025  
**Updated:** Nov 27, 2025  

> **IMPORTANT:** This document captures all algorithmic logic, decisions, choices, rules, and pharmaceutical principles used in the drug tagging pipeline. Update this file with every group of changes.

---

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Pharmaceutical Matching Principles](#pharmaceutical-matching-principles)
3. [Scoring Algorithm](#scoring-algorithm)
4. [Data Sources](#data-sources)
5. [Normalization Rules](#normalization-rules)
6. [Form-Route Mappings](#form-route-mappings)
7. [Synonym Handling](#synonym-handling)
8. [Salt Handling](#salt-handling)
9. [Combination Drug Matching](#combination-drug-matching)
10. [Brand Resolution](#brand-resolution)
11. [Dose Handling](#dose-handling)
12. [Column Definitions](#column-definitions)
13. [Decision Log](#decision-log)

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PART 1: PREPARE                          │
│  Refresh: WHO ATC, DrugBank (R scripts), FDA, PNF              │
│  Output: inputs/drugs/*.parquet                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PART 2: TAG ANNEX F                          │
│  Input: raw/drugs/annex_f.csv                                  │
│  Process: UnifiedTagger assigns ATC + DrugBank ID              │
│  Output: annex_f_with_atc.csv                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PART 3: TAG ESOA                           │
│  Input: esoa_combined.parquet                                  │
│  Process: UnifiedTagger assigns ATC + DrugBank ID              │
│  Output: esoa_with_atc.parquet                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PART 4: BRIDGE TO DRUG CODE                   │
│  Input: esoa_with_atc + annex_f_with_atc                       │
│  Process: Match by ATC/DrugBank, EXACT dose required           │
│  Output: esoa_matched_drug_codes.csv                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pharmaceutical Matching Principles

### Core Principle
**The active ingredient(s) determine the drug identity.** Everything else (salt form, brand name, manufacturer) is secondary.

### Matching Hierarchy (Deterministic, Not Numeric)

1. **Generic Match** - REQUIRED
   - Must match the active ingredient(s)
   - This is the fundamental requirement for any match
   - No match without generic match

2. **Salt Form** - IGNORED (with exception)
   - Salts are delivery mechanisms, not active ingredients
   - `LOSARTAN POTASSIUM` ≈ `LOSARTAN` (same drug)
   - `AMLODIPINE BESYLATE` ≈ `AMLODIPINE` (same drug)
   - **Exception:** Pure salts ARE the active compound
     - `SODIUM CHLORIDE` - sodium and chloride are both active
     - `POTASSIUM CHLORIDE` - potassium is the active ingredient
     - `CALCIUM CARBITE` - calcium is the active ingredient

3. **Dose** - CONTEXT-DEPENDENT
   - **ATC/DrugBank tagging:** Dose-flexible (same drug = same ATC regardless of dose)
   - **Drug Code matching:** Dose-exact (Drug Code is unique down to dose)

4. **Form** - FLEXIBLE with equivalence groups
   - Pharmaceutically equivalent forms can match
   - See [Form-Route Mappings](#form-route-mappings)

5. **Route** - INFERRED from form if missing
   - See [Form-Route Mappings](#form-route-mappings)

6. **Synonyms** - EQUIVALENT
   - `SALBUTAMOL` = `ALBUTEROL` (same drug, different naming conventions)
   - `PARACETAMOL` = `ACETAMINOPHEN`
   - See [Synonym Handling](#synonym-handling)

---

## Scoring Algorithm

### Philosophy
Scoring is deterministic based on pharmaceutical principles, not arbitrary numeric weights.

### Match Decision Tree

```
1. Does generic match? (after synonym normalization)
   NO  → NO MATCH
   YES → Continue

2. Is this a combination drug?
   YES → Do ALL components match? (order-independent)
         NO  → NO MATCH
         YES → Continue
   NO  → Continue

3. For ATC/DrugBank assignment:
   - Ignore salt differences (unless pure salt)
   - Ignore dose differences
   - Prefer form match, but allow equivalents
   - Infer route from form if missing

4. For Drug Code matching:
   - Require EXACT dose match
   - Allow equivalent forms
   - Infer route from form if missing

5. Tie-breaking (when multiple candidates):
   a. Prefer exact form match over equivalent
   b. Prefer single-drug ATC for single drugs
   c. Prefer combo ATC for combination drugs
```

### Match Reasons (for debugging)
- `exact` - All fields match exactly
- `synonym_match` - Matched via synonym
- `form_equivalent` - Form matched via equivalence group
- `route_inferred` - Route was inferred from form
- `combo_match` - Combination drug matched (order-independent)
- `salt_stripped` - Salt form was ignored
- `no_match` - No match found
- `generic_mismatch` - Generic names don't match
- `dose_mismatch` - Doses don't match (Part 4 only)

---

## Data Sources

### Architecture: DrugBank as Base, Enriched by Others

**DrugBank is the primary/base reference.** It is enriched with data from PNF, WHO, and FDA:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRUGBANK (BASE)                              │
│  - drugbank_generics_master (~392K)                            │
│  - drugbank_mixtures_master (~153K)                            │
│  - drugbank_brands_master (~209K)                              │
│  - drugbank_products_export (~456K)                            │
│  - drugbank_salts_master (~3K)                                 │
└─────────────────────────────────────────────────────────────────┘
                    ↑ ENRICHED BY ↑
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│     PNF      │  │     WHO      │  │   FDA DRUG   │  │   FDA FOOD   │
│   (~3K)      │  │   (~6K)      │  │   (~31K)     │  │   (~XXK)     │
│  PH formulary│  │  ATC codes   │  │  PH brands   │  │  Fallback    │
│  synonyms    │  │  DDD info    │  │  dose/form   │  │  for non-drug│
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

### Primary Reference Datasets

| Dataset | Source | Rows | Purpose |
|---------|--------|------|---------|
| `drugbank_generics_master` | DrugBank | ~392K | **BASE** - Generic drug info |
| `drugbank_mixtures_master` | DrugBank | ~153K | Combination drugs |
| `drugbank_brands_master` | DrugBank | ~209K | Brand names |
| `drugbank_products_export` | DrugBank | ~456K | Product variants (dose/form/route) |
| `drugbank_salts_master` | DrugBank | ~3K | Salt forms |

### Enrichment Datasets

| Dataset | Source | Rows | Enriches With |
|---------|--------|------|---------------|
| `pnf_lexicon.parquet` | Philippine National Formulary | ~3K | PH-specific synonyms, ATC codes |
| `who_atc_*.parquet` | WHO ATC Index | ~6K | ATC codes, DDD (Defined Daily Dose) |
| `fda_drug_*.parquet` | FDA Philippines | ~31K | PH brand names, dose/form/route |

### Fallback Dataset

| Dataset | Source | Rows | Purpose |
|---------|--------|------|---------|
| `fda_food_*.parquet` | FDA Philippines | ~135K | Last resort for non-drug items (herbs, supplements) |

### No Source Priority
There is NO preference between sources. All sources contribute to a single unified reference. The unified reference is the authority.

---

## Normalization Rules

### Text Normalization
1. Convert to UPPERCASE
2. Remove extra whitespace
3. Standardize punctuation
4. Handle parentheses (often contain brand names)
5. Filter stop words (see below)

### Stop Words
These words are noise and should be filtered during tokenization:
```
AS, IN, FOR, TO, WITH, EQUIV., AND, OF, OR, NOT, THAN, HAS, DURING, THIS, W/
```

**Exception:** `PER` should be KEPT when indicating ingredient ratio:
- `10MG PER ML` → keep `PER` (it's part of the dose)
- `FOR INJECTION` → filter `FOR` (noise)

### Dose Normalization
1. Normalize to per-1-unit denominators:
   - `500MG/5ML` → `100MG/ML`
   - `1G/10ML` → `100MG/ML`
2. Standardize units:
   - `1G` → `1000MG`
   - `1000MCG` → `1MG`
3. Preserve concentration format:
   - `10MG/ML` (keep as-is)
   - `5%` (keep as-is)

### Generic Name Normalization
1. Strip salt suffixes (unless pure salt)
2. Apply synonym mapping
3. Normalize spacing in multi-word names

---

## Form-Route Mappings

### Form Equivalence Groups
Forms within the same group are pharmaceutically equivalent for matching purposes:

| Group | Forms | Rationale |
|-------|-------|-----------|
| Oral Solid | TABLET, CAPSULE | Both oral solid dosage forms |
| Injectable | AMPULE, VIAL, INJECTION | All parenteral containers |
| Liquid Oral | SOLUTION, SUSPENSION, SYRUP, ELIXIR | All oral liquids |
| Topical | CREAM, OINTMENT, GEL, LOTION | All topical preparations |
| Inhalation | INHALER, NEBULE, MDI, DPI | All respiratory delivery |

### Form → Route Inference
When route is not specified, infer from form:

| Form | Inferred Route |
|------|----------------|
| TABLET | ORAL |
| CAPSULE | ORAL |
| SYRUP | ORAL |
| SUSPENSION | ORAL |
| SACHET | ORAL |
| AMPULE | PARENTERAL |
| VIAL | PARENTERAL |
| INJECTION | PARENTERAL |
| CREAM | TOPICAL |
| OINTMENT | TOPICAL |
| GEL | TOPICAL |
| DROPS | OPHTHALMIC (or context-dependent) |
| INHALER | INHALATION |
| NEBULE | INHALATION |
| SUPPOSITORY | RECTAL |
| PATCH | TRANSDERMAL |

### Form Abbreviations
| Abbreviation | Full Form |
|--------------|-----------|
| TAB | TABLET |
| CAP | CAPSULE |
| AMP | AMPULE |
| INJ | INJECTION |
| SOLN | SOLUTION |
| SUSP | SUSPENSION |
| SUPP | SUPPOSITORY |
| NEB | NEBULE |
| NEBS | NEBULE |
| GTTS | DROPS (from Latin "guttae") |

---

## Synonym Handling

### Principle
Synonyms are different names for the SAME drug. They should be treated as equivalent.

### Common Synonyms
| Name 1 | Name 2 | Region |
|--------|--------|--------|
| SALBUTAMOL | ALBUTEROL | UK/US |
| PARACETAMOL | ACETAMINOPHEN | UK/US |
| ADRENALINE | EPINEPHRINE | UK/US |
| FRUSEMIDE | FUROSEMIDE | UK/US |
| LIGNOCAINE | LIDOCAINE | UK/US |

### Synonym Sources
1. `drugbank$drugs$synonyms` where:
   - `language == 'english'`
   - `coder` is not empty
   - `coder` is not solely "iupac"
2. Hardcoded regional variants (UK/US naming)
3. Combination drug synonyms (e.g., CO-AMOXICLAV = AMOXICILLIN + CLAVULANIC ACID)

### Synonym Application
1. Normalize input text through synonym map
2. For combinations, normalize EACH component
3. Then match against reference

---

## Salt Handling

### Principle
Salts are delivery mechanisms. The base compound is the active ingredient.

### Salt Stripping Rules
1. **Strip salt suffix** from generic name for matching
   - `LOSARTAN POTASSIUM` → `LOSARTAN`
   - `AMLODIPINE BESYLATE` → `AMLODIPINE`
   
2. **Exception: Pure Salts**
   - If stripping would leave empty string, it's a pure salt
   - `SODIUM CHLORIDE` - don't strip, this IS the drug
   - `POTASSIUM CHLORIDE` - don't strip
   - `CALCIUM CARBONATE` - don't strip

3. **Compound Salt Recognition**
   - `SODIUM CHLORIDE` shares anion with other chloride salts
   - Can map `SODIUM` ↔ `SODIUM CHLORIDE` for matching
   - Use `drugbank$salts` for anion→cation mapping

### Common Salt Suffixes
- HYDROCHLORIDE, HCL
- SODIUM, NA
- POTASSIUM, K
- SULFATE
- PHOSPHATE
- ACETATE
- BESYLATE
- MALEATE
- TARTRATE

---

## Combination Drug Matching

### Principle
Combination drugs contain multiple active ingredients. Order doesn't matter.

### Matching Rules
1. **Normalize component order** - sort alphabetically
   - `PIPERACILLIN + TAZOBACTAM` = `TAZOBACTAM + PIPERACILLIN`
   
2. **Apply synonyms to each component**
   - `IPRATROPIUM + SALBUTAMOL` → `IPRATROPIUM + ALBUTEROL`
   
3. **All components must match**
   - Can't match `LOSARTAN` to `LOSARTAN + HCTZ`
   
4. **ATC preference**
   - Single drug → prefer single-drug ATC
   - Combination → prefer combination ATC

### Combination Delimiters
- ` + ` (space-plus-space)
- ` AND `
- `/` (sometimes)
- `,` (sometimes, but also used for subtypes)

---

## Brand Resolution

### Principle
Brand names should be resolved to generic names before matching.

### Brand Sources (in order of reliability)
1. `drugbank$products$name` where `generic == 'false'`
2. `drugbank$drugs$international_brands$brand`
3. `drugbank$drugs$mixtures$name` (brand names of combinations)
4. `fda_drug_*$brand_name` (with swap detection)

### FDA Brand Swap Detection
Some FDA rows have brand/generic swapped. Detect by:
- If `brand_name` matches a known generic exactly
- AND `generic_name` matches no known generic
- THEN the cells are swapped for that row

### Brand Resolution Process
1. Tokenize input text
2. Check each token against brands lookup
3. If brand found, get corresponding generic(s)
4. Replace brand with generic in matching
5. Avoid duplication (don't create "PARACETAMOL (PARACETAMOL)")

---

## Dose Handling

### ATC/DrugBank Tagging (Parts 2 & 3)
- **Dose-flexible**: Same drug at different doses = same ATC
- PARACETAMOL 500MG and PARACETAMOL 650MG both get N02BE01

### Drug Code Matching (Part 4)
- **Dose-exact**: Drug Code is unique down to dose
- PARACETAMOL 500MG TABLET ≠ PARACETAMOL 650MG TABLET
- Different Drug Codes for different doses

### Dose Extraction Patterns
```
\d+(?:\.\d+)?\s*(MG|G|MCG|UG|IU|ML|%|MG/ML|MCG/ML)
```

### Dose Normalization
- `500MG/5ML` → `100MG/ML` (divide to per-1-unit)
- `1G` → `1000MG`
- `0.5%` → `0.5%` (keep percentage as-is)

---

## Column Definitions

### Output Columns

| Column | Description |
|--------|-------------|
| `atc_code` | WHO ATC code (e.g., N02BE01) |
| `drugbank_id` | DrugBank identifier (e.g., DB00316) |
| `generic_name` | Canonical generic name |
| `dose` | Normalized dose (e.g., 500MG, 10MG/ML) |
| `form` | Base dosage form (e.g., TABLET, AMPULE) |
| `route` | Administration route (e.g., ORAL, PARENTERAL) |
| `type_detail` | Subtype after comma (e.g., HUMAN for "ALBUMIN, HUMAN") |
| `release_detail` | Release modifier (e.g., EXTENDED RELEASE) |
| `form_detail` | Form modifier (e.g., FILM COATED) |
| `match_score` | Numeric score for ranking |
| `match_reason` | Why this match was selected |
| `source` | Which reference dataset matched |

### Unified Reference Columns

| Column | Description |
|--------|-------------|
| `drugbank_id` | Primary key |
| `atc_code` | ATC code (exploded if multiple) |
| `generic_name` | Canonical name |
| `form` | Dosage form (exploded) |
| `route` | Route (exploded) |
| `doses` | Pipe-delimited known doses |
| `salt_forms` | Pipe-delimited salt forms |
| `brands_single` | Pipe-delimited brand names |
| `brands_combination` | Brands of combos containing this |
| `mixtures_atc` | ATCs of mixtures containing this |
| `mixtures_drugbank` | DrugBank IDs of mixtures containing this |
| `synonyms` | Pipe-delimited synonyms |
| `sources` | Pipe-delimited data sources |

---

## Decision Log

### 2025-11-27: Initial Architecture
- **Decision:** Use Annex F tagging algorithm as base (97% accuracy)
- **Rationale:** Proven accuracy, well-tested

### 2025-11-27: DuckDB over Aho-Corasick
- **Decision:** Use DuckDB for all lookups instead of Aho-Corasick tries
- **Rationale:** Faster for exact/prefix matching after tokenization, easier to maintain

### 2025-11-27: Salt Handling
- **Decision:** Ignore salts for matching (except pure salts)
- **Rationale:** Salts are delivery mechanisms, not active ingredients

### 2025-11-27: Order-Independent Combos
- **Decision:** Sort components alphabetically before matching
- **Rationale:** `A + B` and `B + A` are the same drug

### 2025-11-27: Dose Flexibility
- **Decision:** Dose-flexible for ATC tagging, dose-exact for Drug Code matching
- **Rationale:** ATC is drug-level, Drug Code is product-level

### 2025-11-27: Scoring Philosophy
- **Decision:** Deterministic pharmaceutical rules over numeric weights
- **Rationale:** More interpretable, based on actual drug science

---

## State of the Pipeline

### Current Metrics (Nov 27, 2025)

| Metric | Value | Status |
|--------|-------|--------|
| **Annex F → ATC** | 94.1% (2,284/2,427) | ✅ Good |
| **Annex F → DrugBank ID** | 73.6% (1,787/2,427) | ⚠️ Needs improvement |
| **ESOA → ATC** | 55.6% (143,900/258,878) | ⚠️ Needs improvement |
| **ESOA → Drug Code** | 40.5% (104,800/258,878) | ⚠️ Target: 60%+ |

### Pipeline Parts Status

#### Part 1: Prepare Dependencies ✅ WORKING
**Script:** `run_drugs_pt_1_prepare_dependencies.py`

| Component | Status | Notes |
|-----------|--------|-------|
| WHO ATC refresh | ✅ Working | Via `dependencies/atcd/` R scripts |
| DrugBank refresh | ✅ Working | Via `dependencies/drugbank_generics/` R scripts |
| FDA brand map | ✅ Working | Via `pipelines/drugs/scripts/brand_map_drugs.py` |
| FDA food catalog | ✅ Working | Via `dependencies/fda_ph_scraper/` |
| PNF preparation | ✅ Working | Via `pipelines/drugs/scripts/prepare_drugs.py` |
| Annex F verification | ✅ Working | Checks `raw/drugs/annex_f.csv` exists |

**Known Issues:**
- ESOA part detection (`esoa_pt_*.csv`) not working properly - currently manually combined

---

#### Part 2: Tag Annex F ✅ WORKING
**Script:** `run_drugs_pt_2_annex_f_atc.py` → `pipelines/drugs/scripts/runners.py`

| Feature | Status | Notes |
|---------|--------|-------|
| UnifiedTagger | ✅ Working | Uses DuckDB for lookups |
| Generic matching | ✅ Working | Synonym normalization, salt stripping |
| ATC assignment | ✅ Working | 94.1% coverage |
| DrugBank ID assignment | ⚠️ Partial | 73.6% coverage |

**Output:** `outputs/drugs/annex_f_with_atc.csv`

---

#### Part 3: Tag ESOA ⚠️ NEEDS IMPROVEMENT
**Script:** `run_drugs_pt_3_esoa_atc.py` → `pipelines/drugs/scripts/runners.py`

| Feature | Status | Notes |
|---------|--------|-------|
| UnifiedTagger | ✅ Working | Same as Part 2 |
| Batch processing | ⚠️ Slow | 5K batch size, no true batch tagging |
| Brand → Generic | ❌ Not implemented | Major gap |
| ATC assignment | ⚠️ Partial | 55.6% coverage |

**Output:** `outputs/drugs/esoa_with_atc.csv`

**Known Issues:**
- No brand name resolution (BIOGESIC → PARACETAMOL)
- Slow row-by-row processing (258K rows)
- Many common drugs not getting ATC codes

---

#### Part 4: Bridge ESOA to Drug Code ⚠️ NEEDS IMPROVEMENT
**Script:** `run_drugs_pt_4_esoa_to_annex_f.py`

| Feature | Status | Notes |
|---------|--------|-------|
| ATC-based matching | ✅ Working | Primary matching method |
| DrugBank ID matching | ✅ Working | Secondary matching method |
| Dose-exact matching | ✅ Working | Required for Drug Code |
| Form equivalence | ✅ Working | TABLET ≈ CAPSULE |
| Fallback matching | ⚠️ Basic | Molecule-based fallback exists |
| Order-independent combos | ❌ Not implemented | PIPERACILLIN + TAZOBACTAM ≠ TAZOBACTAM + PIPERACILLIN |

**Output:** `outputs/drugs/esoa_matched_drug_codes.csv`

**Known Issues:**
- Low match rate (40.5%) due to Part 3 gaps
- Combination drug order matters (shouldn't)
- Missing synonym mappings for common drugs

---

### Reference Datasets Status

#### Generated Lookups (in `outputs/drugs/`)

| File | Rows | Status |
|------|------|--------|
| `generics_lookup.parquet` | ~6K | ✅ Generated |
| `brands_lookup.parquet` | ~126K | ✅ Generated |
| `mixtures_lookup.parquet` | ~153K | ✅ Generated |
| `form_route_validity.parquet` | ~15K | ✅ Generated |
| `unified_drug_reference.parquet` | ~15K | ⚠️ Needs rebuild with new schema |

#### Input Datasets (in `inputs/drugs/`)

| File | Rows | Status |
|------|------|--------|
| `drugbank_generics_master.csv` | ~392K | ✅ Fresh |
| `drugbank_mixtures_master.csv` | ~153K | ✅ Fresh |
| `drugbank_brands_master.csv` | ~209K | ✅ Fresh |
| `drugbank_products_export.csv` | ~456K | ✅ Fresh |
| `drugbank_salts_master.csv` | ~3K | ✅ Fresh |
| `pnf_lexicon.csv` | ~3K | ✅ Fresh |
| `who_atc_2025-11-27.parquet` | ~6K | ✅ Fresh |
| `fda_drug_2025-11-12.parquet` | ~31K | ✅ Fresh |
| `fda_food_2025-11-23.parquet` | ~135K | ✅ Fresh |
| `esoa_combined.csv` | 258,878 | ⚠️ Has duplicates (~145K unique) |

---

### Scripts Status

#### Active Scripts (in `pipelines/drugs/scripts/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `build_unified_reference.py` | Build unified dataset | ⚠️ Needs refactor for new schema |
| `runners.py` | Part 2/3 entry points | ✅ Working |
| `prepare_drugs.py` | PNF preparation | ✅ Working |
| `brand_map_drugs.py` | FDA brand map | ✅ Working |
| `reference_synonyms.py` | Synonym loading | ✅ Working |
| `dose_drugs.py` | Dose extraction | ⚠️ Needs improvement |
| `routes_forms_drugs.py` | Form/route parsing | ✅ Working |

#### Tagging Module (in `pipelines/drugs/scripts/tagging/`)

| Script | Purpose | Status |
|--------|---------|--------|
| `tagger.py` | UnifiedTagger class | ✅ Working, needs batch method |
| `tokenizer.py` | Text tokenization | ✅ Working |
| `scoring.py` | Candidate selection | ⚠️ Has deprecated source priority |
| `lookup.py` | Reference lookups | ✅ Working |
| `constants.py` | Token categories | ✅ Working |
| `form_route_mapping.py` | Form-route inference | ⚠️ Needs data-driven approach |

#### Potentially Unused/Legacy Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `aho_drugs.py` | Aho-Corasick tries | ⚠️ Deprecated (using DuckDB now) |
| `combos_drugs.py` | Combination handling | ⚠️ May have useful logic |
| `concurrency_drugs.py` | Parallel processing | ⚠️ May be unused |
| `debug_drugs.py` | Debug utilities | ✅ Utility |
| `generic_normalization.py` | Generic name normalization | ⚠️ Check if used |
| `pnf_aliases_drugs.py` | PNF aliases | ⚠️ Check if used |
| `pnf_partial_drugs.py` | PNF partial matching | ⚠️ Check if used |
| `resolve_unknowns_drugs.py` | Unknown resolution | ⚠️ Check if used |
| `generate_route_form_mapping.py` | Generate mappings | ⚠️ One-time script |

---

### State of Submodules

All submodules are in `./dependencies/`:

#### `dependencies/atcd/` (WHO ATC Scraper)
**Status:** ✅ WORKING

| File | Purpose | Status |
|------|---------|--------|
| `atcd.R` | Main scraper | ✅ Working |
| `export.R` | Export to CSV/Parquet | ✅ Working |
| `filter.R` | Filter ATC data | ✅ Working |

**Output:** `who_atc_YYYY-MM-DD.csv` in `output/`

**Notes:**
- Scrapes from WHO website
- Parallelized with `future` package
- Exports both CSV and Parquet

---

#### `dependencies/drugbank_generics/` (DrugBank Extractor)
**Status:** ✅ WORKING (but slow)

| File | Purpose | Status |
|------|---------|--------|
| `drugbank_all.R` | Orchestrator | ✅ Working |
| `drugbank_generics.R` | Extract generics | ✅ Working, slow |
| `drugbank_mixtures.R` | Extract mixtures | ✅ Working |
| `drugbank_brands.R` | Extract brands | ✅ Working |
| `drugbank_salts.R` | Extract salts | ✅ Working |

**Outputs:**
- `drugbank_generics_master.csv` (~392K rows)
- `drugbank_mixtures_master.csv` (~153K rows)
- `drugbank_brands_master.csv` (~209K rows)
- `drugbank_products_export.csv` (~456K rows)
- `drugbank_salts_master.csv` (~3K rows)
- `drugbank_pure_salts.csv` (51 rows)
- `drugbank_salt_suffixes.csv` (58 rows)

**Known Issues:**
- `drugbank_generics.R` is very slow (TODO #26)
- May be over-parallelizing or under-vectorizing
- Uses `dbdataset` package from GitHub

---

#### `dependencies/fda_ph_scraper/` (FDA Philippines Scraper)
**Status:** ✅ WORKING

| File | Purpose | Status |
|------|---------|--------|
| `drug_scraper.py` | Scrape FDA drug list | ✅ Working |
| `food_scraper.py` | Scrape FDA food list | ✅ Working |
| `routes_forms.py` | Form/route parsing | ✅ Working |
| `text_utils.py` | Text normalization | ✅ Working |

**Outputs:**
- `fda_drug_YYYY-MM-DD.csv` (~31K rows)
- `fda_food_YYYY-MM-DD.csv` (~135K rows)

**Notes:**
- Scrapes from FDA PH verification website
- Supports both CSV download and HTML scraping fallback
- Exports both CSV and Parquet

---

### Hardcoded Data Locations

The following scripts contain hardcoded data that should be externalized (TODO #22):

| Script | Hardcoded Data |
|--------|----------------|
| `tagging/constants.py` | `NATURAL_STOPWORDS`, `FORM_CANON`, `ROUTE_CANON`, `SALT_TOKENS`, `PURE_SALT_COMPOUNDS` |
| `tagging/scoring.py` | `FORM_EQUIVALENCE_GROUPS`, source priority (deprecated) |
| `tagging/lookup.py` | Hardcoded synonyms (partially cleaned) |
| `run_drugs_pt_4_esoa_to_annex_f.py` | `EQUIVALENT_FORMS`, `FORM_NORMALIZE`, `FORM_TO_ROUTE`, `GENERIC_SYNONYMS` |
| `reference_synonyms.py` | Regional variant synonyms |

---

### Planned Improvements (from implementation_plan_v2.md)

| Priority | Item | Impact |
|----------|------|--------|
| HIGH | Brand → Generic swapping (#1) | +5-10% match rate |
| HIGH | Order-independent combos (#2, #5) | +3% match rate |
| HIGH | Expand synonyms (#11) | +5% match rate |
| MEDIUM | Batch tagging (#10) | 10x performance |
| MEDIUM | Fix ESOA deduplication (#16) | Data quality |
| MEDIUM | Rebuild unified reference (#17) | Foundation |
| LOW | Fuzzy matching (#3) | +1-2% match rate |
| LOW | FDA food fallback (#21) | Edge cases |

---

## Appendix: Pharmaceutical Glossary

| Term | Definition |
|------|------------|
| **ATC** | Anatomical Therapeutic Chemical classification |
| **Generic** | Non-proprietary drug name (active ingredient) |
| **Brand** | Proprietary/trade name |
| **Salt** | Chemical form for drug delivery (e.g., hydrochloride) |
| **Form** | Physical dosage form (tablet, capsule, etc.) |
| **Route** | Administration pathway (oral, IV, etc.) |
| **DDD** | Defined Daily Dose (WHO standard) |
| **Combination** | Drug with multiple active ingredients |
| **Mixture** | Same as combination |
