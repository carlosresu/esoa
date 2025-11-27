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
