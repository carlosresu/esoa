# Annex F Tagging Quality Audit

## Status: FIXED (2025-01-06)

### Results After Fixes
- **Annex F match rate**: 2,316/2,427 (95.4%) ← improved from 93.9%
- **ESOA match rate**: 104,674/146,189 (71.6%)
- **ESOA to Drug Code**: 81,272/146,189 (55.6%)

### Fixes Implemented

#### 1. GELATIN false matches (unified_constants.py, tokenizer.py)
- Added `FORM_MODIFIER_IGNORE` set with words that are drug names but should be ignored when appearing as form modifiers
- Added position-based filtering in `extract_generic_tokens` to exclude matches after form words (CAPSULE, TABLET, etc.)

#### 2. Salt pattern "(as SALT)" handling (tokenizer.py)
- Filter out multiword generic matches that appear inside "( as ...)" patterns
- Updated both `normalize_tokens` and `extract_generic_tokens` to check for salt pattern ranges
- Added `original_text` parameter to `normalize_tokens` for salt pattern detection

#### 3. Diluent/solvent stripping (tokenizer.py)
- Added comprehensive patterns to strip:
  - `+ X mL diluent/solvent`
  - `+ reconstitution fluid`
  - `FREEZE-DRIED POWDER + DILUENT`
  - `monodose/multidose vial + diluent`
  - Trailing `SOLUTION VIAL`, `POWDER BOTTLE`, etc.
- Added vaccine-specific potency stripping (`1000 DL 50 mouse min`, `not less than X PFU`)

#### 4. Combination products (lookup.py)
- Added original-order and reverse-order combo keys with "+" separator
- Previously only generated alphabetically sorted keys which missed some reference entries

#### 5. Contrast agents (scoring.py)
- Added length-based priority in `rank_candidate` to prefer longer/more specific generic names
- Ensures IODAMIDE is preferred over IODINE when both match

#### 6. Combination "+" separator handling (tokenizer.py)
- Fixed combination handling to skip diluent/solvent parts
- Clean salt parentheticals from combo parts before extracting generic names
- Skip stopwords and salt tokens in combo part extraction

---

## Summary of Issues (2025-12-03)

Based on comprehensive review of `annex_f_with_atc.csv`, the following issue categories were identified:

---

## Category 1: FORM WORDS MATCHED AS GENERIC (CRITICAL)

**Problem**: Dosage form words like "GELATIN" from "CAPSULE SOFT GELATIN" are being matched as the generic name.

**Examples**:
- VITAMIN A 200,000 IU CAPSULE SOFT GELATIN → matched to GELATIN (should be VITAMIN A)
- RETINOL (VITAMIN A) 10,000 IU CAPSULE SOFT GELATIN → matched to GELATIN
- IODIZED OIL FLUID 500 mg CAPSULE SOFT GELATIN → matched to GELATIN

**Root Cause**: The tagger is finding "GELATIN" in the text and matching it as a drug before finding the actual generic.

**Fix Required**: 
1. Add form-related words to exclusion list (GELATIN, SOFT, CAPSULE, etc.)
2. Prioritize matches at the start of text over matches in form descriptions

---

## Category 2: SALT PATTERN "(as SALT)" NOT HANDLED

**Problem**: Drugs with `( as SODIUM PHOSPHATE)` pattern are matched to SODIUM PHOSPHATE instead of base drug.

**Examples**:
- PREDNISOLONE ( as SODIUM PHOSPHATE) → matched to SODIUM PHOSPHATE
- DEXAMETHASONE ( as SODIUM PHOSPHATE) → matched to SODIUM PHOSPHATE  
- MEDROXYPROGESTERONE ( as ACETATE) → some fail to match
- LIDOCAINE ( as HYDROCHLORIDE) 2%, 1.8mL w/ epinephrine → matched to EPINEPHRINE

**Root Cause**: The text before `( as ...)` should be the primary generic; the salt should go to salt_details.

**Fix Required**:
1. In tokenizer, detect `( as SALT)` pattern
2. Extract base drug name before the parenthetical
3. Move salt to salt_details column

---

## Category 3: DILUENT/SOLVENT CAUSING NO-MATCH

**Problem**: Drugs with "+ diluent" or "+ solvent" fail to match entirely.

**Examples**:
- VINCRISTINE ( as SULFATE) 1 mg + diluent → no_match
- METHYLPREDNISOLONE 500mg/7.7mL + Diluent → no_match
- RISPERIDONE 50 mg prolonged-release powder + 2 mL diluent → no_match
- CEFTRIAXONE ( as DISODIUM) 250 mg + 5 mL diluent → no_match
- Many vaccines with "+ diluent" → no_match

**Root Cause**: The "+ diluent" text is confusing the parser.

**Fix Required**:
1. Strip "+ diluent", "+ solvent", "+ reconstitution fluid" before parsing
2. These are packaging details, not drug components

---

## Category 4: COMBINATION PRODUCTS NOT DETECTED

**Problem**: Two-component drug combinations not detected.

**Examples**:
- VALSARTAN + HYDROCHLOROTHIAZIDE 80 mg + 12.5 mg → no_match
- TELMISARTAN + HYDROCHLOROTHIAZIDE 80 mg + 12.5 mg → no_match
- LOSARTAN + HYDROCHLOROTHIAZIDE → no_match
- LEVODOPA + CARBIDOPA → no_match
- TOBRAMYCIN + DEXAMETHASONE → no_match
- ISONIAZID + RIFAMPICIN + PYRAZINAMIDE → no_match (3+ components)
- FERROUS SULFATE + FOLIC ACID → no_match

**Root Cause**: 
1. Combination detection may not be triggering
2. Missing combination drugs in reference

**Fix Required**:
1. Ensure "+" separator detection works
2. Add common combinations to reference or ensure WHO ATC combinations are loaded

---

## Category 5: CONTRAST AGENTS MATCHED TO IODINE

**Problem**: Iodinated contrast agents are matched to generic "IODINE" instead of their actual names.

**Examples**:
- IODAMIDE 495 mg/mL → matched to IODINE
- IOPAMIDOL 612 mg/mL → matched to IODINE
- IOPROMIDE 300 mg/mL → matched to IODINE
- IOVERSOL 636 mg/mL → matched to IODINE

Only IOHEXOL is correctly matched.

**Root Cause**: 
1. Reference may not have these drugs
2. Scoring prefers shorter match (IODINE) over specific match

**Fix Required**:
1. Add IODAMIDE, IOPAMIDOL, IOPROMIDE, IOVERSOL to reference
2. Prioritize longer/more specific matches in scoring

---

## Category 6: STANDARDIZATION ISSUES

### 6a. AND vs + in combinations
- WHO ATC uses "AND" (e.g., "SALMETEROL AND FLUTICASONE")
- Annex F uses "+" (e.g., "SALMETEROL + FLUTICASONE")
- Should standardize to "+" format

### 6b. Vaccine naming
- Yellow fever, pneumococcal, influenza vaccines need standardization
- Valency info (7-valent, 13-valent) should be preserved
- Strain info should go to details

### 6c. Interferon/Immunoglobulin subtypes
- INTERFERON ALFA 2A vs 2B - must preserve the subtype
- IMMUNOGLOBULIN (IGIM) vs (IGIV) - must preserve type designation

### 6d. Zinc Sulfate
- Should be ZINC with salt_details=SULFATE (it's not a pure salt)

---

## Category 7: MISSING REFERENCE DATA

Drugs that have `no_candidates` and likely need to be added to reference:

- THIACETAZONE
- TERIZODONE
- L-ASPARAGINASE
- MOLGRAMOSTIN
- MICRONUTRIENT
- ISOSORBIDE-5-MONONITRATE (distinct from ISOSORBIDE DINITRATE)
- LIQUID BICARBONATE CONCENTRATE
- HYPERTONIC SALINE

---

## Category 8: COMPLEX IV SOLUTIONS

**Problem**: Multi-component IV solutions matched to just one component.

**Examples**:
- 5% DEXTROSE IN 0.3% SODIUM CHLORIDE → matched only to DEXTROSE
- 5% DEXTROSE IN LACTATED RINGER'S → matched only to DEXTROSE
- BALANCED MULTIPLE MAINTENANCE SOLUTION WITH 5% DEXTROSE → matched to DEXTROSE

**Root Cause**: First-found component wins.

**Fix Required**: These should be matched as combination products or specific solution types.

---

## Priority Order for Fixes

1. **CRITICAL**: GELATIN false matches (form word as generic)
2. **HIGH**: Salt pattern "(as SALT)" handling
3. **HIGH**: Diluent/solvent stripping
4. **MEDIUM**: Combination product detection
5. **MEDIUM**: Contrast agents reference
6. **LOW**: Standardization and normalization
