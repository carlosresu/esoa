# Drugs Pipeline - Complete Status Report

**Generated**: 2025-01-06

---

## Pipeline Summary

| Stage | Total | Matched | Match Rate |
|-------|-------|---------|------------|
| **Part 2: Annex F Tagging** | 2,427 | 2,311 | **95.2%** |
| **Part 3: ESOA Tagging** | 146,189 | 104,401 | **71.4%** |
| **Part 4: ESOA to Drug Code** | 146,189 | 46,471 | **31.8%** |

---

## Part 2: Annex F Tagging

Matches drug descriptions in Annex F to ATC codes and DrugBank IDs.

| Status | Count | Percentage |
|--------|-------|------------|
| matched | 2,311 | 95.2% |
| no_match | 68 | 2.8% |
| no_candidates | 48 | 2.0% |

**Output file**: `outputs/drugs/annex_f_with_atc.csv`

---

## Part 3: ESOA Tagging

Matches ESOA drug descriptions to ATC codes and DrugBank IDs.

| Status | Count | Percentage |
|--------|-------|------------|
| matched | 104,401 | 71.4% |
| no_candidates | 32,834 | 22.5% |
| no_match | 8,954 | 6.1% |

**Output file**: `outputs/drugs/esoa_with_atc.csv`

---

## Part 4: ESOA to Drug Code (STRICT MATCHING)

Maps ESOA entries to Annex F Drug Codes using **perfect matching** criteria:
- Generic name must match (salt form can vary)
- Dose must match exactly
- Form must match (if available in both)
- Route must match (if available in both)

| Status | Count | Percentage |
|--------|-------|------------|
| **matched_perfect** | 46,471 | 31.8% |
| generic_not_in_annex | 64,597 | 44.2% |
| no_perfect_match | 34,289 | 23.5% |
| no_generic | 832 | 0.6% |

**Output file**: `outputs/drugs/esoa_with_drug_code.csv`

### Match Reason Definitions

- **matched_perfect**: Generic + Dose + Form + Route all match
- **generic_not_in_annex**: Generic name not found in Annex F reference
- **no_perfect_match**: Generic found but dose/form/route don't match
- **no_generic**: No generic name could be extracted from ESOA entry

---

## Fixes Applied in This Session

### For Annex F Issues (from previous session)
1. **GELATIN false matches** - Added form modifier filtering
2. **Salt pattern "(as SALT)"** - Correctly strip salt to extract base drug
3. **Diluent/solvent stripping** - Remove packaging info from drug names
4. **Combination products** - Fixed order-independent combo key matching
5. **Contrast agents** - Prefer longer/more specific generic names

### For ESOA-Specific Issues (this session)
1. **Trailing salt suffix** - Strip "DEXAMETHASONE SODIUM PHOSPHATE" → "DEXAMETHASONE"
2. **Plus sign normalization** - Handle "DRUG+DRUG" (no spaces)
3. **Mixture lookup case fix** - Fixed uppercase/lowercase key mismatch
4. **Multiword generic preservation** - Keep "BENZOIC ACID" intact in combos
5. **Standalone element handling** - Keep "ZINC" as drug in "ASCORBIC ACID+ZINC"

---

## Files Modified

- `pipelines/drugs/scripts/tokenizer.py`
- `pipelines/drugs/scripts/tagger.py`
- `pipelines/drugs/scripts/runners.py`

---

## Output Files Location

All outputs are in: `outputs/drugs/`

- `annex_f_with_atc.csv` - Annex F with ATC codes
- `esoa_with_atc.csv` - ESOA with ATC codes
- `esoa_with_drug_code.csv` - ESOA with Drug Codes (strict matching)
