# Drugs Pipeline: ESOA → Drug Code Matching

This document describes the new 4-part ESOA → Drug Code matching pipeline that uses ATC/DrugBank ID as an intermediate step.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OLD APPROACH (being replaced)                        │
│  ESOA row ──────────────────────────────────────────────────► Drug Code     │
│            (direct fuzzy match against 2,427 Annex F rows)                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              NEW 4-PART APPROACH                             │
│                                                                              │
│  Part 1: Prepare Dependencies                                                │
│          (WHO ATC, DrugBank, FDA brand/food, PNF, Annex F)                  │
│                                                                              │
│  Part 2: Annex F ──► ATC/DrugBank ID                                        │
│          (pre-index Annex F with codes for fast lookup)                     │
│                                                                              │
│  Part 3: ESOA ──► ATC/DrugBank ID                                           │
│          (match via PNF, DrugBank, WHO, FDA brand swaps)                    │
│                                                                              │
│  Part 4: ESOA + Annex F ──► Drug Code                                       │
│          (bridge via ATC/DrugBank ID, score by dose/form/route)             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Run all 4 parts in sequence
python run_drugs_pt_1_prepare_dependencies.py
python run_drugs_pt_2_annex_f_atc.py --workers 8
python run_drugs_pt_3_esoa_atc.py
python run_drugs_pt_4_esoa_to_annex_f.py
```

## Why This Approach?

| Aspect | Old Approach | New Approach |
|--------|--------------|--------------|
| Search space per ESOA row | 2,427 Annex F rows | ~5-20 candidates |
| Matching complexity | O(n × m) fuzzy matching | O(n) lookup + O(k) scoring |
| ATC/DrugBank coverage | Computed per-row | Pre-computed once |
| Reusability | None | Annex F index reusable |
| Modularity | Monolithic | 4 independent parts |

## Part Scripts

### Part 1: `run_drugs_pt_1_prepare_dependencies.py`

Prepares all reference data needed for matching.

**Refreshes:**
- WHO ATC (via `dependencies/atcd` R scripts)
- DrugBank generics/mixtures (via `dependencies/drugbank_generics` R scripts)
- FDA brand map (from FDA drug exports)
- FDA food catalog (from FDA food exports)
- PNF lexicon (normalize and parse dose/route/form)
- Annex F (verify exists)

**Usage:**
```bash
python run_drugs_pt_1_prepare_dependencies.py [--skip-who] [--skip-drugbank] [--skip-fda-brand] [--skip-fda-food]
```

### Part 2: `run_drugs_pt_2_annex_f_atc.py`

Tags each Annex F row with ATC code and DrugBank ID.

**Runs:** `pipelines/drugs/scripts/match_annex_f_with_atc.py`

**Outputs:**
- `outputs/drugs/annex_f_with_atc.csv` - Annex F with ATC codes and DrugBank IDs
- `outputs/drugs/annex_f_atc_ties.csv` - Rows where multiple references tied
- `outputs/drugs/annex_f_atc_unresolved.csv` - Unresolved ATC conflicts

**Usage:**
```bash
python run_drugs_pt_2_annex_f_atc.py --workers 8
```

**Current Results (2025-11-26):**
| Metric | Value |
|--------|-------|
| Total Annex F rows | 2,427 |
| Matched with ATC | 2,136 (88.0%) |
| Has DrugBank ID | 1,478 (60.9%) |

### Part 3: `run_drugs_pt_3_esoa_atc.py`

Matches ESOA rows to ATC codes and DrugBank IDs using the existing pipeline.

**Uses:**
- `pipelines/drugs/scripts/match_features_drugs.py` (molecule detection)
- `pipelines/drugs/scripts/match_scoring_drugs.py` (scoring and classification)

**Outputs:**
- `outputs/drugs/esoa_with_atc.csv` - ESOA with ATC codes

**Usage:**
```bash
python run_drugs_pt_3_esoa_atc.py [--esoa PATH] [--out FILENAME]
```

### Part 4: `run_drugs_pt_4_esoa_to_annex_f.py`

Bridges ESOA rows to Annex F Drug Codes via ATC/DrugBank ID.

**Process:**
1. Load ESOA with ATC (from Part 3)
2. Load Annex F with ATC (from Part 2)
3. For each ESOA row, find Annex F candidates with matching ATC/DrugBank ID
4. Score candidates by dose, form, route similarity
5. Select best Drug Code

**Outputs:**
- `outputs/drugs/esoa_matched_drug_codes.csv` - Final ESOA → Drug Code mapping

**Usage:**
```bash
python run_drugs_pt_4_esoa_to_annex_f.py [--esoa-atc PATH] [--annex-atc PATH] [--out FILENAME]
```

## Supporting Scripts

### `pipelines/drugs/scripts/generate_route_form_mapping.py`

Builds valid route-form combinations from reference data.

**Outputs:**
- `outputs/drugs/route_form_mapping.csv` - Valid route-form pairs (393 pairs)
- `outputs/drugs/route_form_unencountered.csv` - Forms in Annex F not in reference data

## Key Features

### Parallelized Annex F Processing
- Uses `ProcessPoolExecutor` with configurable worker count
- Chunks 2,427 Annex F rows across workers
- Default: 8 workers, auto-tuned chunk size

### Dose Standardization
- All doses normalized to milligrams (mg)
- 500 mcg → 0.5 mg, 1 g → 1000 mg

### DrugBank ID Resolution
- Direct assignment for DrugBank matches
- Lookup by generic name for PNF matches
- ~61% of matched rows have DrugBank IDs

## Integration Points

The `annex_f_with_atc.csv` output integrates with the existing pipeline:

1. **Build ATC → Annex F index** from `annex_f_with_atc.csv`
2. **After ESOA matching** (existing `match_drugs.py`), use the assigned ATC/DrugBank ID
3. **Lookup Annex F candidates** using the index
4. **Score and select** best Drug Code by dose/form/route alignment
