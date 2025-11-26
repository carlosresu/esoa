# Annex F ATC/DrugBank Matching

This document describes the new ESOA → Drug Code matching architecture that uses ATC/DrugBank ID as an intermediate step.

## New Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OLD APPROACH (being replaced)                        │
│  ESOA row ──────────────────────────────────────────────────► Drug Code     │
│            (direct fuzzy match against 2,427 Annex F rows)                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              NEW APPROACH                                    │
│                                                                              │
│  Step 1: ESOA row ──► ATC/DrugBank ID                                       │
│          (match generic name via PNF, DrugBank, WHO, FDA brand swaps)       │
│                                                                              │
│  Step 2: ATC/DrugBank ID ──► Annex F candidates (pre-indexed lookup)        │
│          (instant lookup, not fuzzy matching)                               │
│                                                                              │
│  Step 3: Annex F candidates ──► Best Drug Code                              │
│          (score by dose, form, route - only ~5-20 candidates per lookup)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why This Approach?

| Aspect | Old Approach | New Approach |
|--------|--------------|--------------|
| Search space per ESOA row | 2,427 Annex F rows | ~5-20 candidates |
| Matching complexity | O(n × m) fuzzy matching | O(n) lookup + O(k) scoring |
| ATC/DrugBank coverage | Computed per-row | Pre-computed once |
| Reusability | None | Annex F index reusable |

## Scripts

### 1. `pipelines/drugs/scripts/match_annex_f_with_atc.py` - Pre-index Annex F

Tags each Annex F row with ATC code and DrugBank ID by matching against PNF and DrugBank reference data.

**Outputs:**
- `outputs/drugs/annex_f_with_atc.csv` - Annex F with ATC codes and DrugBank IDs
- `outputs/drugs/annex_f_atc_ties.csv` - Rows where multiple references tied
- `outputs/drugs/annex_f_atc_unresolved.csv` - Unresolved ATC conflicts

**Usage:**
```bash
python -m pipelines.drugs.scripts.match_annex_f_with_atc --workers 8
```

**Current Results (2025-11-26):**
| Metric | Value |
|--------|-------|
| Total Annex F rows | 2,427 |
| Matched with ATC | 2,136 (88.0%) |
| Has DrugBank ID | 1,478 (60.9%) |

### 2. `pipelines/drugs/scripts/generate_route_form_mapping.py` - Route/Form Validation

Builds valid route-form combinations from reference data and identifies unencountered forms.

**Outputs:**
- `outputs/drugs/route_form_mapping.csv` - Valid route-form pairs (393 pairs)
- `outputs/drugs/route_form_unencountered.csv` - Forms in Annex F not in reference data

### 3. ESOA → ATC/DrugBank Matching (existing pipeline)

The existing `pipelines/drugs/` pipeline already matches ESOA rows to ATC codes via:
- PNF lexicon matching
- WHO ATC detection
- FDA brand → generic swaps
- DrugBank generic overlay

### 4. ATC/DrugBank → Drug Code Selection (to be implemented)

Once ESOA has an ATC/DrugBank ID:
1. Lookup all Annex F rows with that code (from `annex_f_with_atc.csv`)
2. Score candidates by dose, form, route similarity
3. Select best Drug Code

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
