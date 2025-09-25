# eSOA Drug Matching Pipeline

This repository implements the **eSOA (electronic Statement of Account) drug matching pipeline**, a dose- and form-aware system that maps free-text medicine descriptions from hospital bills (eSOAs) to standardized drug references:

- **Philippine National Formulary (PNF)**
- **FDA brand map** (brand → generic links from FDA PH online export)
- **WHO ATC classification** (international codes)

It prepares raw CSVs, parses text into structured features, detects candidate generics, scores/classifies matches, and outputs a labeled dataset with a detailed distribution summary and unknown token report. The goal is to support **public health operations and oversight**, not commercial decision-making.

---

## 🚀 Pipeline Overview

### Flowchart

```mermaid
flowchart TD
    A[PNF CSV] --> B[Prepare PNF (normalize, parse dose/form/route)]
    A2[eSOA CSV] --> C[Prepare eSOA (normalize raw_text)]
    B --> D[PNF prepared CSV]
    C --> E[eSOA prepared CSV]

    F[FDA Portal Export] --> G[Build FDA Brand Map]
    G --> H[fda_brand_map_YYYY-MM-DD.csv]

    D --> I[Build features]
    E --> I
    H --> I
    I --> J[Brand → Generic Swaps]
    I --> K[Dose/Route/Form parsing]
    I --> L[PNF + WHO molecule detection]
    I --> M[Combination vs Salt detection]
    I --> N[Unknown token extraction]

    J & K & L & M & N --> O[Scoring & Classification]
    O --> P[Matched dataset (CSV, Excel)]
    O --> Q[Summary.txt (distribution)]
    O --> R[Unknown_words.csv]

    R --> S[Resolve unknowns.py (optional)]
```

---

## 🧠 Core Algorithmic Logic

1. Dose Parsing (scripts/dose.py)

Understands expressions like:

- Amounts: 500 mg, 250 mcg, 1 g
- Ratios: 5 mg/5 mL, 1 g/100 mL, x mg per spray
- Packs: 10 × 500 mg (unmasks to 500 mg, not 5000 mg)
- Percents: 0.9%, optionally w/v or w/w
- Compact unit-dose nouns: mg/tab, mg/cap, mg/spray, mg/puff

Normalization:

- Converts g↦mg, µg/μg↦mcg, L↦1000 mL
- Computes mg per mL for ratio doses when possible

Dose similarity (dose_sim):

- ≤0.1% error → 1.0
- ≤5% error → 0.8
- ≤10% error → 0.6
- 10% → 0.0

Public health/program implications  
These thresholds affect how many items are deemed adequate matches automatically vs. flagged for review.

Supervisor input needed

- Is ±10% a suitable tolerance for dose match in this program?
- Should different forms (e.g., inhalation vs. oral solutions) have stricter or looser tolerance?

---

2. Route & Form Detection (scripts/routes_forms.py)

- Recognizes forms (tablet, cap, MDI, DPI, susp, soln, spray, supp, etc.) and maps to canonical routes (oral, inhalation, nasal, rectal, etc.)
- Expands aliases (PO, per os, intranasal, SL, IM, IV, etc.)
- If route is missing but form is known, imputes route from form

Public health/program implications  
Strict route equality may over-flag valid cases when PNF route lists are incomplete.

Supervisor input needed

- Should we require exact route alignment, or allow imputed/fuzzy matches?

---

3. Brand → Generic Swap (scripts/brand_map.py, used in match_features.py)

- Builds Aho–Corasick automata of FDA brand names; maps each to one or more FDA-listed generics with optional dose/form metadata.
- For each eSOA line, we detect brands in the main text and replace with the FDA generic, recording did_brand_swap = True.
- We do not swap text inside parentheses — assumption: parentheses annotate brands when the generic already leads (e.g., paracetamol (Biogesic)).

Public health/program implications  
Brand→generic swaps substantially increase coverage/standardization but can hide dose/form discrepancies when brand packaging differs.

Supervisor input needed

- If a brand maps to multiple generics or variants, should we:
  - prefer PNF-present generics,
  - prioritize dose/form corroboration, or
  - always select the longest/most specific generic token?

---

4. Combination vs. Salt Detection (scripts/combos.py)

- Splits on +, /, with, but masks dose ratios (mg/mL) to avoid false positives.
- Treats known salt/ester/hydrate tails (e.g., hydrochloride, hemisuccinate, palmitate, pamoate, decanoate) as formulation modifiers and not separate molecules.
- Identifies likely combinations when two or more known generic tokens are present (PNF, WHO, or FDA-generic sets).

Public health/program implications  
Impacts whether a line is processed as a single molecule or combination product.

Supervisor input needed

- Confirm policy: should certain esters/salts be treated as distinct actives in surveillance, or as the same base molecule?

---

5. Best-Variant Selection & Scoring (scripts/match_scoring.py)

For each eSOA entry with at least one PNF candidate:

- Scoring
- Form match: +40
- Route match: +30
- Dose similarity (dose_sim × 30)
- Picks the highest-scoring candidate as the best variant and reports:
  - match_quality ∈ {OK, no/poor dose, no/poor form, no/poor route}
  - selected_form, selected_variant, dose_sim, form_ok, route_ok, atc_code_final
- Confidence score (for context):  
  +60 if generic found, +15 dose parsed, +10 route evidence, +15 ATC code, +0–10 from dose_sim, +10 bonus for clean brand swap.
- Auto-Accept policy (relaxed):
  - Auto-Accept if:
    - PNF generic identified
    - ATC code present
    - form_ok and route_ok
  - Dose mismatch no longer blocks Auto-Accept, but such rows are flagged for supervisors:
    - why_final = "OK, dose mismatch" (or "OK, brand->generic swap (dose mismatch)")
    - reason_final = "no/poor dose match"

Public health/program implications  
Dose mismatches still allow ATC assignment and Auto-Accept, reducing false negatives while keeping supervisor oversight by flagging these cases for review.

Supervisor input needed

- Keep the strict Auto-Accept (~15% auto-accept), or relax back toward confidence-based (~56% auto-accept)?

---

6. Unknown Handling (scripts/match_features.py → unknown_words.csv)

- Extracts tokens not recognized in PNF/WHO/FDA sets.
- Categorizes:
  - Single - Unknown
  - Multiple - All Unknown
  - Multiple - Some Unknown

Public health/program implications  
Frequent unknowns may mean missing formulary entries, local shorthand, or data quality issues.

Supervisor input needed

- Confirm triage workflow: which unknowns trigger formulary enrichment vs local mapping vs data cleaning?

---

## 📊 Outputs

- outputs/esoa_matched.csv — Main matched dataset
- outputs/esoa_matched.xlsx — Excel version
- outputs/summary.txt — Classification distribution
- outputs/unknown_words.csv — Frequency of unmatched tokens

Example summary:

Distribution Summary  
Auto-Accept: 17,089 (15.16%)  
 brand→generic swap: 12,255 (10.88%)  
 OK, no changes: 4,834 (4.29%)  
Needs review: 77,305 (68.6%)  
 Needs review: no/poor dose match: 38,942 (34.56%)  
 Needs review: no/poor form match: 23,333 (20.71%)  
 Needs review: no/poor route match: 12 (0.01%)  
 Needs review: no/poor dose/form/route: 5,389 (4.78%)  
Valid Molecule with ATC (WHO/FDA, not in PNF): 9,629 (8.55%)  
Others: 18,295 (16.23%)  
 Unknown: Multiple - All Unknown: 13,760 (12.21%)  
 Unknown: Multiple - Some Unknown: 2,812 (2.5%)  
 Unknown: Single - Unknown: 1,723 (1.53%)

---

## 🛠️ Running the Pipeline

```bash
# Python 3.10+ (R optional for ATC preprocessing)
pip install -r requirements.txt

# Run full pipeline (writes to ./outputs)
python run.py --pnf inputs/pnf.csv --esoa inputs/esoa.csv --out esoa_matched.csv
```

Optional flags

- --skip-install — Skip pip install
- --skip-r — Skip ATC preprocessing
- --skip-brandmap — Reuse existing FDA brand map

---

## 📂 Repository Structure

- scripts/
- aho.py — Aho–Corasick automata for PNF names
- combos.py — Combination vs. salt logic
- dose.py — Dose parsing & similarity
- match_features.py — Feature engineering (normalization, brand swap, detectors)
- match_scoring.py — Variant selection, scoring, classification
- match_outputs.py — CSV/Excel writers and summary
- match.py — Orchestrates matching
- prepare.py — Prepares PNF/eSOA inputs
- routes_forms.py — Form→route maps and parsing
- text_utils.py — Normalization helpers
- who_molecules.py — WHO ATC loader/detector
- fda_ph_drug_scraper.py — FDA PH brand map scraper
- brand_map.py — Brand automata builder
- main.py — API wrapper (prepare, match, run_all)
- run.py — Full pipeline runner with spinner & timing
- debug/master.py — All-in-one concatenated script for debugging
- outputs/ — Generated files

---

## 👥 Intended Users

- Program analysts / pharmacists / clinical reviewers — interpret match criteria, validate outputs, guide policy choices
- Data engineers / data scientists — maintain/extend logic, add sources
- Supervisors / public-health leads — provide direction on tolerance, acceptance policy, handling of unknowns

---

## ✅ Summary of Items for Supervisor Input

1. Dose tolerance  
   Is ±10% acceptable? Should some forms/routes be stricter?
2. Route/Form strictness  
   Require exact matches or allow imputed/fuzzy?
3. Auto-Accept policy  
   Keep strict (≈15% auto-accept) or relax toward confidence-based (≈56%)?
4. Salts/esters policy  
   Treat as distinct actives or same molecule?
5. Unknowns triage  
   Prioritize formulary enrichment, local mapping, or data quality remediation?

---

## 🔒 Data & Operational Notes

- Standardizes drug text for public-health monitoring, auditing, coverage analytics
- Ensure use aligns with data-sharing agreements & privacy protections
- FDA PH data is fetched from public portal’s CSV export; availability may vary
