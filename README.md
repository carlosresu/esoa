# eSOA Drug Matching Pipeline

This repository implements the **eSOA (electronic Statement of Account) drug matching pipeline**, a dose- and form-aware system that maps free-text medicine descriptions from hospital bills (eSOAs) to standardized drug references.  
**Annex F is now the authoritative source of truth**: every record is keyed by the Annex F Drug Code first, with remaining references acting as supporting evidence or fallbacks.

Reference hierarchy used during matching:

1. **Annex F** (normalized to `annex_f_prepared.csv`; primary identifier = Drug Code)
2. **Philippine National Formulary (PNF)** (dose/form/route enrichment, ATC codes when Annex has gaps)
3. **WHO ATC classification** (international codes; still required for program reporting but no longer the primary ID)
4. **DrugBank generics** (synonym coverage)
5. **FDA brand map** (brand → generic swaps feeding Annex/PNF matches)
6. **FDA Philippines food product catalog** (non-therapeutic identification and debug trail)

It prepares raw CSVs, parses text into structured features, detects candidate generics, scores/classifies matches, and outputs a labeled dataset with a detailed distribution summary and unknown token report. The goal is to support **public health operations and oversight**, not commercial decision-making.

## Requirements

- `pip install -r requirements.txt`

**Python packages**
- `numpy>=1.21`
- `openpyxl>=3.1`
- `pandas>=1.5`
- `pyahocorasick>=2.0`
- `requests>=2.31`
- `XlsxWriter>=3.0`

**R packages for WHO ATC preprocessing**
- `pacman`
- `rvest`
- `dplyr`
- `readr`
- `xml2`
- `purrr`
- `future`
- `furrr`
- `memoise`
- `httr2`
- `tibble`
- `stringr`

## Documentation maintenance

Recent housekeeping added explicit module docstrings and refreshed inline
comments across the preparation (`scripts/prepare.py`), feature engineering
(`scripts/match_features.py`), scoring (`scripts/match_scoring.py`), output
(`scripts/match_outputs.py`), and shared text utility (`scripts/text_utils.py`)
layers.  Refer to those modules directly when you need an authoritative
explanation of a transformation or policy constant; the comments call out why a
field exists and how it is consumed downstream.

---

## 🚀 Pipeline Overview

### Flowchart

```mermaid
flowchart TD
    A[Annex F CSV] --> B[Prepare Annex F: normalize, parse dose, infer route/form]
    A2[PNF CSV] --> C[Prepare PNF: normalize + parse dose/route/form]
    A3[eSOA CSV] --> D[Prepare eSOA: normalize raw_text]
    B --> E[annex_f_prepared.csv]
    C --> F[pnf_prepared.csv]
    D --> G[esoa_prepared.csv]

    H[FDA Portal Export] --> I[Build FDA Brand Map]
    I --> J[fda_brand_map_YYYY-MM-DD.csv]

    E --> K[Build features]
    F --> K
    G --> K
    J --> K
    K --> L[Brand to generic swaps]
    K --> Lb[DrugBank generic overlay]
    K --> L2[FDA food / non-therapeutic detection]
    K --> M[Dose, route, form parsing]
    K --> N[Annex + PNF + WHO molecule detection]
    K --> O[Combination detection]
    K --> P[Unknown token extraction]

    L --> Q[Scoring and classification]
    Lb --> Q
    L2 --> Q
    M --> Q
    N --> Q
    O --> Q
    P --> Q
    Q --> R[Matched dataset CSV and XLSX]
    Q --> S[Summary text reports]
    Q --> T[Unknown words CSV]

    T --> U[Resolve unknowns helper]
```

---

## 🧠 Core Algorithmic Logic

1. Dose Parsing ([scripts/dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py))

Understands expressions like:

- Amounts: 500 mg, 250 mcg, 1 g
- Ratios: 5 mg/5 mL, 1 g/100 mL, x mg per spray
- Packs: 10 × 500 mg (unmasks to 500 mg, not 5000 mg)
- Percents: 0.9%, optionally w/v or w/w
- Compact unit-dose nouns: mg/tab, mg/cap, mg/spray, mg/puff

Normalization:

- Converts g↦mg, µg/μg↦mcg, L↦1000 mL
- Computes mg per mL for ratio doses when possible
- Dose alignment (dose_sim):

  - Returns 1.0 only when the eSOA dose equals the PNF payload exactly after unit conversion. Anything else scores 0.0, keeping auto-accepts precise.
  - Ratio doses require the same mg/mL (or equivalent) and percent strengths must match exactly.
  - Special equivalence: modified-release trimetazidine capsules (55–90 mg) are still accepted against the 35 mg base strength in the PNF.

Public health/program implications  
Exact matching maximizes precision but increases manual review load when facilities document rounded strengths. Program choices determine whether more therapeutic equivalence rules should be added.

Supervisor input needed

- Should we introduce configurable tolerances or extend `_SPECIAL_AMOUNT_EQUIVALENCE` to other molecules with consistent packaging differences?
- For which clinical areas (e.g., pediatric dilutions) would a near-match rule still be acceptable without compromising safety?

---

2. Route & Form Detection ([scripts/routes_forms.py](https://github.com/carlosresu/esoa/blob/main/scripts/routes_forms.py), [scripts/prepare_annex_f.py](https://github.com/carlosresu/esoa/blob/main/scripts/prepare_annex_f.py), and [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py))

- Recognizes forms (tablet, cap, MDI, DPI, susp, soln, spray, supp, etc.) and maps them to canonical routes (oral, inhalation, nasal, rectal, etc.), expanding common aliases (PO, per os, SL, IM, IV, etc.). Annex F gets additional heuristics for drops, ovules, shampoos, and packaging-driven inference (ampule/vial → injectable, large-volume bottles/bags → intravenous, nebules → inhalation).
- When an eSOA entry is missing the route but the form is present, the Annex/PNF route is imputed and logged in `route_evidence`. If only WHO metadata is available, we map WHO Adm.R/UOM codes to the same canonical routes/forms so alignment with Annex/PNF logic remains consistent.
- The scorer keeps a per-route interchange whitelist (`APPROVED_ROUTE_FORMS`) so, for example, tablets and capsules under the oral route both satisfy `form_ok`. Accepted substitutions are surfaced in `route_form_imputations`.
- Suspicious but historically observed combinations (e.g., oral vials) live in `FLAGGED_ROUTE_FORM_EXCEPTIONS`; they remain valid yet receive a `flagged:` annotation so reviewers know why the match passed.
- Solid oral forms are not auto-imputed when the detected dose is a ratio (mg/mL) to avoid creating impossible solid/liquid pairings.

Public health/program implications  
This policy controls where the system treats forms as interchangeable. Expanding the whitelist reduces manual review but risks masking clinically important distinctions (e.g., sachet vs suspension).

Supervisor input needed

- Does the current `APPROVED_ROUTE_FORMS` mapping reflect program policy—for instance, should syrups, suspensions, and sachets all count as interchangeable oral forms?
- Are there additional flagged route/form exceptions that should always require human review, or conversely, any that can be safely promoted into the approved list?

---

3. Brand → Generic Swap ([scripts/brand_map.py](https://github.com/carlosresu/esoa/blob/main/scripts/brand_map.py), used in [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py))

- Builds Aho–Corasick automata of FDA brand names; maps each to one or more FDA-listed generics with optional dose/form metadata.
- For each eSOA line, we detect brands in the main text and replace with the FDA generic, recording did_brand_swap = True.
- Flags `brand_swap_added_generic` when a swap actually injects new generic tokens; the confidence bonus only applies when this flag is true and the swapped row still satisfies dose/form/route checks.
- We do not swap text inside parentheses — assumption: parentheses annotate brands when the generic already leads (e.g., paracetamol (Biogesic)).

Public health/program implications  
Brand→generic swaps substantially increase coverage/standardization but can hide dose/form discrepancies when brand packaging differs.

Supervisor input needed

- If a brand maps to multiple generics or variants, should we:
  - prefer PNF-present generics,
  - prioritize dose/form corroboration, or
  - always select the longest/most specific generic token?

---

4. DrugBank Generic Overlay ([scripts/reference_data.py](https://github.com/carlosresu/esoa/blob/main/scripts/reference_data.py), used in [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) & [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py))

- Loads the curated DrugBank generics export generated by `dependencies/drugbank/drugbank.R` (fallback: `inputs/generics.csv`), normalizes to lowercase ASCII, and indexes by token.
- After brand swaps run, scans the updated text for contiguous DrugBank token sequences and records matches in `drugbank_generics_list`; the boolean `present_in_drugbank` bubbles into scoring and summaries.
- The same token universe feeds the unknown-token guardrails and `resolve_unknowns.py`, ensuring we do not flag valid DrugBank synonyms as unresolved terms.
- When a DrugBank-only hit explains the text (no PNF/WHO/FDA coverage), scoring tags the row as `ValidMoleculeInDrugBank` so reviewers see the provenance.

Public health/program implications  
DrugBank synonyms cover molecules licensed abroad or pending local formulary inclusion. Surfacing them separately lets reviewers spot emerging therapies without misclassifying them as unknown noise.

Supervisor input needed

- Should DrugBank-only matches promote to Auto-Accept when supporting route/form/dose data exists, or always remain review items?
- Do we need additional regional synonym feeds beyond DrugBank to capture local trade names?

---

5. FDA Food / Non-therapeutic Catalog Detection ([scripts/fda_ph_food_scraper.py](https://github.com/carlosresu/esoa/blob/main/scripts/fda_ph_food_scraper.py), [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py), [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py))

- Optionally loads `inputs/fda_food_products.csv` (scraped via `scripts/fda_ph_food_scraper.py`) and builds an automaton of brand and product strings.
- Captures every hit in `non_therapeutic_hits`, distills canonical tokens in `non_therapeutic_tokens`, and highlights the highest scoring entry in `non_therapeutic_best`/`non_therapeutic_detail` so reviewers can see which registration number or company triggered the flag.
- `non_therapeutic_summary` emits `non_therapeutic_detected` to make the Unknown-bucket routing obvious in pivots without expanding the nested JSON columns.
- When a line resolves to an FDA food/non-therapeutic item and no PNF/WHO/FDA-drug molecule exists, scoring routes the row to the `Unknown` bucket with `why_final = "Unknown"` and `reason_final = "non_therapeutic_detected"`.
- Tokens from the catalog are excluded from `unknown_words` so food-only strings don’t pollute the missed-generic report.

Public health/program implications

Identifying non-therapeutic entries early keeps food supplements and supply items from inflating medicine utilization metrics while still surfacing rich metadata for investigation (brand, product, company, registration number).

Supervisor input needed

- Confirm whether additional FDA or DOH catalogs should be merged to expand the non-therapeutic detection net.
- Decide if certain borderline items (e.g., nutraceuticals) should escalate to review instead of landing directly in the Unknown bucket.

---

6. Lexical Normalization & Phonetic Matching ([scripts/pnf_aliases.py](https://github.com/carlosresu/esoa/blob/main/scripts/pnf_aliases.py), [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py))

- Performs “hard” normalization on every PNF name and detected text fragment: ASCII fold, strip punctuation/whitespace, collapse separators, drop trailing salt/form suffixes (HCl, SR, tab, inj, etc.), and sort tokens for combination products.
- Applies a focused set of pharma spelling transforms to cover UK↔US and historical spellings (ceph→cef, sulph→sulf, oes→es, haem→hem, amoxycill→amoxicill, etc.), emitting both the raw-normalized and rule-adjusted variants as lookup keys.
- Builds the expanded alias dictionary (including parenthetical trade names, slash/plus-separated actives, curated abbreviations like Pen G, ATS, ALMG, ISMN/ISDN) and indexes every key into a trigram inverted index for millisecond lookups.
- Uses the trigram index plus Damerau-Levenshtein/Jaccard scoring (threshold 0.88) as a final safety net when exact/partial matches fail, ensuring near-miss spellings such as “Acetylcistine”, “Cephalexin”, or “Trimetazidiine” still attach to the proper PNF molecule before WHO/FDA heuristics engage.

Public health/program implications  
Provides resilient coverage for common misspellings and legacy branches while keeping the matching deterministic—reviewers see `generic_final` populated even when billing text is noisy.

Supervisor input needed

- Are there additional salt/form suffixes or institutional abbreviations that should be treated as aliases (e.g., new ISMN/ISDN style jargon)?
- Should the acceptance score (currently ≥0.88 combined similarity) be tuned tighter/looser for specific therapeutic classes?

---

7. Combination vs. Salt Detection ([scripts/combos.py](https://github.com/carlosresu/esoa/blob/main/scripts/combos.py))

- Splits on +, /, with, but masks dose ratios (mg/mL) to avoid false positives.
- Treats known salt/ester/hydrate tails (e.g., hydrochloride, hemisuccinate, palmitate, pamoate, decanoate) as formulation modifiers and not separate molecules.
- Identifies likely combinations when two or more known generic tokens are present (PNF, WHO, or FDA-generic sets).
- Auto-builds combination aliases from PNF names (parenthetical trade names, slash/plus-separated actives, curated abbreviations such as Pen G, ATS, ALMG, ISMN/ISDN) so free-text variants like “Piperacillin Tazobactam”, “Co-amoxiclav”, or “Proparacaine” still map to the canonical molecule.

Public health/program implications  
Impacts whether a line is processed as a single molecule or combination product.

Supervisor input needed

- Confirm policy: should certain esters/salts be treated as distinct actives in surveillance, or as the same base molecule?

---

8. Best-Variant Selection & Scoring ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py))

For each eSOA entry with at least one PNF candidate:

- Filters out PNF variants whose `route_allowed` does not contain the detected route.
- Recomputes `dose_sim` against the selected PNF row (exact equality as described above) and prefers liquid formulations when the source dose is a ratio.
- Normalizes route/form pairs against the `APPROVED_ROUTE_FORMS` whitelist, marking accepted substitutions or flagged exceptions in `route_form_imputations`.
- Safely imputes missing form/route (`form_source`/`route_source`) from the chosen PNF variant when the text is silent and the inferred form would be coherent.
- Emits `selected_form`, `selected_variant`, detailed dose fields, and `dose_recognized` when the dose matches exactly.
- Falls back to a fuzzy PNF lookup (difflib) when exact and partial matches fail, catching common misspellings and UK/US spelling differences (e.g., Acetylcistine → Acetylcysteine, Cephalexin → Cefalexin, Trimetazidiine → Trimetazidine) before WHO/FDA heuristics engage.
- `match_quality` now tags every record explicitly: Auto-Accept rows report `auto_exact_dose_route_form`, `dose_mismatch_same_atc`, or `dose_mismatch_varied_atc`, while review rows surface `dose_mismatch`, `route_mismatch`, `form_mismatch`, `who_*` metadata gaps, missing-dose/form/route indicators, or the new non-therapeutic / unknown-token signals. No rows fall back to a generic `unspecified` bucket.
- Route and form substitutions that pass `route_ok`/`form_ok` remain Auto-Accept without additional tagging; the new dose tags focus solely on non-exact doses.
- Confidence scoring: +60 generic present, +15 dose parsed, +10 route evidence, +15 ATC, +⌊dose_sim×10⌋, +10 extra when `brand_swap_added_generic` is true and the swap still aligns on dose/form/route.
- Auto-Accept when a PNF generic with ATC is present and both `form_ok` and `route_ok` are true. Dose mismatches therefore become visible through `dose_sim`/`match_quality` (and optional flagged notes) but do not block Auto-Accept.

Public health/program implications  
The acceptance criteria balance reviewer workload against the risk of approving near-but-not-exact matches. Requiring dose equality keeps the column transparent, but the Auto-Accept gate still needs policy oversight.

Supervisor input needed

- Should Auto-Accept additionally require `dose_sim == 1.0`, or remain tolerant provided route/form are aligned?
- Are the confidence-score weights (60/15/10/15/±10 + bonus) appropriate for triaging review queues, or should brand swaps/dose corroboration carry different weight?

---

9. Unknown Handling ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) → unknown_words.csv)

- Extracts tokens not recognized in the combined PNF, WHO, FDA, or DrugBank vocabularies and prunes anything on the shared English/stopword whitelist.
- Categorizes:
  - Single - Unknown
  - Multiple - All Unknown
  - Multiple - Some Unknown
- Post-processing via [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py) scans those tokens against PNF/WHO/FDA catalogues (token n-grams) while reusing the same DrugBank and ignore-word dictionaries, and writes `missed_generics.csv` with suggested reference hits.

Public health/program implications  
Frequent unknowns may mean missing formulary entries, local shorthand, or data quality issues.

Supervisor input needed

- Confirm triage workflow: which unknowns trigger formulary enrichment vs local mapping vs data cleaning?
- Suggested triage options:
  - Formulary enrichment (e.g., add missing drugs to PNF)
  - Local mapping (e.g., create local aliases or shorthand mappings)
  - Data quality remediation (e.g., fix typos, OCR errors, or inconsistent formats)

---

## 📊 Outputs

- outputs/esoa_matched.csv — Main matched dataset (includes `route_form_imputations`, `dose_sim`, `confidence`, etc.)
- outputs/esoa_matched.xlsx — Filterable Excel view of the same records
- outputs/summary.txt — Default bucket breakdown; `summary_molecule.txt` and `summary_match.txt` provide molecule- and reason-focused pivots
- outputs/unknown_words.csv — Frequency of unmatched tokens (fed into post-processing)
- outputs/missed_generics.csv — Suggestions from [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py) that map unknown tokens back to PNF/WHO/FDA references (whole or partial n-gram matches)

Key columns to review inside `outputs/esoa_matched.csv` include:

- `molecules_recognized` — the canonical pipe-delimited list of generics the scorer accepts for ATC alignment and downstream matching
- `generic_final` — the normalized molecule identifier(s) the pipeline ultimately relied on (PNF `generic_id`, WHO molecule, or FDA generic fallback)
- `probable_brands` — FDA brand display names detected before any swap, useful for auditing `did_brand_swap` outcomes
- `drugbank_generics_list` / `present_in_drugbank` — DrugBank synonyms that explain the text whenever local formularies (PNF/WHO/FDA) are silent
- `qty_pnf` / `qty_who` / `qty_fda_drug` / `qty_drugbank` / `qty_fda_food` / `qty_unknown` — integer roll-ups that quantify how each source (and unresolved tokens) contributed to the row; the distribution summaries aggregate these columns per bucket.

The complete column reference lives in `data_dictionary.md`.

Example summary (structure of the new distribution report — counts below are illustrative):

Distribution Summary  
Auto-Accept: 17,089 (15.16%)  
  PNF: 18,002, WHO: 0, FDA drug: 1,134, FDA food: 0, Unknowns: 0  
Candidates: 45,021 (39.93%)  
  PNF: 26,407, WHO: 15,662, FDA drug: 5,113, FDA food: 0, Unknowns: 0  
Needs review: 42,284 (37.53%)  
  PNF: 18,990, WHO: 11,227, FDA drug: 6,315, FDA food: 1,440, Unknowns: 9,806  
Unknown: 9,836 (7.38%)  
  PNF: 0, WHO: 0, FDA drug: 0, FDA food: 3,245, Unknowns: 16,912

Additional lines list the top molecules or match-quality drivers per bucket when `summary_molecule.txt` or `summary_match.txt` is requested.

---

## 🛠️ Running the Pipeline

`run.py` now self-bootstraps `requirements.txt`, ensures `inputs/` and `outputs/` exist, concatenates partitioned `inputs/esoa_pt_*.csv` files into a temporary `esoa_combined.csv`, and shows live spinners plus grouped timing totals for every major stage. R remains optional and is only required when you want to refresh the WHO ATC exports.

```bash
# Python 3.10+ (optional virtualenv setup shown for reproducibility)
pip install -r requirements.txt  # run.py will do this automatically if you skip it

# Run full pipeline (writes fresh artifacts to ./outputs)
python run.py --annex inputs/annex_f.csv --pnf inputs/pnf.csv --esoa inputs/esoa.csv --out esoa_matched.csv
```

**DrugBank prerequisite**  
Refresh `dependencies/drugbank/output/generics.csv` (or drop a curated copy in `inputs/generics.csv`) by running `python run_drugbank.py` whenever the upstream DrugBank dataset changes. The pipeline will automatically consume the freshest export.

Optional flags

- `--skip-r` — Skip the WHO ATC R preprocessing stage
- `--skip-brandmap` — Reuse the most recent FDA brand map instead of rebuilding
- `--skip-excel` — Skip writing the XLSX workbook (CSV and summaries only)
- `--annex` / `--pnf` / `--esoa` — Override default input paths (relative paths fall back to `inputs/`)
- `--out` — Override the matched CSV filename (always placed under `./outputs`)

### Minimal/local run

For incremental testing without touching external data sources or emitting Excel, use:

```bash
python run_minimal.py --pnf inputs/pnf.csv --esoa inputs/esoa.csv --out esoa_matched.csv
```

This helper invokes `run.py` with `--skip-r --skip-brandmap --skip-excel`, keeping dependency bootstrapping and path resolution identical to the full runner while limiting the workload to CSV and summary generation.

### Profiling the pipeline

To inspect runtime hotspots, execute the profiled wrapper:

```bash
python debug.py --pnf inputs/pnf.csv --esoa inputs/esoa.csv --out esoa_matched.csv
```

`debug.py` mirrors `run.py` but wraps execution with [pyinstrument](https://github.com/joerick/pyinstrument). Profiling reports are saved to `./outputs/pyinstrument_profile_<timestamp>.html` and `.txt`, allowing you to review the call tree in a browser or terminal.

---

## 📂 Repository Structure

- scripts/
- [aho.py](https://github.com/carlosresu/esoa/blob/main/scripts/aho.py) — Aho–Corasick automata for PNF names
- [combos.py](https://github.com/carlosresu/esoa/blob/main/scripts/combos.py) — Combination vs. salt logic
- [dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py) — Dose parsing & similarity
- [match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) — Feature engineering (normalization, brand swap, detectors)
- [match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) — Variant selection, scoring, classification
- [match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py) — CSV/Excel writers and summary
- [match.py](https://github.com/carlosresu/esoa/blob/main/scripts/match.py) — Orchestrates matching
- [prepare.py](https://github.com/carlosresu/esoa/blob/main/scripts/prepare.py) — Prepares PNF/eSOA inputs
- [prepare_annex_f.py](https://github.com/carlosresu/esoa/blob/main/scripts/prepare_annex_f.py) — Prepares Annex F authoritative catalogue (names, dose, route/form, packaging metadata)
- [reference_data.py](https://github.com/carlosresu/esoa/blob/main/scripts/reference_data.py) — Shared loaders for DrugBank generics and ignore-word lists reused across pipeline stages
- [routes_forms.py](https://github.com/carlosresu/esoa/blob/main/scripts/routes_forms.py) — Form→route maps and parsing
- [text_utils.py](https://github.com/carlosresu/esoa/blob/main/scripts/text_utils.py) — Normalization helpers
- [who_molecules.py](https://github.com/carlosresu/esoa/blob/main/scripts/who_molecules.py) — WHO ATC loader/detector
- [fda_ph_drug_scraper.py](https://github.com/carlosresu/esoa/blob/main/scripts/fda_ph_drug_scraper.py) — FDA PH brand map scraper
- [brand_map.py](https://github.com/carlosresu/esoa/blob/main/scripts/brand_map.py) — Brand automata builder
- [main.py](https://github.com/carlosresu/esoa/blob/main/main.py) — API wrapper (prepare, match, run_all)
- [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) — Full pipeline runner with spinner & timing
- [run_minimal.py](https://github.com/carlosresu/esoa/blob/main/run_minimal.py) — Skip installs/R/brand-map rebuild and Excel export for quick reruns
- [debug/master.py](https://github.com/carlosresu/esoa/blob/main/debug/master.py) — All-in-one concatenated script for debugging
- outputs/ — Generated files

---

## 👥 Intended Users

- Program analysts / pharmacists / clinical reviewers — interpret match criteria, validate outputs, guide policy choices
- Data engineers / data scientists — maintain/extend logic, add sources
- Supervisors / public-health leads — provide direction on tolerance, acceptance policy, handling of unknowns

---

## ✅ Summary of Items for Supervisor Input

1. Dose equivalence rules  
   Dose scoring now demands exact equality (aside from explicit overrides such as modified-release trimetazidine). Do we need more equivalence rules for other molecules, or is strict equality still the desired policy?

2. Route/Form interchange list  
   The `APPROVED_ROUTE_FORMS` whitelist defines which forms are interchangeable within a route, and `FLAGGED_ROUTE_FORM_EXCEPTIONS` document tolerated but suspicious pairings. WHO metadata now feeds additional route/form hints when PNF is silent—please confirm the mapping tables align with program expectations and whether more aliases should be added.

3. Auto-Accept gate  
   Auto-Accept currently ignores dose mismatches as long as a PNF+ATC match with aligned route/form is present. Should we tighten this (e.g., require `dose_sim == 1.0`, brand corroboration, or a higher confidence score) to reduce downstream review risk?

4. Salts/esters policy  
   Should certain salt or ester forms (e.g., hydrochloride, hemisuccinate, palmitate) be treated as distinct actives for surveillance, or grouped with the base molecule as the current salt-list logic does? Clarifying this determines how combination detection and aggregation behave.

5. Unknowns triage  
   `unknown_words.csv` plus `missed_generics.csv` flag candidate additions for the PNF/WHO/FDA dictionaries. Which buckets trigger formulary enrichment, local alias mapping, or data-quality remediation, and who owns each follow-up loop?

6. FDA brand map cadence  
   `probable_brands` and brand-swap scoring depend on the freshness of `fda_brand_map_*.csv`. Confirm how often the brand export should be refreshed and who validates multi-generic mappings before they enter production.

7. Annex F coverage gaps  
   87 Annex F rows (primarily powders, concentrates, and dialysis solutions) still lack confident route/form inference. Decide whether to extend heuristic coverage (e.g., treating “dry powder dispenser” as inhalation) or maintain a manual curation list before enabling Auto-Accept for those cases.

---

## 🔒 Data & Operational Notes

- Standardizes drug text for public-health monitoring, auditing, coverage analytics
- Ensure use aligns with data-sharing agreements & privacy protections
- FDA PH data is fetched from public portal’s CSV export; availability may vary

---

## 📚 Documentation

- `data_dictionary.md` — Column-level definitions and provenance for `outputs/esoa_matched.csv`.
- `pipeline.md` — Step-by-step execution walkthrough of the matching pipeline with module references.

These references stay in sync with the current pipeline logic in `scripts/` and can be used by reviewers or developers who need deeper traceability.
