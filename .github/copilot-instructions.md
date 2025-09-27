# Copilot Instructions for eSOA Drug Matching Pipeline

## Project Overview
- **Purpose:** Maps free-text hospital drug entries (eSOA) to standardized references (PNF, FDA, WHO ATC) for public health analytics.
- **Pipeline:** Prepares, normalizes, parses, matches, and scores drug data; outputs labeled datasets and unknown token reports.
- **Key files:**
  - `run.py`: Main pipeline runner (CLI)
  - `main.py`: API wrapper
  - `scripts/`: Core logic modules (dose, form, brand, scoring, etc.)
  - `outputs/`: Generated results (CSV, Excel, summaries)

## Architecture & Data Flow
- **Inputs:** `inputs/pnf.csv`, `inputs/esoa.csv`, `inputs/fda_brand_map_*.csv`
- **Preprocessing:**
  - `prepare.py`: Normalizes and parses PNF/eSOA
  - `brand_map.py`, `fda_ph_drug_scraper.py`: Builds brand→generic map
- **Feature Engineering:** `match_features.py` (dose, form, brand swap, unknowns)
- **Matching & Scoring:** `match_scoring.py` (route/form/dose logic, confidence)
- **Output:**
  - `outputs/esoa_matched.csv`, `.xlsx`: Main results
  - `outputs/summary.txt`: Match breakdown
  - `outputs/unknown_words.csv`, `missed_generics.csv`: Unknowns & suggestions

## Developer Workflows
- **Install:** `pip install -r requirements.txt` (Python 3.10+)
- **Run pipeline:**
  ```bash
  python run.py --pnf inputs/pnf.csv --esoa inputs/esoa.csv --out esoa_matched.csv
  ```
  - Optional: `--skip-install`, `--skip-r`, `--skip-brandmap`
- **Debug:** Use `debug/master.py` for all-in-one script debugging.
- **R integration:** Needed only for ATC preprocessing (see `dependencies/atcd/`).

## Project-Specific Patterns & Conventions
- **Dose parsing:** Strict unit normalization (mg, mcg, mL, etc.); see `scripts/dose.py`.
- **Route/form logic:** Canonical mapping and whitelists in `routes_forms.py`, `match_scoring.py`.
- **Brand swaps:** Only outside parentheses; see `brand_map.py`.
- **Combination/salt detection:** Handles +, /, and known salt/ester tails; see `combos.py`.
- **Unknowns:** All unknown tokens are logged and post-processed (`resolve_unknowns.py`).
- **Outputs:** Always written to `outputs/`.

## Integration Points
- **FDA PH data:** Fetched via `fda_ph_drug_scraper.py` (public CSV export).
- **WHO ATC:** CSVs in `dependencies/atcd/output/`.
- **R scripts:** Only required for ATC preprocessing; not needed for core Python pipeline.

## Key Policies & Decisions
- **Dose matching:** Only exact matches (after normalization) are auto-accepted, except for explicit overrides (e.g., trimetazidine).
- **Route/form whitelists:** Controlled via `APPROVED_ROUTE_FORMS` and `FLAGGED_ROUTE_FORM_EXCEPTIONS`.
- **Auto-acceptance:** Requires PNF+ATC match and aligned route/form; dose mismatch is flagged but not blocking.
- **Unknowns triage:** `unknown_words.csv` and `missed_generics.csv` drive dictionary enrichment and data quality review.

## Examples
- **Adding a new salt/ester:** Update `combos.py` and `dose.py`.
- **Changing route/form policy:** Edit `routes_forms.py` and `match_scoring.py`.
- **Debugging a match:** Trace through `match_features.py` → `match_scoring.py` → `match_outputs.py`.

---
For more details, see `README.md` and scripts in `scripts/`.
