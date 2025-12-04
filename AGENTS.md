# AGENT INSTRUCTIONS

These rules are meant for GPT agents. Apply them whenever you are editing this repository:

> **IMPORTANT:** Before making changes to the drugs pipeline, read:
> - `debug/pipeline.md` - Algorithmic logic, pharmaceutical rules, and decision rationale
> - `debug/implementation_plan_v2.md` - Current TODO list and implementation status
> - `debug/progress.md` - Phase-based progress tracker (what's done)
>
> After every group of changes, UPDATE THESE FILES with what was changed and any new decisions made.

1. **Keep the standalone FDA scraper dependency files aligned.** Whenever you touch `pipelines/drugs/scripts/` (especially the normalization helpers or third-party imports) make sure the counterpart `dependencies/fda_ph_scraper/text_utils.py` captures the same text-processing logic and the `dependencies/fda_ph_scraper/requirements.txt` lists the packages needed by those scripts so the standalone scraper runs with the same dependencies as the pipeline helpers.
2. **Standalone runner behavior.** The scripts under `dependencies/fda_ph_scraper`, `dependencies/atcd`, and `dependencies/drugbank_generics` must continue to be runnable on their own roots; they should default to writing outputs under their own `output/` directories while downstream runners copy those exports into `inputs/drugs/`.
3. **Commit & submodule workflow.** Before creating a commit, inspect every submodule (`git submodule status`) for unstaged changes. If a submodule changed, commit and push that submodule first using a concise message describing its diff, then update the main repository's submodule pointer (commit and push that change separately). Always pull/push as appropriate within each repository before moving on so every agent run leaves the working tree clean.
4. **CSV-first data policy.** CSV is the primary data format across all pipeline steps. When reading data, prefer `.csv` files over `.parquet` when both exist. When writing data, always export only `.csv` format. All data operations should use CSV as the canonical format.
5. **Canonical file naming.** Use date-stamped naming for reference datasets: `who_atc_YYYY-MM-DD.csv`, `fda_drug_YYYY-MM-DD.csv`, etc. Legacy naming patterns like `*_molecules.csv` are deprecated.

## Pipeline Execution

6. **Use 8 workers for R scripts.** When running DrugBank or other R scripts via Python, set `ESOA_DRUGBANK_WORKERS=8` environment variable.
7. **Pipeline part dependencies.** The drugs pipeline has 4 parts that must run in order:
   - Part 1: Prepare dependencies (DrugBank generics, mixtures, brands, salts, FDA data)
   - Part 2: Match Annex F with ATC/DrugBank IDs → outputs `annex_f_with_atc.csv`
   - Part 3: Match ESOA with ATC/DrugBank IDs → uses unified tagger (same as Part 2)
   - Part 4: Bridge ESOA to Annex F Drug Codes → uses Part 3 output

## Unified Architecture

8. **Unified tagging algorithm.** Both Annex F and ESOA use the SAME tagging algorithm. The Annex F tagger (Part 2) is the base - do not create separate algorithms for different input types.

9. **DuckDB for all queries.** Use DuckDB for all reference lookups instead of Aho-Corasick tries. Load CSV files into DuckDB and query with SQL. This is faster for exact/prefix matching after tokenization.

10. **DrugBank lean tables.** The single R script `drugbank_lean_export.R` exports 8 lean data tables:
    - `generics_lean.csv` - drugbank_id → name (one row per drug)
    - `synonyms_lean.csv` - drugbank_id → synonym (English, INN/BAN/USAN/JAN/USP)
    - `dosages_lean.csv` - drugbank_id × form × route × strength (valid combos)
    - `brands_lean.csv` - brand → drugbank_id (international brands)
    - `salts_lean.csv` - parent drugbank_id → salt info
    - `mixtures_lean.csv` - mixture components with component_key
    - `products_lean.csv` - drugbank_id × dosage_form × strength × route
    - `atc_lean.csv` - drugbank_id → atc_code (with hierarchy levels)

11. **DrugBank lookup tables.** The same R script also exports 6 lookup tables for normalization:
    - `lookup_salt_suffixes.csv` - salt suffixes to strip (HYDROCHLORIDE, SODIUM, etc.)
    - `lookup_pure_salts.csv` - compounds that ARE salts (SODIUM CHLORIDE, etc.)
    - `lookup_form_canonical.csv` - form aliases → canonical form
    - `lookup_route_canonical.csv` - route aliases → canonical route
    - `lookup_form_to_route.csv` - infer route from form
    - `lookup_per_unit.csv` - per-unit normalization (ML, TAB, etc.)

12. **Unified reference building.** Python script `build_unified_reference.py` consumes lean tables and builds:
    - `unified_generics.csv` - drugbank_id → generic_name
    - `unified_synonyms.csv` - drugbank_id → synonyms (pipe-separated)
    - `unified_atc.csv` - drugbank_id → atc_code (one row per combo)
    - `unified_dosages.csv` - drugbank_id × form × route × dose
    - `unified_brands.csv` - brand_name → generic_name, drugbank_id
    - `unified_salts.csv` - drugbank_id → salt forms
    - `unified_mixtures.csv` - mixture components with component_key

## Drug Matching Policies

13. **Salt handling.** Use `lookup_salt_suffixes.csv` and `lookup_pure_salts.csv` for salt detection. Strip salts from matching basis UNLESS the compound is a pure salt (e.g., sodium chloride, calcium carbonate). Auto-detect pure salts: compounds where base would be empty after stripping.

14. **Dose normalization.** Canonical formats:
    - Weight: normalize to `mg` (e.g., `1g` → `1000mg`)
    - Combinations: `500mg+200mg` (fixed combo), `500mg/200mg` (ratio)
    - Concentration: `10mg/mL`, `5%`
    - Volume: `mL` is canonical for liquids

15. **Form-route validity.** Use `lookup_form_to_route.csv` and `dosages_lean.csv` for form-route inference. Only allow form-route combinations that exist in reference datasets. If form or route missing in input, infer the most common one.

16. **Multi-word generic names.** Preserve known multi-word generics as single tokens (e.g., "tranexamic acid", "folic acid", "insulin glargine"). Do not split these into individual words during tokenization.

17. **Single vs combination ATC codes.** When an input row contains a single molecule, prefer single-drug ATC codes over combination ATCs. For example, LOSARTAN alone should get C09CA01, not C09DA01 (losartan+HCTZ combo).

18. **R and Python constants sync.** The DrugBank R script (`dependencies/drugbank_generics/drugbank_lean_export.R`) exports lookup tables that must stay in sync with Python constants (`pipelines/drugs/scripts/unified_constants.py`):
   - Salt suffixes: `lookup_salt_suffixes.csv` ↔ `SALT_TOKENS`
   - Pure salts: `lookup_pure_salts.csv` ↔ `PURE_SALT_COMPOUNDS`
   - Form canonicals: `lookup_form_canonical.csv` ↔ `FORM_CANON`
   - Route canonicals: `lookup_route_canonical.csv` ↔ `ROUTE_CANON`
   When modifying constants in either location, update the other to match.

19. **Scoring algorithm.** Use deterministic pharmaceutical-principled scoring (NOT numeric weights):
    - Generic match is REQUIRED (no match without it)
    - Salt forms are IGNORED (unless pure salt compound)
    - Dose is FLEXIBLE for ATC tagging, EXACT (zero tolerance) for Drug Code matching
    - Form allows equivalents (TABLET ≈ CAPSULE)
    - Route is INFERRED from form if missing
    - ATC preference: single vs combo based on input molecule count
    - Tie-breaking uses `*_details` columns: release > type > form > indication > salt > alias > iv_diluent_type > brand
    - Brand is for resolution only (brand→generic), NOT for preference
    - See `debug/pipeline.md` for full scoring logic

20. **IV diluent equivalence.** Diluent name variants are normalized to canonical forms:
    - WATER = WATER FOR INJECTION = STERILE WATER = WFI
    - NORMAL_SALINE = SODIUM CHLORIDE = NS = 0.9% SODIUM CHLORIDE
    - HALF_SALINE = 0.45% SODIUM CHLORIDE
    - LACTATED_RINGERS = LACTATED RINGER'S = LR = RL (NOT equivalent to Acetated)
    - ACETATED_RINGERS = ACETATED RINGER'S = AR (NOT equivalent to Lactated)

21. **Part 4 form equivalence.** For Drug Code matching, forms within these groups are considered compatible:
    - TABLET ↔ FILM COATED, COATED, CHEWABLE, SUBLINGUAL, ORALLY DISINTEGRATING
    - CAPSULE ↔ SOFTGEL, GELCAP
    - EXTENDED RELEASE ↔ SUSTAINED RELEASE, CONTROLLED RELEASE, MR, SR, XR, ER
    - SYRUP ↔ SUSPENSION, SOLUTION, ELIXIR, DROPS
    - AMPULE ↔ VIAL, INJECTION
    - NEBULE ↔ NEBULIZER, INHALATION
    - INHALER ↔ MDI, INHALATION

22. **Part 4 dose tolerance.** Drug Code matching uses small tolerances for floating point precision:
    - MG matching: 1% relative difference OR 0.5mg absolute difference
    - Concentration matching: 1% relative OR 0.1 mg/mL absolute
    - Volume NOT required for concentration matching (same concentration = same drug)

23. **Part 4 dose inference.** When only volume is present, infer concentration from description:
    - NSS/PNSS + volume → assume 0.9% = 9 mg/mL (Normal Saline)
    - D5 + volume → assume 5% = 50 mg/mL (5% Dextrose)
    - D10 + volume → assume 10% = 100 mg/mL (10% Dextrose)
    - Bare numeric dose (no unit) → assume MG for tablet-range values (0.1-5000)

## Reference Data

24. **WHO ATC data.** Use the canonical `who_atc_YYYY-MM-DD.csv` files from `dependencies/atcd/`. The `load_who_molecules()` function loads CSV files.

25. **FDA data.** Use `dependencies/fda_ph_scraper/` to generate:
    - `fda_drug_YYYY-MM-DD.csv` - brand → generic mapping with dose/form/route
    - `fda_food_YYYY-MM-DD.csv` - food product catalog (fallback for non-drugs)

26. **Keep data dictionary updated.** Whenever you add, rename, or remove columns from any dataset, update the Data Dictionary section below.

27. **Standardized output column naming.** All pipeline outputs must use consistent column names:
    - `atc_code` - WHO ATC code (NOT `atc_code_final`)
    - `drugbank_id` - DrugBank ID (NOT `drugbank_id_final`)
    - `matched_generic_name` - Canonical generic name (NOT `generic_final`)
    - `matched_reference_text` - Reference drug name from lookup
    - `matched_source` - Data source (NOT `reference_source`)
    - `match_score`, `match_reason` - Match result metadata
    - `dose`, `form`, `route` - Extracted drug form info
    - All `*_details` columns - Extracted qualifiers (salt, brand, indication, etc.)
    - All computed columns - `drug_amount_mg`, `concentration_mg_per_ml`, etc.

---

## Data Dictionary

> **IMPORTANT:** Keep this section updated whenever dataset columns change.

### Pipeline Outputs

#### `outputs/drugs/annex_f_with_atc.csv`
Tagged Annex F drug codes with ATC/DrugBank IDs and extracted details.

| Column | Type | Description |
|--------|------|-------------|
| `Drug Code` | str | Original Annex F drug code |
| `Drug Description` | str | Original drug description text |
| `matched_reference_text` | str | Matched reference drug name from lookup |
| `atc_code` | str | Matched WHO ATC code |
| `drugbank_id` | str | Matched DrugBank ID |
| `matched_generic_name` | str | Canonical generic name |
| `match_score` | int | Match confidence (1=matched, 0=no_match) |
| `match_reason` | str | Why matched/unmatched (matched, no_match, no_candidates) |
| `matched_source` | str | Data source (who_atc, drugbank, fda, mixtures) |
| `dose` | str | Extracted dose(s), pipe-separated |
| `form` | str | Extracted dosage form (TABLET, CAPSULE, etc.) |
| `route` | str | Extracted route(s), pipe-separated |
| `type_details` | str | Drug type qualifier (e.g., "ANHYDROUS") |
| `release_details` | str | Release modifier (SR, XR, MR, etc.) |
| `form_details` | str | Form modifier (FC, EC, ODT) |
| `salt_details` | str | Salt form (HYDROCHLORIDE, SODIUM, etc.) |
| `brand_details` | str | Brand name in parentheses |
| `indication_details` | str | Indication qualifier (FOR HEPATIC FAILURE) |
| `alias_details` | str | Alias (VIT. D3, etc.) |
| `diluent_details` | str | Diluent volume (reconstitution info) |
| `iv_diluent_type` | str | IV base solution (WATER, SODIUM CHLORIDE, LACTATED RINGER'S) |
| `iv_diluent_amount` | str | IV diluent concentration (0.9%, 0.3%) |
| `dose_values` | list | Numeric dose values [5.0, 0.9, 500.0] |
| `dose_units` | list | Dose units ["%", "%", "ML"] |
| `dose_types` | list | Dose classifications ["percentage", "percentage", "volume"] |
| `total_volume_ml` | float | Total solution volume in mL |
| `drug_amount_mg` | float | Computed drug amount in mg (w/v calculation) |
| `diluent_amount_mg` | float | Computed diluent amount in mg (for saline) |
| `concentration_mg_per_ml` | float | Drug concentration in mg/mL |

#### `outputs/drugs/esoa_with_atc.csv`
Tagged ESOA entries with ATC/DrugBank IDs. Has same generated columns as `annex_f_with_atc.csv` plus original ESOA columns (ITEM_NUMBER, ITEM_REF_CODE, DESCRIPTION).

**Standardized column names (same as Annex F):**
- `atc_code` - WHO ATC code
- `drugbank_id` - DrugBank ID
- `matched_generic_name` - Canonical generic name
- `matched_reference_text` - Reference drug name from lookup
- `matched_source` - Data source (who_atc, drugbank, fda, mixtures)
- All `*_details` columns
- All dose/IV solution computed columns

#### `outputs/drugs/esoa_with_drug_codes.csv`
ESOA entries bridged to Annex F drug codes via ATC/DrugBank matching.

### Pipeline Inputs

#### `inputs/drugs/pnf_prepared.csv`
Prepared PNF (Philippine National Formulary) reference dataset.

| Column | Type | Description |
|--------|------|-------------|
| `generic_name` | str | Generic drug name (uppercase) |
| `generic_normalized` | str | Normalized generic name (salts stripped) |
| `salt_form` | str | Extracted salt suffixes |
| `atc_code` | str | WHO ATC code |
| `route` | str | Normalized route (standardized from route_tokens) |
| `form` | str | Dosage form (standardized from form_token) |
| `dose_kind` | str | Dose type (simple, ratio, etc.) |
| `strength` | float | Numeric strength value |
| `unit` | str | Strength unit |
| All `*_details` columns | str | Extracted qualifiers |

#### `inputs/drugs/fda_drug_YYYY-MM-DD.csv`
FDA Philippines drug registry (brand → generic mapping).

| Column | Type | Description |
|--------|------|-------------|
| `brand_name` | str | Brand/trade name |
| `generic_name` | str | Generic/INN name |
| `dosage_form` | str | Dosage form |
| `route` | str | Administration route |
| `dosage_strength` | str | Dose strength |
| `registration_number` | str | FDA registration number |

#### `inputs/drugs/fda_food_YYYY-MM-DD.csv`
FDA Philippines food product catalog.

### Unified Reference Tables

Located in `outputs/drugs/` after running `build_unified_reference.py`:

| File | Description |
|------|-------------|
| `unified_generics.csv` | drugbank_id → generic_name mapping |
| `unified_synonyms.csv` | drugbank_id → synonyms (pipe-separated) |
| `unified_atc.csv` | drugbank_id → atc_code (one row per combo) |
| `unified_dosages.csv` | drugbank_id × form × route × dose |
| `unified_brands.csv` | brand_name → generic_name, drugbank_id |
| `unified_salts.csv` | drugbank_id → salt forms |
| `unified_mixtures.csv` | Mixture components with component_key |

### DrugBank Lean Tables

Located in `dependencies/drugbank_generics/output/`:

| File | Description |
|------|-------------|
| `generics_lean.csv` | drugbank_id → generic_name (one row per drug) |
| `synonyms_lean.csv` | drugbank_id → synonyms (English, INN/BAN/USAN/JAN/USP) |
| `dosages_lean.csv` | drugbank_id × form × route × strength |
| `brands_lean.csv` | brand_name → drugbank_id |
| `salts_lean.csv` | drugbank_id → salt_name |
| `mixtures_lean.csv` | Mixture components with component_key |
| `products_lean.csv` | drugbank_id × dosage_form × strength × route |
| `atc_lean.csv` | drugbank_id → atc_code |

### WHO ATC Tables

Located in `dependencies/atcd/output/`:

| File | Description |
|------|-------------|
| `who_atc_YYYY-MM-DD.csv` | WHO ATC hierarchy with molecule names |

### Dose Calculation Notes

**Pharmaceutical percentage = w/v (weight/volume)**
- Definition: `% w/v = grams of solute per 100 mL of solution`
- Formula: `drug_amount_mg = (percentage/100) × volume_mL × 1000`
- Example: 5% Dextrose in 250mL = 12,500 mg dextrose, concentration = 50 mg/mL
