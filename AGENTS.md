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
    - Dose is FLEXIBLE for ATC tagging, EXACT for Drug Code matching
    - Form allows equivalents (TABLET ≈ CAPSULE)
    - Route is INFERRED from form if missing
    - ATC preference: single vs combo based on input molecule count
    - See `debug/pipeline.md` for full scoring logic

## Reference Data

20. **WHO ATC data.** Use the canonical `who_atc_YYYY-MM-DD.csv` files from `dependencies/atcd/`. The `load_who_molecules()` function loads CSV files.

21. **FDA data.** Use `dependencies/fda_ph_scraper/` to generate:
    - `fda_drug_YYYY-MM-DD.csv` - brand → generic mapping with dose/form/route
    - `fda_food_YYYY-MM-DD.csv` - food product catalog (fallback for non-drugs)
