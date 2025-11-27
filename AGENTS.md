# AGENT INSTRUCTIONS

These rules are meant for GPT agents. Apply them whenever you are editing this repository:

> **IMPORTANT:** Before making changes to the drugs pipeline, read:
> - `debug/pipeline.md` - Algorithmic logic, pharmaceutical rules, and decision rationale
> - `debug/implementation_plan_v2.md` - Current TODO list and implementation status
>
> After every group of changes, UPDATE BOTH FILES with what was changed and any new decisions made.

1. **Keep the standalone FDA scraper dependency files aligned.** Whenever you touch `pipelines/drugs/scripts/` (especially the normalization helpers or third-party imports) make sure the counterpart `dependencies/fda_ph_scraper/text_utils.py` captures the same text-processing logic and the `dependencies/fda_ph_scraper/requirements.txt` lists the packages needed by those scripts so the standalone scraper runs with the same dependencies as the pipeline helpers.
2. **Standalone runner behavior.** The scripts under `dependencies/fda_ph_scraper`, `dependencies/atcd`, and `dependencies/drugbank_generics` must continue to be runnable on their own roots; they should default to writing outputs under their own `output/` directories while downstream runners copy those exports into `inputs/drugs/`.
3. **Commit & submodule workflow.** Before creating a commit, inspect every submodule (`git submodule status`) for unstaged changes. If a submodule changed, commit and push that submodule first using a concise message describing its diff, then update the main repository's submodule pointer (commit and push that change separately). Always pull/push as appropriate within each repository before moving on so every agent run leaves the working tree clean.
4. **Parquet-first data policy.** Parquet is the primary data format across all pipeline steps. When reading data, prefer `.parquet` files over `.csv` when both exist. When writing data, always export both `.parquet` (primary) and `.csv` (compatibility fallback). CSV exports exist only for human readability and compatibility with external tools.
5. **Canonical file naming.** Use date-stamped naming for reference datasets: `who_atc_YYYY-MM-DD.parquet`, `fda_drug_YYYY-MM-DD.parquet`, etc. Legacy naming patterns like `*_molecules.csv` are deprecated.

## Pipeline Execution

6. **Use 8 workers for R scripts.** When running DrugBank or other R scripts via Python, set `ESOA_DRUGBANK_WORKERS=8` environment variable.
7. **Pipeline part dependencies.** The drugs pipeline has 4 parts that must run in order:
   - Part 1: Prepare dependencies (DrugBank generics, mixtures, brands, salts, FDA data)
   - Part 2: Match Annex F with ATC/DrugBank IDs → outputs `annex_f_with_atc.csv`
   - Part 3: Match ESOA with ATC/DrugBank IDs → uses unified tagger (same as Part 2)
   - Part 4: Bridge ESOA to Annex F Drug Codes → uses Part 3 output

## Unified Architecture

8. **Unified tagging algorithm.** Both Annex F and ESOA use the SAME tagging algorithm. The Annex F tagger (Part 2) is the base - do not create separate algorithms for different input types.

9. **DuckDB for all queries.** Use DuckDB for all reference lookups instead of Aho-Corasick tries. Load parquet files into DuckDB and query with SQL. This is faster for exact/prefix matching after tokenization.

10. **Three reference tables.** The unified reference consists of:
    - **Generics table**: All single-entity drug names (base generics, normalized)
    - **Brands table**: FDA + DrugBank brands → maps to generic(s)
    - **Mixtures table**: Component combinations → maps to mixture info

11. **Unified reference dataset schema.** The main reference is `unified_drug_reference.parquet`:
    - Exploded by: `drugbank_id × atc_code × form × route` (only valid combos)
    - Aggregated columns: `salts`, `doses`, `brands`, `mixtures` as pipe-delimited
    - Sources tracked: `sources` column with pipe-delimited provenance

## Drug Matching Policies

12. **Salt handling.** Use `drugbank$salts` dataset for salt detection. Strip salts from matching basis UNLESS the compound is a pure salt (e.g., sodium chloride, calcium carbonate). Auto-detect pure salts: compounds where base would be empty after stripping.

13. **Dose normalization.** Canonical formats:
    - Weight: normalize to `mg` (e.g., `1g` → `1000mg`)
    - Combinations: `500mg+200mg` (fixed combo), `500mg/200mg` (ratio)
    - Concentration: `10mg/mL`, `5%`
    - Volume: `mL` is canonical for liquids

14. **Form-route validity.** Maintain `form_route_validity.parquet` with provenance. Only allow form-route combinations that exist in reference datasets. If form or route missing in input, infer the most common one.

15. **Multi-word generic names.** Preserve known multi-word generics as single tokens (e.g., "tranexamic acid", "folic acid", "insulin glargine"). Do not split these into individual words during tokenization.

16. **Single vs combination ATC codes.** When an input row contains a single molecule, prefer single-drug ATC codes over combination ATCs. For example, LOSARTAN alone should get C09CA01, not C09DA01 (losartan+HCTZ combo).

17. **Scoring algorithm.** Use deterministic pharmaceutical-principled scoring (NOT numeric weights):
    - Generic match is REQUIRED (no match without it)
    - Salt forms are IGNORED (unless pure salt compound)
    - Dose is FLEXIBLE for ATC tagging, EXACT for Drug Code matching
    - Form allows equivalents (TABLET ≈ CAPSULE)
    - Route is INFERRED from form if missing
    - ATC preference: single vs combo based on input molecule count
    - See `debug/pipeline.md` for full scoring logic

## Reference Data

18. **DrugBank data extraction.** The `dependencies/drugbank_generics/` directory contains R scripts that extract from the `dbdataset` package:
    - `drugbank_generics.R` → `drugbank_generics_master.csv`
    - `drugbank_mixtures.R` → `drugbank_mixtures_master.csv`
    - `drugbank_brands.R` → `drugbank_brands_master.csv` + `drugbank_products_export.csv`
    - `drugbank_salts.R` → `drugbank_salts_master.csv`

19. **WHO ATC data.** Use the canonical `who_atc_YYYY-MM-DD.parquet` files. The `load_who_molecules()` function supports both parquet and CSV formats.

20. **Unified dataset enrichment.** DrugBank generics is the base dataset. Enrich with:
    - PNF: synonyms, ATC codes
    - WHO: ATC codes, names
    - DrugBank products: dose/form/route variants
    - DrugBank mixtures: combination info
    - FDA drug brands: brand names, dose/form/route
    - Deduplicate across all steps to prevent row explosion
