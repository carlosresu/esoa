# AGENT INSTRUCTIONS

These rules are meant for GPT agents running as `gpt-X.x-codex-high` or `gpt-X.x-codex-mini-high`. Apply them whenever you are editing this repository:

1. **Keep the standalone FDA scraper dependency files aligned.** Whenever you touch `pipelines/drugs/scripts/` (especially the normalization helpers or third-party imports) make sure the counterpart `dependencies/fda_ph_scraper/text_utils.py` captures the same text-processing logic and the `dependencies/fda_ph_scraper/requirements.txt` lists the packages needed by those scripts so the standalone scraper runs with the same dependencies as the pipeline helpers.
2. **Standalone runner behavior.** The scripts under `dependencies/fda_ph_scraper`, `dependencies/atcd`, and `dependencies/drugbank_generics` must continue to be runnable on their own roots; they should default to writing outputs under their own `output/` directories while downstream runners copy those exports into `inputs/drugs/`.
3. **Commit & submodule workflow.** Before creating a commit, inspect every submodule (`git submodule status`) for unstaged changes. If a submodule changed, commit and push that submodule first using a concise message describing its diff, then update the main repository's submodule pointer (commit and push that change separately). Always pull/push as appropriate within each repository before moving on so every agent run leaves the working tree clean.
4. **Parquet-first data policy.** Parquet is the primary data format across all pipeline steps. When reading data, prefer `.parquet` files over `.csv` when both exist. When writing data, always export both `.parquet` (primary) and `.csv` (compatibility fallback). CSV exports exist only for human readability and compatibility with external tools.
5. **Canonical file naming.** Use date-stamped naming for reference datasets: `who_atc_YYYY-MM-DD.parquet`, `fda_drug_YYYY-MM-DD.parquet`, etc. Legacy naming patterns like `*_molecules.csv` are deprecated.

## Pipeline Execution

6. **Use 8 workers for R scripts.** When running DrugBank or other R scripts via Python, set `ESOA_DRUGBANK_WORKERS=8` environment variable.
7. **Pipeline part dependencies.** The drugs pipeline has 4 parts that must run in order:
   - Part 1: Prepare dependencies (DrugBank generics, mixtures, brands, FDA data)
   - Part 2: Match Annex F with ATC/DrugBank IDs → outputs `annex_f_with_atc.csv`
   - Part 3: Match ESOA with ATC/DrugBank IDs → uses Part 2 output as reference
   - Part 4: Bridge ESOA to Annex F Drug Codes → uses Part 3 output

## Drug Matching Policies

8. **Tiered matching strategy.** Match drugs in this priority order:
   - Tier 1: PNF + WHO + DrugBank Generics + DrugBank Mixtures (known pharmaceuticals with ATC/DrugBank IDs)
   - Tier 2: DrugBank Brands + FDA Drug Brands (brand→generic swap, then re-run Tier 1)
   - Tier 3: FDA Food (food/supplement detection for non-drugs)
   - Tier 4: Fallback parsing for unknown categories

9. **Multi-word generic names.** Preserve known multi-word generics as single tokens (e.g., "tranexamic acid", "folic acid", "insulin glargine"). Do not split these into individual words during tokenization.

10. **Form token filtering.** When comparing molecules between ESOA and Annex F, filter out form tokens (tablet, capsule, vial, ampule, bottle, sachet, etc.) from the molecules list. These are dosage forms, not drug names.

11. **Single vs combination ATC codes.** When an ESOA row contains a single molecule, prefer single-drug ATC codes over combination ATCs. For example, LOSARTAN alone should get C09CA01, not C09DA01 (losartan+HCTZ combo).

12. **Brand sources.** Brand→generic mapping uses combined sources:
    - FDA Drug brands (`fda_drug_*.csv`)
    - DrugBank brands (`drugbank_brands_master.csv` from products where generic=false)
    - DrugBank mixture names (brand names of combination products)

## Reference Data

13. **DrugBank data extraction.** The `dependencies/drugbank_generics/` directory contains R scripts that extract from the `dbdataset` package:
    - `drugbank_generics.R` → `drugbank_generics_master.csv`
    - `drugbank_mixtures.R` → `drugbank_mixtures_master.csv`
    - `drugbank_brands.R` → `drugbank_brands_master.csv` + `drugbank_products_export.csv`

14. **WHO ATC data.** Use the canonical `who_atc_YYYY-MM-DD.parquet` files. The `load_who_molecules()` function supports both parquet and CSV formats.
