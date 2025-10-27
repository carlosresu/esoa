# LaboratoryAndDiagnostic Pipeline Walkthrough

This short guide outlines the LaboratoryAndDiagnostic pipeline, from CLI invocation in
[run_labs.py](https://github.com/carlosresu/esoa/blob/main/run_labs.py) to the matching
and output stages in `pipelines/labs/scripts`.

1. **Input preparation** (`pipelines/labs/scripts/prepare_labs.py`)
   - Reads `raw/03 ESOA_ITEM_LIB.csv` and `.tsv`, filters `ITEM_REF_CODE == "LaboratoryAndDiagnostic"`,
     and drops ITEM_NUMBER 1540–1896.
   - Normalizes description text, deduplicates overlaps between CSV/TSV, and writes
     `inputs/labs/esoa_prepared_labs.csv`.
2. **Master catalogue loading**
   - Loads the hospital Labs catalogue from `inputs/labs/Labs.csv` and optional Diagnostics
     reference data from `raw/Diagnostics.xlsx` (columns A–F).
3. **Matching** (`pipelines/labs/scripts/match_labs.py`)
   - Normalizes each eSOA description, tries an exact match against the Labs master list,
     and falls back to the Diagnostics workbook when necessary.
   - Records provenance columns (`match_source`, `lab_*`, `diagnostics_*`) for auditability.
4. **Outputs**
   - Writes `outputs/labs/esoa_matched_labs.csv` (and an XLSX companion unless `--skip-excel`
     is provided) summarizing the standardized description and matching details.
