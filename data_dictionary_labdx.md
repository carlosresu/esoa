# Data Dictionary: `outputs/laboratory_and_diagnostic/esoa_matched_labs.csv`

Each record in `esoa_matched_labs.csv` corresponds to a Laboratory & Diagnostic eSOA
entry after text normalization and matching against the hospital master list and the
Diagnostics secondary catalog.

| Column | Description |
| --- | --- |
| `ITEM_NUMBER` | Original ITEM_NUMBER from the eSOA source (post filtering). |
| `DESCRIPTION` | Raw Laboratory & Diagnostic description pulled from the eSOA source. |
| `normalized_description` | Lowercased alphanumeric-normalized version of `DESCRIPTION`. |
| `match_source` | `LabAndDx`, `Diagnostics`, or `Unmatched` depending on which catalogue provided the standardized text. |
| `standard_description` | Preferred description chosen from either LabAndDx or Diagnostics. |
| `source_file` | Source filename (`03 ESOA_ITEM_LIB.csv` or `.tsv`) where the row originated. |
| `lab_item_number` | ITEM_NUMBER from `inputs/laboratory_and_diagnostic/LabAndDx.csv` when a match is found. |
| `lab_is_official` | `IS_OFFICIAL` flag from the LabAndDx master list. |
| `lab_description` | Exact description stored in the LabAndDx master list. |
| `diagnostics_code` | Column **A** (`code`) from `raw/Diagnostics.xlsx` when a Diagnostics match occurs. |
| `diagnostics_desc` | Column **B** (`desc`) from `raw/Diagnostics.xlsx`. |
| `diagnostics_cat` | Column **C** (`cat`) from `raw/Diagnostics.xlsx`. |
| `diagnostics_spec` | Column **D** (`spec`) from `raw/Diagnostics.xlsx`. |
| `diagnostics_etc` | Column **E** (`etc`) from `raw/Diagnostics.xlsx`. |
| `diagnostics_misc` | Column **F** (`misc`) from `raw/Diagnostics.xlsx`. |
|
