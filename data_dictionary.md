# Data Dictionary: `outputs/esoa_matched.csv`

Each record in `esoa_matched.csv` represents one normalized eSOA free-text row, enriched with features, reference lookups, and classification signals produced by [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py), [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py), and [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py). Line numbers refer to repository-relative files using 1-based indexing.

## Text & Normalization

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `esoa_idx` | Stable index of the eSOA row inside the processed frame. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):164 | Useful when cross-referencing intermediate artifacts.
| `raw_text` | Original eSOA string as provided in `esoa_prepared.csv`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):162 | Canonical source for downstream parsing.
| `parentheticals` | List of phrases extracted from parentheses in `raw_text`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):163 | Stored as Python list; serialized to string in CSV.
| `normalized` | Lower-cased, punctuation-normalized version of `raw_text`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):165 | Serves as the baseline for dose and route parsing.
| `norm_compact` | `normalized` without whitespace/hyphens to aid automata scans. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):166 | Supports brand and molecule detection.
| `match_basis` | Working text after optional brand→generic swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):268 | Drives all reference matching when a swap occurs.
| `match_basis_norm_basic` | Simplified normalization of `match_basis`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):271-272 | Input to WHO regex detection and unknown-token logic.

## Brand Intelligence

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `probable_brands` | Pipe-delimited display names of FDA brands detected in the row. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):184-270 | Blank when no brand triggered; helps auditors trace swaps.
| `did_brand_swap` | Indicates whether any FDA brand token was replaced with its generic. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):268-270 | `True` even if the resulting text already contained the generic.
| `fda_dose_corroborated` | `True` when FDA metadata confirms the detected dose. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):261-270 | Requires both a brand swap and matching FDA dose string.

## Dose, Route, and Form Parsing

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `dosage_parsed_raw` | Dose dictionary parsed from `normalized` text prior to swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):172-175 | Produced by [scripts/dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py); keys include `kind`, `strength`, `unit`, etc.
| `dosage_parsed` | Dose dictionary parsed after brand swaps using `match_basis`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):274-278 | Supersedes the raw dose when brands change the text.
| `dose_recognized` | Friendly dose string when the final dose matches the selected PNF variant exactly. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):175; refined [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):483-487,759-760 | Shows `"N/A"` whenever `dose_sim` < 1.0.
| `route_raw` | Route detected from `normalized` text before swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):176 | Comes with evidence in `route_evidence_raw`.
| `form_raw` | Form detected from `normalized` text before swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):176 | May be blank when no textual cue exists.
| `route_evidence_raw` | Semicolon-separated evidence trail for `route_raw`/`form_raw`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):176 | Captures `form:` and `route:` markers discovered before swapping.
| `route` | Final route after re-parsing `match_basis` and optional PNF inference. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):278; updated [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):402-406 | May be filled from PNF when text is silent and inference is safe.
| `route_source` | Origin of the final route (`text`, `pnf`, or blank). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):396-399 | Blank indicates no confident route.
| `route_text` | Route read directly from `match_basis` before inference. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):399-400 | Supports audits of inferred values.
| `form` | Final dosage form after re-parsing and safe PNF inference. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):278; updated [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):402-404 | Only overwritten when compatible with ratio/solid safeguards.
| `form_source` | Origin of the final form (`text`, `pnf`, or blank). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):396-399 | Mirrors route handling.
| `form_text` | Form read directly from `match_basis` before inference. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):399 | Useful when checking PNF-derived imputations.
| `form_ok` | `True` when the detected or inferred form is compatible with PNF rules. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):504-566 | Evaluated against `APPROVED_ROUTE_FORMS` and exception flags.
| `route_ok` | `True` when the route aligns with the PNF variant’s allowed routes. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):190-223 and `503-566 | False when route evidence conflicts with PNF allowances.
| `route_form_imputations` | Notes about accepted substitutions or flagged invalid route/form combos. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):504-566 | Empty when no alerts are needed.
| `route_evidence` | Final evidence string combining text cues and PNF imputations. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):277-279; appended [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):405-415 | Adds `pnf:` markers when the route was inferred from the selected variant.

## Molecule Detection & Coverage

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `molecules_recognized` | **Pipe-delimited list of confirmed generic molecule strings used for ATC assignment and downstream matching.** | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):592-612 | This is the canonical resolved-generic string reviewers rely on.
| `molecules_recognized_list` | List form of the recognized molecules. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):592-612 | Duplicates removed; serialized as a Python list in CSV.
| `molecules_recognized_count` | Count of entries in `molecules_recognized`. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):612 | Integer value.
| `present_in_pnf` | `True` when `match_basis` hit at least one non-salt PNF entry. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):452-454 | Set after salt filtering and partial fallback.
| `present_in_who` | `True` when WHO detection produced any ATC code. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):455 | Derived from `who_atc_codes`.
| `present_in_fda_generic` | `True` when FDA generic tokens appear in `match_basis`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):456 | Highlights text already aligned with FDA generics.
| `probable_atc` | WHO ATC suggestion when no PNF match exists but WHO coverage does. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):615 | Empty string otherwise.

## PNF Hit Diagnostics

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `generic_id` | Primary PNF generic identifier selected from matches. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):283-319 | May be filled via partial index fallback.
| `molecule_token` | Span from `match_basis` that yielded `generic_id`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):283-319 | Helpful for reviewer context.
| `pnf_hits_gids` | List of all PNF generic IDs matched before primary selection. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):283-299 | Includes candidates filtered out later.
| `pnf_hits_count` | Number of qualifying PNF matches (post salt filter plus partial fallback). | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):283-319 | Integer.
| `pnf_hits_tokens` | Raw PNF tokens that matched the text. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):283-299 | Stored as list.

## WHO ATC Details

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `who_molecules_list` | List of WHO molecule names detected in `match_basis`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):321-357 | Deduplicated; serialized as list.
| `who_molecules` | Pipe-delimited WHO molecule names. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):352-355 | Empty string when no WHO hits.
| `who_atc_codes_list` | List of WHO ATC codes associated with detected molecules. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):321-357 | Sorted for stability.
| `who_atc_codes` | Pipe-delimited WHO ATC codes. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):352-356 | Empty string when none found.
| `who_atc_count` | Count of WHO ATC codes on the row. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):613 | Integer.
| `who_atc_has_ddd` | `True` if any matched WHO code carries a defined daily dose. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):349-357 | Boolean flag.
| `who_atc_adm_r` | Pipe-delimited WHO administration-route annotations from DDD metadata. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):349-357 | Lowercase route abbreviations.

## Selected PNF Variant & Dose Alignment

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `selected_form` | Form token of the chosen PNF variant. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Matches `form_token` from PNF data.
| `selected_route_allowed` | Pipe-delimited routes permitted by the selected PNF row. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Mirrors `route_allowed` in PNF.
| `selected_variant` | Human-readable summary of the chosen PNF dose (kind/strength/ratio). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-471 | Example: `amount:500mg` or `ratio:5mg/5mL`.
| `selected_dose_kind` | PNF dose kind (`amount`, `ratio`, `percent`). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Kept even when dose text was missing.
| `selected_strength` | Strength value from PNF. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | May be `None` when not applicable.
| `selected_unit` | Unit corresponding to `selected_strength`. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Lowercase units.
| `selected_strength_mg` | Strength converted to milligrams for comparisons. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Float or `None`.
| `selected_per_val` | Denominator value for ratio/per-unit doses. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Typically integer.
| `selected_per_unit` | Denominator unit label (e.g., `mL`, `tablet`). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Lowercase text.
| `selected_ratio_mg_per_ml` | Precomputed mg/mL value from PNF. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Float or `None`.
| `selected_pct` | Percent strength from PNF when relevant. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287,462-481 | Float or `None`.
| `dose_sim` | Final dose similarity (0.0 or 1.0) after recomputation with the selected PNF variant. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):210-214,348-368 | Enforces exact equality (after unit conversion) by design.

## Combination & Unknown Analysis

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `looks_combo_final` | `True` when at least two known generics are detected (combination product). | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):368-385 | Built from PNF/WHO/FDA known tokens.
| `combo_reason` | Textual explanation of the combo heuristic outcome. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):381-385 | `combo/known-generics>=2` or `single/heuristic`.
| `combo_known_generics_count` | Number of unique known generic tokens that triggered combo detection. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):381-385 | Integer.
| `unknown_kind` | Qualifies unresolved tokens (`None`, `Single - Unknown`, etc.). | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):411-449 | Guides reviewer triage.
| `unknown_words_list` | List of normalized tokens not recognized in PNF/WHO/FDA vocabularies. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):411-449 | Used by [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py); serialized to string.
| `unknown_words` | Pipe-delimited string version of `unknown_words_list`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):447-449 | Empty when every token is known.

## Confidence & Final Classification

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `atc_code_final` | ATC code inherited from the selected PNF row (if any). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287 | `None` when the PNF row lacks an ATC.
| `confidence` | Composite 0–100 score covering generic, dose, route, ATC, and brand-swap bonus. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):568-590 | Higher values suggest stronger auto-match evidence.
| `match_molecule(s)` | Source labels describing which reference validated the molecule (PNF/WHO/FDA/brand). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):647-652 | Drives reporting pivots.
| `match_quality` | Summary for review (`dose mismatch`, `form mismatch`, etc.). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):230-233,658-708 | Defaults to `unspecified` when no issue noted.
| `bucket_final` | Final workflow bucket (`Auto-Accept`, `Needs review`, `Others`). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):654-757 | Determines downstream handling.
| `why_final` | High-level justification aligned with `bucket_final`. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):658-757 | Often `Needs review` or `Unknown`.
| `reason_final` | Expanded rationale for review or other buckets. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):709-757 | Includes unknown-token annotations when applicable.
