# Data Dictionary: `outputs/drugs_and_medicine_drugs/esoa_matched_drugs.csv`

Each record in `esoa_matched_drugs.csv` represents one normalized eSOA free-text row, enriched with features, reference lookups, and classification signals produced by [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py), [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py), and [pipelines/drugs/scripts/match_outputs_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_outputs_drugs.py).

📘 **Where to find implementation notes:** the modules listed above now start
with docstrings summarizing their responsibilities and contain refreshed inline
comments that describe how each column is derived.  Use them alongside this
table when validating new data or onboarding reviewers.

`run_drugs_and_medicine.py` now orchestrates dependency bootstrapping, input preparation, and the post-run execution of `pipelines/drugs/scripts/resolve_unknowns_drugs.py`, so the companion outputs mentioned below (e.g., `outputs/drugs_and_medicine_drugs/unknown_words.csv`) are regenerated automatically each time the pipeline completes.

⚙️ **Performance note:** Brand swaps, WHO detection, and other CPU-heavy transforms now fan out across a worker pool when large batches are processed. The helpers keep row order stable, so the column definitions below remain unchanged even when the pipeline runs in parallel.

## Text & Normalization

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `esoa_idx` | Stable index of the eSOA row inside the processed frame. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Useful when cross-referencing intermediate artifacts. |
| `raw_text` | Original eSOA string as provided in the prepared eSOA CSV (`esoa_prepared.csv`). | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Canonical source for downstream parsing. |
| `parentheticals` | List of phrases extracted from parentheses in `raw_text`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Stored as Python list; serialized to string in CSV. |
| `normalized` | Lower-cased, punctuation-normalized version of `raw_text`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Serves as the baseline for dose and route parsing. |
| `norm_compact` | `normalized` without whitespace/hyphens to aid automata scans. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Supports brand and molecule detection. |
| `match_basis` | Working text after optional brand→generic swaps. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Drives all reference matching when a swap occurs. |
| `match_basis_norm_basic` | Simplified normalization of `match_basis`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Input to WHO regex detection and unknown-token logic. |

## Brand Intelligence

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `probable_brands` | Pipe-delimited display names of FDA brands detected in the row. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Blank when no brand triggered; helps auditors trace swaps. |
| `did_brand_swap` | Indicates whether any FDA brand token was replaced with its generic. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | `True` even if the resulting text already contained the generic. |
| `brand_swap_added_generic` | `True` only when the swap inserted new generic tokens into the text. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Drives the +10 confidence bonus once dose/form/route stay aligned. |
| `fda_dose_corroborated` | `True` when FDA metadata confirms the detected dose. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Requires both a brand swap and matching FDA dose string. |
| `fda_generics_list` | Canonical FDA generics surfaced during brand swaps. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Serialized as a pipe-delimited string during export; informs `generic_final` fallback. |
| `drugbank_generics_list` | DrugBank generics detected after brand swaps. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Serialized to a pipe-delimited string; fuels `present_in_drugbank` and `qty_drugbank`. |

## Dose, Route, and Form Parsing

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `dosage_parsed_raw` | Dose dictionary parsed from `normalized` text prior to swaps. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Produced by [pipelines/drugs/scripts/dose_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/dose_drugs.py); keys include `kind`, `strength`, `unit`, etc. |
| `dosage_parsed` | Dose dictionary parsed after brand swaps using `match_basis`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Supersedes the raw dose when brands change the text. |
| `dose_recognized` | Friendly dose string when the final dose matches the selected PNF variant exactly. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py); refined [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Shows `"N/A"` whenever `dose_sim` < 1.0. |
| `route_raw` | Route detected from `normalized` text before swaps. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Comes with evidence in `route_evidence_raw`. |
| `form_raw` | Form detected from `normalized` text before swaps. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | May be blank when no textual cue exists. |
| `route_evidence_raw` | Semicolon-separated evidence trail for `route_raw`/`form_raw`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Captures `form:` and `route:` markers discovered before swapping. |
| `route` | Final route after re-parsing `match_basis` and optional PNF inference. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py); updated [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | May be filled from PNF when text is silent and inference is safe. |
| `route_source` | Origin of the final route (`text`, `pnf`, or blank). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Blank indicates no confident route. |
| `route_text` | Route read directly from `match_basis` before inference. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Supports audits of inferred values. |
| `form` | Final dosage form after re-parsing and safe PNF inference. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py); updated [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Only overwritten when compatible with ratio/solid safeguards. |
| `form_source` | Origin of the final form (`text`, `pnf`, or blank). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Mirrors route handling. |
| `form_text` | Form read directly from `match_basis` before inference. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Useful when checking PNF-derived imputations. |
| `form_ok` | `True` when the detected or inferred form is compatible with PNF rules. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Evaluated against `APPROVED_ROUTE_FORMS` and exception flags. |
| `route_ok` | `True` when the route aligns with the PNF variant’s allowed routes. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | False when route evidence conflicts with PNF allowances. |
| `route_form_imputations` | Notes about accepted substitutions or flagged invalid route/form combos. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Empty when no alerts are needed. |
| `route_evidence` | Final evidence string combining text cues and Annex/PNF imputations. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py); appended [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Adds `pnf:` or `reference` markers when inference occurred. |
| `reference_route_details` | Route evidence supplied directly by the Annex/PNF reference row when text was silent. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py); finalized [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Helps auditors trace packaging-based inferences (e.g., ampule→intravenous). |

## Molecule Detection & Coverage

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `molecules_recognized` | **Pipe-delimited list of confirmed generic molecule strings used for ATC assignment and downstream matching.** | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | This is the canonical resolved-generic string reviewers rely on. |
| `molecules_recognized_list` | List form of the recognized molecules. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Duplicates removed; serialized as a Python list in CSV. |
| `molecules_recognized_count` | Count of entries in `molecules_recognized`. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Integer value. |
| `present_in_annex` | `True` when `match_basis` matched an Annex F entry (Drug Code priority). | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Derived from `reference_source == "annex_f"`. |
| `present_in_pnf` | `True` when `match_basis` hit at least one non-salt PNF entry. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Set after salt filtering and partial fallback. |
| `present_in_who` | `True` when WHO detection produced any ATC code. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Derived from `who_atc_codes`. |
| `present_in_fda_generic` | `True` when FDA generic tokens appear in `match_basis`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Highlights text already aligned with FDA generics. |
| `present_in_drugbank` | `True` when any DrugBank generic matched after swaps. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Supports classification when only DrugBank synonyms explain the text. |
| `probable_atc` | WHO ATC suggestion when no PNF match exists but WHO coverage does. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Empty string otherwise. |

## PNF Hit Diagnostics

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `generic_id` | Primary PNF generic identifier selected from matches. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | May be filled via partial index fallback. |
| `molecule_token` | Span from `match_basis` that yielded `generic_id`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Helpful for reviewer context. |
| `pnf_hits_gids` | List of all PNF generic IDs matched before primary selection. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Includes candidates filtered out later. |
| `pnf_hits_count` | Number of qualifying PNF matches (post salt-filter, plus partial fallback). | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Integer. |
| `pnf_hits_tokens` | Raw PNF tokens that matched the text. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Stored as list. |

## WHO ATC Details

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `who_molecules_list` | List of WHO molecule names detected in `match_basis`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Deduplicated; serialized as list. |
| `who_molecules` | Pipe-delimited WHO molecule names. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Empty string when no WHO hits. |
| `who_atc_codes_list` | List of WHO ATC codes associated with detected molecules. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Sorted for stability. |
| `who_atc_codes` | Pipe-delimited WHO ATC codes. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Empty string when none found. |
| `who_atc_count` | Count of WHO ATC codes on the row. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Integer. |
| `who_atc_has_ddd` | `True` if any matched WHO code carries a defined daily dose. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Boolean flag. |
| `who_atc_adm_r` | Pipe-delimited WHO administration-route annotations from DDD metadata. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Lowercase route abbreviations. |
| `who_route_tokens` | List of canonical route tokens inferred from WHO Adm.R metadata. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Used when PNF lacks route allowances. |
| `who_form_tokens` | List of canonical form tokens inferred from WHO Adm.R/UOM metadata. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Supports WHO-only route/form reconciliation. |

## Selected PNF Variant & Dose Alignment

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `selected_form` | Form token of the chosen PNF variant. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Matches `form_token` from PNF data. |
| `selected_route_allowed` | Pipe-delimited routes permitted by the selected PNF row. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Mirrors `route_allowed` in PNF. |
| `selected_variant` | Human-readable summary of the chosen PNF dose (kind/strength/ratio). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Example: `amount:500mg` or `ratio:5mg/5mL`. |
| `selected_dose_kind` | PNF dose kind (`amount`, `ratio`, `percent`). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Kept even when dose text was missing. |
| `selected_strength` | Strength value from PNF. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | May be `None` when not applicable. |
| `selected_unit` | Unit corresponding to `selected_strength`. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Lowercase units. |
| `selected_strength_mg` | Strength converted to milligrams for comparisons. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Float or `None`. |
| `selected_per_val` | Denominator value for ratio/per-unit doses. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Typically integer. |
| `selected_per_unit` | Denominator unit label (e.g., `mL`, `tablet`). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Lowercase text. |
| `selected_ratio_mg_per_ml` | Precomputed mg/mL value from PNF. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Float or `None`. |
| `selected_pct` | Percent strength from PNF when relevant. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Float or `None`. |
| `dose_sim` | Final dose similarity (0.0 or 1.0) after recomputation with the selected PNF variant. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Enforces exact equality (after unit conversion) by design. |

## Reference Catalogue Metadata

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `reference_source` | Source label for the matched catalogue row (`annex_f` or `pnf`). | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Enables reporting split between Annex-first and PNF fallback matches. |
| `reference_priority` | Numeric priority used for tie-breaking (1 = Annex F, 2 = PNF). | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py); consumed [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Lower values outrank higher ones when scores tie. |
| `reference_primary_code` | Preferred identifier for the matched row (Annex Drug Code or PNF ATC). | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Feeds confidence scoring and reporting pivots. |
| `reference_drug_code` | Raw Annex Drug Code when present. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Blank for PNF-only rows. |

## Unknown Token Analysis

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `unknown_kind` | Qualifies unresolved tokens (`None`, `Single - Unknown`, etc.). | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Guides reviewer triage. |
| `unknown_words_list` | List of normalized tokens not recognized in PNF/WHO/FDA vocabularies. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Used by [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py); serialized to string. |
| `unknown_words` | Pipe-delimited string version of `unknown_words_list`. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Empty when every token is known. |
| `qty_pnf` | Count of tokens attributed to PNF matches for the row. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Summed in summaries to show PNF coverage per bucket. |
| `qty_who` | Count of WHO molecule hits contributing to the row. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Excludes PNF overlaps; used in bucket summaries. |
| `qty_fda_drug` | Count of FDA brand→generic detections that survived scoring. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Helps quantify reliance on FDA mappings. |
| `qty_drugbank` | Count of DrugBank generics matched in the row. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Indicates fallback coverage provided by DrugBank synonyms. |
| `qty_fda_food` | Count of FDA food / non-therapeutic catalog tokens matched. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Highlights non-therapeutic influence within each bucket. |
| `qty_unknown` | Final count of unresolved tokens after fallback heuristics. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Includes fallback counts when `unknown_words_list` is empty but the row failed matching. |

## FDA Food / Non-Therapeutic Detection

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `non_therapeutic_summary` | High-level marker when FDA food/non-therapeutic hits were found. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | `non_therapeutic_detected` signals promotion to the Unknown bucket. |
| `non_therapeutic_detail` | Human-readable detail of the highest scoring catalog match. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Includes brand/product/company/registration data pulled from the FDA catalog. |
| `non_therapeutic_tokens` | Canonical tokens extracted from matching catalog entries. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Exported as a pipe-delimited string; excluded from `unknown_words`. |
| `non_therapeutic_hits` | JSON list of every catalog row that matched the text. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Preserved for debugging score choices and catalog coverage. |
| `non_therapeutic_best` | JSON object for the highest scoring catalog row. | [pipelines/drugs/scripts/match_features_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_features_drugs.py) | Mirrors `non_therapeutic_detail` but keeps the original structured fields. |

## Confidence & Final Classification

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `drug_code_final` | Annex F Drug Code selected for the row (blank when PNF-only). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Primary identifier whenever an Annex match exists. |
| `primary_code_final` | Preferred identifier used for confidence (Annex Drug Code or PNF ATC). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Mirrors `reference_primary_code` after final selection. |
| `atc_code_final` | ATC code inherited from the selected PNF row (if any). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | `None` when the chosen variant lacks an ATC or Annex supplants it. |
| `confidence` | Composite 0–100 score covering generic, dose, route, ATC, and brand-swap bonus. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Higher values suggest stronger auto-match evidence. |
| `match_molecule(s)` | Source labels describing which reference validated the molecule (Annex/PNF/WHO/FDA/brand). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Drives reporting pivots. |
| `generic_final` | Canonical molecule identifier(s) chosen after Annex→PNF→WHO→FDA fallback. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Pipe-delimited string; prefers Annex/PNF `generic_id`, else WHO molecules, else FDA generics. |
| `match_quality` | Summary tag indicating why a row auto-accepted or still needs review. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Always populated; see the list of enumerated values below. |
| `detail_final` | Supplemental descriptors describing unknown-token counts and FDA food detection outcomes. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Semicolon-separated phrases (no raw tokens); when unknown tokens remain, shows `Unknown tokens remaining: <count>`. |

**`match_molecule(s)` values in daily reporting**

- `ValidMoleculeWithDrugCodeInAnnex` – Annex F generic (Drug Code) satisfied scoring checks.
- `ValidMoleculeWithATCinPNF` – PNF generic with ATC coverage survived scoring after Annex failed or was absent.
- `ValidBrandSwappedForGenericInAnnex` – FDA brand map swap resolved to an Annex F generic that passed subsequent checks.
- `ValidBrandSwappedForGenericInPNF` – FDA brand map swap resolved to a PNF generic that passed subsequent checks.
- `ValidBrandSwappedForMoleculeWithATCinWHO` – Brand swap landed on a WHO molecule (no PNF coverage) that still carried ATC metadata.
- `ValidMoleculeWithATCinWHO/NotInPNF` – WHO molecule matched without any PNF hit; we rely entirely on WHO metadata.
- `ValidMoleculeNoATCinFDA/NotInPNF` – FDA generic matched but neither PNF nor WHO produced a molecule/ATC pairing.
- `ValidMoleculeNoCodeInReference` – Annex/PNF generic matched but no reference code (Drug Code or ATC) was available.
- `NonTherapeuticFoodWithUnknownTokens` – FDA food/non-therapeutic catalog match present together with residual unknown tokens.
- `NonTherapeuticFoodNoMolecule` – FDA food/non-therapeutic catalog match and no therapeutic molecule confirmed.
- `NonTherapeuticCatalogOnly` – FDA food/non-therapeutic catalog matched while therapeutic catalogs had no coverage.
- `PartiallyKnownTokensFrom_<sources>` – Some tokens remain unmatched even after PNF/WHO/FDA drug lookups; the suffix enumerates the datasets (e.g., `PartiallyKnownTokensFrom_PNF_WHO`, `PartiallyKnownTokensFrom_None`) that covered the known portion of the string.
- `NoReferenceCatalogMatches` – No PNF/WHO/FDA drug or FDA food catalog entries matched any tokens; requires manual inspection.
- `AllTokensUnknownTo_PNF_WHO_FDA` – No catalog (PNF/WHO/FDA) recognized any token in the row; all tokens remain unknown and the row is routed to the Unknown bucket.
- `RowFailedAllMatchingSteps` – All reference matching stages failed; no catalogs provided coverage.

**`match_quality` review / auto-accept tags**

- `auto_exact_dose_route_form` – Auto-Accept row with exact dose, route, and form alignment against the selected Annex/PNF variant.
- `dose_mismatch_same_atc` – Auto-Accept row where the text dose differs but the reference code is unique across variants (policy allows the substitution).
- `dose_mismatch_varied_atc` – Auto-Accept row with a non-exact dose where multiple reference codes exist across variants; escalated for targeted reconciliation.
- `dose_mismatch` – Recognized dose text disagrees with the selected PNF/WHO dose payload after normalization.
- `form_mismatch` – Textual form or inferred form conflicts with the PNF-allowed form family for the chosen route.
- `route_mismatch` – Textual route conflicts with the allowed routes for the candidate (including WHO fallbacks when PNF is missing).
- `route_form_mismatch` – Route and form combination violates the curated `APPROVED_ROUTE_FORMS` whitelist even if each attribute alone might be acceptable.
- `no_dose_available` / `no_form_available` / `no_route_available` – Parsed text lacks the corresponding attribute; we could not impute it with high confidence.
- `no_dose_and_form_available` / `no_dose_and_route_available` / `no_form_and_route_available` / `no_dose_form_and_route_available` – Multiple attributes were simultaneously missing, signalling limited metadata.
- `who_metadata_insufficient_review_required` – Only WHO supplied the molecule and none of the other checks raised a specific conflict, but we still lack enough corroborating detail to auto-accept.
- `who_does_not_provide_dose_info` – We rely on WHO alone and the WHO ATC extract does not expose a DDD; dose comparisons therefore remain unresolved.
- `who_does_not_provide_route_info` – We rely on WHO alone and the WHO ATC extract did not expose any Adm.R tokens; route validation falls back to manual review.
- `nontherapeutic_and_unknown_tokens` – FDA food/non-therapeutic catalog match present together with residual unknown tokens.
- `nontherapeutic_catalog_match` – FDA food/non-therapeutic catalog match with no corroborated therapeutic molecule.
- `unknown_tokens_present` – Partial unknown tokens remain after matching against PNF, WHO, and FDA drug lists.
- `manual_review_required` – No structured matches materialized; escalated for human triage.

**`match_quality` review flags**

- `dose_mismatch` – Recognized dose text disagrees with the selected PNF/WHO dose payload after normalization.
- `form_mismatch` – Textual form or inferred form conflicts with the PNF-allowed form family for the chosen route.
- `route_mismatch` – Textual route conflicts with the allowed routes for the candidate (including WHO fallbacks when PNF is missing).
- `route_form_mismatch` – Route and form combination violates the curated `APPROVED_ROUTE_FORMS` whitelist even if each attribute alone might be acceptable.
- `no_dose_available` / `no_form_available` / `no_route_available` – Parsed text lacks the corresponding attribute; we could not impute it with high confidence.
- `no_dose_and_form_available` / `no_dose_and_route_available` / `no_form_and_route_available` / `no_dose_form_and_route_available` – Multiple attributes were simultaneously missing, signalling limited metadata.
- `who_metadata_insufficient_review_required` – Only WHO supplied the molecule and none of the other checks raised a specific conflict, but we still lack enough corroborating detail to auto-accept.
- `who_does_not_provide_dose_info` – We rely on WHO alone and the WHO ATC extract does not expose a DDD; dose comparisons therefore remain unresolved.
- `who_does_not_provide_route_info` – We rely on WHO alone and the WHO ATC extract did not expose any Adm.R tokens; route validation falls back to manual review.
- `no_reference_catalog_match` – No reference or catalog matched the text; manual reconciliation required.

### Review Outcome Columns

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `bucket_final` | Final workflow bucket (`Auto-Accept`, `Candidates`, `Needs review`, `Unknown`). | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Determines downstream handling. |
| `why_final` | High-level justification aligned with `bucket_final`. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Mirrors the bucket label for readability. |
| `reason_final` | Expanded rationale for review or other buckets. | [pipelines/drugs/scripts/match_scoring_drugs.py](https://github.com/carlosresu/esoa/blob/main/pipelines/drugs/scripts/match_scoring_drugs.py) | Includes unknown-token annotations when applicable. |
