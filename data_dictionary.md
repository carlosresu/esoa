# Data Dictionary: `outputs/esoa_matched.csv`

Each record in `esoa_matched.csv` represents one normalized eSOA free-text row, enriched with features, reference lookups, and classification signals produced by [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py), [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py), and [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py).

📘 **Where to find implementation notes:** the modules listed above now start
with docstrings summarizing their responsibilities and contain refreshed inline
comments that describe how each column is derived.  Use them alongside this
table when validating new data or onboarding reviewers.

## Text & Normalization

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `esoa_idx` | Stable index of the eSOA row inside the processed frame. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Useful when cross-referencing intermediate artifacts. |
| `raw_text` | Original eSOA string as provided in the prepared eSOA CSV (`esoa_prepared.csv`). | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Canonical source for downstream parsing. |
| `parentheticals` | List of phrases extracted from parentheses in `raw_text`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Stored as Python list; serialized to string in CSV. |
| `normalized` | Lower-cased, punctuation-normalized version of `raw_text`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Serves as the baseline for dose and route parsing. |
| `norm_compact` | `normalized` without whitespace/hyphens to aid automata scans. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Supports brand and molecule detection. |
| `match_basis` | Working text after optional brand→generic swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Drives all reference matching when a swap occurs. |
| `match_basis_norm_basic` | Simplified normalization of `match_basis`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Input to WHO regex detection and unknown-token logic. |

## Brand Intelligence

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `probable_brands` | Pipe-delimited display names of FDA brands detected in the row. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Blank when no brand triggered; helps auditors trace swaps. |
| `did_brand_swap` | Indicates whether any FDA brand token was replaced with its generic. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | `True` even if the resulting text already contained the generic. |
| `fda_dose_corroborated` | `True` when FDA metadata confirms the detected dose. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Requires both a brand swap and matching FDA dose string. |
| `fda_generics_list` | Canonical FDA generics surfaced during brand swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Serialized as a pipe-delimited string during export; informs `generic_final` fallback. |

## Dose, Route, and Form Parsing

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `dosage_parsed_raw` | Dose dictionary parsed from `normalized` text prior to swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Produced by [scripts/dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py); keys include `kind`, `strength`, `unit`, etc. |
| `dosage_parsed` | Dose dictionary parsed after brand swaps using `match_basis`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Supersedes the raw dose when brands change the text. |
| `dose_recognized` | Friendly dose string when the final dose matches the selected PNF variant exactly. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py); refined [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Shows `"N/A"` whenever `dose_sim` < 1.0. |
| `route_raw` | Route detected from `normalized` text before swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Comes with evidence in `route_evidence_raw`. |
| `form_raw` | Form detected from `normalized` text before swaps. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | May be blank when no textual cue exists. |
| `route_evidence_raw` | Semicolon-separated evidence trail for `route_raw`/`form_raw`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Captures `form:` and `route:` markers discovered before swapping. |
| `route` | Final route after re-parsing `match_basis` and optional PNF inference. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py); updated [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | May be filled from PNF when text is silent and inference is safe. |
| `route_source` | Origin of the final route (`text`, `pnf`, or blank). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Blank indicates no confident route. |
| `route_text` | Route read directly from `match_basis` before inference. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Supports audits of inferred values. |
| `form` | Final dosage form after re-parsing and safe PNF inference. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py); updated [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Only overwritten when compatible with ratio/solid safeguards. |
| `form_source` | Origin of the final form (`text`, `pnf`, or blank). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Mirrors route handling. |
| `form_text` | Form read directly from `match_basis` before inference. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Useful when checking PNF-derived imputations. |
| `form_ok` | `True` when the detected or inferred form is compatible with PNF rules. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Evaluated against `APPROVED_ROUTE_FORMS` and exception flags. |
| `route_ok` | `True` when the route aligns with the PNF variant’s allowed routes. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | False when route evidence conflicts with PNF allowances. |
| `route_form_imputations` | Notes about accepted substitutions or flagged invalid route/form combos. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Empty when no alerts are needed. |
| `route_evidence` | Final evidence string combining text cues and PNF imputations. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py); appended [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Adds `pnf:` markers when the route was inferred from the selected variant. |

## Molecule Detection & Coverage

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `molecules_recognized` | **Pipe-delimited list of confirmed generic molecule strings used for ATC assignment and downstream matching.** | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | This is the canonical resolved-generic string reviewers rely on. |
| `molecules_recognized_list` | List form of the recognized molecules. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Duplicates removed; serialized as a Python list in CSV. |
| `molecules_recognized_count` | Count of entries in `molecules_recognized`. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Integer value. |
| `present_in_pnf` | `True` when `match_basis` hit at least one non-salt PNF entry. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Set after salt filtering and partial fallback. |
| `present_in_who` | `True` when WHO detection produced any ATC code. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Derived from `who_atc_codes`. |
| `present_in_fda_generic` | `True` when FDA generic tokens appear in `match_basis`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Highlights text already aligned with FDA generics. |
| `probable_atc` | WHO ATC suggestion when no PNF match exists but WHO coverage does. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Empty string otherwise. |

## PNF Hit Diagnostics

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `generic_id` | Primary PNF generic identifier selected from matches. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | May be filled via partial index fallback. |
| `molecule_token` | Span from `match_basis` that yielded `generic_id`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Helpful for reviewer context. |
| `pnf_hits_gids` | List of all PNF generic IDs matched before primary selection. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Includes candidates filtered out later. |
| `pnf_hits_count` | Number of qualifying PNF matches (post salt-filter, plus partial fallback). | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Integer. |
| `pnf_hits_tokens` | Raw PNF tokens that matched the text. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Stored as list. |

## WHO ATC Details

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `who_molecules_list` | List of WHO molecule names detected in `match_basis`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Deduplicated; serialized as list. |
| `who_molecules` | Pipe-delimited WHO molecule names. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Empty string when no WHO hits. |
| `who_atc_codes_list` | List of WHO ATC codes associated with detected molecules. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Sorted for stability. |
| `who_atc_codes` | Pipe-delimited WHO ATC codes. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Empty string when none found. |
| `who_atc_count` | Count of WHO ATC codes on the row. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Integer. |
| `who_atc_has_ddd` | `True` if any matched WHO code carries a defined daily dose. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Boolean flag. |
| `who_atc_adm_r` | Pipe-delimited WHO administration-route annotations from DDD metadata. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Lowercase route abbreviations. |
| `who_route_tokens` | List of canonical route tokens inferred from WHO Adm.R metadata. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Used when PNF lacks route allowances. |
| `who_form_tokens` | List of canonical form tokens inferred from WHO Adm.R/UOM metadata. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Supports WHO-only route/form reconciliation. |

## Selected PNF Variant & Dose Alignment

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `selected_form` | Form token of the chosen PNF variant. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Matches `form_token` from PNF data. |
| `selected_route_allowed` | Pipe-delimited routes permitted by the selected PNF row. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Mirrors `route_allowed` in PNF. |
| `selected_variant` | Human-readable summary of the chosen PNF dose (kind/strength/ratio). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Example: `amount:500mg` or `ratio:5mg/5mL`. |
| `selected_dose_kind` | PNF dose kind (`amount`, `ratio`, `percent`). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Kept even when dose text was missing. |
| `selected_strength` | Strength value from PNF. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | May be `None` when not applicable. |
| `selected_unit` | Unit corresponding to `selected_strength`. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Lowercase units. |
| `selected_strength_mg` | Strength converted to milligrams for comparisons. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Float or `None`. |
| `selected_per_val` | Denominator value for ratio/per-unit doses. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Typically integer. |
| `selected_per_unit` | Denominator unit label (e.g., `mL`, `tablet`). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Lowercase text. |
| `selected_ratio_mg_per_ml` | Precomputed mg/mL value from PNF. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Float or `None`. |
| `selected_pct` | Percent strength from PNF when relevant. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Float or `None`. |
| `dose_sim` | Final dose similarity (0.0 or 1.0) after recomputation with the selected PNF variant. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Enforces exact equality (after unit conversion) by design. |

## Combination & Unknown Analysis

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `looks_combo_final` | `True` when at least two known generics are detected (combination product). | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Built from PNF/WHO/FDA known tokens. |
| `combo_reason` | Textual explanation of the combo heuristic outcome. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | `combo/known-generics>=2` or `single/heuristic`. |
| `combo_known_generics_count` | Number of unique known generic tokens that triggered combo detection. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Integer. |
| `unknown_kind` | Qualifies unresolved tokens (`None`, `Single - Unknown`, etc.). | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Guides reviewer triage. |
| `unknown_words_list` | List of normalized tokens not recognized in PNF/WHO/FDA vocabularies. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Used by [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py); serialized to string. |
| `unknown_words` | Pipe-delimited string version of `unknown_words_list`. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Empty when every token is known. |
| `qty_pnf` | Count of tokens attributed to PNF matches for the row. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Summed in summaries to show PNF coverage per bucket. |
| `qty_who` | Count of WHO molecule hits contributing to the row. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Excludes PNF overlaps; used in bucket summaries. |
| `qty_fda_drug` | Count of FDA brand→generic detections that survived scoring. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Helps quantify reliance on FDA mappings. |
| `qty_fda_food` | Count of FDA food / non-therapeutic catalog tokens matched. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Highlights non-therapeutic influence within each bucket. |
| `qty_unknown` | Final count of unresolved tokens after fallback heuristics. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Includes fallback counts when `unknown_words_list` is empty but the row failed matching. |

## FDA Food / Non-Therapeutic Detection

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `non_therapeutic_summary` | High-level marker when FDA food/non-therapeutic hits were found. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | `non_therapeutic_detected` signals promotion to the Unknown bucket. |
| `non_therapeutic_detail` | Human-readable detail of the highest scoring catalog match. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Includes brand/product/company/registration data pulled from the FDA catalog. |
| `non_therapeutic_tokens` | Canonical tokens extracted from matching catalog entries. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Exported as a pipe-delimited string; excluded from `unknown_words`. |
| `non_therapeutic_hits` | JSON list of every catalog row that matched the text. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Preserved for debugging score choices and catalog coverage. |
| `non_therapeutic_best` | JSON object for the highest scoring catalog row. | [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py) | Mirrors `non_therapeutic_detail` but keeps the original structured fields. |

## Confidence & Final Classification

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `atc_code_final` | ATC code inherited from the selected PNF row (if any). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | `None` when the PNF row lacks an ATC. |
| `confidence` | Composite 0–100 score covering generic, dose, route, ATC, and brand-swap bonus. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Higher values suggest stronger auto-match evidence. |
| `match_molecule(s)` | Source labels describing which reference validated the molecule (PNF/WHO/FDA/brand). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Drives reporting pivots. |
| `generic_final` | Canonical molecule identifier(s) chosen after PNF→WHO→FDA fallback. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Pipe-delimited string; prefers `generic_id`, else WHO molecules, else FDA generics derived from alias/fuzzy matching. |
| `match_quality` | Summary tag indicating why a row auto-accepted or still needs review. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Always populated; see the list of enumerated values below. |
| `detail_final` | Supplemental descriptors describing unknown-token counts and FDA food detection outcomes. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Semicolon-separated phrases (no raw tokens); when unknown tokens remain, shows `Unknown tokens remaining: <count>`. |

**`match_molecule(s)` values in daily reporting**

- `ValidMoleculeWithATCinPNF` – Exact or inferred PNF generic with ATC coverage survived scoring.
- `ValidBrandSwappedForGenericInPNF` – FDA brand map swapped the text to a PNF generic that subsequently passed scoring checks.
- `ValidBrandSwappedForMoleculeWithATCinWHO` – Brand swap landed on a WHO molecule (no PNF coverage) that still carried ATC metadata.
- `ValidMoleculeWithATCinWHO/NotInPNF` – WHO molecule matched without any PNF hit; we rely entirely on WHO metadata.
- `ValidMoleculeNoATCinFDA/NotInPNF` – FDA generic matched but neither PNF nor WHO produced a molecule/ATC pairing.
- `ValidMoleculeNoATCinPNF` – PNF matched but the prepared PNF extract lacks an ATC for the selected variant.
- `NonTherapeuticFoodWithUnknownTokens` – FDA food/non-therapeutic catalog match present together with residual unknown tokens.
- `NonTherapeuticFoodNoMolecule` – FDA food/non-therapeutic catalog match and no therapeutic molecule confirmed.
- `NonTherapeuticCatalogOnly` – FDA food/non-therapeutic catalog matched while therapeutic catalogs had no coverage.
- `PartiallyKnownTokensFrom_<sources>` – Some tokens remain unmatched even after PNF/WHO/FDA drug lookups; the suffix enumerates the datasets (e.g., `PartiallyKnownTokensFrom_PNF_WHO`, `PartiallyKnownTokensFrom_None`) that covered the known portion of the string.
- `NoReferenceCatalogMatches` – No PNF/WHO/FDA drug or FDA food catalog entries matched any tokens; requires manual inspection.
- `AllTokensUnknownTo_PNF_WHO_FDA` – No catalog (PNF/WHO/FDA) recognized any token in the row; all tokens remain unknown and the row is routed to the Unknown bucket.
- `RowFailedAllMatchingSteps` – All reference matching stages failed; no catalogs provided coverage.

**`match_quality` review / auto-accept tags**

- `auto_exact_dose_route_form` – Auto-Accept row with exact dose, route, and form alignment against the selected PNF variant.
- `dose_mismatch_same_atc` – Auto-Accept row where the text dose differs but the PNF variant’s ATC is unique across all doses (policy allows the substitution).
- `dose_mismatch_varied_atc` – Auto-Accept row with a non-exact dose where multiple ATC payloads exist across PNF variants; escalated for targeted reconciliation.
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
| `bucket_final` | Final workflow bucket (`Auto-Accept`, `Candidates`, `Needs review`, `Unknown`). | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Determines downstream handling. |
| `why_final` | High-level justification aligned with `bucket_final`. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Mirrors the bucket label for readability. |
| `reason_final` | Expanded rationale for review or other buckets. | [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) | Includes unknown-token annotations when applicable. |
