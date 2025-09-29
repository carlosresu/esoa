# Data Dictionary: `outputs/esoa_matched.csv`

Each row corresponds to one normalized eSOA free-text line. The columns below describe the derived attributes, their intent, and where in the pipeline they are populated. Code references use repository-relative paths with 1-based line numbers.

| Column | Meaning | First Assigned | Notes |
| --- | --- | --- | --- |
| `raw_text` | Original eSOA line used as the parsing baseline. | `scripts/match_features.py:162` | Pulled directly from `esoa_prepared.csv`.
| `normalized` | Canonically normalized version of `raw_text` (lowercase, spacing, punctuation). | `scripts/match_features.py:165` | All downstream text detectors use this form.
| `match_basis` | Text after optional FDA brand→generic swaps; canonical target for matching. | `scripts/match_features.py:268` | Falls back to `normalized` when no brand map is available.
| `molecules_recognized` | Pipe-delimited list of normalized molecule names recognized in PNF/WHO lookups. | `scripts/match_scoring.py:592-611` | Built from union of PNF and WHO hits.
| `molecules_recognized_count` | Count of elements in `molecules_recognized`. | `scripts/match_scoring.py:612` | Integer value.
| `dose_recognized` | Human-friendly dose string when the detected dose exactly matches the selected PNF variant. | `scripts/match_features.py:175`; refined `scripts/match_scoring.py:483-487,759-760` | Set to "N/A" when `dose_sim` < 1.0.
| `route` | Final route of administration after extraction and safe PNF inference. | `scripts/match_features.py:278`; imputed `scripts/match_scoring.py:402-406` | May be filled from PNF when text is silent and consistent.
| `route_source` | Indicates whether `route` came from text (`text`), PNF inference (`pnf`), or is blank. | `scripts/match_scoring.py:396-399` | Blank means no confident route.
| `route_text` | Route detected directly from the normalized text before inference. | `scripts/match_scoring.py:399-400` | Mirrors the initial extraction.
| `form` | Final dosage form after extraction and safe PNF inference. | `scripts/match_features.py:278`; imputed `scripts/match_scoring.py:402-404` | Only overwritten when consistent with ratio/solid safeguards.
| `form_source` | Indicates whether `form` came from text (`text`), PNF inference (`pnf`), or is blank. | `scripts/match_scoring.py:396-399` | Blank means form remained unknown.
| `form_text` | Dosage form detected directly from the text before inference. | `scripts/match_scoring.py:399` | Useful for auditing PNF-driven imputations.
| `present_in_pnf` | True when `match_basis` matches at least one non-salt PNF molecule. | `scripts/match_features.py:454` | Gallows per row for Aho-Corasick hits.
| `present_in_who` | True when WHO detectors returned any ATC code. | `scripts/match_features.py:455` | Derived from `who_atc_codes`.
| `present_in_fda_generic` | True when FDA generic tokens appear in `match_basis`. | `scripts/match_features.py:456` | Uses FDA generic token set.
| `probable_atc` | WHO ATC suggestion when the molecule is absent from PNF but found in WHO. | `scripts/match_scoring.py:615` | Empty string otherwise.
| `generic_id` | Primary PNF generic identifier selected from matches. | `scripts/match_features.py:298-319` | Partial index fallback fills remaining gaps.
| `molecule_token` | Span of `match_basis` that produced `generic_id`. | `scripts/match_features.py:298-319` | Helpful for reviewer context.
| `pnf_hits_count` | Number of PNF matches (post salt-filter, plus partial fallback). | `scripts/match_features.py:283-319` | Integer.
| `pnf_hits_tokens` | JSON-like list of raw PNF tokens matched. | `scripts/match_features.py:283-299` | Serialized when written to CSV.
| `who_molecules` | Pipe-delimited WHO molecule names found in `match_basis`. | `scripts/match_features.py:352-355` | Empty if no WHO hits.
| `who_atc_codes` | Pipe-delimited WHO ATC codes corresponding to `who_molecules`. | `scripts/match_features.py:352-356` | Empty string when no codes.
| `who_atc_count` | Count of WHO ATC codes attached to the row. | `scripts/match_scoring.py:613` | Mirrors list length.
| `who_atc_has_ddd` | True if any matched WHO code includes a defined daily dose. | `scripts/match_features.py:349-357` | Boolean.
| `who_atc_adm_r` | Pipe-delimited WHO administration-route annotations grouped from DDD metadata. | `scripts/match_features.py:349-357` | Lowercased labels.
| `route_evidence` | Semicolon-separated audit trail of route/form detections and imputations. | `scripts/match_features.py:277-279`; appended `scripts/match_scoring.py:405-415` | Includes `route:` and `pnf:` tags.
| `route_form_imputations` | Notes on route/form validation against the curated whitelist or flags. | `scripts/match_scoring.py:504-566` | Empty string when nothing notable.
| `dosage_parsed` | JSON-style dict describing the parsed dose (kind/strength/unit/per values/pct). | `scripts/match_features.py:277`; via `scripts/dose.py` | `None` when no dose was detected.
| `selected_form` | Form token from the chosen PNF reference row. | `scripts/match_scoring.py:252-287,462-481` | Used for consistency checks.
| `selected_route_allowed` | Pipe-delimited set of PNF-approved routes for the selected variant. | `scripts/match_scoring.py:252-287,462-481` | Supports route validation.
| `selected_variant` | Human-readable summary of the selected PNF dose configuration. | `scripts/match_scoring.py:252-287,462-471` | Example: `amount:500mg` or `ratio:5mg/5mL`.
| `selected_dose_kind` | Dose kind from the chosen PNF payload (`amount`, `ratio`, `percent`). | `scripts/match_scoring.py:252-287,462-481` | Mirrors `dose_kind` field.
| `selected_strength` | Raw strength value from the selected PNF record. | `scripts/match_scoring.py:252-287,462-481` | May be `None` for percent-only entries.
| `selected_unit` | Unit attached to `selected_strength`. | `scripts/match_scoring.py:252-287,462-481` | Lowercase units (`mg`, `mcg`, etc.).
| `selected_strength_mg` | Strength converted to mg for comparisons. | `scripts/match_scoring.py:252-287,462-481` | Float or `None`.
| `selected_per_val` | Denominator value for ratio/per-unit doses. | `scripts/match_scoring.py:252-287,462-481` | Typically integers.
| `selected_per_unit` | Denominator unit label (e.g., `mL`, `tablet`). | `scripts/match_scoring.py:252-287,462-481` | Lowercase text.
| `selected_ratio_mg_per_ml` | Precomputed mg/mL ratio from PNF when available. | `scripts/match_scoring.py:252-287,462-481` | Float or `None`.
| `selected_pct` | Percent strength from PNF when applicable. | `scripts/match_scoring.py:252-287,462-481` | Float or `None`.
| `dose_sim` | Final dose similarity (0.0 or 1.0) after recomputation with the selected PNF variant. | `scripts/match_scoring.py:210-214,348-368` | Exact matches only.
| `did_brand_swap` | True when FDA brand text was replaced with the mapped generic. | `scripts/match_features.py:268-270` | False when no swap occurred or map unavailable.
| `looks_combo_final` | True when at least two known generic tokens were detected (combination product). | `scripts/match_features.py:381-385` | Boolean flag.
| `combo_reason` | Explanation of `looks_combo_final` decision. | `scripts/match_features.py:381-385` | `combo/known-generics>=2` or `single/heuristic`.
| `combo_known_generics_count` | Count of unique known generic tokens supporting combo detection. | `scripts/match_features.py:381-385` | Integer.
| `unknown_kind` | Category describing unresolved tokens (`None`, `Single - Unknown`, etc.). | `scripts/match_features.py:411-449` | Guides reviewer triage.
| `unknown_words` | Pipe-delimited list of normalized tokens not in PNF/WHO/FDA vocabularies. | `scripts/match_features.py:447-449` | Empty string when fully recognized.
| `atc_code_final` | ATC code inherited from the selected PNF row, if present. | `scripts/match_scoring.py:252-287` | `None` when PNF row lacks an ATC.
| `confidence` | Composite confidence score (0–100) combining generic/dose/route/ATC factors. | `scripts/match_scoring.py:568-590` | Higher implies stronger auto-match evidence.
| `match_molecule(s)` | Label describing which reference source validated the molecule (PNF/WHO/FDA/brand swap). | `scripts/match_scoring.py:647-652` | Useful for downstream pivots.
| `match_quality` | Reason text summarizing why the record is flagged (`dose mismatch`, `form mismatch`, etc.). | `scripts/match_scoring.py:230-233,658-708` | Defaults to `unspecified` when no issue highlighted.
| `bucket_final` | Final workflow bucket (`Auto-Accept`, `Needs review`, `Others`). | `scripts/match_scoring.py:654-757` | Drives review prioritization.
| `why_final` | High-level justification aligned with `bucket_final`. | `scripts/match_scoring.py:658-757` | Often `Needs review` or `Unknown`.
| `reason_final` | Expanded human-readable rationale for review/others buckets. | `scripts/match_scoring.py:709-757` | Includes unknown-token context when applicable.
