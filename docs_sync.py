"""Documentation synchronisation utility.

Re-generates key Markdown files so they stay aligned with the
current pipeline description and column metadata. Run this script
whenever pipeline behaviour or output schema changes.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

PIPELINE_MD = Path("pipeline.md")
DATA_DICTIONARY_MD = Path("data_dictionary.md")

PIPELINE_HEADER = dedent(
    """\
    # Pipeline Execution Walkthrough

    Detailed end-to-end view of the matching pipeline, from CLI invocation in [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) through feature building in [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py), scoring in [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py), and export logic in [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py).
    """
)

PIPELINE_STEPS: list[tuple[str, str]] = [
    (
        "Load Prepared Inputs",
        "Resolve CLI paths (defaults under `inputs/`), verify the files exist, and read the prepared PNF and eSOA CSVs (`pnf_prepared.csv`, `esoa_prepared.csv`) into pandas data frames (see [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) and [scripts/match.py](https://github.com/carlosresu/esoa/blob/main/scripts/match.py)).",
    ),
    (
        "Validate Structural Expectations",
        "Ensure the PNF frame exposes the required molecule, dose, route, and ATC fields and that the eSOA frame includes `raw_text` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Index Reference Vocabularies",
        "Build normalized PNF name lookups, load WHO ATC exports (including regex caches and DDD metadata), and prepare the FDA brand map automata plus generic token set (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Construct PNF Search Automata",
        "Create normalized/compact Aho–Corasick automatons and train the partial token index used for fallback matches (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Normalize eSOA Text",
        "Produce the working frame with `raw_text`, parenthetical captures, `esoa_idx`, and normalized text variants (`normalized`, `norm_compact`) (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Initial Dose / Route / Form Parsing",
        "Parse dosage structures via [scripts/dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py), compute `dose_recognized`, and collect `route_raw`, `form_raw`, and `route_evidence_raw` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Apply FDA Brand → Generic Substitutions",
        "Scan normalized text for brand hits, score candidate generics, swap into `match_basis`, and populate `probable_brands`, `did_brand_swap`, and `fda_dose_corroborated`; recompute `match_basis_norm_basic` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Re-parse Dose / Route / Form on `match_basis`",
        "Re-run dose parsing and route/form detection against the swapped text to produce `dosage_parsed`, `route`, `form`, and refreshed evidence fields (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Detect Molecules Across References",
        "Execute PNF automaton scans (`pnf_hits_gids`, `pnf_hits_tokens`), invoke partial fallbacks when needed, and detect WHO molecules with ATC/DDD metadata (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Classify Combinations and Extract Unknowns",
        "Use known-generic heuristics to set `looks_combo_final`, record `combo_reason`, gather `unknown_words_list`, and derive presence flags for PNF, WHO, and FDA generics (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).",
    ),
    (
        "Score Candidates and Select Best PNF Variant",
        "Join eSOA rows to matching PNF variants, enforce route compatibility, compute preliminary scores, and select the best variant per `esoa_idx` along with dose/form metadata (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).",
    ),
    (
        "Refine Dose, Form, and Route Alignment",
        "Recalculate `dose_sim`, infer missing form/route when safe, upgrade selections when ratio logic prefers liquid formulations, and track route/form validations plus `route_form_imputations` (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).",
    ),
    (
        "Aggregate Scoring Attributes",
        "Compute confidence components, aggregate recognized molecule lists, set `probable_atc`, and expose `form_ok`/`route_ok` compatibility flags (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).",
    ),
    (
        "Bucketize and Annotate Outcomes",
        "Populate `bucket_final`, `match_molecule(s)`, `match_quality`, `why_final`, and `reason_final`, ensuring Auto-Accept logic and review annotations remain consistent (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).",
    ),
    (
        "Write Outputs and Summaries",
        "Persist the curated data set to CSV/XLSX, generate distribution summaries, and freeze workbook panes for review convenience (see [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py)).",
    ),
    (
        "Post-processing (Optional)",
        "Run [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py) to analyse the generated `unknown_words.csv` report and produce follow-up clues, then accumulate timing information before exiting (see [run.py](https://github.com/carlosresu/esoa/blob/main/run.py)).",
    ),
]

DATA_HEADER = dedent(
    """\
    # Data Dictionary: `outputs/esoa_matched.csv`

    Each record in `esoa_matched.csv` represents one normalized eSOA free-text row, enriched with features, reference lookups, and classification signals produced by [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py), [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py), and [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py).
    """
)

def _section(title: str, rows: list[tuple[str, str, str, str]]) -> str:
    """Assemble a Markdown section with a heading and tabular column descriptions."""
    # Seed the Markdown rows with the section header and table scaffold.
    out: list[str] = [f"## {title}", "", "| Column | Meaning | First Assigned | Notes |", "| --- | --- | --- | --- |"]
    for column, meaning, first_assigned, notes in rows:
        # Append each column's metadata as a Markdown table row.
        out.append(f"| {column} | {meaning} | {first_assigned} | {notes} |")
    # Insert a trailing blank line to keep sections visually separated.
    out.append("")
    return "\n".join(out)

DATA_SECTIONS: list[tuple[str, list[tuple[str, str, str, str]]]] = [
    (
        "Text & Normalization",
        [
            (
                "`esoa_idx`",
                "Stable index of the eSOA row inside the processed frame.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Useful when cross-referencing intermediate artifacts.",
            ),
            (
                "`raw_text`",
                "Original eSOA string as provided in the prepared eSOA CSV (`esoa_prepared.csv`).",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Canonical source for downstream parsing.",
            ),
            (
                "`parentheticals`",
                "List of phrases extracted from parentheses in `raw_text`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Stored as Python list; serialized to string in CSV.",
            ),
            (
                "`normalized`",
                "Lower-cased, punctuation-normalized version of `raw_text`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Serves as the baseline for dose and route parsing.",
            ),
            (
                "`norm_compact`",
                "`normalized` without whitespace/hyphens to aid automata scans.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Supports brand and molecule detection.",
            ),
            (
                "`match_basis`",
                "Working text after optional brand→generic swaps.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Drives all reference matching when a swap occurs.",
            ),
            (
                "`match_basis_norm_basic`",
                "Simplified normalization of `match_basis`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Input to WHO regex detection and unknown-token logic.",
            ),
        ],
    ),
    (
        "Brand Intelligence",
        [
            (
                "`probable_brands`",
                "Pipe-delimited display names of FDA brands detected in the row.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Blank when no brand triggered; helps auditors trace swaps.",
            ),
            (
                "`did_brand_swap`",
                "Indicates whether any FDA brand token was replaced with its generic.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "`True` even if the resulting text already contained the generic.",
            ),
            (
                "`fda_dose_corroborated`",
                "`True` when FDA metadata confirms the detected dose.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Requires both a brand swap and matching FDA dose string.",
            ),
        ],
    ),
    (
        "Dose, Route, and Form Parsing",
        [
            (
                "`dosage_parsed_raw`",
                "Dose dictionary parsed from `normalized` text prior to swaps.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Produced by [scripts/dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py); keys include `kind`, `strength`, `unit`, etc.",
            ),
            (
                "`dosage_parsed`",
                "Dose dictionary parsed after brand swaps using `match_basis`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Supersedes the raw dose when brands change the text.",
            ),
            (
                "`dose_recognized`",
                "Friendly dose string when the final dose matches the selected PNF variant exactly.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py); refined [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Shows `\"N/A\"` whenever `dose_sim` < 1.0.",
            ),
            (
                "`route_raw`",
                "Route detected from `normalized` text before swaps.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Comes with evidence in `route_evidence_raw`.",
            ),
            (
                "`form_raw`",
                "Form detected from `normalized` text before swaps.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "May be blank when no textual cue exists.",
            ),
            (
                "`route_evidence_raw`",
                "Semicolon-separated evidence trail for `route_raw`/`form_raw`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Captures `form:` and `route:` markers discovered before swapping.",
            ),
            (
                "`route`",
                "Final route after re-parsing `match_basis` and optional PNF inference.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py); updated [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "May be filled from PNF when text is silent and inference is safe.",
            ),
            (
                "`route_source`",
                "Origin of the final route (`text`, `pnf`, or blank).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Blank indicates no confident route.",
            ),
            (
                "`route_text`",
                "Route read directly from `match_basis` before inference.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Supports audits of inferred values.",
            ),
            (
                "`form`",
                "Final dosage form after re-parsing and safe PNF inference.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py); updated [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Only overwritten when compatible with ratio/solid safeguards.",
            ),
            (
                "`form_source`",
                "Origin of the final form (`text`, `pnf`, or blank).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Mirrors route handling.",
            ),
            (
                "`form_text`",
                "Form read directly from `match_basis` before inference.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Useful when checking PNF-derived imputations.",
            ),
            (
                "`form_ok`",
                "`True` when the detected or inferred form is compatible with PNF rules.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Evaluated against `APPROVED_ROUTE_FORMS` and exception flags.",
            ),
            (
                "`route_ok`",
                "`True` when the route aligns with the PNF variant’s allowed routes.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "False when route evidence conflicts with PNF allowances.",
            ),
            (
                "`route_form_imputations`",
                "Notes about accepted substitutions or flagged invalid route/form combos.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Empty when no alerts are needed.",
            ),
            (
                "`route_evidence`",
                "Final evidence string combining text cues and PNF imputations.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py); appended [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Adds `pnf:` markers when the route was inferred from the selected variant.",
            ),
        ],
    ),
    (
        "Molecule Detection & Coverage",
        [
            (
                "`molecules_recognized`",
                "**Pipe-delimited list of confirmed generic molecule strings used for ATC assignment and downstream matching.**",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "This is the canonical resolved-generic string reviewers rely on.",
            ),
            (
                "`molecules_recognized_list`",
                "List form of the recognized molecules.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Duplicates removed; serialized as a Python list in CSV.",
            ),
            (
                "`molecules_recognized_count`",
                "Count of entries in `molecules_recognized`.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Integer value.",
            ),
            (
                "`present_in_pnf`",
                "`True` when `match_basis` hit at least one non-salt PNF entry.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Set after salt filtering and partial fallback.",
            ),
            (
                "`present_in_who`",
                "`True` when WHO detection produced any ATC code.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Derived from `who_atc_codes`.",
            ),
            (
                "`present_in_fda_generic`",
                "`True` when FDA generic tokens appear in `match_basis`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Highlights text already aligned with FDA generics.",
            ),
            (
                "`probable_atc`",
                "WHO ATC suggestion when no PNF match exists but WHO coverage does.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Empty string otherwise.",
            ),
        ],
    ),
    (
        "PNF Hit Diagnostics",
        [
            (
                "`generic_id`",
                "Primary PNF generic identifier selected from matches.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "May be filled via partial index fallback.",
            ),
            (
                "`molecule_token`",
                "Span from `match_basis` that yielded `generic_id`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Helpful for reviewer context.",
            ),
            (
                "`pnf_hits_gids`",
                "List of all PNF generic IDs matched before primary selection.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Includes candidates filtered out later.",
            ),
            (
                "`pnf_hits_count`",
                "Number of qualifying PNF matches (post salt-filter, plus partial fallback).",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Integer.",
            ),
            (
                "`pnf_hits_tokens`",
                "Raw PNF tokens that matched the text.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Stored as list.",
            ),
        ],
    ),
    (
        "WHO ATC Details",
        [
            (
                "`who_molecules_list`",
                "List of WHO molecule names detected in `match_basis`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Deduplicated; serialized as list.",
            ),
            (
                "`who_molecules`",
                "Pipe-delimited WHO molecule names.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Empty string when no WHO hits.",
            ),
            (
                "`who_atc_codes_list`",
                "List of WHO ATC codes associated with detected molecules.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Sorted for stability.",
            ),
            (
                "`who_atc_codes`",
                "Pipe-delimited WHO ATC codes.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Empty string when none found.",
            ),
            (
                "`who_atc_count`",
                "Count of WHO ATC codes on the row.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Integer.",
            ),
            (
                "`who_atc_has_ddd`",
                "`True` if any matched WHO code carries a defined daily dose.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Boolean flag.",
            ),
            (
                "`who_atc_adm_r`",
                "Pipe-delimited WHO administration-route annotations from DDD metadata.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Lowercase route abbreviations.",
            ),
            (
                "`who_route_tokens`",
                "List of canonical route tokens inferred from WHO Adm.R metadata.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Used when PNF lacks route allowances.",
            ),
            (
                "`who_form_tokens`",
                "List of canonical form tokens inferred from WHO Adm.R/UOM metadata.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Supports WHO-only route/form reconciliation.",
            ),
        ],
    ),
    (
        "Selected PNF Variant & Dose Alignment",
        [
            (
                "`selected_form`",
                "Form token of the chosen PNF variant.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Matches `form_token` from PNF data.",
            ),
            (
                "`selected_route_allowed`",
                "Pipe-delimited routes permitted by the selected PNF row.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Mirrors `route_allowed` in PNF.",
            ),
           (
               "`selected_variant`",
               "Human-readable summary of the chosen PNF dose (kind/strength/ratio).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
               "Example: `amount:500mg` or `ratio:5mg/5mL`.",
           ),
            (
                "`selected_dose_kind`",
                "PNF dose kind (`amount`, `ratio`, `percent`).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Kept even when dose text was missing.",
            ),
            (
                "`selected_strength`",
                "Strength value from PNF.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "May be `None` when not applicable.",
            ),
            (
                "`selected_unit`",
                "Unit corresponding to `selected_strength`.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Lowercase units.",
            ),
            (
                "`selected_strength_mg`",
                "Strength converted to milligrams for comparisons.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Float or `None`.",
            ),
            (
                "`selected_per_val`",
                "Denominator value for ratio/per-unit doses.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Typically integer.",
            ),
            (
                "`selected_per_unit`",
                "Denominator unit label (e.g., `mL`, `tablet`).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Lowercase text.",
            ),
            (
                "`selected_ratio_mg_per_ml`",
                "Precomputed mg/mL value from PNF.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Float or `None`.",
            ),
            (
                "`selected_pct`",
                "Percent strength from PNF when relevant.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Float or `None`.",
            ),
            (
                "`dose_sim`",
                "Final dose similarity (0.0 or 1.0) after recomputation with the selected PNF variant.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Enforces exact equality (after unit conversion) by design.",
            ),
        ],
    ),
    (
        "Combination & Unknown Analysis",
        [
            (
                "`looks_combo_final`",
                "`True` when at least two known generics are detected (combination product).",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Built from PNF/WHO/FDA known tokens.",
            ),
            (
                "`combo_reason`",
                "Textual explanation of the combo heuristic outcome.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "`combo/known-generics>=2` or `single/heuristic`.",
            ),
            (
                "`combo_known_generics_count`",
                "Number of unique known generic tokens that triggered combo detection.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Integer.",
            ),
            (
                "`unknown_kind`",
                "Qualifies unresolved tokens (`None`, `Single - Unknown`, etc.).",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Guides reviewer triage.",
            ),
            (
                "`unknown_words_list`",
                "List of normalized tokens not recognized in PNF/WHO/FDA vocabularies.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Used by [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py); serialized to string.",
            ),
            (
                "`unknown_words`",
                "Pipe-delimited string version of `unknown_words_list`.",
                "[scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)",
                "Empty when every token is known.",
            ),
        ],
    ),
    (
        "Confidence & Final Classification",
        [
            (
                "`atc_code_final`",
                "ATC code inherited from the selected PNF row (if any).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "`None` when the PNF row lacks an ATC.",
            ),
            (
                "`confidence`",
                "Composite 0–100 score covering generic, dose, route, ATC, and brand-swap bonus.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Higher values suggest stronger auto-match evidence.",
            ),
            (
                "`match_molecule(s)`",
                "Source labels describing which reference validated the molecule (PNF/WHO/FDA/brand).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Drives reporting pivots.",
            ),
            (
                "`match_quality`",
                "Summary for review (`dose mismatch`, `form mismatch`, etc.).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Defaults to `unspecified` when no issue noted.",
            ),
            (
                "`bucket_final`",
                "Final workflow bucket (`Auto-Accept`, `Needs review`, `Others`).",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Determines downstream handling.",
            ),
            (
                "`why_final`",
                "High-level justification aligned with `bucket_final`.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Often `Needs review` or `Unknown`.",
            ),
            (
                "`reason_final`",
                "Expanded rationale for review or other buckets.",
                "[scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)",
                "Includes unknown-token annotations when applicable.",
            ),
        ],
    ),
]


def render_pipeline() -> str:
    """Compose the pipeline walkthrough Markdown from the static step listing."""
    # Begin with the static introduction text.
    parts = [PIPELINE_HEADER.strip(), ""]
    for idx, (title, body) in enumerate(PIPELINE_STEPS, start=1):
        # Emit a numbered heading for the step.
        parts.append(f"{idx}. **{title}**  ")
        # Pair the heading with its descriptive paragraph.
        parts.append(f"   {body}")
        # Maintain spacing between bullet paragraphs for readability.
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def render_data_dictionary() -> str:
    """Build the data dictionary Markdown capturing every documented column."""
    # Prime the output with the top-level heading.
    sections = [DATA_HEADER.strip(), ""]
    for section_title, rows in DATA_SECTIONS:
        # Generate a table for each thematic grouping.
        sections.append(_section(section_title, rows))
    return "\n".join(sections).strip() + "\n"


def update_file(path: Path, content: str) -> None:
    """Write Markdown when contents change, avoiding churn in version control."""
    # Read existing file contents (if present) to avoid unnecessary writes.
    current = path.read_text() if path.exists() else ""
    if current != content:
        # Persist the freshly rendered Markdown when a diff is detected.
        path.write_text(content)


def main() -> None:
    """Regenerate the pipeline and data dictionary docs in-place."""
    # Update pipeline walkthrough documentation.
    update_file(PIPELINE_MD, render_pipeline())
    # Refresh the data dictionary to mirror the latest schema.
    update_file(DATA_DICTIONARY_MD, render_data_dictionary())
    # Provide a simple confirmation for CLI users.
    print("Documentation synchronised: pipeline.md, data_dictionary.md")


if __name__ == "__main__":
    main()
