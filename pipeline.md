# Pipeline Execution Walkthrough

Detailed end-to-end view of [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) → [scripts/match.py](https://github.com/carlosresu/esoa/blob/main/scripts/match.py) execution for generating `outputs/esoa_matched.*`.

1. **Load prepared inputs**
   - Resolve CLI paths (defaults under `inputs/`) and ensure files exist ([run.py](https://github.com/carlosresu/esoa/blob/main/run.py):358-371).
   - Read `pnf_prepared.csv` and `esoa_prepared.csv` into pandas DataFrames ([scripts/match.py](https://github.com/carlosresu/esoa/blob/main/scripts/match.py):59-60).

2. **Validate structural expectations**
   - Check PNF frame contains core molecule, dose, route, and ATC fields ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):88-98).
   - Confirm the eSOA frame exposes a `raw_text` column; raise errors early if missing.

3. **Index reference vocabularies**
   - Build normalized PNF name → (`generic_id`, `generic_name`) lookup ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):101-109).
   - Load WHO ATC exports, cache name/code regexes, and capture DDD metadata ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):111-128).
   - Load the most recent FDA brand map, then construct automata for normalized and compact tokens plus brand→generic lookups and FDA generic token set ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):130-148).

4. **Construct PNF search automata**
   - Build normalized and compact Aho–Corasick automatons from PNF molecules ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):150-157).
   - Train the partial token index for fallback matching when full-string hits are absent ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):155-157).

5. **Normalize eSOA text**
   - Create the working DataFrame seeded with `raw_text` and derived `esoa_idx` ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):160-166).
   - Extract parenthetical phrases for later brand heuristics and store normalized / compact (no spaces/hyphen) variants ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):162-167).

6. **Initial dose / route / form parsing**
   - Parse dosage structures from `normalized` text using [scripts/dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py); store the raw dictionary in `dosage_parsed_raw` and render `dose_recognized` strings ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):172-175).
   - Detect route/form tokens (`route_raw`, `form_raw`) and capture supporting evidence in `route_evidence_raw` before any substitutions ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):176-177).

7. **Apply FDA brand → generic substitutions**
   - For each row, scan normalized text with brand automatons to identify candidate brands ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):196-205).
   - Score competing generics using dose/form corroboration and PNF-name presence to pick the best replacement ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):206-227).
   - Replace brand mentions with the chosen normalized generic, scrub duplicates when the text already contains the generic, and populate `probable_brands`, `did_brand_swap`, and `fda_dose_corroborated` flags ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):228-270).
   - Recompute `match_basis_norm_basic` for downstream detectors ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):271-272).

8. **Re-parse dose / route / form on `match_basis`**
   - Run dose extraction, route/form detection, and evidence capture on the substituted text ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):274-279).

9. **Detect molecules across references**
   - Execute Aho–Corasick scans to gather PNF hits, filtering out salt-only matches, and assign `generic_id`/`molecule_token`; retain the full `pnf_hits_gids` / `pnf_hits_tokens` payload for auditing ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):281-299).
   - Use the partial token index to recover best-effort matches when full hits are absent ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):302-319).
   - Apply WHO regex detection to produce molecule lists, ATC codes, DDD flags, and administration-route tags ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):321-365).

10. **Classify combinations and unknown tokens**
    - Count known generic tokens (PNF/WHO/FDA) to label likely combination products ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):368-385).
    - Extract residual tokens not covered by the vocabularies to populate `unknown_kind`, `unknown_words_list`, and `unknown_words` ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):387-449).
    - Derive presence flags (`present_in_pnf`, `present_in_who`, `present_in_fda_generic`) ([scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py):452-457).

11. **Score candidates and pick best PNF variant**
    - Join eSOA rows to all matching PNF variants and filter by route compatibility ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):162-189).
    - Compute preliminary dose similarity and composite scores favouring exact form/route agreement ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):190-223).
    - Select the top-scoring PNF row per eSOA index, capturing dose/form metadata and ATC codes ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):224-287).
    - Merge selections back into the working DataFrame, ensuring booleans fall back to `False` ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):289-317).

12. **Refine dose, form, and route alignment**
    - Recompute `dose_sim` against the chosen PNF payload to honour exact matching policies ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):348-368).
    - Track original text form/route, determine if PNF inference is safe, and update fields plus `route_evidence` entries when imputed ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):370-415).
    - Iterate through PNF candidates again to upgrade selections when dose similarity improves (especially for ratio doses) ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):419-494).
    - Validate route/form combinations against the curated whitelist, logging acceptances, flags, or violations in `route_form_imputations` ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):500-566).

13. **Aggregate scoring attributes**
    - Compute presence booleans for dose, route, form, and route evidence; expose `form_ok`/`route_ok` compatibility flags and clip `dose_sim` to [0,1] ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):568-575).
    - Generate the 0–100 `confidence` score with brand-swap bonus handling ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):576-590).
    - Collate recognized molecule unions — including the canonical `molecules_recognized` pipe-delimited generics string alongside `molecules_recognized_list` — WHO ATC counts, and probable ATC fallbacks ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):592-616).

14. **Bucketize and annotate outcomes**
    - Initialize `bucket_final`, `match_molecule(s)`, and `match_quality` then populate source-based labels (PNF/WHO/FDA) ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):617-652).
    - Mark `Auto-Accept` rows (PNF + ATC + aligned form/route) and default others to `Needs review` pending further analysis ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):654-660).
    - Assign detailed `match_quality` messages for dose, form, route, and missing-context scenarios ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):662-708).
    - Enrich `reason_final` / `why_final` based on unknown-token categories and unresolved states ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):709-757).
    - Normalize `dose_recognized` to `N/A` when the final `dose_sim` is not perfect ([scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py):759-760).

15. **Write outputs and summaries**
    - Trim/arrange columns per `OUTPUT_COLUMNS` and emit UTF-8 CSV ([scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py):38-76).
    - Write Excel workbook with frozen header row and autofilter ([scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py):78-90).
    - Produce distribution summaries (`summary*.txt`) covering bucket, molecule-source, and match-quality pivots ([scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py):92-118).

16. **Post-processing (optional)**
    - Invoke [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py) to analyze `unknown_words.csv` and suggest possible reference matches ([run.py](https://github.com/carlosresu/esoa/blob/main/run.py):112-122).
    - Consolidate output paths and timing summaries before exiting ([run.py](https://github.com/carlosresu/esoa/blob/main/run.py):124-160).
