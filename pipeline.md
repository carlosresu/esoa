# Pipeline Execution Walkthrough

Detailed end-to-end view of the matching pipeline, from CLI invocation in [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) through feature building in [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py), scoring in [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py), and export logic in [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py).

1. **Load Prepared Inputs**  
   Resolve CLI paths (defaults under `inputs/`), verify the files exist, and read the prepared PNF and eSOA CSVs (`pnf_prepared.csv`, `esoa_prepared.csv`) into pandas data frames (see [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) and [scripts/match.py](https://github.com/carlosresu/esoa/blob/main/scripts/match.py)).

2. **Validate Structural Expectations**  
   Ensure the PNF frame exposes the required molecule, dose, route, and ATC fields and that the eSOA frame includes `raw_text` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

3. **Index Reference Vocabularies**  
   Build normalized PNF name lookups, load WHO ATC exports (including regex caches and DDD metadata), and prepare the FDA brand map automata plus generic token set (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

4. **Construct PNF Search Automata**  
   Create normalized/compact Aho–Corasick automatons and train the partial token index used for fallback matches (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

5. **Normalize eSOA Text**  
   Produce the working frame with `raw_text`, parenthetical captures, `esoa_idx`, and normalized text variants (`normalized`, `norm_compact`) (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

6. **Initial Dose / Route / Form Parsing**  
   Parse dosage structures via [scripts/dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py), compute `dose_recognized`, and collect `route_raw`, `form_raw`, and `route_evidence_raw` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

7. **Apply FDA Brand → Generic Substitutions**  
   Scan normalized text for brand hits, score candidate generics, swap into `match_basis`, and populate `probable_brands`, `did_brand_swap`, and `fda_dose_corroborated`; recompute `match_basis_norm_basic` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

8. **Re-parse Dose / Route / Form on `match_basis`**  
   Re-run dose parsing and route/form detection against the swapped text to produce `dosage_parsed`, `route`, `form`, and refreshed evidence fields (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

9. **Detect Molecules Across References**  
   Execute PNF automaton scans (`pnf_hits_gids`, `pnf_hits_tokens`), invoke partial fallbacks when needed, and detect WHO molecules with ATC/DDD metadata (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

10. **Classify Combinations and Extract Unknowns**  
   Use known-generic heuristics to set `looks_combo_final`, record `combo_reason`, gather `unknown_words_list`, and derive presence flags for PNF, WHO, and FDA generics (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

11. **Score Candidates and Select Best PNF Variant**  
   Join eSOA rows to matching PNF variants, enforce route compatibility, compute preliminary scores, and select the best variant per `esoa_idx` along with dose/form metadata (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

12. **Refine Dose, Form, and Route Alignment**  
   Recalculate `dose_sim`, infer missing form/route when safe, upgrade selections when ratio logic prefers liquid formulations, and track route/form validations plus `route_form_imputations` (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

13. **Aggregate Scoring Attributes**  
   Compute confidence components, aggregate recognized molecule lists, set `probable_atc`, and expose `form_ok`/`route_ok` compatibility flags (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

14. **Bucketize and Annotate Outcomes**  
   Populate `bucket_final`, `match_molecule(s)`, `match_quality`, `why_final`, and `reason_final`, ensuring Auto-Accept logic and review annotations remain consistent (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

15. **Write Outputs and Summaries**  
   Persist the curated data set to CSV/XLSX, generate distribution summaries, and freeze workbook panes for review convenience (see [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py)).

16. **Post-processing (Optional)**  
   Run [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py) to analyse the generated `unknown_words.csv` report and produce follow-up clues, then accumulate timing information before exiting (see [run.py](https://github.com/carlosresu/esoa/blob/main/run.py)).
