# Pipeline Execution Walkthrough

Detailed end-to-end view of the matching pipeline, from CLI invocation in [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) through feature building in [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py), scoring in [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py), and export logic in [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py).

🆕 **Inline documentation refresh** – the Python modules referenced below now
include descriptive docstrings and comments that mirror this walkthrough.  When
deep-diving into a particular step, the code comments explain the exact
transformations performed and why policy constants are set the way they are.

1. **Load Prepared Inputs**  
   Resolve CLI paths (defaults under `inputs/`), verify the files exist, and read the prepared PNF and eSOA CSVs (`pnf_prepared.csv`, `esoa_prepared.csv`) into pandas data frames (see [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) and [scripts/match.py](https://github.com/carlosresu/esoa/blob/main/scripts/match.py)).

2. **Validate Structural Expectations**  
   Ensure the PNF frame exposes the required molecule, dose, route, and ATC fields and that the eSOA frame includes `raw_text` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

3. **Index Reference Vocabularies**  
   Build normalized PNF name lookups, load WHO ATC exports (including regex caches, DDD metadata, and WHO route/form mappings), and prepare the FDA brand map automata plus generic token set (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

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
   Execute PNF automaton scans (`pnf_hits_gids`, `pnf_hits_tokens`) using expanded alias sets (parenthetical trade names, slash/plus splits, curated abbreviations), fall back to partial and fuzzy matches for near-miss spellings, and detect WHO molecules with ATC/DDD metadata (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

10. **Detect FDA Food / Non-therapeutic Catalog Matches**
    Optionally load `fda_food_products.csv`, capture every matching entry in `non_therapeutic_hits`, summarize the highest scoring registration in `non_therapeutic_detail`, and retain canonical tokens in `non_therapeutic_tokens` for downstream unknown filtering and review context (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)). When no therapeutic molecule is recognized, scoring promotes these rows to the Others bucket with `reason_final = "non_therapeutic_detected"` (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

11. **Classify Combinations and Extract Unknowns**
   Use known-generic heuristics to set `looks_combo_final`, record `combo_reason`, gather `unknown_words_list`, and derive presence flags for PNF, WHO, and FDA generics (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

12. **Score Candidates and Select Best PNF Variant**
   Join eSOA rows to matching PNF variants, enforce route compatibility, compute preliminary scores, and select the best variant per `esoa_idx` along with dose/form metadata (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

13. **Refine Dose, Form, and Route Alignment**
   Recalculate `dose_sim`, infer missing form/route when safe, use WHO route/form tokens when PNF data is absent, upgrade selections when ratio logic prefers liquid formulations, and track route/form validations plus `route_form_imputations` (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

14. **Aggregate Scoring Attributes**
   Compute confidence components, aggregate recognized molecule lists, set `probable_atc`, expose `form_ok`/`route_ok`, and derive `generic_final` (PNF `generic_id` when available, otherwise WHO molecules or FDA generics) as the canonical molecule field (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

15. **Bucketize and Annotate Outcomes**
    Populate `bucket_final`, `match_molecule(s)`, `match_quality`, `why_final`, and `reason_final`, ensuring Auto-Accept logic and review annotations remain consistent (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

16. **Write Outputs and Summaries**
    Persist the curated data set to CSV/XLSX, generate distribution summaries, and freeze workbook panes for review convenience (see [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py)).

17. **Post-processing (Optional)**
    Run [resolve_unknowns.py](https://github.com/carlosresu/esoa/blob/main/resolve_unknowns.py) to analyse the generated `unknown_words.csv` report and produce follow-up clues, then accumulate timing information before exiting (see [run.py](https://github.com/carlosresu/esoa/blob/main/run.py)).

## Auto-Accept Reference (PNF-backed exacts)

Rows land in the **Auto-Accept** bucket when scoring finds a PNF variant with an
ATC code (`present_in_pnf` + `atc_code_final`) and both `form_ok` and `route_ok`
evaluate to `True`. These cases keep `match_quality`/`reason_final` blank because
no review flags were raised. The summary generated by
`scripts/match_outputs.py` also breaks out the high-confidence exacts:

- `ValidGenericInPNF, exact dose/route/form match` – Auto-accepts whose
  canonical text already used the PNF generic and the parsed dose/form/route
  matched perfectly (`dose_sim == 1.0`).
- `ValidBrandSwappedForGenericInPNF: exact dose/form/route match` – Brand swap
  identified the PNF generic, the swap aligned on all clinical metadata, and the
  row stayed Auto-Accept. `did_brand_swap = True` pinpoints these rows.

Rows that auto-accept but are not listed above typically needed minor
inference—e.g., the route was imputed from the PNF variant or the input lacked
dose text (`dose_recognized = "N/A"`). `route_form_imputations` captures any
whitelisted substitutions applied during this step.

## Needs Review Reference (match_molecule × match_quality)

When `bucket_final` stays `Needs review`, downstream summaries pair `match_molecule(s)` with `match_quality`. The combinations below describe how each scenario moved through the pipeline and why manual review is still required.

### ValidBrandSwappedForMoleculeWithATCinWHO

- `who_metadata_insufficient_review_required` – Brand normalization selected a WHO molecule (no PNF coverage) and all hard checks passed, yet WHO lacked enough corroborating metadata to auto-accept. Confirm route, form, and dose manually.
- `route_mismatch` – The input route text conflicts with the WHO Adm.R-derived route tokens (or our allowed per-route mapping). Validate the stated route or adjust the mapping.
- `who_does_not_provide_dose_info` – Dose text exists but the WHO ATC export has no DDD for this molecule, so the comparison stops short. Verify the clinical dose manually.
- `no_dose_available` – No usable dose was parsed from the eSOA text and WHO could not supply one. Add or confirm the dose evidence.
- `no_form_and_route_available` – Neither form nor route surfaced in the text, leaving WHO hints as the only guide. Ensure the administration details match expectations.
- `route_form_mismatch` – The route/form pairing gleaned from text (or WHO) is outside our `APPROVED_ROUTE_FORMS` whitelist. Review whether an exception is acceptable or the data needs correction.
- `no_form_available` – A WHO-only molecule with no form parsed from the record; confirm the dosage form before approval.
- `no_dose_and_form_available` – Both dose and form are missing in the textual evidence. Supply the missing context or reject.
- `who_does_not_provide_route_info` – WHO provided the molecule but no Adm.R tokens, so we cannot validate the route stated (or missing) in text. Check clinical route manually.

### ValidMoleculeWithATCinPNF

- `no_dose_available` – The text mapped cleanly to a PNF generic with ATC coverage but contained no dose evidence. Review the record for missing strength information.
- `no_dose_and_form_available` – Dose and form are absent in the text, so we cannot verify alignment with the PNF variant. Confirm both before finalizing.
- `no_form_available` – Route and dose may be present, yet the dosage form was not parsed. Ensure the form matches the PNF listing.
- `dose_mismatch` – Parsed dosing disagrees with the selected PNF strength after normalization. Correct the dose wording or pick a better variant.
- `no_form_and_route_available` – Both form and route were missing, preventing validation against PNF allowances. Provide the missing administration context.
- `route_form_mismatch` – The route/form combination from text violates our allowed pairings even though the PNF molecule matched. Decide whether to adjust the whitelist or fix the source data.
- `no_dose_and_route_available` – Dose and route are absent, so the PNF hit lacks key metadata cross-checks. Supply the route and dosing information.
- `no_dose_form_and_route_available` – None of the three attributes were captured; the molecule alone is insufficient for approval. Complete the clinical details.
- `no_route_available` – The route could not be parsed or inferred, leaving the PNF route list unchecked. Confirm administration route.

### ValidBrandSwappedForGenericInPNF

- `no_dose_available` – FDA brand mapping surfaced a PNF generic, but the record lacked dose evidence. Validate the strength prior to acceptance.
- `route_form_mismatch` – Textual route/form disagrees with the PNF whitelist even after the brand swap. Investigate whether the source data or whitelist needs correction.
- `no_form_and_route_available` – Neither form nor route was available to compare against the PNF allowances. Add the missing administration details.
- `no_dose_and_form_available` – Dose and form were absent; reviewers must confirm both attributes manually.
- `dose_mismatch` – Parsed dose conflicts with the PNF variant chosen after the brand swap. Adjust the dose text or reconcile with an alternate variant.
- `no_form_available` – Form could not be parsed, so the substitution lacks confirmation of dosage form. Provide or verify the form.
- `no_dose_and_route_available` – Dose and route were both missing; ensure the record documents them before approving.
- `no_route_available` – Route was absent in the text; confirm it against the PNF option.
- `no_dose_form_and_route_available` – Dose, form, and route are all missing, leaving only the molecule identification; complete the clinical details.

### ValidMoleculeWithATCinWHO/NotInPNF

- `who_metadata_insufficient_review_required` – WHO supplied the molecule and ATC, yet no additional signals were present. Confirm full clinical metadata before use.
- `who_does_not_provide_dose_info` – WHO lacks a DDD entry, so dose verification cannot proceed. Ensure the prescribed dose is sound.
- `no_form_and_route_available` – The eSOA text omitted both form and route; WHO hints alone are insufficient. Add the missing administration info.
- `who_does_not_provide_route_info` – WHO did not provide Adm.R tokens, preventing route validation. Confirm the route manually.
- `no_form_available` – Form is missing from the text and WHO data; capture the dosage form before approval.
- `route_form_mismatch` – WHO-derived or textual route/form pairing fell outside our approved combinations. Determine whether the pairing is clinically acceptable.
- `route_mismatch` – The stated route conflicts with WHO’s Adm.R mapping. Validate the correct administration route.

### ValidMoleculeNoATCinFDA/NotInPNF

- `route_mismatch` – Only the FDA generic matched and the inferred route conflicts with our whitelist. Confirm the correct route or adjust mappings.
- `no_dose_available` – Neither the text nor references provided a checkable dose. Supply the strength information.
- `no_dose_form_and_route_available` – All three attributes were absent, so the FDA-only identification lacks supporting metadata. Complete the clinical details.
- `no_form_and_route_available` – Form and route are missing; the FDA-only hit needs additional info.
- `no_form_available` – Dosage form is absent. Add or validate the missing form.
- `route_form_mismatch` – The available route/form combination is not allowed in our whitelist. Confirm whether the pairing should be permitted.
- `no_dose_and_form_available` – Dose and form are missing while only the FDA source supplied the molecule. Provide both before approving.

### UnspecifiedSource

- `auto_exact_dose_route_form` – Auto-Accept rows where the text matched the PNF dose/route/form exactly.
- `dose_mismatch_same_atc` – Auto-Accept rows with non-exact doses, but the PNF variant’s ATC covers all dose presentations.
- `dose_mismatch_varied_atc` – Auto-Accept rows with non-exact doses when multiple ATC payloads exist; flagged for reconciliation.
- Approved route/form substitutions (e.g., sachet counted as tablet) are folded into these tags and no longer tracked separately.
- `nontherapeutic_and_unknown_tokens` / `nontherapeutic_catalog_match` – FDA food/non-therapeutic catalog hits, optionally with residual unknown tokens.
- `unknown_tokens_present` – Partial unknown tokens remain even after PNF/WHO/FDA lookups; see `match_molecule(s)` suffix (e.g., `PartiallyKnownTokensFrom_PNF_WHO`) for which datasets covered the known portion.
- `AllTokensUnknownTo_PNF_WHO_FDA` – No catalog matched any token; routed to Others with `match_quality = N/A` and `detail_final` showing the exact count of unknown tokens remaining.
- `manual_review_required` – No structured signals materialised; requires human triage.
- `route_mismatch` – The text points to an unverified molecule and also carries a route conflict. Resolve the molecule identification and the route evidence.
- `no_dose_available` – No reference match plus a missing dose. Provide dose context or determine if the entry should be excluded.
- `route_form_mismatch` – The captured route/form pairing is outside our whitelist for an unverified molecule. Clarify both the molecule and administration details.
- `no_dose_and_form_available` – Both dose and form are absent while the molecule remains unverified. Supplement the missing fields or reject.
- `no_form_available` – Form is missing and no reference confirmed the molecule. Capture the dosage form before moving forward.

## Others Bucket Reference (why_final × reason_final)

The **Others** bucket collects rows that are explicitly non-therapeutic or that
remain unknown even after FDA food catalog checks. Reviewers can pivot on
`why_final`/`reason_final` to decide next steps:

### Non-Therapeutic Medical Products (`reason_final = non_therapeutic_detected`)

- Triggered when the FDA food/non-therapeutic automaton matches the text and no
  PNF/WHO/FDA-drug molecule is present. `non_therapeutic_detail` shows the
  highest scoring catalog entry (brand/product/company/registration number),
  while `non_therapeutic_hits` enumerates all candidates considered.
- `non_therapeutic_tokens` captures the canonicalized tokens that caused the
  match so analysts can confirm why the row bypassed the therapeutic pipelines.

### Unknown (`reason_final` annotated with source presence)

- `Single - Unknown (...)` – Only one token remains unknown. The parenthetical
  explains whether any part of the string was recognized by PNF/WHO/FDA before
  falling back here.
- `Multiple - Some Unknown (...)` – Some tokens resolved to known molecules but
  at least one token stayed unknown. The annotation again lists the references
  that did recognize part of the row.
- `Multiple - All Unknown (...)` – Every token stayed unrecognized even after
  accounting for FDA food tokens; investigate whether the entry is a new drug,
  a miscoded item, or an additional non-therapeutic product.
