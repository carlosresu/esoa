# Pipeline Execution Walkthrough

Detailed end-to-end view of the matching pipeline, from CLI invocation in [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) (now a thin orchestrator that resolves the appropriate `ITEM_REF_CODE` pipeline from `pipelines/registry.py`) through feature building in [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py), scoring in [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py), and export logic in [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py).

🆕 **Inline documentation refresh** – the Python modules referenced below now
include descriptive docstrings and comments that mirror this walkthrough.  When
deep-diving into a particular step, the code comments explain the exact
transformations performed and why policy constants are set the way they are.

Before the numbered stages below, the `DrugsAndMedicinePipeline` (invoked via `run.py`) now handles shared orchestration:

- Bootstraps `requirements.txt` with pip when needed, so a separate `--skip-install` flag is no longer required.
- Ensures `inputs/` and `outputs/` exist, then prunes dated WHO ATC exports and brand-map snapshots after the run.
- Concatenates partitioned `esoa_pt_*.csv` files into a temporary `esoa_combined.csv` before invoking the preparation step.
- Optionally runs the WHO ATC R preprocessors (guarded by `--skip-r`) to keep ATC and DDD extracts fresh.
- Builds or reuses the FDA brand map (`--skip-brandmap`), silencing console output while still surfacing failures.
- Wraps each stage in a live spinner, records per-step timings, and prints a grouped summary once matching completes.
- Chooses a safe worker pool size (auto-tuned via `resolve_worker_count`, overridable with `ESOA_MAX_WORKERS`) so CPU-heavy phases can execute concurrently without starving smaller laptops.

1. **Prepare and Load Inputs**  
   Resolve CLI paths (defaults under `inputs/`), concatenate partitioned eSOA files when present, and run the preparation layer: [scripts/prepare_annex_f.py](https://github.com/carlosresu/esoa/blob/main/scripts/prepare_annex_f.py) produces `annex_f_prepared.csv` while [scripts/prepare.py](https://github.com/carlosresu/esoa/blob/main/scripts/prepare.py) emits `pnf_prepared.csv` and `esoa_prepared.csv`. The matching core then reads these normalized CSVs (see [run.py](https://github.com/carlosresu/esoa/blob/main/run.py) and [scripts/match.py](https://github.com/carlosresu/esoa/blob/main/scripts/match.py)).

2. **Validate Structural Expectations**  
   Ensure the unified reference catalogue exposes Annex/PNF molecule, dose, route, priority, and identifier fields, and that the eSOA frame includes `raw_text` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

3. **Index Reference Vocabularies**  
   Build normalized Annex F + PNF name lookups (Annex priority), load WHO ATC exports (including regex caches, DDD metadata, and WHO route/form mappings), and prepare the FDA brand map automata plus generic token set (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

4. **Construct Search Automata**  
   Create normalized/compact Aho–Corasick automatons for the combined reference set and train the partial token index used for fallback matches (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

5. **Normalize eSOA Text**  
   Produce the working frame with `raw_text`, parenthetical captures, `esoa_idx`, and normalized text variants (`normalized`, `norm_compact`) (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

6. **Initial Dose / Route / Form Parsing**  
   Parse dosage structures via [scripts/dose.py](https://github.com/carlosresu/esoa/blob/main/scripts/dose.py), compute `dose_recognized`, and collect `route_raw`, `form_raw`, and `route_evidence_raw` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

7. **Apply FDA Brand → Generic Substitutions**  
   Scan normalized text for brand hits, score candidate generics, swap into `match_basis`, and populate `probable_brands`, `did_brand_swap`, `brand_swap_added_generic`, `fda_generics_list`, and `fda_dose_corroborated`; recompute `match_basis_norm_basic` (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)). Large batches spill across a worker pool so each process builds the Aho–Corasick automata once and applies swaps concurrently.

8. **Re-parse Dose / Route / Form on `match_basis`**  
   Re-run dose parsing and route/form detection against the swapped text to produce `dosage_parsed`, `route`, `form`, and refreshed evidence fields (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)).

9. **Detect Molecules Across References**  
   Execute Annex/PNF automaton scans (`pnf_hits_gids`, `pnf_hits_tokens`) using expanded alias sets (parenthetical trade names, slash/plus splits, curated abbreviations), fall back to partial and fuzzy matches for near-miss spellings, and detect WHO molecules with ATC/DDD metadata. Annex hits are prioritized via `source_priority` before scoring (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)). WHO regex passes now run in parallel when millions of tokens are involved, keeping the wall-clock impact in check.

10. **Overlay DrugBank Generics**  
    Load the DrugBank generics export via [scripts/reference_data.py](https://github.com/carlosresu/esoa/blob/main/scripts/reference_data.py), scan `match_basis` for contiguous token matches, populate `drugbank_generics_list`, and set `present_in_drugbank`. These tokens are also merged into the shared ignore-word pool so valid DrugBank synonyms never appear as unknowns, and [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py) labels rows as `ValidMoleculeInDrugBank` when no other catalogue explains the text.

11. **Detect FDA Food / Non-therapeutic Catalog Matches**  
    Optionally load `fda_food_products.csv`, capture every matching entry in `non_therapeutic_hits`, summarize the highest scoring registration in `non_therapeutic_detail`, record the structured winner in `non_therapeutic_best`, emit `non_therapeutic_summary`, and retain canonical tokens in `non_therapeutic_tokens` for downstream unknown filtering and review context (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)). When no therapeutic molecule is recognized, scoring promotes these rows to the Unknown bucket with `reason_final = "non_therapeutic_detected"` (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

12. **Extract Unknown Tokens and Presence Flags**  
   Gather `unknown_words_list`, classify `unknown_kind`, and derive presence counts for Annex F, PNF, WHO, FDA, DrugBank, and FDA food catalog hits (see [scripts/match_features.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_features.py)). DrugBank tokens contribute to `qty_drugbank` and are stripped from the unknown pool before scoring.

13. **Score Candidates and Select Best PNF Variant**
   Join eSOA rows to matching Annex F/PNF variants, enforce route compatibility, compute preliminary scores, and select the best variant per `esoa_idx` along with dose/form metadata. Annex rows carry `source_priority = 1`, ensuring Drug Codes outrank ATC-only matches (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

14. **Refine Dose, Form, and Route Alignment**
   Recalculate `dose_sim`, infer missing form/route when safe, use WHO route/form tokens when local data is absent, upgrade selections when ratio logic prefers liquid formulations, and track route/form validations plus `route_form_imputations`. Annex route evidence is injected via `reference_route_details` when the raw text lacked explicit clues (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

15. **Aggregate Scoring Attributes**
   Compute confidence components (including the +10 bonus when `brand_swap_added_generic` is true and dose/form/route align), aggregate recognized molecule lists, set `probable_atc`, expose `form_ok`/`route_ok`, and derive `generic_final` plus Annex-specific metadata: `reference_source`, `reference_priority`, `drug_code_final`, `primary_code_final`, and `reference_route_details` (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).

16. **Bucketize and Annotate Outcomes**
    Populate `bucket_final`, `match_molecule(s)`, `match_quality`, `why_final`, and `reason_final`, ensuring Auto-Accept logic and review annotations remain consistent (see [scripts/match_scoring.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_scoring.py)).
    - **Auto-Accept** — Annex F or PNF match with a reference code (`drug_code_final` or `primary_code_final`), `form_ok`/`route_ok`, and no unresolved tokens.
    - **Candidates** — Reference code assigned via Annex, PNF, WHO, or FDA brand swap but Auto-Accept criteria not met; no unknown tokens remain.
    - **Needs review** — Reference code assigned yet one or more tokens stay unresolved; `match_quality` pinpoints the outstanding issues.
    - **Unknown** — No therapeutic reference matched, including FDA food/non-therapeutic detections.

17. **Write Outputs and Summaries**
    Persist the curated data set to CSV/XLSX, generate distribution summaries, and freeze workbook panes for review convenience. Outputs now include Annex metadata columns so reviewers can pivot on Drug Code and source priority (see [scripts/match_outputs.py](https://github.com/carlosresu/esoa/blob/main/scripts/match_outputs.py)).

18. **Post-processing & Timing Summary**
    Automatically run whichever `resolve_unknowns.py` is available (project root or `scripts/`) to analyse `unknown_words.csv`, then emit the grouped timing roll-up captured during each spinner stage. Finished runs also prune dated ATC/brand-map exports to keep disk usage in check (see [run.py](https://github.com/carlosresu/esoa/blob/main/run.py)).

## Auto-Accept Reference (Annex/PNF-backed exacts)

Rows land in the **Auto-Accept** bucket when scoring finds an Annex F or PNF
variant with a trusted identifier (`drug_code_final` or `primary_code_final`),
`form_ok` and `route_ok` are `True`, and no unresolved tokens remain. These
cases keep `match_quality`/`reason_final` blank because no review flags were
raised. The summary generated by `scripts/match_outputs.py` also breaks out the
high-confidence exacts:

- `ValidMoleculeWithDrugCodeInAnnex: exact dose/route/form match` – Auto-accepts
  whose canonical text aligned perfectly with the Annex F listing (`dose_sim == 1.0`).
- `ValidGenericInPNF, exact dose/route/form match` – Auto-accepts where the PNF
  variant (rather than Annex) provided the reference code and the parsed dose/form/route
  matched exactly.
- `ValidBrandSwappedForGenericInAnnex: exact dose/form/route match` / `ValidBrandSwappedForGenericInPNF: exact dose/form/route match` – Brand swaps that aligned on all clinical metadata and retained the Auto-Accept status. `did_brand_swap = True` pinpoints these rows.

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

### ValidMoleculeWithDrugCodeInAnnex

- `no_dose_available` – Annex F supplied the drug code, but the record lacked dose evidence. Capture the strength before approval.
- `route_form_mismatch` – Textual route/form conflicts with Annex-implied allowances. Determine whether to adjust heuristics or correct the source data.
- `no_form_and_route_available` – Neither form nor route was available to validate against Annex hints. Provide administration details.
- `no_form_available` / `no_route_available` – Missing individual attributes that prevent verification; supplement before accepting.
- `no_dose_and_form_available`, `no_dose_and_route_available`, `no_dose_form_and_route_available` – Combinations of missing fields that block automatic approval. Resolve the gaps or escalate.
- `dose_mismatch` – Parsed dosing disagrees with the Annex strength. Confirm packaging vs. billing text and reconcile.
- `no_dose_available_annex_heuristic` – (When present) indicates Annex heuristics inferred the form/route but dose proof is still outstanding.

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

### ValidBrandSwappedForGenericInAnnex / ValidBrandSwappedForGenericInPNF

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

### ValidMoleculeInDrugBank

- `needs_formulary_mapping` – DrugBank supplied the synonym but no local formulary ID exists yet. Confirm whether the molecule should be ingested into PNF/WHO or mapped to an existing local alias.
- `no_dose_available` – The free text lacked dose evidence despite a DrugBank hit. Capture the strength prior to approval.
- `no_route_available` – Route was not documented; confirm administration route before accepting a DrugBank-only identification.
- `route_form_mismatch` – The detected route/form pairing violates current policy. Validate whether the synonym reflects a legitimate presentation or requires escalation.
- `no_dose_form_and_route_available` – None of the cardinal fields were observed, leaving the synonym unverified. Collect the missing metadata before progressing.

### UnspecifiedSource

- `auto_exact_dose_route_form` – Auto-Accept rows where the text matched the PNF dose/route/form exactly.
- `dose_mismatch_same_atc` – Auto-Accept rows with non-exact doses, but the PNF variant’s ATC covers all dose presentations.
- `dose_mismatch_varied_atc` – Auto-Accept rows with non-exact doses when multiple ATC payloads exist; flagged for reconciliation.
- Approved route/form substitutions (e.g., sachet counted as tablet) are folded into these tags and no longer tracked separately.
- `nontherapeutic_and_unknown_tokens` / `nontherapeutic_catalog_match` – FDA food/non-therapeutic catalog hits, optionally with residual unknown tokens.
- `unknown_tokens_present` – Partial unknown tokens remain even after PNF/WHO/FDA lookups; see `match_molecule(s)` suffix (e.g., `PartiallyKnownTokensFrom_PNF_WHO`) for which datasets covered the known portion.
- `AllTokensUnknownTo_PNF_WHO_FDA` – No catalog (PNF, WHO, FDA, or DrugBank) matched any token; routed to Unknown with `match_quality = contains_unknown_tokens` and `detail_final` showing the unresolved token count.
- `manual_review_required` – No structured signals materialised; requires human triage.
- `route_mismatch` – The text points to an unverified molecule and also carries a route conflict. Resolve the molecule identification and the route evidence.
- `no_dose_available` – No reference match plus a missing dose. Provide dose context or determine if the entry should be excluded.
- `route_form_mismatch` – The captured route/form pairing is outside our whitelist for an unverified molecule. Clarify both the molecule and administration details.
- `no_dose_and_form_available` – Both dose and form are absent while the molecule remains unverified. Supplement the missing fields or reject.
- `no_form_available` – Form is missing and no reference confirmed the molecule. Capture the dosage form before moving forward.

## Unknown Bucket Reference (why_final × reason_final)

The **Unknown** bucket collects rows that are explicitly non-therapeutic or that
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
