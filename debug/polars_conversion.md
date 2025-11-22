
# Init–Concurrency

A. pipelines/drugs/scripts/__init__.py: Rewrite module exports to point at Polars/Parquet-first implementations, update the docstring to note Polars-first defaults, ensure no pandas references, and keep __all__ in sync without touching other files.
B. pipelines/drugs/scripts/aho_drugs.py: Make the automata builder fully Polars-first by accepting pl.DataFrame/LazyFrame, using Polars expressions for base/synonym selection and deduping instead of per-row loops where feasible, assuming Parquet inputs, and ensuring no pandas usage while preserving behavior.
C. pipelines/drugs/scripts/brand_map_drugs.py: Convert all I/O and transforms to Polars lazy/streaming with Parquet as the default format; remove any pandas fallbacks; keep load_latest_brandmap, build_brand_automata, and fda_generics_set returning Polars DataFrames and documented as Parquet-first.
D. pipelines/drugs/scripts/combos_drugs.py: Keep combination-parsing helpers pure but document and adjust signatures so they are Polars-friendly (usable via pl.col().map_elements), add no pandas dependencies, and assume upstream tabular callers are Parquet/Polars.
E. pipelines/drugs/scripts/concurrency_drugs.py: Ensure concurrency helpers are documented for Polars workloads (e.g., mapping over Polars columns/series), avoid pandas assumptions, keep behavior intact, and note Parquet-first pipeline context.

# Debug–Features

F. pipelines/drugs/scripts/debug_drugs.py: Keep the profiler wrapper intact but align docs and any file handling with the Polars/Parquet pipeline, avoiding pandas assumptions and leaving behavior unchanged
G. pipelines/drugs/scripts/dose_drugs.py: Make dose parsing utilities Polars-ready by ensuring they work cleanly with pl.col().map_elements/expressions, add no pandas imports, and keep outputs deterministic; document Parquet-first pipeline context.
H. pipelines/drugs/scripts/generic_normalization.py: Align normalization helpers with Polars usage (expression-friendly, no pandas), keep behavior identical, and clarify in docs they are meant for Parquet/Polars pipelines.
I. pipelines/drugs/scripts/match_drugs.py: Refactor the full matcher to be pure Polars (lazy/streaming where possible) and Parquet-first: load inputs with scan_parquet, perform joins/dedup/filtering with Polars expressions, drop any pandas remnants, preserve CLI/outputs, and only leave TODO(polars): convert if a third-party forces pandas.
J. pipelines/drugs/scripts/match_features_drugs.py: Remove the pandas fallback and convert every feature-building step to Polars-first/Parquet-first: operate on Polars DataFrames/LazyFrames for brand/WHO/FDA lookups, tokenization, fuzzy matching, route/form extraction, and build_features; replace iterrows/Series logic with Polars expressions; keep outputs as Polars and default to Parquet writes, with TODO(polars): convert only if absolutely necessary.

# Outputs–Prepare

K. pipelines/drugs/scripts/match_outputs_drugs.py: Rewrite loading, summarization, and writing to be Polars-first and Parquet-first: use scan_parquet for inputs, do bucket/summary aggregation in Polars, replace _write_csv_and_parquet with Polars I/O (CSV optional), and keep pandas only if required for Excel with a TODO(polars): convert note; preserve CLI/output parity.
L. pipelines/drugs/scripts/match_scoring_drugs.py: Convert the scoring/classification engine to pure Polars: rewrite helpers and score_and_classify to operate on Polars DataFrames/LazyFrames with expressions for route/form/dose/priority logic, eliminate pandas/Series usage, default all I/O to Parquet, and maintain the existing columns/behavior.
M. pipelines/drugs/scripts/pnf_aliases_drugs.py: Ensure alias expansion utilities are Polars-friendly (no pandas), document expected Polars/Parquet usage if DataFrames are involved, and keep existing behavior.
N. pipelines/drugs/scripts/pnf_partial_drugs.py: Keep the partial matcher strictly Polars/Parquet-first: accept pl.DataFrame/LazyFrame, favor Polars expressions over row iteration where possible, avoid pandas, and preserve current outputs.
O. pipelines/drugs/scripts/prepare_drugs.py: Rewrite the preparation stage to Polars-first and Parquet-first: load PNF/eSOA with Polars (scan_csv/scan_parquet), compute all derived columns via Polars expressions (no pandas or iterrows), explode routes/forms in Polars, write Parquet as the canonical output (CSV optional), and keep validations/schema identical.

# Reference–WHO

P. pipelines/drugs/scripts/reference_data_drugs.py: Convert reference loaders to Polars/Parquet: use Polars scans instead of pandas reads, rewrite _iter_csv_column, load_drugbank_generics, and load_ignore_words to operate via Polars while returning the same Python collections, and default outputs to Parquet where applicable.
Q. pipelines/drugs/scripts/resolve_unknowns_drugs.py: Make unknown-resolution helpers Polars-ready by operating on/expecting Polars columns or frames, avoid pandas entirely, and document Parquet-first assumptions while keeping behavior unchanged.
R. pipelines/drugs/scripts/routes_forms_drugs.py: Keep route/form parsers pure but document Polars usage (vectorizable via pl.col().map_elements), ensure no pandas dependencies, and preserve logic.
S. pipelines/drugs/scripts/text_utils_drugs.py: Ensure text/token utilities remain Polars-first (usable in Polars expressions, no pandas), clarify Parquet-first pipeline context, and keep outputs identical.
T. pipelines/drugs/scripts/who_molecules_drugs.py: Rewrite the WHO loader to Polars/Parquet-first: load with scan_parquet, compute normalized columns with Polars expressions instead of pandas maps/iterrows, return the same dict/list structures, avoid pandas (or mark any unavoidable use with TODO(polars): convert), and preserve behavior.