"""Match Annex F entries to ATC codes using PNF, DrugBank, and WHO lexicons."""

from __future__ import annotations

import concurrent.futures
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DRUGS_DIR = BASE_DIR / "inputs" / "drugs"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DRUGS_DIR = OUTPUTS_DIR / "drugs"
_REFERENCE_ROWS_SHARED: list[dict] | None = None


def _strip_commas(token: str) -> str:
    return token.rstrip(",").strip()


def split_with_parentheses(text: str) -> List[str]:
    """Split on spaces while keeping parentheses content together; drop commas and parens."""
    if text is None:
        return []
    chars = str(text)
    tokens: List[str] = []
    current: List[str] = []
    depth = 0

    for ch in chars:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            if depth:
                depth -= 1
            continue
        if ch.isspace() and depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(ch)

    if current:
        tokens.append("".join(current))

    cleaned = [_strip_commas(tok) for tok in tokens]
    return [tok.upper() for tok in cleaned if tok]


def format_number_token(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text.upper() if text else None
    try:
        num = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text.upper() if text else None
    if math.isnan(num):
        return None
    if num.is_integer():
        return str(int(num))
    text = f"{num:.15g}"
    return text.rstrip("0").rstrip(".")


def tokens_from_field(value) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).replace("_", " ")
    return split_with_parentheses(text)


def parse_pipe_tokens(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    return [tok for tok in str(value).split("|") if tok]


def _maybe_none(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_str_or_empty(value) -> str:
    val = _maybe_none(value)
    if val is None:
        return ""
    return str(val)


def _ordered_overlap(
    fuzzy_tokens: List[str], fuzzy_counts: Counter, candidate_tokens: Iterable[str]
) -> List[str]:
    candidate_counts = Counter(candidate_tokens)
    overlap_counts = fuzzy_counts & candidate_counts
    overlap: List[str] = []
    for tok in fuzzy_tokens:
        if overlap_counts.get(tok, 0) > 0:
            overlap.append(tok)
            overlap_counts[tok] -= 1
    return overlap


def _reference_sort_key(rec: dict) -> tuple[int, str, str]:
    source_priority = {"pnf": 0, "drugbank": 1, "who": 2}
    priority = source_priority.get(rec.get("source"), 99)
    name = _as_str_or_empty(rec.get("name")).upper()
    ident = _as_str_or_empty(rec.get("id")).upper()
    return (priority, name, ident)


def build_pnf_reference(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        lexicon = _as_str_or_empty(row.get("lexicon"))
        lexicon_secondary = _as_str_or_empty(row.get("lexicon_secondary"))
        rows.append(
            {
                "source": "pnf",
                "id": _maybe_none(row.get("generic_id")),
                "name": _maybe_none(row.get("generic_name")),
                "lexicon": lexicon,
                "lexicon_secondary": lexicon_secondary,
                "primary_tokens": parse_pipe_tokens(lexicon),
                "secondary_tokens": parse_pipe_tokens(lexicon_secondary),
                "atc_code": _maybe_none(row.get("atc_code")),
            }
        )
    return rows


def build_drugbank_reference(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        primary_tokens: List[str] = []
        for col in ("lexeme", "generic_components_key", "canonical_generic_name"):
            primary_tokens.extend(split_with_parentheses(row.get(col)))

        secondary_tokens: List[str] = []
        for col in ("route_norm", "form_norm", "salt_names", "dose_norm"):
            secondary_tokens.extend(split_with_parentheses(row.get(col)))

        name = _maybe_none(row.get("canonical_generic_name")) or _maybe_none(
            row.get("lexeme")
        )

        rows.append(
            {
                "source": "drugbank",
                "id": _maybe_none(row.get("drugbank_id")),
                "name": name,
                "lexicon": "|".join(primary_tokens),
                "lexicon_secondary": "|".join(secondary_tokens),
                "primary_tokens": primary_tokens,
                "secondary_tokens": secondary_tokens,
                "atc_code": _maybe_none(row.get("atc_code")),
            }
        )
    return rows


def build_who_reference(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        primary_tokens = split_with_parentheses(row.get("atc_name"))

        secondary_tokens: List[str] = []
        for col in ("adm_r", "uom"):
            secondary_tokens.extend(split_with_parentheses(row.get(col)))

        rows.append(
            {
                "source": "who",
                "id": _maybe_none(row.get("atc_code")),
                "name": _maybe_none(row.get("atc_name")),
                "lexicon": "|".join(primary_tokens),
                "lexicon_secondary": "|".join(secondary_tokens),
                "primary_tokens": primary_tokens,
                "secondary_tokens": secondary_tokens,
                "atc_code": _maybe_none(row.get("atc_code")),
            }
        )
    return rows


def build_reference_rows(
    pnf_df: pd.DataFrame, drugbank_df: pd.DataFrame, who_df: pd.DataFrame
) -> list[dict]:
    refs: list[dict] = []
    refs.extend(build_pnf_reference(pnf_df))
    refs.extend(build_drugbank_reference(drugbank_df))
    refs.extend(build_who_reference(who_df))
    return refs


def _empty_match_record(annex_row: pd.Series) -> dict:
    return {
        "Drug Code": annex_row.get("Drug Code"),
        "Drug Description": annex_row.get("Drug Description"),
        "fuzzy_basis": _as_str_or_empty(annex_row.get("fuzzy_basis")),
        "matched_source": None,
        "matched_generic_name": None,
        "matched_lexicon": None,
        "match_count": None,
        "matched_secondary_lexicon": None,
        "secondary_match_count": None,
        "atc_code": None,
    }


def _build_match_record(
    annex_row: pd.Series, ref: dict, primary_score: int, secondary_score: int
) -> dict:
    return {
        "Drug Code": annex_row.get("Drug Code"),
        "Drug Description": annex_row.get("Drug Description"),
        "fuzzy_basis": _as_str_or_empty(annex_row.get("fuzzy_basis")),
        "matched_source": ref.get("source"),
        "matched_generic_name": ref.get("name"),
        "matched_lexicon": ref.get("lexicon"),
        "match_count": primary_score,
        "matched_secondary_lexicon": ref.get("lexicon_secondary"),
        "secondary_match_count": secondary_score,
        "atc_code": ref.get("atc_code"),
    }


def _score_annex_row(
    annex_row: dict, reference_rows: list[dict]
) -> tuple[dict, list[dict], list[dict]]:
    matched_rows: list[dict] = []
    tie_rows: list[dict] = []
    unresolved_rows: list[dict] = []

    fuzzy_tokens = parse_pipe_tokens(annex_row.get("fuzzy_basis"))
    fuzzy_counts = Counter(fuzzy_tokens)
    if not fuzzy_tokens:
        matched_rows.append(_empty_match_record(annex_row))
        return matched_rows[0], tie_rows, unresolved_rows

    best_primary = 0
    best_primary_records: list[dict] = []
    for ref in reference_rows:
        primary_overlap = _ordered_overlap(
            fuzzy_tokens, fuzzy_counts, ref.get("primary_tokens", ())
        )
        primary_score = len(primary_overlap)
        if primary_score == 0 or primary_score < best_primary:
            continue
        if primary_score > best_primary:
            best_primary_records = []
            best_primary = primary_score
        best_primary_records.append(
            ref
            | {
                "primary_overlap": primary_overlap,
                "primary_score": primary_score,
            }
        )

    if not best_primary_records or best_primary == 0:
        matched_rows.append(_empty_match_record(annex_row))
        return matched_rows[0], tie_rows, unresolved_rows

    best_secondary = -1
    finalists: list[dict] = []
    for rec in best_primary_records:
        secondary_overlap = _ordered_overlap(
            fuzzy_tokens, fuzzy_counts, rec.get("secondary_tokens", ())
        )
        secondary_score = len(secondary_overlap)
        if secondary_score > best_secondary:
            finalists = []
            best_secondary = secondary_score
        if secondary_score == best_secondary:
            finalists.append(
                rec
                | {
                    "secondary_overlap": secondary_overlap,
                    "secondary_score": secondary_score,
                }
            )

    if not finalists:
        matched_rows.append(_empty_match_record(annex_row))
        return matched_rows[0], tie_rows, unresolved_rows

    if len(finalists) == 1:
        winner = finalists[0]
        matched_rows.append(
            _build_match_record(annex_row, winner, best_primary, best_secondary)
        )
        return matched_rows[0], tie_rows, unresolved_rows

    sorted_finalists = sorted(finalists, key=_reference_sort_key)
    atc_set = {_as_str_or_empty(rec.get("atc_code")) for rec in sorted_finalists}
    if len(atc_set) == 1:
        winner = sorted_finalists[0]
        matched_rows.append(
            _build_match_record(annex_row, winner, best_primary, best_secondary)
        )
        for rec in sorted_finalists:
            tie_rows.append(
                _build_match_record(
                    annex_row,
                    rec,
                    rec.get("primary_score", best_primary),
                    rec.get("secondary_score", best_secondary),
                )
            )
        return matched_rows[0], tie_rows, unresolved_rows

    matched_rows.append(_empty_match_record(annex_row))
    for rec in sorted_finalists:
        unresolved_rows.append(
            _build_match_record(
                annex_row,
                rec,
                rec.get("primary_score", best_primary),
                rec.get("secondary_score", best_secondary),
            )
        )
    return matched_rows[0], tie_rows, unresolved_rows


def _init_reference_rows(reference_rows: list[dict]) -> None:
    global _REFERENCE_ROWS_SHARED
    _REFERENCE_ROWS_SHARED = reference_rows


def _process_annex_row_worker(annex_row: dict) -> tuple[dict, list[dict], list[dict]]:
    reference_rows = _REFERENCE_ROWS_SHARED or []
    return _score_annex_row(annex_row, reference_rows)


def match_annex_with_atc(
    annex_df: pd.DataFrame, reference_rows: list[dict], max_workers: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annex_records = annex_df.to_dict(orient="records")
    matched_rows: list[dict] = []
    tie_rows: list[dict] = []
    unresolved_rows: list[dict] = []

    worker_count = max_workers
    if worker_count is None:
        worker_count = max(1, min(32, os.cpu_count() or 1))
    if worker_count <= 1:
        for rec in annex_records:
            match_row, ties, unresolved = _score_annex_row(rec, reference_rows)
            matched_rows.append(match_row)
            tie_rows.extend(ties)
            unresolved_rows.extend(unresolved)
        return (
            pd.DataFrame(matched_rows),
            pd.DataFrame(tie_rows),
            pd.DataFrame(unresolved_rows),
        )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_reference_rows,
        initargs=(reference_rows,),
    ) as executor:
        for match_row, ties, unresolved in executor.map(
            _process_annex_row_worker, annex_records, chunksize=25
        ):
            matched_rows.append(match_row)
            tie_rows.extend(ties)
            unresolved_rows.extend(unresolved)

    return (
        pd.DataFrame(matched_rows),
        pd.DataFrame(tie_rows),
        pd.DataFrame(unresolved_rows),
    )


def main() -> None:
    annex_path = DRUGS_DIR / "annex_f_lexicon.csv"
    pnf_path = DRUGS_DIR / "pnf_lexicon.csv"
    drugbank_path = DRUGS_DIR / "drugbank_generics_master.csv"
    who_path = DRUGS_DIR / "who_atc_2025-11-20.csv"

    annex_df = pd.read_csv(annex_path)
    pnf_df = pd.read_csv(pnf_path)
    drugbank_df = pd.read_csv(drugbank_path)
    who_df = pd.read_csv(who_path)

    reference_rows = build_reference_rows(pnf_df, drugbank_df, who_df)
    match_df, tie_df, unresolved_df = match_annex_with_atc(annex_df, reference_rows)

    OUTPUTS_DRUGS_DIR.mkdir(parents=True, exist_ok=True)
    match_path = OUTPUTS_DRUGS_DIR / "annex_f_with_atc.csv"
    ties_path = OUTPUTS_DRUGS_DIR / "annex_f_atc_ties.csv"
    unresolved_path = OUTPUTS_DRUGS_DIR / "annex_f_atc_unresolved.csv"

    match_df.to_csv(match_path, index=False)
    tie_df.to_csv(ties_path, index=False)
    unresolved_df.to_csv(unresolved_path, index=False)

    print(f"Annex F ATC matches saved to {match_path}")
    print(f"Acceptable ties saved to {ties_path}")
    print(f"Unresolved ties saved to {unresolved_path}")


if __name__ == "__main__":
    main()
