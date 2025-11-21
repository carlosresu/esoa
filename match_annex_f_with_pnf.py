"""Build PNF and Annex F lexicons then score word matches between them."""

from __future__ import annotations

import math
from pathlib import Path
from collections import Counter
from typing import Iterable, List, Sequence

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DRUGS_DIR = BASE_DIR / "inputs" / "drugs"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DRUGS_DIR = OUTPUTS_DIR / "drugs"


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


def build_pnf_lexicon(df: pd.DataFrame) -> pd.DataFrame:
    def split_tokens(row) -> tuple[list[str], list[str]]:
        base_tokens = split_with_parentheses(row.get("generic_name", ""))

        secondary: List[str] = []
        secondary.extend(tokens_from_field(row.get("route_allowed")))
        secondary.extend(tokens_from_field(row.get("form_token")))

        for col in ("strength", "per_val", "pct", "strength_mg", "ratio_mg_per_ml"):
            token = format_number_token(row.get(col))
            if token:
                secondary.append(token)

        secondary.extend(tokens_from_field(row.get("unit")))
        secondary.extend(tokens_from_field(row.get("per_unit")))

        return base_tokens, secondary

    df = df.copy()
    token_pairs = df.apply(split_tokens, axis=1, result_type="expand")
    df["lexicon_tokens"] = token_pairs[0]
    df["lexicon_secondary_tokens"] = token_pairs[1]
    df["lexicon"] = df["lexicon_tokens"].apply(lambda toks: "|".join(toks))
    df["lexicon_secondary"] = df["lexicon_secondary_tokens"].apply(
        lambda toks: "|".join(toks)
    )
    return df


def build_annex_fuzzy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fuzzy_basis_tokens"] = df["Drug Description"].apply(split_with_parentheses)
    df["fuzzy_basis"] = df["fuzzy_basis_tokens"].apply(lambda toks: "|".join(toks))
    return df


def best_matches(
    annex_df: pd.DataFrame, pnf_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pnf_rows = []
    for _, row in pnf_df.iterrows():
        primary_tokens = parse_pipe_tokens(row.get("lexicon", ""))
        secondary_tokens = parse_pipe_tokens(row.get("lexicon_secondary", ""))
        pnf_rows.append(
            {
                "generic_id": row.get("generic_id"),
                "generic_name": row.get("generic_name"),
                "lexicon": row.get("lexicon", ""),
                "lexicon_secondary": row.get("lexicon_secondary", ""),
                "primary_tokens": primary_tokens,
                "secondary_tokens": secondary_tokens,
            }
        )

    matched_rows = []
    tied_rows = []
    for _, annex_row in annex_df.iterrows():
        fuzzy_tokens = parse_pipe_tokens(annex_row.get("fuzzy_basis", ""))
        fuzzy_counts = Counter(fuzzy_tokens)
        if not fuzzy_tokens:
            matched_rows.append(
                {
                    "Drug Code": annex_row.get("Drug Code"),
                    "Drug Description": annex_row.get("Drug Description"),
                    "matched_generic_name": None,
                    "matching_tokens": None,
                    "secondary_matching_tokens": None,
                    "fuzzy_basis": None,
                    "matched_lexicon": None,
                    "match_count": None,
                    "matched_secondary_lexicon": None,
                    "secondary_match_count": None,
                }
            )
            continue

        best_primary = 0
        best_primary_records: List[dict] = []
        for pnf_row in pnf_rows:
            primary_counts = Counter(pnf_row["primary_tokens"])
            primary_overlap_counts = fuzzy_counts & primary_counts
            primary_overlap: List[str] = []
            for tok in fuzzy_tokens:
                if primary_overlap_counts.get(tok, 0) > 0:
                    primary_overlap.append(tok)
                    primary_overlap_counts[tok] -= 1
            primary_count = len(primary_overlap)
            if primary_count == 0 or primary_count < best_primary:
                continue
            if primary_count > best_primary:
                best_primary = primary_count
                best_primary_records = []
            best_primary_records.append(
                pnf_row | {"primary_overlap": primary_overlap, "primary_count": primary_count}
            )

        if not best_primary_records or best_primary == 0:
            matched_rows.append(
                {
                    "Drug Code": annex_row.get("Drug Code"),
                    "Drug Description": annex_row.get("Drug Description"),
                    "matched_generic_name": None,
                    "matching_tokens": None,
                    "secondary_matching_tokens": None,
                    "fuzzy_basis": None,
                    "matched_lexicon": None,
                    "match_count": None,
                    "matched_secondary_lexicon": None,
                    "secondary_match_count": None,
                }
            )
            continue

        best_secondary = -1
        final_records: List[dict] = []
        for rec in best_primary_records:
            sec_counts = Counter(rec["secondary_tokens"])
            secondary_overlap_counts = fuzzy_counts & sec_counts
            secondary_overlap: List[str] = []
            for tok in fuzzy_tokens:
                if secondary_overlap_counts.get(tok, 0) > 0:
                    secondary_overlap.append(tok)
                    secondary_overlap_counts[tok] -= 1
            secondary_count = len(secondary_overlap)
            if secondary_count > best_secondary:
                best_secondary = secondary_count
                final_records = []
            if secondary_count == best_secondary:
                final_records.append(
                    rec
                    | {
                        "secondary_overlap": secondary_overlap,
                        "secondary_count": secondary_count,
                    }
                )

        if len(final_records) != 1:
            matched_rows.append(
                {
                    "Drug Code": annex_row.get("Drug Code"),
                    "Drug Description": annex_row.get("Drug Description"),
                    "matched_generic_name": None,
                    "matching_tokens": None,
                    "secondary_matching_tokens": None,
                    "fuzzy_basis": None,
                    "matched_lexicon": None,
                    "match_count": None,
                    "matched_secondary_lexicon": None,
                    "secondary_match_count": None,
                }
            )
            for rec in final_records:
                tied_rows.append(
                    {
                        "Drug Code": annex_row.get("Drug Code"),
                        "Drug Description": annex_row.get("Drug Description"),
                        "matched_generic_name": rec["generic_name"],
                        "matching_tokens": "|".join(rec["primary_overlap"]),
                        "secondary_matching_tokens": "|".join(
                            rec.get("secondary_overlap", ())
                        ),
                        "fuzzy_basis": annex_row.get("fuzzy_basis", ""),
                        "matched_lexicon": rec["lexicon"],
                        "match_count": best_primary,
                        "matched_secondary_lexicon": rec["lexicon_secondary"],
                        "secondary_match_count": best_secondary,
                    }
                )
            continue

        rec = final_records[0]
        matched_rows.append(
            {
                "Drug Code": annex_row.get("Drug Code"),
                "Drug Description": annex_row.get("Drug Description"),
                "matched_generic_name": rec["generic_name"],
                "matching_tokens": "|".join(rec["primary_overlap"]),
                "secondary_matching_tokens": "|".join(rec.get("secondary_overlap", ())),
                "fuzzy_basis": annex_row.get("fuzzy_basis", ""),
                "matched_lexicon": rec["lexicon"],
                "match_count": best_primary,
                "matched_secondary_lexicon": rec["lexicon_secondary"],
                "secondary_match_count": best_secondary,
            }
        )

    return pd.DataFrame(matched_rows), pd.DataFrame(tied_rows)


def main() -> None:
    pnf_path = DRUGS_DIR / "pnf_prepared.csv"
    annex_path = DRUGS_DIR / "annex_f.csv"

    pnf_df = pd.read_csv(pnf_path)
    annex_df = pd.read_csv(annex_path)

    pnf_with_lex = build_pnf_lexicon(pnf_df)
    annex_with_basis = build_annex_fuzzy(annex_df)

    pnf_lex_path = DRUGS_DIR / "pnf_lexicon.csv"
    annex_lex_path = DRUGS_DIR / "annex_f_lexicon.csv"
    pnf_with_lex.drop(
        columns=["lexicon_tokens", "lexicon_secondary_tokens"], errors="ignore"
    ).to_csv(
        pnf_lex_path, index=False
    )
    annex_with_basis.drop(
        columns=["fuzzy_basis_tokens"], errors="ignore"
    ).to_csv(annex_lex_path, index=False)

    match_df, tie_df = best_matches(annex_with_basis, pnf_with_lex)
    OUTPUTS_DRUGS_DIR.mkdir(parents=True, exist_ok=True)
    match_path = OUTPUTS_DRUGS_DIR / "annex_f_pnf_matches.csv"
    match_df.to_csv(match_path, index=False)
    tie_path = OUTPUTS_DRUGS_DIR / "annex_f_pnf_ties.csv"
    tie_df.to_csv(tie_path, index=False)

    print(f"PNF lexicon saved to {pnf_lex_path}")
    print(f"Annex F lexicon saved to {annex_lex_path}")
    print(f"Match results saved to {match_path}")
    print(f"Tied matches saved to {tie_path}")


if __name__ == "__main__":
    main()
