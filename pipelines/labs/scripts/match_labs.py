#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matching helpers for Laboratory & Diagnostic eSOA records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import pandas as pd


def _normalize(text: str | float | int | None) -> str:
    if text is None:
        return ""
    value = str(text).strip().lower()
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _build_lookup(df: pd.DataFrame, text_column: str) -> Dict[str, pd.Series]:
    lookup: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        key = _normalize(row.get(text_column))
        if not key:
            continue
        lookup.setdefault(key, row)
    return lookup


def match_labs_records(
    esoa_csv: Path,
    master_csv: Path,
    diagnostics_xlsx: Path,
    out_csv: Path,
    *,
    skip_excel: bool = False,
) -> Path:
    esoa_df = pd.read_csv(esoa_csv, dtype=str)
    if esoa_df.empty:
        raise ValueError("Prepared Laboratory & Diagnostic eSOA CSV is empty.")

    master_df = pd.read_csv(master_csv, dtype=str)
    if master_df.empty:
        raise ValueError("Labs master CSV is empty.")

    diagnostics_df = pd.read_excel(diagnostics_xlsx, dtype=str) if diagnostics_xlsx.is_file() else pd.DataFrame()

    master_lookup = _build_lookup(master_df, "DESCRIPTION")
    diagnostics_lookup = _build_lookup(diagnostics_df, "desc") if not diagnostics_df.empty else {}

    results = []
    for _, row in esoa_df.iterrows():
        item_number = row.get("ITEM_NUMBER")
        description = row.get("DESCRIPTION", "")
        norm = _normalize(description)
        master_row = master_lookup.get(norm)
        diag_row = diagnostics_lookup.get(norm)

        output_row = {
            "ITEM_NUMBER": item_number,
            "DESCRIPTION": description,
            "normalized_description": norm,
            "match_source": "Unmatched",
            "standard_description": description,
            "source_file": row.get("SOURCE_FILE"),
            "lab_item_number": pd.NA,
            "lab_is_official": pd.NA,
            "lab_description": pd.NA,
            "diagnostics_code": pd.NA,
            "diagnostics_desc": pd.NA,
            "diagnostics_cat": pd.NA,
            "diagnostics_spec": pd.NA,
            "diagnostics_etc": pd.NA,
            "diagnostics_misc": pd.NA,
        }

        if master_row is not None:
            output_row.update(
                {
                    "match_source": "Labs",
                    "standard_description": master_row.get("DESCRIPTION"),
                    "lab_item_number": master_row.get("ITEM_NUMBER"),
                    "lab_is_official": master_row.get("IS_OFFICIAL"),
                    "lab_description": master_row.get("DESCRIPTION"),
                }
            )
        elif diag_row is not None:
            output_row.update(
                {
                    "match_source": "Diagnostics",
                    "standard_description": diag_row.get("desc"),
                    "diagnostics_code": diag_row.get("code"),
                    "diagnostics_desc": diag_row.get("desc"),
                    "diagnostics_cat": diag_row.get("cat"),
                    "diagnostics_spec": diag_row.get("spec"),
                    "diagnostics_etc": diag_row.get("etc"),
                    "diagnostics_misc": diag_row.get("misc"),
                }
            )
        results.append(output_row)

    matched_df = pd.DataFrame(results)
    matched_df.sort_values(by=["ITEM_NUMBER"], inplace=True, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(out_csv, index=False, encoding="utf-8")

    if not skip_excel:
        xlsx_path = out_csv.with_suffix(".xlsx")
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            matched_df.to_excel(writer, index=False, sheet_name="matched")
            ws = writer.sheets["matched"]
            ws.freeze_panes(1, 0)
            nrows, ncols = matched_df.shape
            ws.autofilter(0, 0, nrows, ncols - 1)

    return out_csv
