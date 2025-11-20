#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preparation helpers for Laboratory & Diagnostic eSOA records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

LABS_CODE = "LaboratoryAndDiagnostic"
ITEM_EXCLUDE_START = 1540
ITEM_EXCLUDE_END = 1896

def _write_csv_and_parquet(frame: pd.DataFrame, csv_path: Path) -> None:
    """Persist a dataframe to CSV and Parquet using the same stem."""
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    parquet_path = Path(csv_path).with_suffix(".parquet")
    frame.to_parquet(parquet_path, index=False)


def _load_esoa_file(path: Path, *, sep: str = ",") -> pd.DataFrame:
    if not path or not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, sep=sep)
    if df.empty:
        return df
    needed = {"ITEM_NUMBER", "ITEM_REF_CODE", "DESCRIPTION"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    df["SOURCE_FILE"] = path.name
    return df[list(needed) + ["SOURCE_FILE"]]


def _filter_category(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame[frame["ITEM_REF_CODE"].astype(str) == LABS_CODE].copy()
    if filtered.empty:
        return filtered
    # Exclude ITEM_NUMBER 1540–1896 inclusive
    item_numbers = pd.to_numeric(filtered["ITEM_NUMBER"], errors="coerce")
    mask = (item_numbers >= ITEM_EXCLUDE_START) & (item_numbers <= ITEM_EXCLUDE_END)
    filtered = filtered.loc[~mask].copy()
    filtered.dropna(subset=["DESCRIPTION"], inplace=True)
    filtered["DESCRIPTION"] = filtered["DESCRIPTION"].astype(str).str.strip()
    filtered = filtered[filtered["DESCRIPTION"].astype(bool)]
    return filtered


def prepare_labs_inputs(
    csv_source: Path,
    tsv_source: Path,
    extra_sources: Iterable[Path] | None,
    dest_csv: Path,
) -> Path:
    """Create the prepared Laboratory & Diagnostic eSOA CSV."""
    frames: list[pd.DataFrame] = []
    frames.append(_filter_category(_load_esoa_file(csv_source, sep=",")))
    frames.append(_filter_category(_load_esoa_file(tsv_source, sep="\t")))

    if extra_sources:
        for extra in extra_sources:
            frames.append(_filter_category(_load_esoa_file(extra)))

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        raise FileNotFoundError("No LaboratoryAndDiagnostic rows found in provided sources.")

    # Deduplicate by ITEM_NUMBER + DESCRIPTION to avoid duplicates between CSV/TSV.
    combined.drop_duplicates(subset=["ITEM_NUMBER", "DESCRIPTION"], inplace=True)
    combined.sort_values(by=["ITEM_NUMBER", "DESCRIPTION"], inplace=True)

    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_and_parquet(combined, dest_csv)
    return dest_csv
