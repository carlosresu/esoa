"""
I/O utility functions for drug pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Write DataFrame to CSV (canonical format)."""
    df.to_csv(csv_path, index=False)


# Backward compatibility alias
def write_csv_and_parquet(df: pd.DataFrame, csv_path: Path) -> None:
    """Deprecated: use write_csv instead. Now writes CSV only."""
    write_csv(df, csv_path)


def reorder_columns_after(
    df: pd.DataFrame,
    target_col: str,
    move_col: str,
) -> pd.DataFrame:
    """Move a column to be right after another column."""
    cols = list(df.columns)
    target_idx = cols.index(target_col) if target_col in cols else -1
    move_idx = cols.index(move_col) if move_col in cols else -1
    
    if target_idx >= 0 and move_idx >= 0 and move_idx != target_idx + 1:
        cols.remove(move_col)
        cols.insert(target_idx + 1, move_col)
        return df[cols]
    
    return df
