#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared I/O helpers for reading/writing tabular data in the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow
    from pyarrow.lib import ArrowKeyError as _ArrowKeyError  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency at runtime
    pyarrow = None  # type: ignore[assignment]
    _ArrowKeyError = RuntimeError  # type: ignore[assignment]

_PARQUET_SUFFIXES = {".parquet", ".pq", ".ipc", ".arrow"}
_FEATHER_SUFFIXES = {".feather", ".ft"}
_CSV_SUFFIXES = {".csv"}
_COMPRESSED_CSV_SUFFIXES = {".csv.gz", ".csv.bz2", ".csv.zip"}


def _suffix(path: Path) -> str:
    """Return the lowercase composite suffix (supports multi-part like .csv.gz)."""
    if not path.suffixes:
        return ""
    return "".join(s.lower() for s in path.suffixes[-2:]) if len(path.suffixes) > 1 else path.suffix.lower()


def ensure_parquet_suffix(path: Path) -> Path:
    """Ensure the provided path ends with a parquet-friendly suffix."""
    if path.suffix.lower() in _PARQUET_SUFFIXES:
        return path
    return path.with_suffix(".parquet")


def read_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a DataFrame from CSV/Parquet/Feather, inferring the format from the suffix."""
    p = Path(path)
    suffix = _suffix(p)
    if p.suffix.lower() in _PARQUET_SUFFIXES or suffix in _PARQUET_SUFFIXES:
        return pd.read_parquet(p, **kwargs)
    if p.suffix.lower() in _FEATHER_SUFFIXES or suffix in _FEATHER_SUFFIXES:
        return pd.read_feather(p, **kwargs)
    if p.suffix.lower() in _CSV_SUFFIXES or suffix in _CSV_SUFFIXES or suffix in _COMPRESSED_CSV_SUFFIXES:
        return pd.read_csv(p, **kwargs)
    # Fallback to CSV as the most permissive reader when extension is missing.
    return pd.read_csv(p, **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, *, compression: str = "snappy", **kwargs: Any) -> None:
    """Persist a DataFrame as Parquet with a sane default compression codec."""
    p = ensure_parquet_suffix(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(p, index=False, compression=compression, **kwargs)
    except _ArrowKeyError as exc:
        if _needs_pyarrow_extension_patch(exc):
            _install_pyarrow_extension_hotfix()
            df.to_parquet(p, index=False, compression=compression, **kwargs)
        else:
            raise


def _needs_pyarrow_extension_patch(exc: Exception) -> bool:
    return "arrow.py_extension_type" in str(exc)


def _install_pyarrow_extension_hotfix() -> None:
    if pyarrow is None:
        raise RuntimeError(
            "pyarrow is required to write parquet files but is not installed."
        )

    if getattr(pyarrow, "_esoa_extension_patched", False):
        return

    class _SentinelExtensionType(pyarrow.ExtensionType):  # type: ignore[misc]
        def __arrow_ext_serialize__(self) -> bytes:
            return b""

        @classmethod
        def __arrow_ext_deserialize__(
            cls, storage_type: pyarrow.DataType, serialized: bytes
        ) -> "pyarrow.ExtensionType":
            raise RuntimeError(
                "Unexpected attempt to deserialize placeholder extension type."
            )

    try:
        pyarrow.register_extension_type(
            _SentinelExtensionType(pyarrow.null(), "arrow.py_extension_type")
        )
    except pyarrow.ArrowKeyError:
        pass

    try:
        pyarrow.unregister_extension_type("arrow.py_extension_type")
    except pyarrow.ArrowKeyError:
        pass

    pyarrow._esoa_extension_patched = True
