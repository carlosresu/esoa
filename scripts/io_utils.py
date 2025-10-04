#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared I/O helpers for reading/writing tabular data in the pipeline."""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow
    from pyarrow.lib import ArrowKeyError as _ArrowKeyError  # type: ignore[attr-defined]
    from pyarrow.lib import ArrowMemoryError as _ArrowMemoryError  # type: ignore[attr-defined]
    import pyarrow.parquet as _pyarrow_parquet
except ImportError:  # pragma: no cover - optional dependency at runtime
    pyarrow = None  # type: ignore[assignment]
    _ArrowKeyError = RuntimeError  # type: ignore[assignment]
    _ArrowMemoryError = MemoryError  # type: ignore[assignment]
    _pyarrow_parquet = None  # type: ignore[assignment]

_PARQUET_SUFFIXES = {".parquet", ".pq", ".ipc", ".arrow"}
_FEATHER_SUFFIXES = {".feather", ".ft"}
_CSV_SUFFIXES = {".csv"}
_COMPRESSED_CSV_SUFFIXES = {".csv.gz", ".csv.bz2", ".csv.zip"}
_DEFAULT_PARQUET_CHUNK_SIZE = 50_000


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


def _is_memory_error(exc: Exception) -> bool:
    if isinstance(exc, (MemoryError, _ArrowMemoryError)):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOMEM:
        return True
    return False


def _write_parquet_streaming(
    df: pd.DataFrame,
    path: Path,
    *,
    compression: str,
    preserve_index: bool,
    chunk_size: int,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """Write large DataFrames in row-group chunks to avoid huge contiguous allocations."""
    if pyarrow is None or _pyarrow_parquet is None:
        raise RuntimeError("pyarrow is required to write parquet files but is not installed.")

    opts = dict(extra_kwargs or {})
    engine = opts.pop("engine", None)
    if engine and engine != "pyarrow":
        raise RuntimeError("Streaming parquet fallback only supports the pyarrow engine.")

    table_kwargs: dict[str, Any] = {"preserve_index": preserve_index}
    for key in ("schema", "types_mapper", "safe", "use_threads"):
        if key in opts:
            table_kwargs[key] = opts.pop(key)

    writer_kwargs: dict[str, Any] = {"compression": compression}
    for key in (
        "use_dictionary",
        "write_statistics",
        "data_page_size",
        "coerce_timestamps",
        "allow_truncated_timestamps",
        "version",
        "use_deprecated_int96_timestamps",
    ):
        if key in opts:
            writer_kwargs[key] = opts.pop(key)

    if opts:
        raise TypeError(f"Unsupported parquet kwargs in streaming fallback: {sorted(opts.keys())}")

    total_rows = len(df)
    if total_rows == 0:
        table = pyarrow.Table.from_pandas(df, **table_kwargs)
        _pyarrow_parquet.write_table(table, str(path), **writer_kwargs)
        return

    chunk_size = max(int(chunk_size), 1)
    first_chunk = df.iloc[: min(chunk_size, total_rows)]
    table = pyarrow.Table.from_pandas(first_chunk, **table_kwargs)
    with _pyarrow_parquet.ParquetWriter(str(path), table.schema, **writer_kwargs) as writer:
        writer.write_table(table)
        written = len(first_chunk)
        while written < total_rows:
            chunk = df.iloc[written : min(written + chunk_size, total_rows)]
            table = pyarrow.Table.from_pandas(chunk, **table_kwargs)
            writer.write_table(table)
            written += len(chunk)


def write_parquet(df: pd.DataFrame, path: str | Path, *, compression: str = "snappy", **kwargs: Any) -> None:
    """Persist a DataFrame as Parquet with a sane default compression codec."""
    p = ensure_parquet_suffix(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)

    write_kwargs = dict(kwargs)
    compression_value = write_kwargs.pop("compression", compression)
    index_flag = bool(write_kwargs.pop("index", False))
    columns = write_kwargs.pop("columns", None)
    target_df = df if columns is None else df.loc[:, columns]

    def _write_once() -> None:
        target_df.to_parquet(
            p,
            compression=compression_value,
            index=index_flag,
            **write_kwargs,
        )

    try:
        _write_once()
        return
    except _ArrowKeyError as exc:
        if _needs_pyarrow_extension_patch(exc):
            _install_pyarrow_extension_hotfix()
            try:
                _write_once()
                return
            except Exception as inner_exc:
                if _is_memory_error(inner_exc):
                    _write_parquet_streaming(
                        target_df,
                        p,
                        compression=compression_value,
                        preserve_index=index_flag,
                        chunk_size=_DEFAULT_PARQUET_CHUNK_SIZE,
                        extra_kwargs=write_kwargs,
                    )
                    return
                raise
        raise
    except Exception as exc:
        if _is_memory_error(exc):
            _write_parquet_streaming(
                target_df,
                p,
                compression=compression_value,
                preserve_index=index_flag,
                chunk_size=_DEFAULT_PARQUET_CHUNK_SIZE,
                extra_kwargs=write_kwargs,
            )
            return
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
