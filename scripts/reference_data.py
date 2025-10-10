#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightweight loaders for reusable reference datasets (DrugBank generics, ignore-word lists)."""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from .text_utils import _normalize_text_basic

_TOKEN_RX = re.compile(r"[a-z]+")


def _project_root(project_root: str | Path | None = None) -> Path:
    """Resolve the repository root so loaders work regardless of caller cwd."""
    if project_root is None:
        return Path(__file__).resolve().parent.parent
    return Path(project_root).resolve()


def _iter_csv_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Iterable[str]:
    """Yield trimmed strings from the first matching column name, or the first column as fallback."""
    selected: List[str] = []
    for name in candidates:
        if name in frame.columns:
            selected = frame[name].dropna().astype(str).tolist()
            break
    if not selected and len(frame.columns):
        selected = frame.iloc[:, 0].dropna().astype(str).tolist()
    for value in selected:
        clean = value.strip()
        if clean:
            yield clean


@lru_cache(maxsize=None)
def load_drugbank_generics(project_root: str | Path | None = None) -> Tuple[Set[str], Set[str], Dict[str, Set[Tuple[str, ...]]]]:
    """
    Load DrugBank generics (prefer the freshly exported dependencies/drugbank/output/generics.csv).

    Returns:
        - normalized_names: Unique normalized generic phrases (lowercase, punctuation-stripped).
        - token_pool: All individual tokens present across the normalized names.
        - token_index: first-token -> set of token tuples representing each generic phrase.
    """
    root = _project_root(project_root)
    candidates = [
        root / "dependencies" / "drugbank" / "output" / "generics.csv",
        root / "inputs" / "generics.csv",
    ]

    normalized_names: Set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path, dtype=str)
        except Exception:
            continue
        for raw in _iter_csv_column(frame, ("name", "generic")):
            norm = _normalize_text_basic(raw)
            if norm:
                normalized_names.add(norm)
        if normalized_names:
            # Prefer the first successfully loaded dataset (dependencies path takes precedence).
            break

    token_pool: Set[str] = set()
    token_index: Dict[str, Set[Tuple[str, ...]]] = {}
    for name in normalized_names:
        tokens = tuple(name.split())
        if not tokens:
            continue
        token_pool.update(tokens)
        first = tokens[0]
        bucket = token_index.setdefault(first, set())
        bucket.add(tokens)
    return normalized_names, token_pool, token_index


DEFAULT_IGNORE_TOKENS: Set[str] = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "with",
}

_IGNORE_FILENAMES: Tuple[str, ...] = (
    "english_words.txt",
    "english_words.csv",
    "stopwords.txt",
    "stopwords.csv",
    "ignore_words.txt",
    "ignore_words.csv",
)


def _iter_txt_words(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            word = line.strip()
            if not word or word.startswith("#"):
                continue
            yield word


@lru_cache(maxsize=None)
def load_ignore_words(project_root: str | Path | None = None) -> Set[str]:
    """
    Load caller-provided stopwords / English words that should never surface as unknown tokens.

    Returns a set of lowercase alphanumeric tokens.
    """
    root = _project_root(project_root)
    inputs_dir = root / "inputs"

    tokens: Set[str] = set(DEFAULT_IGNORE_TOKENS)

    def _consume(words: Iterable[str]) -> None:
        for raw in words:
            norm = _normalize_text_basic(raw)
            if not norm:
                continue
            for match in _TOKEN_RX.findall(norm):
                if match:
                    tokens.add(match)

    for filename in _IGNORE_FILENAMES:
        path = inputs_dir / filename
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".txt":
                _consume(_iter_txt_words(path))
            else:
                frame = pd.read_csv(path, dtype=str)
                _consume(_iter_csv_column(frame, ("word", "token", "value")))
        except Exception:
            continue

    return tokens
