#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import ahocorasick  # type: ignore
import pandas as pd

from .text_utils import _base_name, _normalize_text_basic, normalize_compact


@dataclass
class BrandMatch:
    """Structured payload describing an FDA brand and its normalized attributes."""
    brand: str
    generic: str
    dosage_form: str
    route: str
    dosage_strength: str


def _latest_brandmap_path(inputs_dir: str) -> Optional[str]:
    """Return the newest brand map file path, supporting legacy naming schemes."""
    # Prefer renamed pattern
    pattern_new = os.path.join(inputs_dir, "fda_brand_map_*.csv")
    candidates = glob.glob(pattern_new)
    if not candidates:
        # Backward-compatibility with old name
        pattern_old = os.path.join(inputs_dir, "brand_map_*.csv")
        candidates = glob.glob(pattern_old)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def load_latest_brandmap(inputs_dir: str) -> Optional[pd.DataFrame]:
    """Load the most recent brand map CSV into a dataframe with expected columns present."""
    path = _latest_brandmap_path(inputs_dir)
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        for c in ["brand_name", "generic_name", "dosage_form", "route", "dosage_strength"]:
            if c not in df.columns:
                df[c] = ""
        return df
    except Exception:
        return None


def build_brand_automata(brand_df: pd.DataFrame) -> Tuple[ahocorasick.Automaton, ahocorasick.Automaton, Dict[str, List[BrandMatch]]]:
    """Compile Aho–Corasick automatons and lookup tables for brand→generic substitutions."""
    A_norm = ahocorasick.Automaton()
    A_comp = ahocorasick.Automaton()
    mapping: Dict[str, List[BrandMatch]] = {}
    seen_norm = set()
    seen_comp = set()
    for _, r in brand_df.iterrows():
        b = str(r.get("brand_name") or "").strip()
        g = str(r.get("generic_name") or "").strip()
        if not b or not g:
            continue
        dosage_form = str(r.get("dosage_form") or "").strip()
        route = str(r.get("route") or "").strip()
        dosage_strength = str(r.get("dosage_strength") or "").strip()
        bn = _normalize_text_basic(_base_name(b))
        bc = normalize_compact(b)
        if not bn:
            continue
        mapping.setdefault(bn, []).append(BrandMatch(
            brand=b, generic=g, dosage_form=dosage_form, route=route, dosage_strength=dosage_strength
        ))
        if bn not in seen_norm:
            A_norm.add_word(bn, bn); seen_norm.add(bn)
        if bc and bc not in seen_comp:
            A_comp.add_word(bc, bn); seen_comp.add(bc)
    A_norm.make_automaton()
    A_comp.make_automaton()
    return A_norm, A_comp, mapping


def scan_brands(text_norm: str, text_comp: str,
                A_norm: ahocorasick.Automaton,
                A_comp: ahocorasick.Automaton) -> List[str]:
    """Return normalized brand keys detected via either normalized or compact searches."""
    found: Dict[str, int] = {}
    for _, bn in A_norm.iter(text_norm):
        found[bn] = max(found.get(bn, 0), len(bn))
    for _, bn in A_comp.iter(text_comp):
        found[bn] = max(found.get(bn, 0), len(bn))
    return [k for k, _ in sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))]


def fda_generics_set(brand_df: pd.DataFrame) -> Set[str]:
    """Return a set of normalized base generic names present in FDA brand map."""
    gens: Set[str] = set()
    if not isinstance(brand_df, pd.DataFrame) or "generic_name" not in brand_df.columns:
        return gens
    for g in brand_df["generic_name"].fillna("").astype(str).tolist():
        gb = _normalize_text_basic(_base_name(g))
        if gb:
            gens.add(gb)
    return gens
