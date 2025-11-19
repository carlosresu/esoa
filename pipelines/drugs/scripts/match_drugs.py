#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thin orchestration layer that chains feature building, scoring, and outputs."""

from __future__ import annotations
import sys, time
import os, pandas as pd
from typing import Callable

# Local lightweight spinner so this module is self-contained
def _run_with_spinner(label: str, func: Callable[[], None]) -> float:
    """Wrap a callable with a lightweight spinner to show progress inside module-level scripts."""
    import threading
    done = threading.Event()
    err = []

    def worker():
        try:
            func()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"; i = 0
    while not done.is_set():
        elapsed = time.perf_counter() - t0
        sys.stdout.write(f"\r{frames[i % len(frames)]} {elapsed:7.2f}s {label}")
        sys.stdout.flush()
        time.sleep(0.1); i += 1
    th.join()
    elapsed = time.perf_counter() - t0
    sys.stdout.write(f"\r✓ {elapsed:7.2f}s {label}\n")
    sys.stdout.flush()
    if err:
        raise err[0]
    return elapsed

from .match_features_drugs import build_features
from .match_scoring_drugs import score_and_classify
from .match_outputs_drugs import write_outputs


def _assemble_reference_catalogue(annex_df: pd.DataFrame, pnf_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Annex F and PNF catalogues into a single reference frame with priority metadata."""
    annex = annex_df.copy()
    pnf = pnf_df.copy()

    def _prep_annex_synonyms(row: pd.Series) -> str:
        candidates = []
        for field in ("raw_description", "normalized_description", "generic_name"):
            value = row.get(field)
            if isinstance(value, str):
                value = value.strip()
                if value:
                    candidates.append(value)
        seen = set()
        ordered = []
        for cand in candidates:
            if cand not in seen:
                seen.add(cand)
                ordered.append(cand)
        return "|".join(ordered)

    annex_ref = pd.DataFrame(
        {
            "generic_id": annex["drug_code"].astype(str),
            "generic_name": annex["generic_name"].astype(str),
            "synonyms": annex.apply(_prep_annex_synonyms, axis=1),
            "atc_code": [""] * len(annex),
            "route_allowed": annex["route_allowed"].fillna("").astype(str),
            "form_token": annex["form_token"].fillna("").astype(str),
            "dose_kind": annex["dose_kind"],
            "strength": annex["strength"],
            "unit": annex["unit"],
            "per_val": annex["per_val"],
            "per_unit": annex["per_unit"],
            "pct": annex["pct"],
            "strength_mg": annex["strength_mg"],
            "ratio_mg_per_ml": annex["ratio_mg_per_ml"],
            "source": ["annex_f"] * len(annex),
            "source_priority": [1] * len(annex),
            "drug_code": annex["drug_code"].astype(str),
            "route_evidence_reference": annex["route_evidence"].fillna("").astype(str),
        }
    )
    annex_ref["primary_code"] = annex_ref["drug_code"]

    pnf["source"] = "pnf"
    pnf["source_priority"] = 2
    pnf["drug_code"] = ""
    pnf["route_evidence_reference"] = ""
    pnf["primary_code"] = pnf["atc_code"].fillna("").astype(str)

    combined = pd.concat(
        [annex_ref, pnf[
            [
                "generic_id",
                "generic_name",
                "synonyms",
                "atc_code",
                "route_allowed",
                "form_token",
                "dose_kind",
                "strength",
                "unit",
                "per_val",
                "per_unit",
                "pct",
                "strength_mg",
                "ratio_mg_per_ml",
                "source",
                "source_priority",
                "drug_code",
                "primary_code",
                "route_evidence_reference",
            ]
        ]],
        ignore_index=True,
    )
    return combined

def match(
    annex_prepared_csv: str,
    pnf_prepared_csv: str,
    esoa_prepared_csv: str,
    out_csv: str = "esoa_matched.csv",
    *,
    timing_hook: Callable[[str, float], None] | None = None,
    skip_excel: bool = False,
) -> str:
    """Run the feature build, scoring, and output-writing stages on prepared inputs exactly as outlined in pipeline_drugs.md steps 6–15."""
    def _timed(label: str, func: Callable[[], None]) -> float:
        elapsed = _run_with_spinner(label, func)
        if timing_hook:
            timing_hook(label, elapsed)
        return elapsed

    # Load inputs
    # Use small mutable containers so closures can assign to outer scope by reference.
    reference_df = [None]
    esoa_df = [None]
    _timed(
        "Load reference catalogues",
        lambda: reference_df.__setitem__(
            0,
            _assemble_reference_catalogue(
                pd.read_csv(annex_prepared_csv),
                pd.read_csv(pnf_prepared_csv),
            ),
        ),
    )
    _timed("Load eSOA prepared CSV", lambda: esoa_df.__setitem__(0, pd.read_csv(esoa_prepared_csv)))

    # Build features — inner function prints its own sub-spinners; do not show outer spinner.
    # Feeding the matcher-specific feature engineering step ties together all reference data.
    features_df = build_features(reference_df[0], esoa_df[0], timing_hook=timing_hook)

    # Score & classify
    # Prepare container for the scored DataFrame so the closure can mutate it.
    out_df = [None]
    _timed("Score & classify", lambda: out_df.__setitem__(0, score_and_classify(features_df, reference_df[0])))

    # Write outputs — inner module prints its own sub-spinners; do not show outer spinner
    out_path = os.path.abspath(out_csv)
    # Persist the outputs (CSV, XLSX, summaries) while capturing timing metrics.
    write_outputs(out_df[0], out_path, timing_hook=timing_hook, skip_excel=skip_excel)

    return out_path
