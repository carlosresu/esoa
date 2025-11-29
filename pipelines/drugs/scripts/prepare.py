#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-processing helpers that normalize PNF and eSOA inputs for the matcher.

The preparation stage is intentionally verbose because it shoulders the
responsibility of turning raw spreadsheets into a predictable schema used by the
rest of the pipeline.  Rich inline comments call out the expectations we enforce
and why particular derived columns exist so future maintainers do not need to
reverse engineer the data frame mutations.
"""

import os
from pathlib import Path
import pandas as pd

from .routes_forms import map_route_token, parse_form_from_text
from .dose import parse_dose_struct_from_text, to_mg, safe_ratio_mg_per_ml
from .text_utils import (
    clean_atc,
    extract_base_and_salts,
    normalize_text,
    serialize_salt_list,
    slug_id,
)
from .concurrency import maybe_parallel_map


def _calc_strength_mg(payload: tuple[object, object]) -> float | None:
    """Convert arbitrary (strength, unit) payloads into canonical mg where possible."""
    strength, unit = payload
    if pd.notna(strength) and isinstance(unit, str) and unit:
        return to_mg(strength, unit)
    return None


def _calc_ratio_mg_per_ml(payload: tuple[object, object, object, object, object]) -> float | None:
    """Derive mg/mL ratios only when the payload describes a ratio with ml denominator."""
    dose_kind, strength, unit, per_val, per_unit = payload
    if dose_kind == "ratio" and isinstance(per_unit, str) and per_unit.lower() == "ml":
        return safe_ratio_mg_per_ml(strength, unit, per_val)
    return None


def _write_csv_and_parquet(frame: pd.DataFrame, csv_path: str) -> None:
    """Persist a dataframe to CSV and Parquet using the same stem."""
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    parquet_path = Path(csv_path).with_suffix(".parquet")
    frame.to_parquet(parquet_path, index=False)


def prepare(pnf_csv: str, esoa_csv: str, outdir: str = ".") -> tuple[str, str]:
    """Normalize PNF and eSOA inputs, deriving helper columns and writing prepared CSVs."""
    os.makedirs(outdir, exist_ok=True)

    # Load and immediately validate the PNF payload so downstream assumptions
    # remain explicit and testable.
    pnf = pd.read_csv(pnf_csv)
    for col in ["Molecule", "Route", "ATC Code"]:
        if col not in pnf.columns:
            raise ValueError(f"pnf.csv is missing required column: {col}")

    # Canonicalize the identifying columns that later stages depend on.  These
    # are split out early so failures surface before any heavy parsing work.
    molecule_values = pnf["Molecule"].fillna("").astype(str)
    pnf["raw_molecule"] = molecule_values
    pnf["generic_name"] = molecule_values.str.strip().str.upper()
    molecule_list = molecule_values.tolist()
    split_values = maybe_parallel_map(molecule_list, extract_base_and_salts)
    pnf["generic_normalized"] = [
        base or original.strip().upper()
        for (base, _), original in zip(split_values, molecule_list)
    ]
    pnf["salt_form"] = [serialize_salt_list(salts) for _, salts in split_values]
    generic_names = pnf["generic_normalized"].astype(str).tolist()
    pnf["generic_id"] = maybe_parallel_map(generic_names, slug_id)
    pnf["synonyms"] = ""
    route_values = pnf["Route"].fillna("").astype(str).tolist()
    pnf["route_tokens"] = maybe_parallel_map(route_values, map_route_token)
    atc_values = pnf["ATC Code"].fillna("").astype(str).tolist()
    pnf["atc_code"] = maybe_parallel_map(atc_values, clean_atc)

    # Consolidate all textual dose evidence into a single normalized field that
    # the dose parser can read once.  The parser expects clean text, hence the
    # normalization step.
    text_cols = [c for c in ["Technical Specifications", "Specs", "Specification"] if c in pnf.columns]
    pnf["_tech"] = pnf[text_cols[0]].fillna("") if text_cols else ""
    parse_src_raw = (pnf["generic_normalized"].astype(str) + " " + pnf["_tech"].astype(str)).str.strip()
    parse_src_list = parse_src_raw.tolist()
    pnf["_parse_src"] = maybe_parallel_map(parse_src_list, normalize_text)

    # Break the parsed dose payload into explicit columns so the matching stage
    # can work with scalars instead of repeatedly walking nested dictionaries.
    parsed = maybe_parallel_map(pnf["_parse_src"], parse_dose_struct_from_text)
    pnf["dose_kind"] = [d.get("dose_kind") if isinstance(d, dict) else None for d in parsed]
    pnf["strength"] = [d.get("strength") if isinstance(d, dict) else None for d in parsed]
    pnf["unit"] = [d.get("unit") if isinstance(d, dict) else None for d in parsed]
    pnf["per_val"] = [d.get("per_val") if isinstance(d, dict) else None for d in parsed]
    pnf["per_unit"] = [d.get("per_unit") if isinstance(d, dict) else None for d in parsed]
    pnf["pct"] = [d.get("pct") if isinstance(d, dict) else None for d in parsed]
    pnf["form_token"] = maybe_parallel_map(pnf["_parse_src"], parse_form_from_text)

    # Derive canonical strength units for quick equality checks (e.g., mg vs g
    # conversions) and compute ratio helpers where enough information exists.
    strength_inputs = list(zip(pnf["strength"], pnf["unit"]))
    pnf["strength_mg"] = maybe_parallel_map(strength_inputs, _calc_strength_mg)
    ratio_inputs = list(zip(pnf["dose_kind"], pnf["strength"], pnf["unit"], pnf["per_val"], pnf["per_unit"]))
    pnf["ratio_mg_per_ml"] = maybe_parallel_map(ratio_inputs, _calc_ratio_mg_per_ml)

    # Expand the multi-route allowances so each row describes a single canonical
    # route.  This mirrors the matching logic that expects one allowed route per
    # record when validating compatibility.
    exploded = pnf.explode("route_tokens", ignore_index=True)
    exploded.rename(columns={"route_tokens": "route_allowed"}, inplace=True)
    keep = exploded[exploded["generic_name"].astype(bool)].copy()

    pnf_prepared = keep[[
        "generic_id", "generic_name", "generic_normalized", "raw_molecule", "salt_form", "synonyms", "atc_code",
        "route_allowed", "form_token", "dose_kind",
        "strength", "unit", "per_val", "per_unit", "pct",
        "strength_mg", "ratio_mg_per_ml",
    ]].copy()

    pnf_out = os.path.join(outdir, "pnf_prepared.csv")
    _write_csv_and_parquet(pnf_prepared, pnf_out)

    # eSOA preparation is intentionally light-weight: only rename the primary
    # text column but still validate that the source CSV carries it.
    esoa = pd.read_csv(esoa_csv)
    if "DESCRIPTION" not in esoa.columns:
        raise ValueError("esoa.csv is missing required column: DESCRIPTION")
    esoa_prepared = esoa.rename(columns={"DESCRIPTION": "raw_text"}).copy()
    esoa_out = os.path.join(outdir, "esoa_prepared.csv")
    _write_csv_and_parquet(esoa_prepared, esoa_out)

    return pnf_out, esoa_out
