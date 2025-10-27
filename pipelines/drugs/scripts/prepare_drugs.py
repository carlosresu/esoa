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
import pandas as pd

from .routes_forms_drugs import map_route_token, parse_form_from_text
from .dose_drugs import parse_dose_struct_from_text, to_mg, safe_ratio_mg_per_ml
from .text_utils_drugs import clean_atc, normalize_text, slug_id


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
    pnf["generic_name"] = pnf["Molecule"].fillna("").astype(str)
    pnf["generic_id"] = pnf["generic_name"].map(slug_id)
    pnf["synonyms"] = ""
    pnf["route_tokens"] = pnf["Route"].map(map_route_token)
    pnf["atc_code"] = pnf["ATC Code"].map(clean_atc)

    # Consolidate all textual dose evidence into a single normalized field that
    # the dose parser can read once.  The parser expects clean text, hence the
    # normalization step.
    text_cols = [c for c in ["Technical Specifications", "Specs", "Specification"] if c in pnf.columns]
    pnf["_tech"] = pnf[text_cols[0]].fillna("") if text_cols else ""
    pnf["_parse_src"] = (pnf["generic_name"].astype(str) + " " + pnf["_tech"].astype(str)).str.strip().map(normalize_text)

    # Break the parsed dose payload into explicit columns so the matching stage
    # can work with scalars instead of repeatedly walking nested dictionaries.
    parsed = pnf["_parse_src"].map(parse_dose_struct_from_text)
    pnf["dose_kind"] = parsed.map(lambda d: d.get("dose_kind"))
    pnf["strength"] = parsed.map(lambda d: d.get("strength"))
    pnf["unit"] = parsed.map(lambda d: d.get("unit"))
    pnf["per_val"] = parsed.map(lambda d: d.get("per_val"))
    pnf["per_unit"] = parsed.map(lambda d: d.get("per_unit"))
    pnf["pct"] = parsed.map(lambda d: d.get("pct"))
    pnf["form_token"] = pnf["_parse_src"].map(parse_form_from_text)

    # Derive canonical strength units for quick equality checks (e.g., mg vs g
    # conversions) and compute ratio helpers where enough information exists.
    pnf["strength_mg"] = pnf.apply(
        lambda r: to_mg(r.get("strength"), r.get("unit"))
        if (pd.notna(r.get("strength")) and isinstance(r.get("unit"), str) and r.get("unit"))
        else None,
        axis=1,
    )
    pnf["ratio_mg_per_ml"] = pnf.apply(
        lambda r: safe_ratio_mg_per_ml(r.get("strength"), r.get("unit"), r.get("per_val"))
        if (r.get("dose_kind") == "ratio" and str(r.get("per_unit")).lower() == "ml")
        else None,
        axis=1,
    )

    # Expand the multi-route allowances so each row describes a single canonical
    # route.  This mirrors the matching logic that expects one allowed route per
    # record when validating compatibility.
    exploded = pnf.explode("route_tokens", ignore_index=True)
    exploded.rename(columns={"route_tokens": "route_allowed"}, inplace=True)
    keep = exploded[exploded["generic_name"].astype(bool)].copy()

    pnf_prepared = keep[[
        "generic_id", "generic_name", "synonyms", "atc_code",
        "route_allowed", "form_token", "dose_kind",
        "strength", "unit", "per_val", "per_unit", "pct",
        "strength_mg", "ratio_mg_per_ml",
    ]].copy()

    pnf_out = os.path.join(outdir, "pnf_prepared.csv")
    pnf_prepared.to_csv(pnf_out, index=False, encoding="utf-8")

    # eSOA preparation is intentionally light-weight: only rename the primary
    # text column but still validate that the source CSV carries it.
    esoa = pd.read_csv(esoa_csv)
    if "DESCRIPTION" not in esoa.columns:
        raise ValueError("esoa.csv is missing required column: DESCRIPTION")
    esoa_prepared = esoa.rename(columns={"DESCRIPTION": "raw_text"}).copy()
    esoa_out = os.path.join(outdir, "esoa_prepared.csv")
    esoa_prepared.to_csv(esoa_out, index=False, encoding="utf-8")

    return pnf_out, esoa_out
