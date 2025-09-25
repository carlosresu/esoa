#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

from .routes_forms import map_route_token, parse_form_from_text
from .dose import parse_dose_struct_from_text, to_mg, safe_ratio_mg_per_ml
from .text_utils import clean_atc, normalize_text, slug_id


def prepare(pnf_csv: str, esoa_csv: str, outdir: str = ".") -> tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)

    pnf = pd.read_csv(pnf_csv)
    for col in ["Molecule", "Route", "ATC Code"]:
        if col not in pnf.columns:
            raise ValueError(f"pnf.csv is missing required column: {col}")

    pnf["generic_name"] = pnf["Molecule"].fillna("").astype(str)
    pnf["generic_id"] = pnf["generic_name"].map(slug_id)
    pnf["synonyms"] = ""
    pnf["route_tokens"] = pnf["Route"].map(map_route_token)
    pnf["atc_code"] = pnf["ATC Code"].map(clean_atc)

    text_cols = [c for c in ["Technical Specifications", "Specs", "Specification"] if c in pnf.columns]
    pnf["_tech"] = pnf[text_cols[0]].fillna("") if text_cols else ""
    pnf["_parse_src"] = (pnf["generic_name"].astype(str) + " " + pnf["_tech"].astype(str)).str.strip().map(normalize_text)

    parsed = pnf["_parse_src"].map(parse_dose_struct_from_text)
    pnf["dose_kind"] = parsed.map(lambda d: d.get("dose_kind"))
    pnf["strength"] = parsed.map(lambda d: d.get("strength"))
    pnf["unit"] = parsed.map(lambda d: d.get("unit"))
    pnf["per_val"] = parsed.map(lambda d: d.get("per_val"))
    pnf["per_unit"] = parsed.map(lambda d: d.get("per_unit"))
    pnf["pct"] = parsed.map(lambda d: d.get("pct"))
    pnf["form_token"] = pnf["_parse_src"].map(parse_form_from_text)

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

    esoa = pd.read_csv(esoa_csv)
    if "DESCRIPTION" not in esoa.columns:
        raise ValueError("esoa.csv is missing required column: DESCRIPTION")
    esoa_prepared = esoa.rename(columns={"DESCRIPTION": "raw_text"}).copy()
    esoa_out = os.path.join(outdir, "esoa_prepared.csv")
    esoa_prepared.to_csv(esoa_out, index=False, encoding="utf-8")

    return pnf_out, esoa_out
