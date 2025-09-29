#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import requests

from scripts.text_utils import normalize_text
from scripts.routes_forms import FORM_TO_ROUTE, parse_form_from_text

BASE_URL = "https://verification.fda.gov.ph"
HUMAN_DRUGS_URL = f"{BASE_URL}/drug_productslist.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; eSOA-BrandMap/1.0; +https://github.com/)"
}


def fetch_csv_export() -> List[Dict[str, str]]:
    """Download the FDA PH drug CSV export and return it as a list of dict rows."""
    with requests.Session() as s:
        r0 = s.get(HUMAN_DRUGS_URL, headers=HEADERS, timeout=30)
        r0.raise_for_status()
        # Request the CSV export endpoint once the session is established.
        r = s.get(f"{HUMAN_DRUGS_URL}?export=csv", headers=HEADERS, timeout=120)
        r.raise_for_status()
        text = r.text

    rows: List[Dict[str, str]] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        # Normalize whitespace around keys and values for consistency.
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    if not rows:
        raise RuntimeError("FDA export returned no rows.")
    return rows


def normalize_columns(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Rename raw FDA columns into predictable snake_case keys."""
    key_map = {
        "Registration Number": "registration_number",
        "Generic Name": "generic_name",
        "Brand Name": "brand_name",
        "Dosage Strength": "dosage_strength",
        "Dosage Form": "dosage_form",
        "Pharmacologic Category": "pharmacologic_category",
        "Manufacturer": "manufacturer",
        "Country of Origin": "country_of_origin",
        "Application Type": "application_type",
        "Issuance Date": "issuance_date",
        "Expiry Date": "expiry_date",
        "Product Information": "product_information",
    }
    out: List[Dict[str, str]] = []
    for r in rows:
        nr: Dict[str, str] = {}
        for k, v in r.items():
            kk = key_map.get(k, k.lower().replace(" ", "_"))
            # Map or fallback to a deterministic snake_case key.
            nr[kk] = v
        out.append(nr)
    return out


def infer_form_and_route(dosage_form: Optional[str]) -> (Optional[str], Optional[str]):
    """Derive normalized form and route tokens from the FDA dosage form string."""
    if not isinstance(dosage_form, str) or not dosage_form.strip():
        return None, None
    norm = normalize_text(dosage_form)
    form_token = parse_form_from_text(norm)
    route = FORM_TO_ROUTE.get(form_token) if form_token else None
    return form_token, route


def build_brand_map(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Trim, dedupe, and enrich FDA rows for downstream brand→generic lookups."""
    seen = set()
    out: List[Dict[str, str]] = []
    for r in rows:
        brand = (r.get("brand_name") or "").strip()
        generic = (r.get("generic_name") or "").strip()
        if not brand or not generic:
            continue

        dosage_form = (r.get("dosage_form") or "").strip()
        dosage_strength = (r.get("dosage_strength") or "").strip()
        regno = (r.get("registration_number") or "").strip()

        form_token, route = infer_form_and_route(dosage_form)

        key = (
            brand.lower(),
            generic.lower(),
            (form_token or "").lower(),
            (route or "").lower(),
            dosage_strength.lower(),
        )
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "brand_name": brand,
                "generic_name": generic,
                "dosage_form": form_token or dosage_form or "",
                "route": route or "",
                "dosage_strength": dosage_strength,
                "registration_number": regno,
            }
        )
    return out


def main() -> None:
    """CLI wrapper that downloads, normalizes, and writes the brand-map export."""
    ap = argparse.ArgumentParser(description="Build FDA PH brand→generic map (CSV export only)")
    ap.add_argument("--outdir", default="inputs", help="Output directory")
    ap.add_argument("--outfile", default=None, help="Optional explicit output CSV filename")
    args = ap.parse_args()

    rows = fetch_csv_export()
    rows = normalize_columns(rows)
    brand_map = build_brand_map(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.outfile) if args.outfile else outdir / f"fda_brand_map_{Path.cwd().name}.csv"
    # Use explicit filename if provided; otherwise the caller supplies one via subprocess args

    fieldnames = ["brand_name", "generic_name", "dosage_form", "route", "dosage_strength", "registration_number"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Stream rows to disk preserving the deduped ordering.
        w.writerows(brand_map)


if __name__ == "__main__":
    main()
