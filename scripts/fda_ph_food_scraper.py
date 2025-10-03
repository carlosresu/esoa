#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

BASE_URL = "https://verification.fda.gov.ph"
FOOD_PRODUCTS_URL = f"{BASE_URL}/All_FoodProductslist.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; eSOA-BrandMap/1.0; +https://github.com/)"
}

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

DEFAULT_TIMEOUT = 300  # Food export is noticeably slower; allow up to 5 minutes.


def _parse_csv_rows(lines: Iterable[str]) -> List[Dict[str, str]]:
    reader = csv.DictReader(lines)
    rows: List[Dict[str, str]] = []
    for row in reader:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    if not rows:
        raise RuntimeError("FDA PH food export returned no rows.")
    return rows


def fetch_food_csv(timeout: int = DEFAULT_TIMEOUT) -> str:
    """Download the raw FDA PH food CSV export."""
    with requests.Session() as session:
        # Establish the session to mimic the behaviour used by the drug scraper.
        session.get(FOOD_PRODUCTS_URL, headers=HEADERS, timeout=30)
        response = session.get(f"{FOOD_PRODUCTS_URL}?export=csv", headers=HEADERS, timeout=timeout)
        response.raise_for_status()
    return response.text


def normalize_columns(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    key_map = {
        "Brand Name": "brand_name",
        "Product Name": "product_name",
        "Company Name": "company_name",
        "Registration Number": "registration_number",
    }
    out: List[Dict[str, str]] = []
    for row in rows:
        normalized: Dict[str, str] = {}
        for key, value in row.items():
            normalized[key_map.get(key, key.lower().replace(" ", "_"))] = value
        out.append(normalized)
    return out


def build_catalog(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Trim to the required columns and deduplicate entries."""
    required = ("brand_name", "product_name", "company_name", "registration_number")
    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for row in rows:
        brand = (row.get("brand_name") or "").strip()
        product = (row.get("product_name") or "").strip()
        company = (row.get("company_name") or "").strip()
        reg = (row.get("registration_number") or "").strip()
        if not brand and not product:
            continue
        key = (brand.lower(), product.lower(), company.lower(), reg.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "brand_name": brand,
            "product_name": product,
            "company_name": company,
            "registration_number": reg,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the FDA Philippines food product catalog. Disabled by default because "
            "the source page does not publish an 'as of' date; rerun with --enable-download "
            "to fetch the CSV (allow up to 5 minutes for the export)."
        )
    )
    parser.add_argument("--enable-download", action="store_true", help="Actually download the CSV export")
    parser.add_argument("--outdir", default="inputs", help="Directory where the processed CSV will be written")
    parser.add_argument("--outfile", default="fda_food_products.csv", help="Processed output filename")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout (seconds) for the CSV export request")
    args = parser.parse_args()

    if not args.enable_download:
        print("Download skipped: the FDA PH food catalog lacks an 'as of' date. Pass --enable-download to fetch.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    csv_text = fetch_food_csv(timeout=args.timeout)
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    raw_path = RAW_DIR / f"FDA_PH_FOOD_PRODUCTS_{timestamp}.csv"
    raw_path.write_text(csv_text, encoding="utf-8")

    rows = _parse_csv_rows(csv_text.splitlines())
    rows = normalize_columns(rows)
    catalog = build_catalog(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / args.outfile

    fieldnames = ["brand_name", "product_name", "company_name", "registration_number"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog)

    print(
        "FDA PH food catalog downloaded (raw={raw}, processed={processed}, entries={count})".format(
            raw=raw_path,
            processed=out_csv,
            count=len(catalog),
        )
    )


if __name__ == "__main__":
    main()

