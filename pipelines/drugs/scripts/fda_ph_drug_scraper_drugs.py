#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from ..constants import PIPELINE_INPUTS_DIR, PIPELINE_RAW_DIR
from .text_utils_drugs import normalize_text
from .routes_forms_drugs import FORM_TO_ROUTE, parse_form_from_text

BASE_URL = "https://verification.fda.gov.ph"
HUMAN_DRUGS_URL = f"{BASE_URL}/drug_productslist.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; eSOA-BrandMap/1.0; +https://github.com/)"
}


RAW_DIR = PIPELINE_RAW_DIR

AS_OF_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"as of\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4})", re.IGNORECASE),
    re.compile(r"as of\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", re.IGNORECASE),
    re.compile(r"as of\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
)

DATE_FILE_RX = re.compile(r"FDA_PH_DRUGS_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)


def _strip_ordinals(value: str) -> str:
    """Remove ordinal suffixes (1st→1, 2nd→2, …) for reliable datetime parsing."""
    return re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", value)


def _parse_date_candidates(value: str) -> Optional[date]:
    """Try multiple date formats until one succeeds; return None if all fail."""
    cleaned = _strip_ordinals(value.strip())
    for fmt in ("%B %d, %Y", "%B %d %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _extract_as_of_date(html_text: str) -> date:
    """Locate the "as of" date on the landing page and return it as a date object."""
    for pattern in AS_OF_PATTERNS:
        match = pattern.search(html_text)
        if match:
            parsed = _parse_date_candidates(match.group(1))
            if parsed:
                return parsed
    raise RuntimeError("Unable to determine the FDA PH drug catalog 'as of' date.")


def _existing_raw_dates(raw_dir: Path) -> List[Tuple[date, Path]]:
    """Collect already-downloaded raw CSVs and their associated dates."""
    items: List[Tuple[date, Path]] = []
    if not raw_dir.is_dir():
        return items
    for path in raw_dir.glob("FDA_PH_DRUGS_*.csv"):
        match = DATE_FILE_RX.search(path.name)
        if not match:
            continue
        parsed = _parse_date_candidates(match.group(1))
        if parsed:
            items.append((parsed, path))
    items.sort()
    return items


def _parse_csv_rows(lines: Iterable[str]) -> List[Dict[str, str]]:
    """Parse CSV text into a list of dictionaries with trimmed keys/values."""
    reader = csv.DictReader(lines)
    rows: List[Dict[str, str]] = []
    for row in reader:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    if not rows:
        raise RuntimeError("FDA export returned no rows.")
    return rows


def fetch_csv_export() -> Tuple[List[Dict[str, str]], date, Path, bool]:
    """Download (or reuse) the FDA PH drug CSV export.

    Returns:
        rows: Parsed CSV rows.
        catalog_date: Date advertised on the FDA PH site or latest cached date.
        raw_path: File path of the raw CSV backing these rows.
        downloaded: True when a fresh download occurred during this run.
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = _existing_raw_dates(RAW_DIR)
    latest_local_date = existing[-1][0] if existing else None

    with requests.Session() as session:
        landing = session.get(HUMAN_DRUGS_URL, headers=HEADERS, timeout=30)
        landing.raise_for_status()
        as_of = _extract_as_of_date(landing.text)

        target_date = as_of
        downloaded = False

        if latest_local_date and latest_local_date >= as_of:
            # Reuse the most recent cached export when it is at least as new as the site.
            target_date = latest_local_date
            raw_path = RAW_DIR / f"FDA_PH_DRUGS_{target_date.isoformat()}.csv"
            if not raw_path.is_file():
                raise RuntimeError(f"Expected cached raw file {raw_path} is missing.")
            text = raw_path.read_text(encoding="utf-8")
        else:
            raw_path = RAW_DIR / f"FDA_PH_DRUGS_{target_date.isoformat()}.csv"
            if raw_path.is_file():
                text = raw_path.read_text(encoding="utf-8")
            else:
                export = session.get(f"{HUMAN_DRUGS_URL}?export=csv", headers=HEADERS, timeout=120)
                export.raise_for_status()
                text = export.text
                raw_path.write_text(text, encoding="utf-8")
                downloaded = True

    rows = _parse_csv_rows(text.splitlines())
    return rows, target_date, raw_path, downloaded


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

    rows, catalog_date, raw_path, downloaded = fetch_csv_export()
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

    status_note = "downloaded" if downloaded else "reused cached"
    print(f"FDA PH drug catalog ({catalog_date.isoformat()}) {status_note}: raw={raw_path}")


if __name__ == "__main__":
    main()
