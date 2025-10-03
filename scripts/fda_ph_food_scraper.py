#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
from urllib.parse import parse_qs, urljoin, urlparse
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

BASE_URL = "https://verification.fda.gov.ph"
FOOD_PRODUCTS_URL = f"{BASE_URL}/All_FoodProductslist.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; eSOA-BrandMap/1.0; +https://github.com/)"
}

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

DEFAULT_TIMEOUT = 300  # Allow up to 5 minutes; page render is notably slow.

RECORD_SUMMARY_RX = re.compile(r"Records\s+\d+\s+to\s+([\d,]+)\s+of\s+([\d,]+)", re.IGNORECASE)


class _FoodTableParser(HTMLParser):
    """Minimal HTML parser for the FDA food products grid."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.in_tbody = False
        self.in_row = False
        self.in_cell = False
        self.current_row: List[str] = []
        self.current_text: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("id") == "tbl_All_FoodProductslist":
            self.in_table = True
            return
        if not self.in_table:
            return
        if tag == "tbody":
            self.in_tbody = True
            return
        if self.in_tbody and tag == "tr":
            self.in_row = True
            self.current_row = []
            return
        if self.in_row and tag == "td":
            self.in_cell = True
            self.current_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self.in_table:
            self.in_table = False
            self.in_tbody = False
            self.in_row = False
            self.in_cell = False
            return
        if not self.in_table:
            return
        if tag == "tbody":
            self.in_tbody = False
            return
        if tag == "tr" and self.in_row:
            self.in_row = False
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []
            return
        if tag == "td" and self.in_cell:
            text = "".join(self.current_text).strip()
            self.current_row.append(text)
            self.in_cell = False
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_text.append(data)


def _parse_record_summary(html: str) -> Optional[int]:
    match = RECORD_SUMMARY_RX.search(html)
    if not match:
        return None
    total = match.group(2).replace(",", "")
    try:
        return int(total)
    except ValueError:
        return None


def _parse_food_rows(html: str) -> List[Dict[str, str]]:
    parser = _FoodTableParser()
    parser.feed(html)
    parsed: List[Dict[str, str]] = []
    for cells in parser.rows:
        if len(cells) < 5:
            continue
        parsed.append(
            {
                "registration_number": cells[1].strip(),
                "company_name": cells[2].strip(),
                "product_name": cells[3].strip(),
                "brand_name": cells[4].strip(),
            }
        )
    return parsed


def _dedupe_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[Tuple[str, str, str, str]] = set()
    unique: List[Dict[str, str]] = []
    for row in rows:
        key = (
            (row.get("brand_name") or "").strip().lower(),
            (row.get("product_name") or "").strip().lower(),
            (row.get("company_name") or "").strip().lower(),
            (row.get("registration_number") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _fetch_page(session: requests.Session, *, recperpage: str, start: Optional[int], timeout: int) -> str:
    params: Dict[str, str] = {"recperpage": recperpage}
    if start is not None:
        params["start"] = str(start)
    response = session.get(FOOD_PRODUCTS_URL, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def _extract_next_start_values(html: str, *, current: int) -> List[int]:
    values: set[int] = set()
    for href in re.findall(r"href=\"([^\"]+)\"", html, flags=re.IGNORECASE):
        if "All_FoodProductslist.php" not in href:
            continue
        parsed = urlparse(urljoin(FOOD_PRODUCTS_URL, href))
        query = parse_qs(parsed.query)
        rec_vals = query.get("recperpage")
        if rec_vals and rec_vals[0].lower() != "100":
            continue
        for start_val in query.get("start", []):
            try:
                num = int(start_val)
            except ValueError:
                continue
            if num > current:
                values.add(num)
    return sorted(values)


def _scrape_all_mode(session: requests.Session, timeout: int) -> Tuple[List[Dict[str, str]], Optional[int], str]:
    html = _fetch_page(session, recperpage="ALL", start=None, timeout=timeout)
    rows = _parse_food_rows(html)
    total = _parse_record_summary(html)
    return rows, total, html


def _scrape_paginated(session: requests.Session, timeout: int, expected_total: Optional[int]) -> Tuple[List[Dict[str, str]], List[str]]:
    aggregated: List[Dict[str, str]] = []
    html_pages: List[str] = []
    start = 1
    previous_first_reg: Optional[str] = None

    while True:
        html = _fetch_page(session, recperpage="100", start=start, timeout=timeout)
        html_pages.append(html)
        rows = _parse_food_rows(html)
        if not rows:
            break

        first_reg = rows[0].get("registration_number")
        if previous_first_reg is not None and first_reg == previous_first_reg:
            # Guard against being stuck on the same page.
            break
        previous_first_reg = first_reg

        aggregated.extend(rows)
        aggregated = _dedupe_rows(aggregated)

        if expected_total is not None and len(aggregated) >= expected_total:
            break

        if len(rows) < 100:
            break

        next_candidates = _extract_next_start_values(html, current=start)
        if next_candidates:
            start = min(next_candidates)
        else:
            start += len(rows)

    return aggregated, html_pages


def scrape_food_catalog(timeout: int = DEFAULT_TIMEOUT) -> Tuple[List[Dict[str, str]], List[str], bool]:
    with requests.Session() as session:
        rows_all, total_expected, html_all = _scrape_all_mode(session, timeout)
        html_pages = [html_all]
        loaded_all = bool(total_expected) and len(rows_all) == total_expected

        if loaded_all:
            return _dedupe_rows(rows_all), html_pages, True

        paginated_rows, extra_html = _scrape_paginated(session, timeout, total_expected)
        if paginated_rows:
            html_pages.extend(extra_html)
            return _dedupe_rows(paginated_rows), html_pages, False

        if rows_all:
            # We managed to scrape at least something from the ALL page.
            return _dedupe_rows(rows_all), html_pages, False

        raise RuntimeError("Unable to scrape FDA PH food products list (no rows captured).")


def build_catalog(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    for row in rows:
        brand = (row.get("brand_name") or "").strip()
        product = (row.get("product_name") or "").strip()
        company = (row.get("company_name") or "").strip()
        reg = (row.get("registration_number") or "").strip()
        if not (brand or product or company or reg):
            continue
        filtered.append(
            {
                "brand_name": brand,
                "product_name": product,
                "company_name": company,
                "registration_number": reg,
            }
        )
    return _dedupe_rows(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the FDA Philippines food product catalog by scraping the verification portal. "
            "The site does not publish an 'as of' date and the CSV export is unreliable, so we "
            "prefer the ALL view and fall back to 100-row pagination when necessary."
        )
    )
    parser.add_argument("--enable-download", action="store_true", help="Actually perform the scrape")
    parser.add_argument("--outdir", default="inputs", help="Directory for the processed CSV")
    parser.add_argument("--outfile", default="fda_food_products.csv", help="Processed output filename")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout (seconds)")
    args = parser.parse_args()

    if not args.enable_download:
        print("Download skipped: pass --enable-download to scrape the FDA PH food catalog.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows, html_pages, used_all = scrape_food_catalog(timeout=args.timeout)
    catalog = build_catalog(rows)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    raw_suffix = "ALL" if used_all else "PAGED"
    raw_path = RAW_DIR / f"FDA_PH_FOOD_PRODUCTS_{raw_suffix}_{timestamp}.html"
    raw_path.write_text(html_pages[0], encoding="utf-8")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / args.outfile

    fieldnames = ["brand_name", "product_name", "company_name", "registration_number"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog)

    mode = "ALL view" if used_all else "paginated fallback"
    print(
        "FDA PH food catalog scraped via {mode} (raw={raw}, processed={processed}, entries={count})".format(
            mode=mode,
            raw=raw_path,
            processed=out_csv,
            count=len(catalog),
        )
    )


if __name__ == "__main__":
    main()
