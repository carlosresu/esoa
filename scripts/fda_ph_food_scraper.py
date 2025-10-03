#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import requests

BASE_URL = "https://verification.fda.gov.ph"
FOOD_PRODUCTS_URL = f"{BASE_URL}/All_FoodProductslist.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; eSOA-BrandMap/1.0; +https://github.com/)"
}

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

PAGE_SIZES = ("100",)
DEFAULT_TIMEOUT = 300

RECORD_SUMMARY_RX = re.compile(r"Records\s+\d+\s+to\s+([\d,]+)\s+of\s+([\d,]+)", re.IGNORECASE)


def _render_progress(prefix: str, current: int, total: Optional[int], *, done: bool = False) -> None:
    if total and total > 0:
        msg = f"{prefix}: {current:,}/{total:,}"
    else:
        msg = f"{prefix}: {current:,}"
    sys.stdout.write("\r" + msg.ljust(80))
    sys.stdout.flush()
    if done:
        sys.stdout.write("\n")
        sys.stdout.flush()


class _FoodTableParser(HTMLParser):
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


def _row_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        (row.get("brand_name") or "").strip().lower(),
        (row.get("product_name") or "").strip().lower(),
        (row.get("company_name") or "").strip().lower(),
        (row.get("registration_number") or "").strip().lower(),
    )


def _cells_to_rows(cells_list: Iterable[List[str]]) -> List[Dict[str, str]]:
    parsed: List[Dict[str, str]] = []
    for cells in cells_list:
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


def _parse_record_summary(html: str) -> Optional[int]:
    match = RECORD_SUMMARY_RX.search(html)
    if not match:
        return None
    total = match.group(2).replace(",", "")
    try:
        return int(total)
    except ValueError:
        return None


def _parse_food_rows(html: str) -> Tuple[List[Dict[str, str]], Optional[int]]:
    parser = _FoodTableParser()
    parser.feed(html)
    return _cells_to_rows(parser.rows), _parse_record_summary(html)


def _dedupe_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[Tuple[str, str, str, str]] = set()
    unique: List[Dict[str, str]] = []
    for row in rows:
        key = _row_key(row)
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


def _extract_next_start_values(html: str, *, current: int, recperpage: str) -> List[int]:
    values: set[int] = set()
    for href in re.findall(r"href=\"([^\"]+)\"", html, flags=re.IGNORECASE):
        if "All_FoodProductslist.php" not in href:
            continue
        parsed = urlparse(urljoin(FOOD_PRODUCTS_URL, href))
        query = parse_qs(parsed.query)
        if query.get("recperpage", [""])[0].lower() != recperpage.lower():
            continue
        for start_val in query.get("start", []):
            try:
                num = int(start_val)
            except ValueError:
                continue
            if num > current:
                values.add(num)
    return sorted(values)


def _scrape_paginated(
    session: requests.Session,
    recperpage: str,
    timeout: int,
    expected_total: Optional[int],
) -> Tuple[List[Dict[str, str]], List[str]]:
    aggregated: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    html_pages: List[str] = []
    start = 1
    previous_first_reg: Optional[str] = None
    last_report = time.monotonic()

    while True:
        html = _fetch_page(session, recperpage=recperpage, start=start, timeout=timeout)
        html_pages.append(html)
        rows, total = _parse_food_rows(html)
        if expected_total is None and total is not None:
            expected_total = total
        if not rows:
            break

        first_reg = rows[0].get("registration_number")
        if previous_first_reg is not None and first_reg == previous_first_reg:
            break
        previous_first_reg = first_reg

        for row in rows:
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            aggregated.append(row)

        now = time.monotonic()
        if now - last_report >= 1:
            _render_progress(f"Scraping recperpage={recperpage}", len(aggregated), expected_total)
            last_report = now

        if expected_total is not None and len(aggregated) >= expected_total:
            break

        if len(rows) < int(recperpage):
            break

        next_candidates = _extract_next_start_values(html, current=start, recperpage=recperpage)
        if next_candidates:
            start = min(next_candidates)
        else:
            start += len(rows)

    if aggregated:
        _render_progress(f"Scraping recperpage={recperpage}", len(aggregated), expected_total, done=True)

    return aggregated, html_pages


def scrape_food_catalog(timeout: int = DEFAULT_TIMEOUT) -> Tuple[List[Dict[str, str]], List[str], str]:
    with requests.Session() as session:
        expected_total: Optional[int] = None
        for size in PAGE_SIZES:
            rows, html_pages = _scrape_paginated(session, size, timeout, expected_total)
            if rows:
                expected_total = expected_total or _parse_record_summary("".join(html_pages))
                if expected_total is None or len(rows) >= expected_total:
                    return _dedupe_rows(rows), html_pages, size
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape the FDA Philippines food product catalog by iterating paginated views (1000 first, then 100)."
        )
    )
    parser.add_argument("--outdir", default="inputs", help="Directory for the processed CSV")
    parser.add_argument("--outfile", default="fda_food_products.csv", help="Processed output filename")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout (seconds)")
    parser.add_argument("--force", action="store_true", help="Force re-scraping even if output exists")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    out_csv = outdir / args.outfile
    if out_csv.exists() and not args.force:
        print(f"Existing catalog found at {out_csv}; use --force to refresh.")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    rows, html_pages, page_size = scrape_food_catalog(timeout=args.timeout)
    catalog = build_catalog(rows)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    raw_path = RAW_DIR / f"FDA_PH_FOOD_PRODUCTS_{page_size}_{timestamp}.html"
    raw_content = "\n<!-- page break -->\n".join(html_pages) if html_pages else ""
    raw_path.write_text(raw_content, encoding="utf-8")

    fieldnames = ["brand_name", "product_name", "company_name", "registration_number"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog)

    print(
        "FDA PH food catalog scraped via recperpage={page} (raw={raw}, processed={processed}, entries={count})".format(
            page=page_size,
            raw=raw_path,
            processed=out_csv,
            count=len(catalog),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
