#!/usr/bin/env python3
"""
BeefPriceWatch (DAFM) Factory Price Scraper
=============================================
Source: publicapps.agriculture.gov.ie/bpw-api
  - Per-factory, per-category headline prices
  - Per-factory, per-conformation/fat detail prices
  - National aggregate prices

Run bi-weekly via cron:
    0 7 * * 1,4   /path/to/.venv/bin/python /path/to/scrape_factory_prices.py

Appends only new rows to factory_prices.csv.

Output columns:
    scraped_date, source, report_date, week_number, year,
    country, category, grade, factory,
    price_euro_per_kg, price_raw, unit, notes
"""

import csv
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from data_utils import safe_append_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DIR        = Path(__file__).parent
OUTPUT_CSV = DIR / "factory_prices.csv"

CSV_FIELDS = [
    "scraped_date", "source", "report_date", "week_number", "year",
    "country", "category", "grade", "factory",
    "price_euro_per_kg", "price_raw", "unit", "notes",
]

DEDUP_KEY = ("source", "report_date", "category", "grade", "factory")

REQUEST_TIMEOUT = 20
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
}


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_existing() -> set:
    if not OUTPUT_CSV.exists():
        return set()
    seen = set()
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add(tuple(row.get(k, "") for k in DEDUP_KEY))
    return seen


def _append_rows(rows: list[dict], seen: set) -> int:
    """Safely append new rows using atomic write."""
    return safe_append_csv(OUTPUT_CSV, rows, CSV_FIELDS, dedup_key=DEDUP_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
# Price helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cents_to_euro(raw: str) -> str:
    """Convert c/kg value string to €/kg. Handles ranges like '600/670'."""
    try:
        parts  = [float(re.sub(r"[^\d.]", "", p)) for p in str(raw).split("/") if re.search(r"\d", p)]
        if not parts:
            return ""
        mid = sum(parts) / len(parts)
        # If value looks like cents (> 20), divide by 100
        return f"{mid / 100:.4f}" if mid > 20 else f"{mid:.4f}"
    except (ValueError, ZeroDivisionError):
        return ""


def _euro_str_to_float(raw: str) -> str:
    """Strip € symbol and normalise euro/kg string."""
    cleaned = re.sub(r"[€\s]", "", str(raw))
    try:
        return f"{float(cleaned):.4f}"
    except ValueError:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — BeefPriceWatch REST API (DAFM)
# ═══════════════════════════════════════════════════════════════════════════════

BPW_BASE = "https://publicapps.agriculture.gov.ie/bpw-api/api/v1"


def _bpw_get_dates(weeks: int = 12) -> list[str]:
    """Return the most recent `weeks` available week-end dates."""
    try:
        r = requests.get(f"{BPW_BASE}/prices/dates", headers=HEADERS,
                         timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        dates = r.json().get("date", [])
        return dates[:weeks]
    except Exception as e:
        log.warning("BPW dates endpoint failed: %s", e)
        # Fall back: generate last N Saturdays
        today = date.today()
        return [
            (today - timedelta(days=(today.weekday() + 2 - 7 * i) % 7 + 7 * i)
             ).strftime("%Y-%m-%d")
            for i in range(weeks)
        ]


def _bpw_prices_for_range(start: str, end: str) -> list[dict]:
    """Fetch per-factory per-category prices for a date range."""
    rows = []
    try:
        r = requests.get(f"{BPW_BASE}/prices",
                         params={"start": start, "end": end},
                         headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("BPW /prices %s–%s failed: %s", start, end, e)
        return rows

    scraped = str(date.today())
    for item in data:
        factory  = item.get("factory", {}).get("name", "")
        cat      = item.get("category", {})
        cat_name = cat.get("name", "")
        report_d = item.get("dateCreated", end)
        headline = str(item.get("headline", ""))
        headline_euro = _cents_to_euro(headline) if headline else ""

        # Headline row (summary price for this factory × category)
        if headline_euro:
            rows.append({
                "scraped_date":    scraped,
                "source":          "BeefPriceWatch",
                "report_date":     report_d,
                "week_number":     "",
                "year":            report_d[:4],
                "country":         "IE",
                "category":        cat_name,
                "grade":           cat.get("headlineClassification", ""),
                "factory":         factory,
                "price_euro_per_kg": headline_euro,
                "price_raw":       headline,
                "unit":            "c/kg",
                "notes":           "headline",
            })

        # Detail rows (per conformation × fat grade)
        for detail in item.get("details", []):
            conf  = detail.get("conformation", "")
            fat   = detail.get("fat", "")
            grade = f"{conf}/{fat}" if conf or fat else ""
            cent  = str(detail.get("cent", ""))
            count = str(detail.get("count", ""))
            wt    = str(detail.get("weight", ""))
            euro  = _cents_to_euro(cent) if cent else ""
            rows.append({
                "scraped_date":    scraped,
                "source":          "BeefPriceWatch",
                "report_date":     report_d,
                "week_number":     "",
                "year":            report_d[:4],
                "country":         "IE",
                "category":        cat_name,
                "grade":           grade,
                "factory":         factory,
                "price_euro_per_kg": euro,
                "price_raw":       cent,
                "unit":            "c/kg",
                "notes":           f"count={count} avg_wt={wt}kg",
            })

    return rows


def _bpw_national_for_range(start: str, end: str) -> list[dict]:
    """Fetch national aggregate prices for a date range."""
    rows = []
    try:
        r = requests.get(f"{BPW_BASE}/prices/national",
                         params={"start": start, "end": end},
                         headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("BPW /prices/national %s–%s failed: %s", start, end, e)
        return rows

    scraped = str(date.today())
    for item in data:
        cat      = item.get("category", {})
        cat_name = cat.get("name", "")
        report_d = item.get("dateCreated", end)

        for detail in item.get("details", []):
            conf  = detail.get("conformation", "")
            fat   = detail.get("fat", "")
            grade = f"{conf}/{fat}" if conf or fat else ""
            cent  = str(detail.get("cent", ""))
            c_min = str(detail.get("centMin", ""))
            c_max = str(detail.get("centMax", ""))
            count = str(detail.get("count", ""))
            wt    = str(detail.get("weight", ""))
            euro  = _cents_to_euro(cent) if cent else ""
            rows.append({
                "scraped_date":    scraped,
                "source":          "BeefPriceWatch-National",
                "report_date":     report_d,
                "week_number":     "",
                "year":            report_d[:4],
                "country":         "IE",
                "category":        cat_name,
                "grade":           grade,
                "factory":         "National",
                "price_euro_per_kg": euro,
                "price_raw":       cent,
                "unit":            "c/kg",
                "notes":           f"min={c_min} max={c_max} count={count} avg_wt={wt}kg",
            })

    return rows


def scrape_bpw() -> list[dict]:
    """Scrape BeefPriceWatch via its REST API.

    Smart week fetching:
    - If factory_prices.csv exists, fetch only last 3 weeks (safety margin for edits)
    - If new file, fetch 12 weeks (bootstrap)
    """
    # Determine weeks to fetch
    if OUTPUT_CSV.exists():
        weeks_to_fetch = 3  # Just recent weeks — API rarely changes old data
        log.info("BPW: existing data found, fetching recent %d weeks", weeks_to_fetch)
    else:
        weeks_to_fetch = 12  # Bootstrap: get full history
        log.info("BPW: no existing data, bootstrapping with %d weeks", weeks_to_fetch)

    dates = _bpw_get_dates(weeks=weeks_to_fetch)
    if not dates:
        log.warning("BPW: no dates returned")
        return []

    all_rows = []
    log.info("BPW: fetching %d weeks (%s → %s)", len(dates), dates[-1], dates[0])
    for d in dates:
        all_rows.extend(_bpw_prices_for_range(d, d))
        all_rows.extend(_bpw_national_for_range(d, d))
        time.sleep(0.15)

    log.info("BPW: %d rows", len(all_rows))
    return all_rows


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Factory Price Scraper — %s", date.today())
    log.info("=" * 60)

    seen     = _load_existing()
    all_rows = []

    log.info("Scraping BeefPriceWatch (DAFM REST API)...")
    all_rows.extend(scrape_bpw())

    added = _append_rows(all_rows, seen)
    log.info("Done — %d new rows added to %s  (%d collected total)",
             added, OUTPUT_CSV.name, len(all_rows))


if __name__ == "__main__":
    main()
