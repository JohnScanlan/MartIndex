#!/usr/bin/env python3
"""
MartBids.ie Daily Sold Lots Scraper (with per-animal detail)
=============================================================
Fetches sold lot data + individual animal health/quality records
from all marts via the REST API. Run once daily via cron.

Appends only new lots to sold_lots.csv (val_code is the unique key).
Skips marts where nothing has changed since the last run.

Usage:
    python3 martbids_scraper.py           # normal daily run
    python3 martbids_scraper.py --reset   # wipe CSV and re-scrape everything

Output: sold_lots.csv
"""

import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# ── Configuration ─────────────────────────────────────────────────────────────
EMAIL    = "johnscanlan52@yahoo.ie"
PASSWORD = "HaroldsCross1!"

API_BASE  = "https://bidding.martbids.ie/martbidding/v1"
LOGIN_URL = f"{API_BASE}/receiverapp-v5.php"
AWS_URL   = f"{API_BASE}/awstempcalls.php"

OUTPUT_CSV = Path(__file__).parent / "sold_lots.csv"

# Parallel workers for per-lot detail fetches (keep polite — don't go above 10)
DETAIL_WORKERS  = 8
REQUEST_TIMEOUT = 20
MART_DELAY      = 0.3   # seconds between mart-level API calls
RETRY_ATTEMPTS  = 3
RETRY_DELAY     = 5

CSV_FIELDS = [
    # ── Basic sold data ──────────────────────────────────────────
    "mart", "mart_id", "sale_id",
    "lot", "breed", "sex", "weight", "price", "age_months",
    # ── Per-animal detail (LOTCONTENTjson) ──────────────────────
    "dob", "dam_breed", "days_in_herd",
    "tb_test", "herd_tb_test", "export_status",
    "no_of_owners", "quality_assurance", "bvd_tested",
    "price_per_kg",
    # ── ICBF sub-fields ──────────────────────────────────────────
    "icbf_genomic_eval",
    "icbf_cbv",
    "icbf_across_breed",
    "icbf_replacement_index",
    "icbf_ebi",
    "icbf_milk",
    "icbf_fertility",
    "icbf_terminal_index",
    # ── Metadata ─────────────────────────────────────────────────
    "val_code", "scraped_date",
]

# Maps the API's "prompt" strings to our CSV column names
INFO_FIELD_MAP = {
    "DOB":               "dob",
    "Dam Breed":         "dam_breed",
    "Days in Herd":      "days_in_herd",
    "TB Test":           "tb_test",
    "Herd TB Test":      "herd_tb_test",
    "Export":            "export_status",
    "No of Owners":      "no_of_owners",
    "Quality Assurance": "quality_assurance",
    "BVD Tested":        "bvd_tested",
    "Price per KG":      "price_per_kg",
}

# Maps ICBF sub-field sTitle values to CSV column names
ICBF_FIELD_MAP = {
    "Genomic Eval":      "icbf_genomic_eval",
    "CBV":               "icbf_cbv",
    "Across Breed":      "icbf_across_breed",
    "Replacement Index": "icbf_replacement_index",
    "EBI":               "icbf_ebi",
    "Milk":              "icbf_milk",
    "Fertility":         "icbf_fertility",
    "Terminal Index":    "icbf_terminal_index",
}


# ── Auth ───────────────────────────────────────────────────────────────────────
def login() -> tuple[str, str]:
    resp = _post(LOGIN_URL, {
        "Task": "FULLChkLogin",
        "email": EMAIL,
        "password": PASSWORD,
    }, token=None)
    return resp["token"], str(resp["userid"])


# ── Mart list ──────────────────────────────────────────────────────────────────
def get_mart_list(token: str, userid: str) -> list[dict]:
    return _post(AWS_URL, {
        "Task": "martlistjson",
        "userid": userid,
        "jwtusertoken": token,
    }, token=token)


# ── Sold lots (basic) ─────────────────────────────────────────────────────────
def get_sold_lots(token: str, userid: str, nid: int) -> list[dict]:
    result = _post(AWS_URL, {
        "Task": "SOLDDETAILSTABjson",
        "userid": userid,
        "jwtusertoken": token,
        "nid": nid,
    }, token=token)
    return result if isinstance(result, list) else []


# ── Per-lot animal detail ─────────────────────────────────────────────────────
def get_lot_detail(token: str, userid: str, nid: int,
                   base_lot: str, expected_sale_id: str) -> list[dict] | None:
    """
    Fetch animal detail for a base lot (e.g. '35' for sold entries 35A/35B/35C).
    Returns the 'data' array (list of animals) if the sale ID matches, else None.
    """
    result = _post(AWS_URL, {
        "Task": "LOTCONTENTjson",
        "userid": userid,
        "jwtusertoken": token,
        "ringid": 1,
        "sLotNo": base_lot,
        "nid": nid,
    }, token=token)

    if not isinstance(result, list) or not result:
        return None

    entry = result[0]
    if str(entry.get("sSaleId", "")) != str(expected_sale_id):
        return None          # stale/wrong sale — skip

    return entry.get("data") or []


def parse_animal_info(info_list: list[dict]) -> dict:
    """Flatten an animal's info [{prompt, data}, …] into a dict keyed by CSV column."""
    out: dict[str, str] = {}
    for item in info_list:
        prompt   = item.get("prompt", "")
        data_val = str(item.get("data", "")).strip()

        if prompt == "ICBF":
            # Parse embedded JSON: [{"sTitle":"Genomic Eval","sValue":"Yes"}, …]
            if data_val.startswith("["):
                try:
                    for sub in json.loads(data_val):
                        col = ICBF_FIELD_MAP.get(sub.get("sTitle", ""))
                        if col:
                            out[col] = str(sub.get("sValue", "")).strip()
                except (ValueError, TypeError):
                    pass
        else:
            col = INFO_FIELD_MAP.get(prompt)
            if col:
                out[col] = data_val
    return out


# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_seen_valcodes() -> set[str]:
    seen: set[str] = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("val_code"):
                    seen.add(row["val_code"])
    return seen


def append_rows(rows: list[dict]) -> None:
    needs_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


# ── Parse ─────────────────────────────────────────────────────────────────────
def extract_sale_id(val_code: str) -> str:
    parts = val_code.split("-")
    return parts[1] if len(parts) >= 2 else ""


def base_lot_number(lot_no: str) -> str:
    """Strip trailing letter suffix: '35B' → '35',  '1R' → '1',  '4' → '4'."""
    return lot_no.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") or lot_no


# ── HTTP helper ───────────────────────────────────────────────────────────────
def _post(url: str, payload: dict, token: str | None) -> dict | list:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"BEARER {token}"

    last_err = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers,
                                 timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text or text == "0":
                return []
            try:
                return resp.json()
            except ValueError:
                return []
        except Exception as exc:
            last_err = exc
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"All {RETRY_ATTEMPTS} attempts failed for {url}: {last_err}")


# ── Core scrape for one mart ──────────────────────────────────────────────────
def scrape_mart(token: str, userid: str, mart: dict,
                seen: set[str], today: str) -> list[dict]:
    """
    Fetch sold lots for one mart, enrich with animal detail for new lots only.
    Returns list of new rows ready to append to CSV.
    """
    nid  = mart["nId"]
    name = mart["sMart"]

    sold_lots = get_sold_lots(token, userid, nid)
    if not sold_lots:
        return []

    # Filter to only lots not already in the CSV
    new_lots = [lot for lot in sold_lots
                if lot.get("sValCode") and lot["sValCode"] not in seen]
    if not new_lots:
        return []

    sale_id = extract_sale_id(new_lots[0]["sValCode"])

    # ── Fetch per-lot detail in parallel ─────────────────────────────────────
    # Group new lots by base lot number to avoid duplicate API calls
    base_to_lots: dict[str, list[dict]] = {}
    for lot in new_lots:
        base = base_lot_number(lot["sLotNo"])
        base_to_lots.setdefault(base, []).append(lot)

    # detail_cache[base_lot] = list of animal info dicts (or empty list)
    detail_cache: dict[str, list[dict]] = {}

    def fetch_one(base_lot: str) -> tuple[str, list[dict]]:
        animals = get_lot_detail(token, userid, nid, base_lot, sale_id) or []
        return base_lot, [parse_animal_info(a.get("info", [])) for a in animals]

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        futures = {pool.submit(fetch_one, base): base for base in base_to_lots}
        for future in as_completed(futures):
            base, parsed_animals = future.result()
            detail_cache[base] = parsed_animals

    # ── Build rows ────────────────────────────────────────────────────────────
    rows: list[dict] = []
    for lot in new_lots:
        base    = base_lot_number(lot["sLotNo"])
        animals = detail_cache.get(base, [])

        # Match this sold entry to the right animal by sub-lot index
        # e.g. 35B → index 1 (B is the 2nd letter, 0-indexed = 1)
        # Only attempt when suffix is a single letter; fall back to first animal
        suffix = lot["sLotNo"][len(base):]            # '' or 'B' or 'R' etc.
        if suffix and len(suffix) == 1 and len(animals) > 1:
            idx = ord(suffix.upper()) - ord("A")      # A→0, B→1, C→2 …
            animal_info = animals[idx] if 0 <= idx < len(animals) else animals[0]
        else:
            animal_info = animals[0] if animals else {}

        row: dict = {
            "mart":         name,
            "mart_id":      nid,
            "sale_id":      sale_id,
            "lot":          lot.get("sLotNo", ""),
            "breed":        lot.get("sBreed", ""),
            "sex":          lot.get("sSex", ""),
            "weight":       lot.get("xWeight", ""),
            "price":        lot.get("sPrice", ""),
            "age_months":   lot.get("sMonths", ""),
            "val_code":     lot.get("sValCode", ""),
            "scraped_date": today,
        }
        # Merge in the animal detail (empty strings where not available)
        for col in (*INFO_FIELD_MAP.values(), *ICBF_FIELD_MAP.values()):
            row[col] = animal_info.get(col, "")

        rows.append(row)

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    reset_mode = "--reset" in sys.argv

    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"[{ts}] MartBids scraper starting{' (RESET MODE)' if reset_mode else ''}…")

    if reset_mode and OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
        print(f"  Deleted existing {OUTPUT_CSV.name} — full re-scrape")

    token, userid = login()
    print(f"  Logged in  (userid={userid})")

    marts = get_mart_list(token, userid)
    print(f"  {len(marts)} marts found")

    seen = load_seen_valcodes()
    print(f"  {len(seen)} val_codes already in {OUTPUT_CSV.name}")

    total_new = 0
    skipped   = 0

    for mart in marts:
        name = mart["sMart"]
        time.sleep(MART_DELAY)

        try:
            rows = scrape_mart(token, userid, mart, seen, today)
        except Exception as exc:
            print(f"  {name:24s}  ERROR: {exc}")
            continue

        if not rows:
            # Determine whether it's "no data" or "unchanged"
            try:
                sold = get_sold_lots(token, userid, mart["nId"])
            except Exception:
                sold = []
            if not sold:
                print(f"  {name:24s}  no sold data")
            else:
                sale_id = extract_sale_id(sold[0].get("sValCode", "-"))
                skipped += 1
                print(f"  {name:24s}  unchanged  (sale_id={sale_id}, {len(sold)} lots)")
            continue

        append_rows(rows)
        for r in rows:
            seen.add(r["val_code"])

        sale_id   = rows[0]["sale_id"]
        detail_ct = sum(1 for r in rows if r.get("dob"))
        print(f"  {name:24s}  +{len(rows):3d} lots  "
              f"(sale_id={sale_id}, {detail_ct}/{len(rows)} with animal detail)")
        total_new += len(rows)

    print(f"\n  Done.  {total_new} new rows → {OUTPUT_CSV.name}  |  {skipped} mart(s) unchanged")


if __name__ == "__main__":
    main()
