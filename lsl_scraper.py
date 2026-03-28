#!/usr/bin/env python3
"""
Livestock-Live.com daily cattle scraper
========================================
Scrapes the "Last Sale" catalogue for all Irish cattle marts.
No login required — the last sale is publicly accessible.
Appends to lsl_lots.csv, deduplicated by val_code (mart_code-auction_id).

CSV columns:
  mart, mart_code, sale_date, lot, auction_id, breed, sex,
  age_months, weight, price, price_per_kg, icbf_stars, status,
  scraped_date, val_code
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import date as dt_date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

DIR        = Path(__file__).parent
OUTPUT_CSV = DIR / "lsl_lots.csv"
BASE_URL   = "https://www.livestock-live.com"
WORKERS    = 6
DELAY      = 0.5   # seconds between requests per thread

CSV_FIELDS = [
    "mart", "mart_code", "sale_date", "lot", "auction_id",
    "breed", "sex", "age_months", "weight",
    "price", "price_per_kg", "icbf_stars", "status",
    "scraped_date", "val_code",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    })
    return s


def get_mart_codes(session) -> list[str]:
    """Fetch all Irish mart codes from the locations page."""
    try:
        resp = session.get(f"{BASE_URL}/Locations-Livestock", timeout=15)
        codes = re.findall(r'/OnlineCatalogue-([A-Z0-9]+)', resp.text)
        # Filter out non-Irish / non-cattle codes (UK, Spain, etc.)
        exclude = {"ALEXL", "ASTRAS", "BENTH", "BROCKA", "CANECFU", "CMFA",
                   "COOLEY", "DOWNP", "DRAPER", "DUNGA", "DUNGAN", "EHGC",
                   "FEAGASCTRU", "FEOIL", "GRANA", "GVMKC", "INISHO", "KINGD",
                   "MIDKER", "MILFORD", "OMAGH", "OZBEK", "PLUMBR", "POORF",
                   "RUGBY", "SUBGAN", "TAAFFE", "TJCOXL", "TKHF", "TULLSHOW",
                   "WFRTBRS"}
        return sorted(set(codes) - exclude)
    except Exception as e:
        print(f"  [WARN] Could not fetch mart list: {e}")
        # Fallback hardcoded list of known Irish cattle marts
        return [
            "BALLIN", "BALLYM", "CAHIR", "CASTL", "CORRIN", "DINGLE",
            "DROMC", "ENNIS", "GLENA", "GORT", "HEADF", "KANTURK",
            "KENMA", "KILFEN", "KILRU", "LISTO", "M100", "M101", "M102",
            "M162", "M166", "M170", "M186", "M198", "M248", "M262",
            "M295", "M303", "MOHILL", "MWLIS", "NEWPORT", "SCARIFF",
            "SIXMIL", "TIPPE", "TIPPEM", "WRM",
        ]


def parse_age_months(age_str: str):
    """'107m,27d' → 107.9  |  '24m' → 24.0"""
    m = re.match(r'(\d+)m(?:,(\d+)d)?', str(age_str).strip())
    if not m:
        return None
    months = int(m.group(1))
    days   = int(m.group(2)) if m.group(2) else 0
    return round(months + days / 30, 1)


# Species codes that are NOT cattle — skip these lots entirely
NON_CATTLE = {"LB", "L1", "L2", "L3", "L4", "SH", "PIG", "LAMB"}

# Species/category codes that ARE cattle (non-exhaustive; everything else
# that isn't in NON_CATTLE and isn't a bare "Sold" is also treated as cattle)
CATTLE_CODES = {
    "CO", "BLKS", "WHF", "WB", "WH", "WBH", "WBF",
    "WEAN", "BULLOCK", "HEIFER", "CA", "DC", "HB", "HF", "HFR",
    "WS", "SUCK", "BULL", "STEER", "COW", "C", "B",
}


def parse_lot_text(text: str) -> dict:
    """
    Parse the raw text block for one lot.  Handles several formats:

    1. Standard / category:
       SPECIES [SEX] [BREED] [AGE] [STARS] STATUS PRICE WEIGHTKg
       'CO F CHX 107m,27d **** Lot Sold 2820.00 785.00Kg'
       'BLKS M CHX 23m,1d **** Lot Sold 2250.00 500.00Kg'

    2. Bare-sold (CORRIN, WRM):
       'Sold 2320.00 510.00Kg'

    3. Quantity-only (no price available — skip):
       'BULLOCK AA 12m,13d x2 25/03'
       'WEAN MALE 6m,18d x4 26/03'
    """
    tokens = text.split()
    result = {
        "species": None, "sex": None, "breed": None,
        "age_months": None, "icbf_stars": None,
        "status": None, "price": None, "weight": None,
    }
    if not tokens:
        return result

    # ── Format 2: bare "Sold PRICE [WEIGHTKg]" ───────────────────────────────
    if tokens[0] == "Sold":
        result["status"] = "Lot Sold"
        if len(tokens) >= 2 and re.match(r'^\d+\.?\d*$', tokens[1]):
            try:
                result["price"] = float(tokens[1])
            except ValueError:
                pass
        if len(tokens) >= 3:
            wm = re.match(r'([\d.]+)[Kk][Gg]$', tokens[2])
            if wm:
                result["weight"] = float(wm.group(1))
        return result

    # ── Formats 1 & 3: species/category first ────────────────────────────────
    result["species"] = tokens[0]
    i = 1

    # Sex (single letter M/F/N/B/MALE/FEMALE — handle both)
    if i < len(tokens):
        t = tokens[i].upper()
        if t in {"M", "F", "N", "B", "MALE", "FEMALE"}:
            result["sex"] = tokens[i]; i += 1

    # Breed — token before the age pattern (skip if it looks like a quantity or date)
    if i < len(tokens):
        t = tokens[i]
        if (not re.match(r'\d+m', t)
                and not re.match(r'^x\d+$', t, re.I)
                and not re.match(r'^\d{2}/\d{2}', t)
                and t not in {"Lot", "Sold", "In", "Awaiting"}):
            result["breed"] = t; i += 1

    # Age
    if i < len(tokens) and re.match(r'\d+m', tokens[i]):
        result["age_months"] = parse_age_months(tokens[i]); i += 1

    # Quantity indicator e.g. x1, x2 — means no individual price → skip
    if i < len(tokens) and re.match(r'^x\d+$', tokens[i], re.I):
        return result   # status stays None → filtered out later

    # ICBF stars (* or +)
    if i < len(tokens) and re.match(r'^[\*\+]+$', tokens[i]):
        result["icbf_stars"] = len(tokens[i]); i += 1

    # Status ("Lot Sold", "In Auction", "Awaiting Auction", etc.)
    status_parts = []
    while i < len(tokens):
        t = tokens[i]
        if re.match(r'^\d+\.?\d*$', t):       # price
            break
        if re.match(r'[\d.]+[Kk][Gg]$', t):   # weight
            break
        if re.match(r'^x\d+$', t, re.I):       # late quantity → no price
            return result
        status_parts.append(t); i += 1
    if status_parts:
        result["status"] = " ".join(status_parts)

    # Price
    if i < len(tokens) and re.match(r'^\d+\.?\d*$', tokens[i]):
        try:
            result["price"] = float(tokens[i]); i += 1
        except ValueError:
            pass

    # Weight
    if i < len(tokens):
        wm = re.match(r'([\d.]+)[Kk][Gg]$', tokens[i])
        if wm:
            result["weight"] = float(wm.group(1))

    return result


def get_mart_name(soup: BeautifulSoup, mart_code: str) -> str:
    """Extract mart name from page."""
    for sel in ["h1", ".mart-title", ".page-title", "h2"]:
        el = soup.select_one(sel)
        if el:
            txt = el.text.strip().split("\n")[0].strip()
            if txt and len(txt) > 2:
                return txt
    if soup.title:
        parts = re.split(r'[|\-]', soup.title.text)
        if parts:
            return parts[0].strip()
    return mart_code


# ── Per-mart scraper ──────────────────────────────────────────────────────────

def scrape_mart(mart_code: str, existing_ids: set) -> tuple[list[dict], str, int]:
    """
    Scrape last sale for one mart.
    Returns (new_rows, mart_code, total_lots_on_page).
    """
    session = make_session()
    url = f"{BASE_URL}/OnlineCatalogue-{mart_code}"
    try:
        resp = session.get(url, params={"SearchDateRange": "Last"},
                           timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return [], mart_code, 0

    soup = BeautifulSoup(resp.text, "html.parser")
    lot_divs = soup.select(".singleitem-container")
    if not lot_divs:
        return [], mart_code, 0

    mart_name = get_mart_name(soup, mart_code)
    today     = dt_date.today().isoformat()
    rows      = []

    for div in lot_divs:
        pin = div.select_one(".pinbtn")
        if not pin:
            continue

        auction_id = pin.get("auctionid", "")
        lot_num    = pin.get("lot", "")
        sale_date  = pin.get("lotdate", "")

        # Convert DD/MM/YYYY → YYYY-MM-DD
        try:
            sale_date = datetime.strptime(sale_date, "%d/%m/%Y").strftime("%Y-%m-%d")
        except Exception:
            pass

        val_code = f"{mart_code}-{auction_id}"
        if val_code in existing_ids:
            continue

        # Parse lot text
        data_div = div.select_one(".col-xl-9") or div.select_one(".col-lg-9")
        if not data_div:
            continue
        raw = " ".join(data_div.text.split())
        p   = parse_lot_text(raw)

        # Skip non-cattle species (lambs, sheep, etc.)
        if p["species"] in NON_CATTLE:
            continue
        # Skip lots with no price data (quantity-only, unsold, no parse)
        if p["status"] != "Lot Sold":
            continue
        # Skip lots with no price or weight
        if not p["price"] or not p["weight"]:
            continue
        # Sanity check: implausible weight (M186 sometimes shows 0.01Kg)
        if p["weight"] < 50:
            continue

        ppkg = round(p["price"] / p["weight"], 4) if p["weight"] > 0 else None

        rows.append({
            "mart":         mart_name,
            "mart_code":    mart_code,
            "sale_date":    sale_date,
            "lot":          lot_num,
            "auction_id":   auction_id,
            "breed":        p["breed"],
            "sex":          p["sex"],
            "age_months":   p["age_months"],
            "weight":       p["weight"],
            "price":        p["price"],
            "price_per_kg": ppkg,
            "icbf_stars":   p["icbf_stars"],
            "status":       p["status"],
            "scraped_date": today,
            "val_code":     val_code,
        })

    time.sleep(DELAY)
    return rows, mart_code, len(lot_divs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[LSL] Livestock-Live.com scraper starting…")

    # Load existing val_codes for deduplication
    if OUTPUT_CSV.exists():
        existing_df  = pd.read_csv(OUTPUT_CSV, usecols=["val_code"])
        existing_ids = set(existing_df["val_code"].astype(str))
    else:
        existing_ids = set()
    print(f"  Existing records: {len(existing_ids):,}")

    session    = make_session()
    mart_codes = get_mart_codes(session)
    print(f"  Scraping {len(mart_codes)} Irish cattle marts…\n")

    all_rows  = []
    unchanged = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(scrape_mart, code, existing_ids): code
            for code in mart_codes
        }
        for future in as_completed(futures):
            try:
                rows, code, total = future.result()
                if rows:
                    all_rows.extend(rows)
                    print(f"  {code:15s} +{len(rows):3d} new lots  "
                          f"({total} on page)")
                else:
                    unchanged += 1
            except Exception as e:
                print(f"  ERROR {e}")

    print(f"\n  {unchanged} mart(s) unchanged / no cattle sold")

    if all_rows:
        new_df = pd.DataFrame(all_rows, columns=CSV_FIELDS)
        if OUTPUT_CSV.exists():
            old_df = pd.read_csv(OUTPUT_CSV)
            df = pd.concat([old_df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["val_code"])
        else:
            df = new_df
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  Done. {len(all_rows):,} new rows → {OUTPUT_CSV.name}  "
              f"| {len(df):,} total rows")
    else:
        print("  No new data today.")


if __name__ == "__main__":
    main()
