#!/usr/bin/env python3
"""
Weather data fetcher for MartBids sold lots.
Reads sold_lots.csv, fetches daily weather from Open-Meteo (free, no key)
for each mart+date combo, writes/updates weather_cache.csv.

Columns in weather_cache.csv:
  mart, date, temp_max_c, temp_min_c, precipitation_mm, wind_speed_kmh
"""

import time
import requests
import pandas as pd
from pathlib import Path
from mart_coords import MART_COORDS

DIR           = Path(__file__).parent
LOTS_CSV      = DIR / "sold_lots.csv"
WEATHER_CSV   = DIR / "weather_cache.csv"
WEATHER_COLS  = ["mart", "date", "temp_max_c", "temp_min_c",
                 "precipitation_mm", "wind_speed_kmh"]
ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather_for_mart(mart: str, dates: list[str]) -> list[dict]:
    """Fetch daily weather for a mart over a list of dates (YYYY-MM-DD strings)."""
    if mart not in MART_COORDS:
        print(f"  [WARN] No coords for mart '{mart}', skipping.")
        return []

    lat, lon = MART_COORDS[mart]
    start = min(dates)
    end   = max(dates)

    try:
        resp = requests.get(ARCHIVE_URL, params={
            "latitude":        lat,
            "longitude":       lon,
            "start_date":      start,
            "end_date":        end,
            "daily":           "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "timezone":        "Europe/Dublin",
            "wind_speed_unit": "kmh",
        }, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] Weather fetch failed for {mart}: {e}")
        return []

    daily = data.get("daily", {})
    rows = []
    date_set = set(dates)
    for i, d in enumerate(daily.get("time", [])):
        if d in date_set:
            rows.append({
                "mart":              mart,
                "date":              d,
                "temp_max_c":        daily["temperature_2m_max"][i],
                "temp_min_c":        daily["temperature_2m_min"][i],
                "precipitation_mm":  daily["precipitation_sum"][i],
                "wind_speed_kmh":    daily["wind_speed_10m_max"][i],
            })
    return rows


def main():
    # Load sold lots
    if not LOTS_CSV.exists():
        print(f"No sold_lots.csv found at {LOTS_CSV}")
        return

    df = pd.read_csv(LOTS_CSV, usecols=["mart", "scraped_date"])
    df["date"] = pd.to_datetime(df["scraped_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])

    # Load existing cache
    if WEATHER_CSV.exists():
        cache = pd.read_csv(WEATHER_CSV)
        existing = set(zip(cache["mart"], cache["date"]))
    else:
        cache = pd.DataFrame(columns=WEATHER_COLS)
        existing = set()

    # Find mart+date combos not yet cached
    needed = (
        df[["mart", "date"]]
        .drop_duplicates()
        .apply(lambda r: (r["mart"], r["date"]), axis=1)
    )
    to_fetch = [(m, d) for m, d in needed if (m, d) not in existing]

    if not to_fetch:
        print("Weather cache is up to date, nothing to fetch.")
        return

    print(f"Fetching weather for {len(to_fetch)} mart+date combos...")

    # Group by mart so we batch date ranges per mart
    from collections import defaultdict
    mart_dates = defaultdict(list)
    for mart, date in to_fetch:
        mart_dates[mart].append(date)

    new_rows = []
    for mart, dates in mart_dates.items():
        print(f"  {mart}: {len(dates)} date(s)")
        rows = fetch_weather_for_mart(mart, dates)
        new_rows.extend(rows)
        time.sleep(0.3)   # be polite to the free API

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=WEATHER_COLS)
        cache = pd.concat([cache, new_df], ignore_index=True)
        cache = cache.drop_duplicates(subset=["mart", "date"])
        cache.to_csv(WEATHER_CSV, index=False)
        print(f"Weather cache updated → {len(new_rows)} new rows, "
              f"{len(cache)} total rows.")
    else:
        print("No new weather data retrieved.")


if __name__ == "__main__":
    main()
