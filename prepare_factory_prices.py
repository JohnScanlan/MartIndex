#!/usr/bin/env python3
"""
Prepare and clean factory_prices.csv
=====================================
Output: factory_prices_clean.csv

Key transformations:
  - Drop rows with invalid prices (< 3 or > 12 €/kg)
  - Split BPW detail grade "O=/3=" into conformation + fat_class
  - Standardise category names
  - Add is_headline flag
  - Parse count and avg_weight from notes field
  - report_date as proper date column
"""

import re
from pathlib import Path

import pandas as pd

DIR   = Path(__file__).parent
INPUT = DIR / "factory_prices.csv"
OUT   = DIR / "factory_prices_clean.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, parse_dates=["report_date", "scraped_date"])

print(f"Raw rows: {len(df):,}")

# ── Drop bad prices ───────────────────────────────────────────────────────────
df = df[(df["price_euro_per_kg"] >= 3.0) & (df["price_euro_per_kg"] <= 12.0)].copy()
print(f"After price filter (3–12 €/kg): {len(df):,}")

# ── Standardise category names ─────────────────────────────────────────────
CAT_MAP = {
    "Steer":      "Steer",
    "Heifer":     "Heifer",
    "Cow":        "Cow",
    "Young Bull": "Young Bull",
    "Bull":       "Bull",
    "Steers":     "Steer",
    "Heifers":    "Heifer",
    "Cows":       "Cow",
    "Young Bulls":"Young Bull",
    "Bulls":      "Bull",
}
df["category"] = df["category"].str.strip()
df["category"] = df["category"].map(CAT_MAP).fillna(df["category"])

# ── is_headline flag ──────────────────────────────────────────────────────────
df["is_headline"] = df["notes"].str.strip() == "headline"

# ── Split BPW detail grades into conformation + fat_class ─────────────────
# Headline grades look like "R3", "O4", "U3" — these are classification codes
# Detail grades look like "O=/3=", "R-/3+", "U-/2+" — conformation/fat

def _split_grade(row):
    grade = str(row["grade"]) if pd.notna(row["grade"]) else ""
    if row["is_headline"] or "/" not in grade:
        return pd.Series({"conformation": "", "fat_class": "", "classification": grade})
    parts = grade.split("/", 1)
    return pd.Series({"conformation": parts[0], "fat_class": parts[1], "classification": ""})

split = df.apply(_split_grade, axis=1)
df["conformation"]   = split["conformation"]
df["fat_class"]      = split["fat_class"]
df["classification"] = split["classification"]   # e.g. R3, O4 for headlines

# ── Parse count + avg_weight from notes ───────────────────────────────────────
df["lot_count"]  = df["notes"].str.extract(r"count=(\d+)").astype(float)
df["avg_weight_kg"] = df["notes"].str.extract(r"avg_wt=([\d.]+)kg").astype(float)

# ── Parse min/max cents from national notes ───────────────────────────────────
df["price_min_euro"] = df["notes"].str.extract(r"min=([\d.]+)").astype(float) / 100
df["price_max_euro"] = df["notes"].str.extract(r"max=([\d.]+)").astype(float) / 100

# ── Week number from report_date ──────────────────────────────────────────────
df["week_number"] = df["report_date"].dt.isocalendar().week.astype("Int64")
df["year"]        = df["report_date"].dt.year

# ── Select and order final columns ────────────────────────────────────────────
FINAL_COLS = [
    "report_date", "week_number", "year", "scraped_date",
    "source", "country",
    "category", "is_headline", "classification", "conformation", "fat_class",
    "factory",
    "price_euro_per_kg", "price_raw", "unit",
    "lot_count", "avg_weight_kg", "price_min_euro", "price_max_euro",
    "notes",
]
df = df[FINAL_COLS].sort_values(["report_date", "source", "category", "factory"])

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows → {OUT.name}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Sources ──")
print(df["source"].value_counts().to_string())

print("\n── Categories ──")
print(df["category"].value_counts().to_string())

print("\n── Date range ──")
print(df["report_date"].min().date(), "→", df["report_date"].max().date())

print("\n── Headline prices: latest week by category ──")
latest = df[df["is_headline"] & (df["source"] == "BeefPriceWatch")]
latest = latest[latest["report_date"] == latest["report_date"].max()]
print(
    latest.groupby("category")["price_euro_per_kg"]
    .agg(["mean", "min", "max", "count"])
    .round(4)
    .to_string()
)
