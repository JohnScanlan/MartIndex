#!/usr/bin/env python3
"""
Daily PDF Report Generator for MartBids.
Builds a 5-page farmer-focused PDF, then emails it to recipients.

Pages:
  1 - National Market Summary (KPIs, sex split, week-on-week)
  2 - Breed × Weight Bracket (Males)
  3 - Breed × Weight Bracket (Females)
  4 - Regional Breakdown + Market Intelligence
  5 - ML Model Metrics

Email credentials: set env vars SMTP_USER and SMTP_PASS,
or create email_config.json: {"user": "...", "pass": "..."}
"""

import io
import os
import json
import smtplib
import warnings
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import date, timedelta

FACTORY_CSV = Path(__file__).parent / "factory_prices_clean.csv"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

warnings.filterwarnings("ignore")

DIR          = Path(__file__).parent
LOTS_CSV     = DIR / "sold_lots.csv"
META_JSON    = DIR / "model_metadata.json"
REPORT_PATH  = DIR / "daily_report.pdf"
EMAIL_CONFIG = DIR / "email_config.json"

RECIPIENTS = [
    "johnscanlan52@yahoo.ie",
    "michaelscanlan05@yahoo.com",
    "margaretscanlan14@hotmail.com",
]

SMTP_HOST    = "smtp.gmail.com"
SMTP_PORT    = 587
SMTP_TIMEOUT = 30   # seconds - prevents overnight hangs

REGION_MAP = {
    "Athenry": "Connacht", "Balla": "Connacht", "Ballinasloe": "Connacht",
    "Ballinrobe": "Connacht", "Ballymote": "Connacht", "Carrigallen": "Connacht",
    "Castlerea": "Connacht", "Drumshanbo": "Connacht", "Elphin": "Connacht",
    "Headford": "Connacht", "Loughrea": "Connacht", "Mohill": "Connacht",
    "Portumna": "Connacht", "Roscommon": "Connacht", "Tuam": "Connacht",
    "Cashel": "Munster", "Corrin": "Munster", "Ennis": "Munster",
    "Iveragh": "Munster", "Kilfenora": "Munster", "Kilrush": "Munster",
    "Mid Tipp Mart": "Munster", "Nenagh": "Munster", "Roscrea": "Munster",
    "Scarriff": "Munster", "Templemore": "Munster",
    "Baltinglass": "Leinster", "Ballymahon LWFM": "Leinster", "Birr": "Leinster",
    "Carnew": "Leinster", "Granard": "Leinster", "Kilkenny": "Leinster",
    "Midland and Western": "Leinster",
    "Clogher": "Ulster", "Donegal": "Ulster", "Donegal Suffolks": "Ulster",
    "Lisnaskea": "Ulster", "Raphoe": "Ulster", "Rathfriland": "Ulster",
}

WEIGHT_BINS   = [0, 200, 300, 400, 500, 600, 9999]
WEIGHT_LABELS = ["<200", "200-300", "300-400", "400-500", "500-600", "600+"]
TOP_BREEDS    = ["LMX", "AAX", "CHX", "FR", "HEX", "FRX", "BBX", "SIX", "LM", "CH"]

NAVY  = (27, 42, 74)
GOLD  = (201, 168, 76)
LIGHT = (240, 241, 243)


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_eur(s):
    if pd.isna(s) or str(s).strip() == "":
        return np.nan
    return pd.to_numeric(str(s).replace("€", "").replace(",", "").strip(),
                         errors="coerce")

def eur(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    fmt = f",.{decimals}f"
    return f"EUR {val:{fmt}}"

def pct_arrow(change):
    if pd.isna(change):
        return "N/A"
    sym = "+" if change >= 0 else ""
    return f"{sym}{change:.1f}%"

def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data

def save_tmp(data: bytes, name: str) -> Path:
    p = DIR / name
    p.write_bytes(data)
    return p


# ── Data preparation ──────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(LOTS_CSV, low_memory=False)
    df["price_num"]      = df["price"].apply(parse_eur)
    df["weight_num"]     = pd.to_numeric(df["weight"], errors="coerce")
    df["ppkg"]           = df["price_num"] / df["weight_num"].replace(0, np.nan)
    df["scraped_date"]   = pd.to_datetime(df["scraped_date"], errors="coerce")
    df["region"]         = df["mart"].map(REGION_MAP).fillna("Other")
    df["icbf_stars_n"]   = (df["icbf_across_breed"].fillna("")
                             .apply(lambda s: str(s).count("☆") + str(s).count("★")))
    df["weight_bracket"] = pd.cut(df["weight_num"], bins=WEIGHT_BINS,
                                  labels=WEIGHT_LABELS, right=False)
    # Only rows with valid ppkg for price analysis
    df_valid = df[(df["ppkg"] >= 0.5) & (df["ppkg"] <= 20)].copy()
    return df, df_valid


def get_today_df(df_valid: pd.DataFrame):
    today_str = date.today().isoformat()
    t = df_valid[df_valid["scraped_date"].dt.strftime("%Y-%m-%d") == today_str]
    if t.empty:
        latest = df_valid["scraped_date"].max()
        t = df_valid[df_valid["scraped_date"] == latest]
        print(f"  No data for today, using latest: {latest.date()}")
    return t.copy()


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_weekly_trend(df_valid: pd.DataFrame) -> Path:
    """Rolling 8-week median €/kg trend."""
    df_valid = df_valid.copy()
    df_valid["week"] = df_valid["scraped_date"].dt.to_period("W")
    weekly = (df_valid.groupby("week")["ppkg"]
              .median()
              .reset_index()
              .tail(10))
    weekly["week_str"] = weekly["week"].astype(str).str[-5:]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(range(len(weekly)), weekly["ppkg"], marker="o",
            color=f"#{NAVY[0]:02x}{NAVY[1]:02x}{NAVY[2]:02x}", linewidth=2)
    ax.fill_between(range(len(weekly)), weekly["ppkg"],
                    alpha=0.12, color=f"#{NAVY[0]:02x}{NAVY[1]:02x}{NAVY[2]:02x}")
    ax.set_xticks(range(len(weekly)))
    ax.set_xticklabels(weekly["week_str"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Median EUR/kg")
    ax.set_title("National Median EUR/kg - Last 10 Weeks", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_trend.png")


def chart_sex_breakdown(df_today: pd.DataFrame) -> Path:
    sex_map = {"M": "Male", "F": "Female", "B": "Bull"}
    counts = (df_today["sex"].map(sex_map).fillna("Other")
              .value_counts())
    colours = ["#1B2A4A", "#C9A84C", "#9B2335", "#888888"]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.barh(counts.index, counts.values,
            color=colours[:len(counts)], edgecolor="white")
    for i, (idx, v) in enumerate(counts.items()):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
    ax.set_title("Volume by Sex (today)", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_sex.png")


def chart_breed_movers(df_valid: pd.DataFrame) -> Path:
    """Bar chart of biggest breed price movers week-on-week."""
    today = df_valid["scraped_date"].max().date()
    w1 = df_valid[df_valid["scraped_date"].dt.date > today - timedelta(days=7)]
    w2 = df_valid[(df_valid["scraped_date"].dt.date > today - timedelta(days=14)) &
                  (df_valid["scraped_date"].dt.date <= today - timedelta(days=7))]
    med1 = w1.groupby("breed")["ppkg"].agg(["median", "count"])
    med2 = w2.groupby("breed")["ppkg"].agg(["median", "count"])
    comp = med1.join(med2, lsuffix="_this", rsuffix="_last")
    comp = comp[(comp["count_this"] >= 5) & (comp["count_last"] >= 5)].copy()
    comp["pct_change"] = (comp["median_this"] - comp["median_last"]) / comp["median_last"] * 100
    comp = comp[comp.index.isin(TOP_BREEDS)].sort_values("pct_change")

    if comp.empty:
        return None

    colours = [("#9B2335" if v < 0 else "#276749") for v in comp["pct_change"]]
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(comp) * 0.45)))
    ax.barh(comp.index, comp["pct_change"], color=colours, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("% change vs last week")
    ax.set_title("Breed Price Movers - Week on Week", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_movers.png")


def chart_feature_importance(meta: dict) -> Path:
    fi = pd.Series(meta.get("feature_importances", {})).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    fi[::-1].plot.barh(ax=ax, color=f"#{NAVY[0]:02x}{NAVY[1]:02x}{NAVY[2]:02x}")
    ax.set_title("Top 10 Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_fi.png")


# ── PDF class ─────────────────────────────────────────────────────────────────

class Report(FPDF):
    def header(self):
        self.set_fill_color(*NAVY)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, "  MartIndex Daily Report", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Generated {date.today().isoformat()} - MartIndex Intelligence", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(*LIGHT)
        self.set_text_color(*NAVY)
        self.cell(0, 7, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kpi_row(self, items: list):
        cell_w = self.epw / len(items)
        for label, val, sub in items:
            x, y = self.get_x(), self.get_y()
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*NAVY)
            self.multi_cell(cell_w, 8, val, align="C")
            self.set_xy(x, y + 8)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(100, 100, 100)
            self.multi_cell(cell_w, 4, label, align="C")
            if sub:
                self.set_xy(x, self.get_y())
                self.set_font("Helvetica", "I", 7)
                self.multi_cell(cell_w, 4, sub, align="C")
            self.set_text_color(0, 0, 0)
            self.set_xy(x + cell_w, y)
        self.ln(18)

    def add_image_full(self, path, h: float = 60):
        if path and Path(path).exists():
            self.image(str(path), x=self.l_margin, w=self.epw, h=h)
            self.ln(3)

    def add_image_half(self, path, side: str = "L", h: float = 55):
        if path and Path(path).exists():
            x = self.l_margin if side == "L" else self.l_margin + self.epw / 2 + 2
            self.image(str(path), x=x, w=self.epw / 2 - 2, h=h)


# ── Page builders ─────────────────────────────────────────────────────────────

def load_factory_data():
    """Load and prep factory prices clean CSV."""
    if not FACTORY_CSV.exists():
        return pd.DataFrame()
    fp = pd.read_csv(FACTORY_CSV, low_memory=False)
    fp["report_date"] = pd.to_datetime(fp["report_date"], errors="coerce")
    fp["price_euro_per_kg"] = pd.to_numeric(fp["price_euro_per_kg"], errors="coerce")
    fp = fp[fp["source"] == "BeefPriceWatch"].copy()
    return fp


def _weekly_factory_steer(fp: pd.DataFrame) -> pd.DataFrame:
    """Return weekly avg price for reference R3/R= steer (headline) over last 12 weeks."""
    ref = fp[
        fp["is_headline"] &
        fp["category"].str.lower().str.contains("steer") &
        fp["factory"].str.lower().ne("national")
    ].copy()
    ref["week"] = ref["report_date"].dt.to_period("W")
    weekly = (ref.groupby("week")["price_euro_per_kg"]
              .mean()
              .reset_index()
              .sort_values("week")
              .tail(12))
    return weekly


def chart_factory_trend(fp: pd.DataFrame) -> Path:
    """12-week factory reference steer price trend."""
    weekly = _weekly_factory_steer(fp)
    if weekly.empty:
        return None
    weeks = [str(w)[-5:] for w in weekly["week"]]
    vals  = weekly["price_euro_per_kg"].values

    fig, ax = plt.subplots(figsize=(9, 3.2))
    gold_hex = f"#{GOLD[0]:02x}{GOLD[1]:02x}{GOLD[2]:02x}"
    navy_hex = f"#{NAVY[0]:02x}{NAVY[1]:02x}{NAVY[2]:02x}"
    ax.plot(range(len(vals)), vals, marker="o", color=gold_hex, linewidth=2.5)
    ax.fill_between(range(len(vals)), vals, alpha=0.15, color=gold_hex)
    # 12-week avg line
    avg12 = vals.mean()
    ax.axhline(avg12, color=navy_hex, linewidth=1, linestyle="--",
               label=f"12-wk avg: EUR {avg12:.2f}/kg")
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("EUR/kg")
    ax.set_title("Factory Reference Steer - 12-Week Trend", fontweight="bold")
    # Set Y-axis limits to 5 cents below min and 5 cents above max
    y_min = vals.min() - 0.05
    y_max = vals.max() + 0.05
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_factory_trend.png")


def chart_mart_trend(df_valid: pd.DataFrame) -> Path:
    """12-week mart median EUR/kg trend."""
    df_valid = df_valid.copy()
    df_valid["week"] = df_valid["scraped_date"].dt.to_period("W")
    weekly = (df_valid.groupby("week")["ppkg"]
              .median()
              .reset_index()
              .sort_values("week")
              .tail(12))
    if weekly.empty:
        return None
    weeks = [str(w)[-5:] for w in weekly["week"]]
    vals  = weekly["ppkg"].values
    avg12 = vals.mean()

    navy_hex = f"#{NAVY[0]:02x}{NAVY[1]:02x}{NAVY[2]:02x}"
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(range(len(vals)), vals, marker="o", color=navy_hex, linewidth=2.5)
    ax.fill_between(range(len(vals)), vals, alpha=0.12, color=navy_hex)
    ax.axhline(avg12, color=f"#{GOLD[0]:02x}{GOLD[1]:02x}{GOLD[2]:02x}",
               linewidth=1, linestyle="--", label=f"12-wk avg: EUR {avg12:.2f}/kg")
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Median EUR/kg")
    ax.set_title("Mart National Median EUR/kg - 12-Week Trend", fontweight="bold")
    # Set Y-axis limits to 5 cents below min and 5 cents above max
    y_min = vals.min() - 0.05
    y_max = vals.max() + 0.05
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_mart_trend.png")


def page1_market_overview(report: Report, df_valid: pd.DataFrame, fp: pd.DataFrame):
    """Page 1 - Factory reference price + mart price, 12-week trends, % changes."""
    report.add_page()
    report.set_font("Helvetica", "", 8)
    report.set_text_color(100, 100, 100)
    report.cell(0, 5, f"Report date: {date.today().isoformat()}    "
                      f"Total mart lots in dataset: {len(df_valid):,}",
                new_x="LMARGIN", new_y="NEXT")
    report.ln(2)

    # ── Factory: latest & prev week, 12-wk avg ────────────────────────────────
    factory_kpis = ("N/A", "N/A", "N/A")
    factory_avg12 = np.nan
    if not fp.empty:
        weekly_f = _weekly_factory_steer(fp)
        if len(weekly_f) >= 1:
            latest_f  = weekly_f.iloc[-1]["price_euro_per_kg"]
            factory_avg12 = weekly_f["price_euro_per_kg"].mean()
            wow_f = np.nan
            if len(weekly_f) >= 2:
                prev_f = weekly_f.iloc[-2]["price_euro_per_kg"]
                wow_f  = (latest_f - prev_f) / prev_f * 100
            factory_kpis = (
                f"EUR {latest_f:.2f}/kg",
                pct_arrow(wow_f) if not np.isnan(wow_f) else "N/A",
                f"EUR {factory_avg12:.2f}/kg",
            )

    # ── Mart: latest & prev week, 12-wk avg ──────────────────────────────────
    df_valid = df_valid.copy()
    df_valid["week"] = df_valid["scraped_date"].dt.to_period("W")
    weekly_m = (df_valid.groupby("week")["ppkg"]
                .median()
                .reset_index()
                .sort_values("week")
                .tail(12))
    mart_kpis = ("N/A", "N/A", "N/A")
    if len(weekly_m) >= 1:
        latest_m  = weekly_m.iloc[-1]["ppkg"]
        avg12_m   = weekly_m["ppkg"].mean()
        wow_m     = np.nan
        if len(weekly_m) >= 2:
            prev_m = weekly_m.iloc[-2]["ppkg"]
            wow_m  = (latest_m - prev_m) / prev_m * 100
        mart_kpis = (
            f"EUR {latest_m:.2f}/kg",
            pct_arrow(wow_m) if not np.isnan(wow_m) else "N/A",
            f"EUR {avg12_m:.2f}/kg",
        )

    # ── KPI boxes ─────────────────────────────────────────────────────────────
    report.section_title("Factory Prices - Reference Steer (R3 Headline, National Avg)")
    report.kpi_row([
        ("Latest Week",         factory_kpis[0], ""),
        ("Change vs Prev Week", factory_kpis[1], ""),
        ("12-Week Average",     factory_kpis[2], ""),
    ])

    report.section_title("Mart Prices - National Median EUR/kg")
    report.kpi_row([
        ("Latest Week",         mart_kpis[0], ""),
        ("Change vs Prev Week", mart_kpis[1], ""),
        ("12-Week Average",     mart_kpis[2], ""),
    ])

    # ── Factory 12-week trend chart ───────────────────────────────────────────
    report.section_title("Factory Reference Steer - 12-Week Price Trend")
    try:
        p = chart_factory_trend(fp)
        if p:
            report.add_image_full(p, h=52)
    except Exception as e:
        report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")

    # ── Mart 12-week trend chart ──────────────────────────────────────────────
    report.section_title("Mart National Median - 12-Week Price Trend")
    try:
        p = chart_mart_trend(df_valid)
        if p:
            report.add_image_full(p, h=52)
    except Exception as e:
        report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")


def page2_price_tables(report: Report, df_valid: pd.DataFrame, fp: pd.DataFrame):
    """Page 2 - Top 5 breed × weight class tables for mart + factory category table."""
    report.add_page()

    # ── Mart: top 5 breeds by volume, last 30 days ───────────────────────────
    cutoff = df_valid["scraped_date"].max() - timedelta(days=30)
    df_30  = df_valid[df_valid["scraped_date"] >= cutoff].copy()
    top5   = df_30["breed"].value_counts().head(5).index.tolist()

    df_30["weight_bracket"] = pd.cut(df_30["weight_num"],
                                     bins=WEIGHT_BINS, labels=WEIGHT_LABELS, right=False)

    for sex, sex_label in [("M", "Males"), ("F", "Females")]:
        report.section_title(f"Mart - Top 5 Breeds × Weight Class ({sex_label}, last 30 days, avg EUR/kg)")
        df_s = df_30[(df_30["sex"] == sex) & (df_30["breed"].isin(top5))].copy()

        pivot = (df_s.groupby(["breed", "weight_bracket"])["ppkg"]
                 .agg(["mean", "count"])
                 .reset_index())

        breed_w  = 30
        bracket_w = (report.epw - breed_w) / len(WEIGHT_LABELS)

        report.set_font("Helvetica", "B", 8)
        report.set_fill_color(*NAVY)
        report.set_text_color(255, 255, 255)
        report.cell(breed_w, 7, "Breed", fill=True, align="C")
        for lbl in WEIGHT_LABELS:
            report.cell(bracket_w, 7, lbl, fill=True, align="C")
        report.ln()
        report.set_text_color(0, 0, 0)

        for i, breed in enumerate(top5):
            fill = i % 2 == 0
            report.set_fill_color(245, 247, 255) if fill else report.set_fill_color(255, 255, 255)
            report.set_font("Helvetica", "B", 8)
            report.cell(breed_w, 6, breed, fill=fill)
            report.set_font("Helvetica", "", 8)
            for bracket in WEIGHT_LABELS:
                row = pivot[(pivot["breed"] == breed) & (pivot["weight_bracket"] == bracket)]
                val = f"{row.iloc[0]['mean']:.2f}" if not row.empty and row.iloc[0]["count"] >= 3 else "-"
                report.cell(bracket_w, 6, val, fill=fill, align="C")
            report.ln()
        report.ln(4)

    # ── Factory: latest week avg price by category ────────────────────────────
    if fp.empty:
        return
    report.section_title("Factory - Latest Week Avg Price by Category (EUR/kg, all factories)")
    latest_w = fp["report_date"].max()
    fp_latest = fp[
        fp["is_headline"] &
        (fp["report_date"] == latest_w) &
        fp["factory"].str.lower().ne("national")
    ].copy()

    cat_summary = (fp_latest.groupby("category")["price_euro_per_kg"]
                   .agg(avg="mean", lo="min", hi="max", factories="count")
                   .reset_index()
                   .sort_values("avg", ascending=False))

    # Compare vs previous week
    dates_sorted = sorted(fp["report_date"].dropna().unique(), reverse=True)
    prev_w = dates_sorted[1] if len(dates_sorted) >= 2 else None
    if prev_w is not None:
        fp_prev = fp[fp["is_headline"] & (fp["report_date"] == prev_w) &
                     fp["factory"].str.lower().ne("national")]
        prev_avg = fp_prev.groupby("category")["price_euro_per_kg"].mean().rename("prev_avg")
        cat_summary = cat_summary.set_index("category").join(prev_avg).reset_index()
        cat_summary["wow"] = ((cat_summary["avg"] - cat_summary["prev_avg"])
                              / cat_summary["prev_avg"] * 100)
    else:
        cat_summary["wow"] = np.nan

    col_w = [45, 30, 25, 25, 30, 35]
    headers = ["Category", "Avg EUR/kg", "Min", "Max", "Factories", "vs Prev Week"]
    report.set_font("Helvetica", "B", 8)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    for h, w in zip(headers, col_w):
        report.cell(w, 7, h, fill=True, align="C")
    report.ln()
    report.set_text_color(0, 0, 0)
    report.set_font("Helvetica", "", 8)

    for i, row in cat_summary.iterrows():
        fill = i % 2 == 0
        report.set_fill_color(245, 247, 255) if fill else report.set_fill_color(255, 255, 255)
        wow_str = pct_arrow(row["wow"]) if "wow" in row and not np.isnan(row["wow"]) else "-"
        report.cell(col_w[0], 6, str(row["category"]), fill=fill)
        report.cell(col_w[1], 6, f"{row['avg']:.2f}", fill=fill, align="C")
        report.cell(col_w[2], 6, f"{row['lo']:.2f}", fill=fill, align="C")
        report.cell(col_w[3], 6, f"{row['hi']:.2f}", fill=fill, align="C")
        report.cell(col_w[4], 6, str(int(row["factories"])), fill=fill, align="C")
        report.cell(col_w[5], 6, wow_str, fill=fill, align="C", new_x="LMARGIN", new_y="NEXT")
    report.ln(4)


def page1_national_summary(report: Report, df_today: pd.DataFrame, df_valid: pd.DataFrame):
    """Legacy summary page - kept as page 3."""
    report.add_page()
    report.set_font("Helvetica", "", 8)
    report.set_text_color(100, 100, 100)
    report.cell(0, 5, f"Report date: {date.today().isoformat()}    "
                      f"Total lots in dataset: {len(df_valid):,}",
                new_x="LMARGIN", new_y="NEXT")
    report.ln(2)

    report.section_title("Volume & Price by Sex (today)")
    sex_map = {"M": "Male (Steer/Bull)", "F": "Female (Heifer)", "B": "Bull"}
    df_today = df_today.copy()
    df_today["sex_label"] = df_today["sex"].map(sex_map).fillna("Other")
    sex_grp = (df_today.groupby("sex_label")
               .agg(lots=("lot", "count"),
                    avg_ppkg=("ppkg", "mean"),
                    med_ppkg=("ppkg", "median"),
                    avg_price=("price_num", "mean"),
                    avg_wt=("weight_num", "mean"))
               .sort_values("lots", ascending=False))

    col_w = [55, 22, 32, 32, 30, 25]
    headers = ["Sex", "Lots", "Avg EUR/kg", "Median EUR/kg", "Avg Price", "Avg Wt (kg)"]
    report.set_font("Helvetica", "B", 8)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    for h, w in zip(headers, col_w):
        report.cell(w, 6, h, border=0, fill=True, align="C")
    report.ln()
    report.set_text_color(0, 0, 0)
    report.set_font("Helvetica", "", 8)
    for i, (sex, row) in enumerate(sex_grp.iterrows()):
        fill = i % 2 == 0
        report.set_fill_color(248, 248, 255) if fill else report.set_fill_color(255, 255, 255)
        report.cell(col_w[0], 5, sex, fill=fill)
        report.cell(col_w[1], 5, str(row["lots"]), fill=fill, align="C")
        report.cell(col_w[2], 5, f"{row['avg_ppkg']:.2f}" if not np.isnan(row["avg_ppkg"]) else "-", fill=fill, align="C")
        report.cell(col_w[3], 5, f"{row['med_ppkg']:.2f}" if not np.isnan(row["med_ppkg"]) else "-", fill=fill, align="C")
        report.cell(col_w[4], 5, eur(row["avg_price"]) if not np.isnan(row["avg_price"]) else "-", fill=fill, align="C")
        report.cell(col_w[5], 5, f"{row['avg_wt']:.0f}" if not np.isnan(row["avg_wt"]) else "-",
                    fill=fill, align="C", new_x="LMARGIN", new_y="NEXT")
    report.ln(4)

    report.section_title("Lots by Mart (today)")
    mart_summary = (df_today.groupby("mart")
                    .agg(lots=("lot", "count"),
                         avg_ppkg=("ppkg", "mean"),
                         med_ppkg=("ppkg", "median"),
                         avg_price=("price_num", "mean"))
                    .sort_values("lots", ascending=False)
                    .head(20))

    col_w2 = [60, 22, 32, 32, 35]
    headers2 = ["Mart", "Lots", "Avg EUR/kg", "Med EUR/kg", "Avg Price"]
    report.set_font("Helvetica", "B", 8)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    for h, w in zip(headers2, col_w2):
        report.cell(w, 6, h, border=0, fill=True, align="C")
    report.ln()
    report.set_text_color(0, 0, 0)
    report.set_font("Helvetica", "", 8)
    for i, (mart, row) in enumerate(mart_summary.iterrows()):
        fill = i % 2 == 0
        report.set_fill_color(248, 248, 255) if fill else report.set_fill_color(255, 255, 255)
        report.cell(col_w2[0], 5, mart, fill=fill)
        report.cell(col_w2[1], 5, str(row["lots"]), fill=fill, align="C")
        report.cell(col_w2[2], 5, f"{row['avg_ppkg']:.2f}" if not np.isnan(row["avg_ppkg"]) else "-", fill=fill, align="C")
        report.cell(col_w2[3], 5, f"{row['med_ppkg']:.2f}" if not np.isnan(row["med_ppkg"]) else "-", fill=fill, align="C")
        report.cell(col_w2[4], 5, eur(row["avg_price"]) if not np.isnan(row["avg_price"]) else "-",
                    fill=fill, align="C", new_x="LMARGIN", new_y="NEXT")
    report.ln(4)

    try:
        p = chart_weekly_trend(df_valid)
        report.add_image_full(p, h=55)
    except Exception as e:
        report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")


def page2_breed_weight_table(report: Report, df_valid: pd.DataFrame, sex: str, sex_label: str):
    """Full breed × weight bracket price table for one sex."""
    report.add_page()
    report.section_title(f"Breed x Weight Bracket - {sex_label} (last 30 days, avg EUR/kg)")

    # Use last 30 days for sufficient volume per cell
    cutoff = df_valid["scraped_date"].max() - timedelta(days=30)
    df_s = df_valid[(df_valid["scraped_date"] >= cutoff) &
                    (df_valid["sex"] == sex) &
                    (df_valid["breed"].isin(TOP_BREEDS))].copy()

    pivot = (df_s.groupby(["breed", "weight_bracket"])["ppkg"]
             .agg(["mean", "count"])
             .reset_index())

    breed_w = 28
    bracket_w = (report.epw - breed_w) / len(WEIGHT_LABELS)

    # Header
    report.set_font("Helvetica", "B", 8)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    report.cell(breed_w, 7, "Breed", fill=True, align="C")
    for lbl in WEIGHT_LABELS:
        report.cell(bracket_w, 7, lbl, fill=True, align="C")
    report.ln()
    report.set_text_color(0, 0, 0)

    # Data rows
    for i, breed in enumerate(TOP_BREEDS):
        fill = i % 2 == 0
        report.set_fill_color(245, 247, 255) if fill else report.set_fill_color(255, 255, 255)
        report.set_font("Helvetica", "B", 8)
        report.cell(breed_w, 6, breed, fill=fill)
        report.set_font("Helvetica", "", 8)
        for bracket in WEIGHT_LABELS:
            row = pivot[(pivot["breed"] == breed) & (pivot["weight_bracket"] == bracket)]
            if row.empty or row.iloc[0]["count"] < 3:
                val = "-"
            else:
                val = f"{row.iloc[0]['mean']:.2f}"
            report.cell(bracket_w, 6, val, fill=fill, align="C")
        report.ln()

    report.ln(3)
    report.set_font("Helvetica", "I", 7)
    report.set_text_color(120, 120, 120)
    report.cell(0, 5, "  Cells with fewer than 3 lots shown as  '-'   |   Values in EUR/kg",
                new_x="LMARGIN", new_y="NEXT")
    report.set_text_color(0, 0, 0)
    report.ln(4)

    # ── Lot counts table (same layout) ────────────────────────────────────────
    report.section_title(f"Lot Counts - {sex_label} (last 30 days)")
    report.set_font("Helvetica", "B", 8)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    report.cell(breed_w, 7, "Breed", fill=True, align="C")
    for lbl in WEIGHT_LABELS:
        report.cell(bracket_w, 7, lbl, fill=True, align="C")
    report.cell(0, 7, " Total", fill=True, align="C")
    report.ln()
    report.set_text_color(0, 0, 0)

    for i, breed in enumerate(TOP_BREEDS):
        fill = i % 2 == 0
        report.set_fill_color(245, 247, 255) if fill else report.set_fill_color(255, 255, 255)
        report.set_font("Helvetica", "B", 8)
        report.cell(breed_w, 6, breed, fill=fill)
        report.set_font("Helvetica", "", 8)
        total = 0
        for bracket in WEIGHT_LABELS:
            row = pivot[(pivot["breed"] == breed) & (pivot["weight_bracket"] == bracket)]
            cnt = int(row.iloc[0]["count"]) if not row.empty else 0
            total += cnt
            report.cell(bracket_w, 6, str(cnt) if cnt > 0 else "-", fill=fill, align="C")
        report.cell(0, 6, str(total), fill=fill, align="C")
        report.ln()


def page4_regional_and_intelligence(report: Report, df_valid: pd.DataFrame):
    report.add_page()

    # ── Regional breakdown ────────────────────────────────────────────────────
    report.section_title("Regional Breakdown - National Average EUR/kg (last 30 days)")
    cutoff = df_valid["scraped_date"].max() - timedelta(days=30)
    df_r = df_valid[df_valid["scraped_date"] >= cutoff].copy()
    df_r["region"] = df_r["mart"].map(REGION_MAP).fillna("Other")

    regional = (df_r.groupby("region")
                .agg(lots=("ppkg", "count"),
                     avg_ppkg=("ppkg", "mean"),
                     med_ppkg=("ppkg", "median"),
                     avg_price=("price_num", "mean"),
                     avg_wt=("weight_num", "mean"))
                .loc[lambda x: x.index.isin(["Connacht", "Munster", "Leinster", "Ulster"])])

    col_w = [40, 25, 35, 35, 35, 30]
    headers = ["Region", "Lots", "Avg EUR/kg", "Med EUR/kg", "Avg Price", "Avg Wt (kg)"]
    report.set_font("Helvetica", "B", 9)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    for h, w in zip(headers, col_w):
        report.cell(w, 7, h, fill=True, align="C")
    report.ln()
    report.set_text_color(0, 0, 0)
    report.set_font("Helvetica", "", 9)
    for i, (region, row) in enumerate(regional.iterrows()):
        fill = i % 2 == 0
        report.set_fill_color(245, 247, 255) if fill else report.set_fill_color(255, 255, 255)
        report.cell(col_w[0], 6, region, fill=fill)
        report.cell(col_w[1], 6, f"{row['lots']:,}", fill=fill, align="C")
        report.cell(col_w[2], 6, f"{row['avg_ppkg']:.2f}", fill=fill, align="C")
        report.cell(col_w[3], 6, f"{row['med_ppkg']:.2f}", fill=fill, align="C")
        report.cell(col_w[4], 6, eur(row["avg_price"]), fill=fill, align="C")
        report.cell(col_w[5], 6, f"{row['avg_wt']:.0f}", fill=fill, align="C",
                    new_x="LMARGIN", new_y="NEXT")
    report.ln(5)

    # Per-breed breakdown by region
    report.section_title("Avg EUR/kg by Region x Breed (last 30 days)")
    breed_region = (df_r[df_r["breed"].isin(TOP_BREEDS)]
                    .groupby(["breed", "region"])["ppkg"]
                    .agg(["mean", "count"])
                    .reset_index())
    regions = ["Connacht", "Munster", "Leinster", "Ulster"]
    breed_w2 = 28
    reg_w = (report.epw - breed_w2) / len(regions)

    report.set_font("Helvetica", "B", 8)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    report.cell(breed_w2, 7, "Breed", fill=True, align="C")
    for r in regions:
        report.cell(reg_w, 7, r, fill=True, align="C")
    report.ln()
    report.set_text_color(0, 0, 0)
    for i, breed in enumerate(TOP_BREEDS):
        fill = i % 2 == 0
        report.set_fill_color(245, 247, 255) if fill else report.set_fill_color(255, 255, 255)
        report.set_font("Helvetica", "B", 8)
        report.cell(breed_w2, 6, breed, fill=fill)
        report.set_font("Helvetica", "", 8)
        for region in regions:
            row = breed_region[(breed_region["breed"] == breed) &
                                (breed_region["region"] == region)]
            val = f"{row.iloc[0]['mean']:.2f}" if not row.empty and row.iloc[0]["count"] >= 3 else "-"
            report.cell(reg_w, 6, val, fill=fill, align="C")
        report.ln()
    report.ln(5)

    # ── ICBF stars premium ────────────────────────────────────────────────────
    report.section_title("ICBF Stars Premium - Avg EUR/kg by Star Rating (all time)")
    stars_grp = (df_valid.groupby("icbf_stars_n")["ppkg"]
                 .agg(["mean", "median", "count"])
                 .reset_index()
                 .rename(columns={"icbf_stars_n": "Stars", "mean": "Avg EUR/kg",
                                  "median": "Med EUR/kg", "count": "Lots"}))
    stars_grp = stars_grp[stars_grp["Lots"] >= 10]

    col_w3 = [30, 30, 35, 35, 60]
    headers3 = ["Stars", "Lots", "Avg EUR/kg", "Med EUR/kg", "vs 0-star premium"]
    base_ppkg = stars_grp.loc[stars_grp["Stars"] == 0, "Avg EUR/kg"]
    base_val  = base_ppkg.iloc[0] if not base_ppkg.empty else np.nan

    report.set_font("Helvetica", "B", 9)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    for h, w in zip(headers3, col_w3):
        report.cell(w, 7, h, fill=True, align="C")
    report.ln()
    report.set_text_color(0, 0, 0)
    report.set_font("Helvetica", "", 9)
    for i, row in stars_grp.iterrows():
        fill = i % 2 == 0
        report.set_fill_color(245, 247, 255) if fill else report.set_fill_color(255, 255, 255)
        stars_disp = f"{int(row['Stars'])} stars" if row["Stars"] > 0 else "No stars"
        premium = row["Avg EUR/kg"] - base_val if not np.isnan(base_val) and row["Stars"] > 0 else np.nan
        prem_str = (f"+EUR {premium:.2f}/kg" if not np.isnan(premium) else "-")
        report.cell(col_w3[0], 6, stars_disp, fill=fill, align="C")
        report.cell(col_w3[1], 6, f"{row['Lots']:,}", fill=fill, align="C")
        report.cell(col_w3[2], 6, f"{row['Avg EUR/kg']:.2f}", fill=fill, align="C")
        report.cell(col_w3[3], 6, f"{row['Med EUR/kg']:.2f}", fill=fill, align="C")
        report.cell(col_w3[4], 6, prem_str, fill=fill, align="C",
                    new_x="LMARGIN", new_y="NEXT")
    report.ln(5)

    # ── Biggest movers chart ───────────────────────────────────────────────────
    try:
        p = chart_breed_movers(df_valid)
        if p:
            report.section_title("Biggest Breed Price Movers - Week on Week")
            report.add_image_full(p, h=65)
    except Exception as e:
        report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")


def page5_ml_metrics(report: Report, meta: dict, df_valid: pd.DataFrame):
    if not meta:
        return
    report.add_page()
    report.section_title("Model Performance Summary")

    report.set_font("Helvetica", "", 9)
    n_tr = meta.get("n_train", "?")
    n_te = meta.get("n_test", "?")
    cv   = meta.get("cv_mae_eur_kg", "?")
    cv_s = meta.get("cv_mae_std", "?")
    report.cell(0, 6,
        f"Training rows: {n_tr:,}    Test rows: {n_te:,}    "
        f"CV MAE: EUR {cv}/kg +/- EUR {cv_s}/kg",
        new_x="LMARGIN", new_y="NEXT")
    report.ln(2)

    tm = meta.get("test_metrics", {})
    col_w = [70, 50]
    rows = [
        ("Test R²",            f"{tm.get('R2', 'N/A')}"),
        ("Test MAE (EUR/kg)",  f"{tm.get('MAE_eur_kg', 'N/A')}"),
        ("Test RMSE (EUR/kg)", f"{tm.get('RMSE_eur_kg', 'N/A')}"),
        ("MAPE (%)",           f"{tm.get('MAPE_%', 'N/A')}"),
        ("Within 5%",          f"{tm.get('within_5pct', 'N/A')}%"),
        ("Within 10%",         f"{tm.get('within_10pct', 'N/A')}%"),
        ("Within 20%",         f"{tm.get('within_20pct', 'N/A')}%"),
        ("Total lots in DB",   f"{len(df_valid):,}"),
    ]
    report.set_font("Helvetica", "B", 9)
    report.set_fill_color(*NAVY)
    report.set_text_color(255, 255, 255)
    report.cell(col_w[0], 7, "Metric", fill=True)
    report.cell(col_w[1], 7, "Value", fill=True)
    report.ln()
    report.set_text_color(0, 0, 0)
    report.set_font("Helvetica", "", 9)
    for i, (k, v) in enumerate(rows):
        fill = i % 2 == 0
        report.set_fill_color(245, 247, 255) if fill else report.set_fill_color(255, 255, 255)
        report.cell(col_w[0], 6, k, fill=fill)
        report.cell(col_w[1], 6, v, fill=fill, new_x="LMARGIN", new_y="NEXT")
    report.ln(5)

    try:
        p = chart_feature_importance(meta)
        report.add_image_full(p, h=65)
    except Exception as e:
        report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")


# ── Email ─────────────────────────────────────────────────────────────────────

def get_smtp_credentials():
    if EMAIL_CONFIG.exists():
        cfg = json.loads(EMAIL_CONFIG.read_text())
        return cfg.get("user"), cfg.get("pass")
    return os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")


def send_email(pdf_path: Path):
    user, pwd = get_smtp_credentials()
    if not user or not pwd:
        print("  [WARN] No SMTP credentials. Skipping email.")
        return

    today = date.today().isoformat()
    msg = MIMEMultipart()
    msg["From"]    = user
    msg["To"]      = ", ".join(RECIPIENTS)
    msg["Subject"] = f"MartIndex Daily Report - {today}"
    msg.attach(MIMEText(
        f"Please find attached the MartIndex daily cattle market report for {today}.\n\n"
        "This email was generated automatically.\n", "plain"))

    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename=MartIndex_Report_{today}.pdf")
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
            server.ehlo()
            server.starttls()
            server.login(user, pwd)
            server.sendmail(user, RECIPIENTS, msg.as_string())
        print(f"  Email sent to {len(RECIPIENTS)} recipients.")
    except Exception as e:
        print(f"  [ERROR] Failed to send email: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not LOTS_CSV.exists():
        print(f"No data at {LOTS_CSV}")
        return

    print("Loading data...")
    df_all, df_valid = load_data()
    df_today = get_today_df(df_valid)
    print(f"Building PDF for {len(df_today):,} lots...")

    meta = json.loads(META_JSON.read_text()) if META_JSON.exists() else None

    report = Report(orientation="P", unit="mm", format="A4")
    report.set_auto_page_break(auto=True, margin=12)

    fp = load_factory_data()

    page1_market_overview(report, df_valid, fp)
    page2_price_tables(report, df_valid, fp)
    page1_national_summary(report, df_today, df_valid)
    page4_regional_and_intelligence(report, df_valid)

    report.output(str(REPORT_PATH))
    print(f"  Report saved → {REPORT_PATH.name}")

    print("Sending email...")
    send_email(REPORT_PATH)

    for f in DIR.glob("_chart_*.png"):
        f.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()
