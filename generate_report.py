#!/usr/bin/env python3
"""
Daily PDF Report Generator for MartBids.
Builds a PDF with market KPIs, charts, and model metrics,
then emails it to configured recipients.

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
from datetime import date

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
TEST_PRED    = DIR / "model_test_predictions.csv"
REPORT_PATH  = DIR / "daily_report.pdf"
EMAIL_CONFIG = DIR / "email_config.json"

RECIPIENTS = [
    "johnscanlan52@yahoo.ie",
    "michaelscanlan05@yahoo.com",
    "margaretscanlan14@hotmail.com",
]

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_eur(s):
    if pd.isna(s) or str(s).strip() == "":
        return np.nan
    return pd.to_numeric(str(s).replace("€", "").replace(",", "").strip(),
                         errors="coerce")

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


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_lots_by_mart(df: pd.DataFrame) -> Path:
    counts = df["mart"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.barh(ax=ax, color="#2196F3")
    ax.set_xlabel("Lots sold today")
    ax.set_title("Lots by Mart (today)")
    ax.invert_yaxis()
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_marts.png")

def chart_price_hist(df: pd.DataFrame) -> Path:
    prices = df["price_num"].dropna()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(prices, bins=40, color="#4CAF50", edgecolor="white")
    ax.set_xlabel("Price (€)")
    ax.set_ylabel("Count")
    ax.set_title("Price Distribution (today)")
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_hist.png")

def chart_breed_ppkg(df: pd.DataFrame) -> Path:
    df2 = df.dropna(subset=["price_per_kg_num", "breed"])
    top = df2["breed"].value_counts().head(8).index
    df2 = df2[df2["breed"].isin(top)]
    order = df2.groupby("breed")["price_per_kg_num"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [df2.loc[df2["breed"] == b, "price_per_kg_num"].dropna().values for b in order]
    ax.boxplot(data, labels=order, patch_artist=True,
               boxprops=dict(facecolor="#FF9800", alpha=0.7))
    ax.set_xlabel("Breed")
    ax.set_ylabel("€/kg")
    ax.set_title("Price/kg by Breed (today)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_breed.png")

def chart_feature_importance(meta: dict) -> Path:
    fi = pd.Series(meta.get("feature_importances", {})).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    fi[::-1].plot.barh(ax=ax, color="#9C27B0")
    ax.set_title("Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return save_tmp(fig_to_png(fig), "_chart_fi.png")


# ── PDF builder ───────────────────────────────────────────────────────────────

def eur(val, fmt=":,.0f"):
    """Format a number as EUR (using 'EUR' prefix for PDF latin-1 compat)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"EUR {val:{fmt.strip(':')}}"


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(33, 150, 243)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "  MartBids Daily Report", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Generated {date.today().isoformat()} - MartBids Intelligence", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(230, 240, 255)
        self.cell(0, 8, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)

    def kpi_row(self, items: list[tuple[str, str]]):
        self.set_font("Helvetica", "", 10)
        cell_w = self.epw / len(items)
        for label, val in items:
            self.set_font("Helvetica", "B", 13)
            x = self.get_x()
            y = self.get_y()
            self.multi_cell(cell_w, 7, val, align="C")
            self.set_xy(x, y + 7)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(100, 100, 100)
            self.multi_cell(cell_w, 5, label, align="C")
            self.set_text_color(0, 0, 0)
            self.set_xy(x + cell_w, y)
        self.ln(14)

    def add_image_full(self, path: Path, h: float = 60):
        if path.exists():
            self.image(str(path), x=self.l_margin, w=self.epw, h=h)
            self.ln(3)

    def metrics_table(self, metrics: dict, title: str):
        self.section_title(title)
        self.set_font("Helvetica", "", 9)
        for k, v in metrics.items():
            self.cell(70, 6, k, border="B")
            self.cell(0, 6, str(v), border="B", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


def build_pdf(df_today, df_all, meta):
    report = Report(orientation="P", unit="mm", format="A4")
    report.set_auto_page_break(auto=True, margin=12)
    report.add_page()

    report.set_font("Helvetica", "", 9)
    report.set_text_color(100, 100, 100)
    report.cell(0, 6, f"Report date: {date.today().isoformat()}    "
                      f"Total lots in dataset: {len(df_all):,}", new_x="LMARGIN", new_y="NEXT")
    report.ln(2)

    # ── Page 1: KPIs ──────────────────────────────────────────────────────────
    report.section_title("Today's Market Summary")
    n_lots   = len(df_today)
    n_marts  = df_today["mart"].nunique()
    avg_p    = df_today["price_num"].mean()
    med_ppkg = df_today["price_per_kg_num"].median()
    avg_wt   = df_today["weight_num"].mean()

    report.kpi_row([
        ("Lots Sold",        str(n_lots)),
        ("Marts Active",     str(n_marts)),
        ("Avg Price",        eur(avg_p)),
        ("Median EUR/kg",    eur(med_ppkg, ":.2f")),
        ("Avg Weight",       f"{avg_wt:.0f} kg" if not np.isnan(avg_wt) else "N/A"),
    ])

    # Mart table
    report.section_title("Lots by Mart")
    mart_summary = (df_today.groupby("mart")
                    .agg(lots=("lot", "count"),
                         avg_price=("price_num", "mean"),
                         avg_ppkg=("price_per_kg_num", "mean"))
                    .sort_values("lots", ascending=False)
                    .head(20))

    report.set_font("Helvetica", "B", 8)
    col_w = [60, 25, 35, 35]
    for h, w in zip(["Mart", "Lots", "Avg Price (EUR)", "Avg EUR/kg"], col_w):
        report.cell(w, 6, h, border=1)
    report.ln()

    report.set_font("Helvetica", "", 8)
    for mart, row in mart_summary.iterrows():
        report.cell(col_w[0], 5, mart, border="B")
        report.cell(col_w[1], 5, str(row["lots"]), border="B", align="C")
        ap = eur(row["avg_price"]) if not np.isnan(row["avg_price"]) else "-"
        ak = eur(row["avg_ppkg"], ":.2f") if not np.isnan(row["avg_ppkg"]) else "-"
        report.cell(col_w[2], 5, ap, border="B", align="R")
        report.cell(col_w[3], 5, ak, border="B", align="R", new_x="LMARGIN", new_y="NEXT")
    report.ln(4)

    # ── Page 2: Charts ────────────────────────────────────────────────────────
    report.add_page()
    report.section_title("Charts")

    try:
        p = chart_lots_by_mart(df_today)
        report.add_image_full(p, h=65)
    except Exception as e:
        report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")

    try:
        p = chart_price_hist(df_today)
        report.add_image_full(p, h=60)
    except Exception as e:
        report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")

    try:
        p = chart_breed_ppkg(df_today)
        report.add_image_full(p, h=65)
    except Exception as e:
        report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")

    # ── Page 3: ML metrics ────────────────────────────────────────────────────
    if meta:
        report.add_page()
        report.section_title("Model Performance")

        report.set_font("Helvetica", "", 9)
        n_tr = meta.get("n_train", "?")
        n_te = meta.get("n_test", "?")
        cv   = meta.get("cv_mae_eur_kg", meta.get("cv_mae_eur", "?"))
        cv_s = meta.get("cv_mae_std", "?")
        report.cell(0, 6,
            f"Training rows: {n_tr:,}    Test rows: {n_te:,}    "
            f"CV MAE: EUR {cv}/kg +/- EUR {cv_s}/kg",
            new_x="LMARGIN", new_y="NEXT")
        report.ln(2)

        report.metrics_table(meta.get("test_metrics", {}), "Test Set Metrics")

        try:
            p = chart_feature_importance(meta)
            report.add_image_full(p, h=65)
        except Exception as e:
            report.cell(0, 6, f"Chart error: {e}", new_x="LMARGIN", new_y="NEXT")

        # Cumulative dataset growth
        report.section_title("Dataset Summary (all time)")
        report.set_font("Helvetica", "", 9)
        n_all    = len(df_all)
        n_breeds = df_all["breed"].nunique()
        n_marts  = df_all["mart"].nunique()
        report.cell(0, 6,
            f"Total lots: {n_all:,}    Breeds: {n_breeds}    Marts: {n_marts}",
            new_x="LMARGIN", new_y="NEXT")

    return report


# ── Email ─────────────────────────────────────────────────────────────────────

def get_smtp_credentials():
    if EMAIL_CONFIG.exists():
        cfg = json.loads(EMAIL_CONFIG.read_text())
        return cfg.get("user"), cfg.get("pass")
    return os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")

def send_email(pdf_path: Path):
    user, pwd = get_smtp_credentials()
    if not user or not pwd:
        print("  [WARN] No SMTP credentials found. Skipping email.")
        print("         Create email_config.json or set SMTP_USER / SMTP_PASS env vars.")
        return

    today = date.today().isoformat()
    msg = MIMEMultipart()
    msg["From"]    = user
    msg["To"]      = ", ".join(RECIPIENTS)
    msg["Subject"] = f"MartBids Daily Report — {today}"

    body = (
        f"Please find attached the MartBids daily cattle market report for {today}.\n\n"
        "This email was generated automatically.\n"
    )
    msg.attach(MIMEText(body, "plain"))

    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename=MartBids_Report_{today}.pdf")
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
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

    df = pd.read_csv(LOTS_CSV)
    df["price_num"]       = df["price"].apply(parse_eur)
    df["weight_num"]      = pd.to_numeric(df["weight"], errors="coerce")
    df["price_per_kg_num"] = df["price_num"] / df["weight_num"].replace(0, np.nan)
    df["scraped_date"]    = pd.to_datetime(df["scraped_date"], errors="coerce")

    today_str  = date.today().isoformat()
    df_today   = df[df["scraped_date"].dt.strftime("%Y-%m-%d") == today_str].copy()
    if df_today.empty:
        # Fall back to most recent date in dataset
        latest = df["scraped_date"].max()
        df_today = df[df["scraped_date"] == latest].copy()
        print(f"No data for today ({today_str}), using latest: {latest.date()}")

    meta = None
    if META_JSON.exists():
        meta = json.loads(META_JSON.read_text())

    print(f"Building PDF for {len(df_today):,} lots...")
    report = build_pdf(df_today, df, meta)
    report.output(str(REPORT_PATH))
    print(f"  Report saved → {REPORT_PATH.name}")

    print("Sending email...")
    send_email(REPORT_PATH)

    # Cleanup temp chart images
    for f in DIR.glob("_chart_*.png"):
        f.unlink(missing_ok=True)

    print("Done.")


if __name__ == "__main__":
    main()
