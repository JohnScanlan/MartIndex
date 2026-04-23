"""
MartIndex — Irish Cattle Market Intelligence
============================================
Run with:  streamlit run dashboard.py
"""

import io
import json
import warnings
import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MartIndex",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

# ── MartIndex colour palette ──────────────────────────────────────────────────
FB_BLUE   = "#1B2A4A"   # primary navy
FB_DARK   = "#1B2A4A"   # text / headings
FB_BG     = "#F4F5F7"   # page background
FB_CARD   = "#FFFFFF"   # card / panel background
FB_GREY   = "#6B7280"   # muted text
FB_GREEN  = "#276749"   # positive / success
FB_RED    = "#9B2335"   # negative / alert
FB_BORDER = "#DDE1E7"   # borders
GOLD      = "#C9A84C"   # accent gold

PALETTE = [FB_BLUE, GOLD, FB_GREEN, FB_RED, "#4B5EA6",
           "#7C6A3A", "#5B8A6A", "#8B3A4A", "#2E4A7A", "#A08030"]

st.markdown(f"""
<style>
    /* ── App background ──────────────────────────────────────────────────── */
    .stApp {{ background-color: {FB_BG}; }}

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background-color: {FB_CARD};
        border-right: 1px solid {FB_BORDER};
    }}
    section[data-testid="stSidebar"] > div {{ padding-top: 1rem; }}

    /* ── Typography (scoped — not applied inside widgets) ────────────────── */
    h1 {{ color: {FB_DARK} !important; font-size: 1.7rem !important;
          font-weight: 800 !important; letter-spacing: -0.5px; }}
    h2, h3 {{ color: {FB_DARK} !important; font-weight: 700 !important; }}
    label {{ color: {FB_DARK} !important; font-weight: 500; }}
    .stMarkdown p {{ color: {FB_DARK} !important; font-size: 0.95rem; }}
    .stCaptionContainer p {{ color: {FB_GREY} !important; font-size: 0.82rem !important; }}

    /* ── Metric cards ────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {FB_CARD};
        border: 1px solid {FB_BORDER};
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 6px rgba(27,42,74,0.08);
    }}
    [data-testid="stMetricLabel"] p {{
        color: {FB_GREY} !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }}
    [data-testid="stMetricValue"] {{
        color: {FB_DARK} !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricDelta"] svg {{ display: none; }}

    /* ── Tabs ────────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {FB_CARD};
        border-radius: 10px;
        padding: 5px;
        gap: 3px;
        border: 1px solid {FB_BORDER};
        box-shadow: 0 1px 4px rgba(27,42,74,0.06);
        flex-wrap: wrap;
    }}
    /* Unselected tab */
    .stTabs [data-baseweb="tab"] {{
        border-radius: 7px;
        padding: 7px 14px;
        white-space: nowrap;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {{
        color: {FB_DARK} !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }}
    /* Selected tab */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {FB_BLUE} !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs [data-baseweb="tab"][aria-selected="true"] div {{
        color: #FFFFFF !important;
    }}
    /* Remove the default underline indicator */
    .stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
    .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

    /* ── Input controls (select box / multiselect outer shell) ───────────── */
    [data-baseweb="select"] > div:first-child,
    [data-baseweb="input"] > div:first-child {{
        background-color: {FB_CARD} !important;
        border-color: {FB_BORDER} !important;
        border-radius: 8px !important;
    }}
    /* All text inside any select control — value, placeholder, options */
    [data-baseweb="select"] *,
    [data-baseweb="select"] input,
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="select"] p {{
        color: {FB_DARK} !important;
        background-color: transparent !important;
    }}
    /* Single-value display (selectbox chosen item) */
    [data-baseweb="select"] [data-testid="stSelectbox"],
    [data-baseweb="select"] [class*="singleValue"],
    [data-baseweb="select"] [class*="SingleValue"] {{
        color: {FB_DARK} !important;
    }}

    /* ── Multiselect tags ────────────────────────────────────────────────── */
    [data-baseweb="tag"] {{
        background-color: #EEF2F8 !important;
        border: 1px solid {FB_BLUE} !important;
        border-radius: 6px !important;
    }}
    [data-baseweb="tag"] span {{ color: {FB_BLUE} !important; font-weight: 600; }}

    /* ── Dropdown list (the popover that opens on click) ─────────────────── */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"] {{
        background-color: {FB_CARD} !important;
        border: 1px solid {FB_BORDER} !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(27,42,74,0.14) !important;
    }}
    /* Each option row */
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"],
    [data-baseweb="list"] li {{
        background-color: {FB_CARD} !important;
        color: {FB_DARK} !important;
    }}
    /* Option text spans */
    [data-baseweb="menu"] li span,
    [data-baseweb="menu"] [role="option"] span,
    [data-baseweb="menu"] li div,
    [data-baseweb="menu"] [role="option"] div {{
        color: {FB_DARK} !important;
    }}
    /* Hover / selected option */
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [aria-selected="true"],
    [data-baseweb="menu"] li[aria-selected="true"] span,
    [data-baseweb="menu"] li:hover span {{
        background-color: #EEF2F8 !important;
        color: {FB_BLUE} !important;
    }}

    /* ── Number / text inputs ────────────────────────────────────────────── */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {{
        color: {FB_DARK} !important;
        background-color: {FB_CARD} !important;
        border-color: {FB_BORDER} !important;
        border-radius: 8px !important;
    }}

    /* ── Sliders ─────────────────────────────────────────────────────────── */
    [data-testid="stSlider"] p {{ color: {FB_DARK} !important; }}
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {{
        color: {FB_GREY} !important; font-size: 0.78rem;
    }}

    /* ── Checkboxes / radio ──────────────────────────────────────────────── */
    [data-testid="stCheckbox"] span {{ color: {FB_DARK} !important; }}
    [data-testid="stRadio"] label span {{ color: {FB_DARK} !important; }}

    /* ── Plotly chart panels ─────────────────────────────────────────────── */
    [data-testid="stPlotlyChart"] {{
        background: {FB_CARD};
        border: 1px solid {FB_BORDER};
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 1px 4px rgba(27,42,74,0.06);
    }}

    /* ── Dataframes ──────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {FB_BORDER} !important;
        border-radius: 10px !important;
        overflow: hidden;
    }}

    /* ── Buttons ─────────────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {{
        background: {FB_BLUE};
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 10px 26px;
        font-size: 0.95rem;
        transition: opacity 0.15s;
    }}
    .stButton > button[kind="primary"]:hover {{ opacity: 0.88; }}
    .stButton > button {{
        color: {FB_DARK} !important;
        background: {FB_CARD} !important;
        border: 1px solid {FB_BORDER} !important;
        border-radius: 8px;
    }}

    /* ── Alert boxes ─────────────────────────────────────────────────────── */
    div[data-testid="stAlert"] {{ border-radius: 8px; }}

    /* ── Divider ─────────────────────────────────────────────────────────── */
    hr {{ border-color: {FB_BORDER} !important; margin: 1rem 0; }}

    /* ── Mobile responsive ───────────────────────────────────────────────── */
    @media (max-width: 768px) {{
        h1 {{ font-size: 1.35rem !important; }}
        h2, h3 {{ font-size: 1.05rem !important; }}
        [data-testid="stMetricValue"] {{ font-size: 1.2rem !important; }}
        [data-testid="stMetricLabel"] p {{ font-size: 0.68rem !important; }}
        .stTabs [data-baseweb="tab"] {{
            padding: 6px 10px;
        }}
        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span {{
            font-size: 0.78rem !important;
        }}
        [data-testid="stPlotlyChart"] {{ padding: 4px; }}
        section[data-testid="stSidebar"] {{ min-width: 240px !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_eur(s):
    if pd.isna(s) or str(s).strip() == "":
        return np.nan
    return pd.to_numeric(str(s).replace("€", "").replace(",", "").strip(), errors="coerce")

def count_stars(s):
    if pd.isna(s) or str(s).strip() == "":
        return 0
    return str(s).count("☆") + str(s).count("★")

def export_score_fn(s):
    return {"Yes": 2, "ReTest": 1, "No": 0}.get(str(s).strip(), np.nan)

NUMERIC_FEATURES = [
    "weight", "age_months", "days_in_herd", "no_of_owners",
    "icbf_cbv_num", "icbf_replacement_num", "icbf_ebi_num", "icbf_stars",
    "log_weight", "weight_per_month", "icbf_has_data",
    "has_genomic", "quality_assured", "bvd_ok", "export_score",
    "temp_max_c", "temp_min_c", "precipitation_mm", "wind_speed_kmh",
    "sale_month",
]
CATEGORICAL_FEATURES = ["breed_grp", "sex_clean", "mart", "dam_breed_grp", "breed_sex", "source"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _chart_layout(**kw):
    """Common Plotly layout defaults — explicit colours so dark-mode never bleeds in."""
    base = dict(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color=FB_DARK, family="system-ui, -apple-system, sans-serif"),
        title_font=dict(color=FB_DARK),
    )
    base.update(kw)
    return base


def _show_chart(fig):
    """Apply axis colours then render — use everywhere instead of st.plotly_chart."""
    fig.update_xaxes(
        tickfont=dict(color=FB_DARK),
        gridcolor="#EAECEF",
        linecolor=FB_BORDER,
        zerolinecolor="#EAECEF",
    )
    fig.update_yaxes(
        tickfont=dict(color=FB_DARK),
        gridcolor="#EAECEF",
        linecolor=FB_BORDER,
        zerolinecolor="#EAECEF",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    # ── MartBids ──────────────────────────────────────────────────────────────
    mb = pd.read_csv(BASE / "sold_lots.csv")
    mb["source"]               = "MartBids"
    mb["price_num"]            = mb["price"].apply(parse_eur)
    mb["weight"]               = pd.to_numeric(mb["weight"], errors="coerce")
    mb["age_months"]           = pd.to_numeric(mb["age_months"], errors="coerce")
    mb["days_in_herd"]         = pd.to_numeric(mb["days_in_herd"], errors="coerce")
    mb["no_of_owners"]         = pd.to_numeric(mb["no_of_owners"], errors="coerce")
    mb["price_per_kg_num"]     = mb["price_num"] / mb["weight"].replace(0, np.nan)
    mb["icbf_cbv_num"]         = mb["icbf_cbv"].apply(parse_eur)
    mb["icbf_replacement_num"] = mb["icbf_replacement_index"].apply(parse_eur)
    mb["icbf_ebi_num"]         = mb["icbf_ebi"].apply(parse_eur)
    mb["icbf_stars"]           = mb["icbf_across_breed"].apply(count_stars)
    mb["has_genomic"]          = (mb["icbf_genomic_eval"] == "Yes").astype(int)
    mb["quality_assured"]      = (mb["quality_assurance"] == "Yes").astype(int)
    mb["bvd_ok"]               = (mb["bvd_tested"] == "Yes").astype(int)
    mb["export_score"]         = mb["export_status"].apply(export_score_fn)
    mb["sex_clean"]            = mb["sex"].map({"M": "M", "F": "F", "B": "B"}).fillna("Unknown")
    mb_dt                      = pd.to_datetime(mb["scraped_date"], errors="coerce")
    mb["sale_date"]            = mb_dt
    mb["sale_month"]           = mb_dt.dt.month.fillna(0).astype(int)

    frames = [mb]

    # ── Livestock-Live ────────────────────────────────────────────────────────
    lsl_path = BASE / "lsl_lots.csv"
    if lsl_path.exists():
        lsl = pd.read_csv(lsl_path)
        lsl["source"]           = "Livestock-Live"
        lsl["price_num"]        = pd.to_numeric(lsl["price"], errors="coerce")
        lsl["weight"]           = pd.to_numeric(lsl["weight"], errors="coerce")
        lsl["age_months"]       = pd.to_numeric(lsl["age_months"], errors="coerce")
        lsl["price_per_kg_num"] = pd.to_numeric(lsl["price_per_kg"], errors="coerce")
        lsl["icbf_stars"]       = pd.to_numeric(lsl["icbf_stars"], errors="coerce").fillna(0)
        lsl["sex_clean"]        = lsl["sex"].map({"M": "M", "F": "F", "B": "B"}).fillna("Unknown")
        lsl_dt                  = pd.to_datetime(lsl["sale_date"], errors="coerce")
        lsl["sale_date"]        = lsl_dt
        lsl["scraped_date"]     = lsl_dt
        lsl["sale_month"]       = lsl_dt.dt.month.fillna(0).astype(int)
        # MartBids-only columns absent from LSL → NaN
        for col in ["days_in_herd", "no_of_owners", "dam_breed",
                    "icbf_cbv_num", "icbf_replacement_num", "icbf_ebi_num",
                    "has_genomic", "quality_assured", "bvd_ok", "export_score"]:
            lsl[col] = np.nan
        frames.append(lsl)

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["price_num"] > 0].dropna(subset=["price_num", "weight"]).copy()

    # ── Global hard filters ────────────────────────────────────────────────
    df = df[df["weight"] > 0].copy()
    df = df[df["price_num"] <= 10_000].copy()                  # remove outliers
    df = df[df["age_months"].isna() | (df["age_months"] <= 96)].copy()  # max 8 years
    df = df[df["sex_clean"] != "Unknown"].copy()                         # known sex only

    # ── Clean invalid breed codes (pure numbers, single chars, blanks) ─────
    df["breed"] = df["breed"].astype(str).str.strip()
    df.loc[
        df["breed"].str.fullmatch(r'\d+') |
        (df["breed"].str.len() < 2) |
        df["breed"].isin(["", "nan", "None"]),
        "breed"
    ] = np.nan

    # ── Shared feature engineering ─────────────────────────────────────────
    top_breeds      = df["breed"].value_counts().head(20).index
    df["breed_grp"] = df["breed"].where(df["breed"].isin(top_breeds), "Other")
    df["breed_sex"] = df["breed_grp"] + "_" + df["sex_clean"]

    if "dam_breed" in df.columns:
        top_dam = df["dam_breed"].value_counts().head(15).index
        df["dam_breed_grp"] = (df["dam_breed"]
                               .where(df["dam_breed"].isin(top_dam), "Other")
                               .fillna("Unknown"))
    else:
        df["dam_breed_grp"] = "Unknown"

    # ── Weather merge ─────────────────────────────────────────────────────
    wx_path = BASE / "weather_cache.csv"
    if wx_path.exists():
        wx = pd.read_csv(wx_path)
        wx["date"] = pd.to_datetime(wx["date"]).dt.strftime("%Y-%m-%d")
        df["sale_date_str"] = pd.to_datetime(df["sale_date"]).dt.strftime("%Y-%m-%d")
        df = df.merge(wx.rename(columns={"date": "sale_date_str"}),
                      on=["mart", "sale_date_str"], how="left")
    else:
        for col in ["temp_max_c", "temp_min_c", "precipitation_mm", "wind_speed_kmh"]:
            df[col] = np.nan

    return df


@st.cache_resource
def load_model():
    path = BASE / "cattle_model.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_meta():
    path = BASE / "model_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_test_preds():
    path = BASE / "model_test_predictions.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource
def load_shap_objects():
    sv_path = BASE / "shap_values.pkl"
    bg_path = BASE / "shap_background.pkl"
    if sv_path.exists() and bg_path.exists():
        return joblib.load(sv_path), joblib.load(bg_path)
    return None, None


@st.cache_data
def load_weather_latest():
    wx_path = BASE / "weather_cache.csv"
    if not wx_path.exists():
        return {}
    wx = pd.read_csv(wx_path)
    wx["date"] = pd.to_datetime(wx["date"])
    latest = wx.sort_values("date").groupby("mart").last().reset_index()
    return latest.set_index("mart").to_dict("index")


@st.cache_data
def compute_growth_model(df_hash):
    """
    Fit weight = a + b*sqrt(age_months) per breed_grp × sex_clean.
    Sqrt model captures natural deceleration of growth as animals mature.
    Returns dict: (breed, sex) → (a, b).
    """
    df = load_data()
    grp = df.dropna(subset=["age_months", "weight"])
    grp = grp[grp["age_months"] > 0]

    def _fit(ages, weights):
        try:
            popt, _ = curve_fit(
                lambda t, a, b: a + b * np.sqrt(t),
                ages, weights,
                p0=[weights.mean() - 50 * np.sqrt(ages.mean()), 50],
                maxfev=3000,
            )
            return float(popt[0]), max(float(popt[1]), 1.0)
        except Exception:
            lr = LinearRegression().fit(ages.reshape(-1, 1), weights)
            return float(lr.intercept_), max(float(lr.coef_[0]) * 3, 1.0)

    params = {}
    for (breed, sex), g in grp.groupby(["breed_grp", "sex_clean"]):
        if len(g) < 15:
            continue
        params[(breed, sex)] = _fit(g["age_months"].values, g["weight"].values)

    params[("_default", "_default")] = _fit(
        grp["age_months"].values, grp["weight"].values
    )
    return params


def project_weight(cur_weight, cur_age, months_ahead, _a, b):
    """
    Project weight using the sqrt growth model anchored to the animal's
    current weight. Only the b (slope) term drives the projection — the
    intercept is discarded so the curve passes through (cur_age, cur_weight).
    """
    future_age = cur_age + months_ahead
    gain = b * (np.sqrt(max(future_age, 0.01)) - np.sqrt(max(cur_age, 0.01)))
    return cur_weight + max(gain, 0.5 * months_ahead)


# ── Sidebar filters ───────────────────────────────────────────────────────────

def sidebar_filters(df):
    st.sidebar.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0 16px;">
      <div>
        <div style="font-size:1.1rem;font-weight:800;color:{FB_DARK};">MartIndex</div>
        <div style="font-size:0.72rem;color:{FB_GREY};">Irish Cattle Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"<p style='color:{FB_GREY};font-weight:700;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.5px;'>Filters</p>", unsafe_allow_html=True)

    marts = sorted(df["mart"].unique())
    sel_marts = st.sidebar.multiselect("Mart", marts, default=marts, placeholder="All marts")

    breeds = sorted(df["breed"].dropna().unique())
    sel_breeds = st.sidebar.multiselect("Breed", breeds, default=[], placeholder="All breeds")

    sexes = sorted(df["sex_clean"].unique())
    sel_sex = st.sidebar.multiselect("Sex", sexes, default=sexes, placeholder="All")

    min_w, max_w = int(df["weight"].min()), int(df["weight"].max())
    weight_range = st.sidebar.slider("Weight (kg)", min_w, max_w, (min_w, max_w))

    min_a = int(df["age_months"].dropna().min())
    max_a = int(df["age_months"].dropna().max())
    age_range = st.sidebar.slider("Age (months)", min_a, max_a, (min_a, max_a))

    mask = (
        df["mart"].isin(sel_marts if sel_marts else marts)
        & df["sex_clean"].isin(sel_sex if sel_sex else sexes)
        & df["weight"].between(*weight_range)
        & df["age_months"].between(*age_range)
    )
    if sel_breeds:
        mask &= df["breed"].isin(sel_breeds)

    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Overview
# ═══════════════════════════════════════════════════════════════════════════════

def tab_overview(df):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Lots",   f"{len(df):,}")
    k2.metric("Avg Price",    f"€{df['price_num'].mean():,.0f}")
    k3.metric("Avg Weight",   f"{df['weight'].mean():.0f} kg")
    k4.metric("Avg €/kg",     f"€{df['price_per_kg_num'].mean():.2f}")
    k5.metric("Active Marts", str(df["mart"].nunique()))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        mart_avg = (df.groupby("mart")["price_num"]
                      .agg(["mean", "count"])
                      .rename(columns={"mean": "avg_price", "count": "lots"})
                      .sort_values("avg_price", ascending=True)
                      .reset_index())
        fig = px.bar(
            mart_avg, x="avg_price", y="mart", orientation="h",
            text=mart_avg["avg_price"].map("€{:,.0f}".format),
            hover_data={"lots": True, "avg_price": ":,.0f"},
            color="avg_price", color_continuous_scale=[[0, "#E7F3FF"], [1, FB_BLUE]],
            title="Average Sold Price by Mart",
            labels={"avg_price": "Avg Price (€)", "mart": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=520,
                          margin=dict(l=10, r=80), **_chart_layout())
        _show_chart(fig)

    with col_b:
        breed_stats = (df.groupby("breed")
                         .agg(lots=("price_num", "count"),
                              avg_price=("price_num", "mean"),
                              avg_weight=("weight", "mean"))
                         .reset_index()
                         .sort_values("lots", ascending=False)
                         .head(25))
        fig2 = px.scatter(
            breed_stats, x="avg_weight", y="avg_price",
            size="lots", color="breed",
            hover_name="breed",
            hover_data={"lots": True, "avg_price": ":,.0f", "avg_weight": ":.0f"},
            title="Breed: Avg Weight vs Avg Price (bubble = volume)",
            labels={"avg_price": "Avg Price (€)", "avg_weight": "Avg Weight (kg)"},
            color_discrete_sequence=PALETTE,
        )
        fig2.update_layout(height=520, showlegend=False, **_chart_layout())
        _show_chart(fig2)

    col_c, col_d = st.columns(2)

    with col_c:
        sex_stats = (df.groupby("sex_clean")["price_num"]
                       .agg(["mean", "count"])
                       .rename(columns={"mean": "avg_price", "count": "lots"})
                       .reset_index())
        fig3 = px.bar(
            sex_stats, x="sex_clean", y="avg_price",
            text=sex_stats["avg_price"].map("€{:,.0f}".format),
            color="sex_clean", color_discrete_sequence=PALETTE,
            title="Average Price by Sex",
            labels={"sex_clean": "Sex", "avg_price": "Avg Price (€)"},
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False, height=360, **_chart_layout())
        _show_chart(fig3)

    with col_d:
        top_breed_vol = df["breed"].value_counts().head(12).reset_index()
        top_breed_vol.columns = ["breed", "count"]
        fig4 = px.pie(
            top_breed_vol, names="breed", values="count",
            title="Lot Volume by Breed (Top 12)",
            color_discrete_sequence=PALETTE, hole=0.38,
        )
        fig4.update_layout(height=360, **_chart_layout())
        _show_chart(fig4)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Price Explorer
# ═══════════════════════════════════════════════════════════════════════════════

def tab_explorer(df):
    st.subheader("Weight vs Price Scatter")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    colour_by = ctrl1.selectbox("Colour by", ["breed", "sex_clean", "mart",
                                               "export_status", "quality_assurance", "bvd_tested"])
    size_by   = ctrl2.selectbox("Size by",   ["None", "age_months", "days_in_herd",
                                               "no_of_owners", "icbf_cbv_num"])
    trend     = ctrl3.checkbox("Show trendline", value=True)

    size_col = None if size_by == "None" else size_by
    plot_df  = df.dropna(subset=["weight", "price_num"])
    plot_df  = plot_df[(plot_df["weight"] > 0) & (plot_df["price_num"] > 0)]
    if size_col:
        plot_df = plot_df.dropna(subset=[size_col])

    fig = px.scatter(
        plot_df, x="weight", y="price_num",
        color=colour_by, size=size_col,
        hover_name="breed",
        hover_data={"mart": True, "lot": True, "age_months": True,
                    "sex_clean": True, "price_per_kg_num": ":.2f"},
        trendline="ols" if trend else None,
        trendline_scope="overall",
        opacity=0.6,
        title=f"Weight vs Price — coloured by {colour_by}",
        labels={"weight": "Weight (kg)", "price_num": "Price (€)",
                "sex_clean": "Sex", "price_per_kg_num": "€/kg"},
        color_discrete_sequence=PALETTE,
        height=560,
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                   font=dict(color=FB_DARK)),
                      **_chart_layout())
    _show_chart(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = px.histogram(
            df, x="price_num", nbins=60, color="sex_clean", barmode="overlay",
            opacity=0.7, title="Price Distribution by Sex",
            labels={"price_num": "Price (€)", "sex_clean": "Sex"},
            color_discrete_sequence=PALETTE,
        )
        fig2.update_layout(height=380, **_chart_layout())
        _show_chart(fig2)

    with col_b:
        fig3 = px.histogram(
            df[df["price_per_kg_num"] > 0], x="price_per_kg_num", nbins=60, color="sex_clean", barmode="overlay",
            opacity=0.7, title="Price per KG Distribution by Sex",
            labels={"price_per_kg_num": "Price per KG (€)", "sex_clean": "Sex"},
            color_discrete_sequence=PALETTE,
        )
        fig3.update_layout(height=380, **_chart_layout())
        _show_chart(fig3)

    st.subheader("Price per KG vs Age")
    plot_df2 = df.dropna(subset=["age_months", "price_per_kg_num"])
    plot_df2 = plot_df2[(plot_df2["price_per_kg_num"] > 0) & (plot_df2["age_months"] > 0) & (plot_df2["price_per_kg_num"] < 12) & (plot_df2["age_months"] <= 45)]
    fig4 = px.scatter(
        plot_df2, x="age_months", y="price_per_kg_num",
        color="sex_clean", hover_name="breed",
        hover_data={"mart": True, "weight": True, "price_num": ":,.0f"},
        opacity=0.5, trendline="lowess", trendline_scope="trace",
        title="How Price per KG changes with Age",
        labels={"age_months": "Age (months)", "price_per_kg_num": "€/kg", "sex_clean": "Sex"},
        color_discrete_sequence=PALETTE, height=440,
    )
    fig4.update_layout(**_chart_layout())
    _show_chart(fig4)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Breed & Mart Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════

def tab_breed_mart(df):
    col_a, col_b = st.columns(2)

    with col_a:
        top_b = df["breed"].value_counts().head(15).index
        breed_df = df[df["breed"].isin(top_b)].copy()
        order = (breed_df.groupby("breed")["price_num"]
                          .median().sort_values(ascending=False).index.tolist())
        fig = px.box(
            breed_df, x="breed", y="price_num", color="breed",
            category_orders={"breed": order},
            title="Price Distribution by Breed (Top 15)",
            labels={"price_num": "Price (€)", "breed": "Breed"},
            color_discrete_sequence=PALETTE, height=460,
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-30, **_chart_layout())
        _show_chart(fig)

    with col_b:
        ppkg = (df[df["breed"].isin(top_b)]
                  .groupby("breed")["price_per_kg_num"]
                  .median()
                  .sort_values(ascending=False)
                  .reset_index())
        fig2 = px.bar(
            ppkg, x="breed", y="price_per_kg_num",
            color="price_per_kg_num",
            color_continuous_scale=[[0, "#E7F3FF"], [1, FB_BLUE]],
            text=ppkg["price_per_kg_num"].map("€{:.2f}".format),
            title="Median Price per KG by Breed",
            labels={"price_per_kg_num": "Median €/kg", "breed": "Breed"},
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=460,
                           xaxis_tickangle=-30, **_chart_layout())
        _show_chart(fig2)

    st.subheader("Price Distribution across Marts")
    mart_order = (df.groupby("mart")["price_num"]
                    .median().sort_values(ascending=False).index.tolist())
    fig3 = px.box(
        df, x="mart", y="price_num", color="mart",
        category_orders={"mart": mart_order},
        title="Sold Price Distribution by Mart",
        labels={"price_num": "Price (€)", "mart": ""},
        color_discrete_sequence=PALETTE, height=500,
    )
    fig3.update_layout(showlegend=False, xaxis_tickangle=-35, **_chart_layout())
    _show_chart(fig3)

    st.subheader("Breed × Sex — Average Price Heatmap")
    pivot_breeds = df["breed"].value_counts().head(15).index
    heat_df = (df[df["breed"].isin(pivot_breeds)]
                 .groupby(["breed", "sex_clean"])["price_num"]
                 .mean()
                 .unstack(fill_value=np.nan))
    fig4 = px.imshow(
        heat_df, text_auto=".0f",
        color_continuous_scale=[[0, "#E7F3FF"], [1, FB_BLUE]],
        aspect="auto",
        title="Average Price (€) — Breed × Sex",
        labels={"x": "Sex", "y": "Breed", "color": "Avg Price (€)"},
        height=500,
    )
    fig4.update_layout(**_chart_layout())
    _show_chart(fig4)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Price Tracker  (stock-chart style)
# ═══════════════════════════════════════════════════════════════════════════════

def tab_tracker(df):
    df2 = df.dropna(subset=["sale_date"]).copy()
    if df2.empty:
        st.info("No date information yet. Data will appear here as daily scrapes accumulate.")
        return

    # ── Daily aggregation ─────────────────────────────────────────────────────
    daily = (df2.groupby(df2["sale_date"].dt.date)
               .agg(avg_ppkg=("price_per_kg_num", "mean"),
                    avg_price=("price_num", "mean"),
                    lots=("price_num", "count"))
               .reset_index())
    daily.columns = ["date", "avg_ppkg", "avg_price", "lots"]
    daily = daily.sort_values("date")
    daily["date"]  = pd.to_datetime(daily["date"])
    daily["ma7"]   = daily["avg_ppkg"].rolling(7, min_periods=1).mean()
    daily["ma7_p"] = daily["avg_price"].rolling(7, min_periods=1).mean()

    last_ppkg  = daily["avg_ppkg"].iloc[-1]
    first_ppkg = daily["avg_ppkg"].iloc[0]
    pct_chg    = (last_ppkg - first_ppkg) / first_ppkg * 100
    hi7 = daily["avg_ppkg"].tail(7).max()
    lo7 = daily["avg_ppkg"].tail(7).min()

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Avg €/kg",   f"€{last_ppkg:.2f}",
              delta=f"{pct_chg:+.1f}% since start")
    k2.metric("7-day High €/kg",    f"€{hi7:.2f}")
    k3.metric("7-day Low €/kg",     f"€{lo7:.2f}")
    k4.metric("Days of Data",       str(len(daily)))

    st.divider()

    # ── Stock chart — €/kg ────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["avg_ppkg"],
        fill="tozeroy",
        fillcolor="rgba(24, 119, 242, 0.10)",
        line=dict(color=FB_BLUE, width=2),
        name="Avg €/kg",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Avg €/kg: €%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["ma7"],
        line=dict(color=FB_RED, width=1.8, dash="dot"),
        name="7-day moving avg",
        hovertemplate="7d avg: €%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Irish Cattle Market — Avg Price per KG",
        yaxis_title="€ per kg",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color=FB_DARK)),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05, bgcolor="#F0F2F5"),
            rangeselector=dict(
                buttons=[
                    dict(count=7,  label="1W", step="day",   stepmode="backward"),
                    dict(count=1,  label="1M", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor=FB_BG,
                activecolor=FB_BLUE,
                font=dict(color=FB_DARK),
            ),
        ),
        **_chart_layout(),
    )
    fig.update_xaxes(gridcolor="#F0F2F5")
    fig.update_yaxes(gridcolor="#F0F2F5")
    _show_chart(fig)

    # ── Volume bars ───────────────────────────────────────────────────────────
    fig_vol = go.Figure(go.Bar(
        x=daily["date"], y=daily["lots"],
        marker_color=FB_BLUE, opacity=0.65,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Lots sold: %{y:,}<extra></extra>",
        name="Lots Sold",
    ))
    fig_vol.update_layout(
        title="Daily Lots Sold",
        yaxis_title="Lots", height=220,
        **_chart_layout(),
    )
    fig_vol.update_xaxes(gridcolor="#F0F2F5")
    fig_vol.update_yaxes(gridcolor="#F0F2F5")
    _show_chart(fig_vol)

    st.divider()

    # ── Resolution toggle for deeper analysis ────────────────────────────────
    res = st.radio("Resolution for analysis below", ["Daily", "Weekly", "Monthly"],
                   horizontal=True, index=1)
    df2["week"]  = df2["sale_date"].dt.to_period("W").apply(lambda p: p.start_time)
    df2["month"] = df2["sale_date"].dt.to_period("M").apply(lambda p: p.start_time)
    grp_col = {"Daily": df2["sale_date"].dt.date,
               "Weekly": df2["week"],
               "Monthly": df2["month"]}[res]

    ts = (df2.groupby(grp_col)
              .agg(avg_price=("price_num", "mean"),
                   median_price=("price_num", "median"),
                   avg_ppkg=("price_per_kg_num", "mean"),
                   lots=("price_num", "count"))
              .reset_index())
    ts.columns = ["date", "avg_price", "median_price", "avg_ppkg", "lots"]

    st.subheader("Price per KG by Breed Over Time")
    top_breeds = df2["breed"].value_counts().head(6).index.tolist()
    breed_ts = (df2[df2["breed"].isin(top_breeds)]
                  .groupby([grp_col, "breed"])["price_per_kg_num"]
                  .mean()
                  .reset_index())
    breed_ts.columns = ["date", "breed", "avg_ppkg"]
    fig_bt = px.line(
        breed_ts, x="date", y="avg_ppkg", color="breed",
        markers=True,
        title="Average €/kg by Breed (Top 6)",
        labels={"date": "Date", "avg_ppkg": "Avg €/kg", "breed": "Breed"},
        color_discrete_sequence=PALETTE, height=420,
    )
    fig_bt.update_layout(legend=dict(orientation="h", y=1.08, font=dict(color=FB_DARK)), **_chart_layout())
    fig_bt.update_xaxes(gridcolor="#F0F2F5")
    fig_bt.update_yaxes(gridcolor="#F0F2F5")
    _show_chart(fig_bt)

    if "temp_max_c" in df2.columns and df2["temp_max_c"].notna().any():
        st.subheader("Weather vs Price Correlation")
        wx_ts = (df2.groupby(grp_col)
                    .agg(avg_ppkg=("price_per_kg_num", "mean"),
                         avg_temp=("temp_max_c", "mean"),
                         avg_rain=("precipitation_mm", "mean"))
                    .reset_index())
        wx_ts.columns = ["date", "avg_ppkg", "avg_temp", "avg_rain"]
        col_wx1, col_wx2 = st.columns(2)
        with col_wx1:
            fig_wx1 = px.scatter(wx_ts, x="avg_temp", y="avg_ppkg", trendline="ols",
                                 title="Avg Temp vs Avg €/kg",
                                 labels={"avg_temp": "Avg Max Temp (°C)", "avg_ppkg": "Avg €/kg"},
                                 height=350, color_discrete_sequence=[FB_BLUE])
            fig_wx1.update_layout(**_chart_layout())
            _show_chart(fig_wx1)
        with col_wx2:
            fig_wx2 = px.scatter(wx_ts, x="avg_rain", y="avg_ppkg", trendline="ols",
                                 title="Rainfall vs Avg €/kg",
                                 labels={"avg_rain": "Precipitation (mm)", "avg_ppkg": "Avg €/kg"},
                                 height=350, color_discrete_sequence=[FB_GREEN])
            fig_wx2.update_layout(**_chart_layout())
            _show_chart(fig_wx2)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Growth & Value Calculator
# ═══════════════════════════════════════════════════════════════════════════════

def tab_calculator(df):
    st.markdown(f"""
    <div style="background:{FB_BLUE};border-radius:12px;padding:18px 24px;margin-bottom:20px;color:white;">
      <div style="font-size:1.3rem;font-weight:800;margin-bottom:4px;">🧮 Cattle Growth & Value Calculator</div>
      <div style="opacity:0.85;font-size:0.9rem;">
        Sqrt-anchored growth model · LightGBM price prediction ·
        Best sell window · Comparable sales · Mart comparison · Break-even
      </div>
    </div>
    """, unsafe_allow_html=True)

    growth_params = compute_growth_model(str(len(df)))
    model        = load_model()
    all_breeds   = sorted(df["breed_grp"].dropna().unique())
    all_dam      = sorted(df["dam_breed_grp"].dropna().unique())
    all_marts    = sorted(df["mart"].dropna().unique())

    st.markdown("### Animal Details")
    c1, c2, c3 = st.columns(3)
    breed_val  = c1.selectbox("Breed", all_breeds,
                               index=all_breeds.index("AAX") if "AAX" in all_breeds else 0,
                               key="calc_breed")
    sex_val    = c1.selectbox("Sex", ["M", "F", "B", "Unknown"], key="calc_sex")
    dam_val    = c2.selectbox("Dam Breed", all_dam, key="calc_dam")
    mart_val   = c2.selectbox("Primary Mart", all_marts, key="calc_mart")
    cur_weight = c3.number_input("Current Weight (kg)", 50, 800, 330, step=5)
    cur_age    = c3.number_input("Current Age (months)", 0, 60, 8, step=1)
    months_fwd = st.slider("Months ahead to project", 1, 24, 6)

    # Resolve sqrt model params for this breed/sex
    a, b = growth_params.get(
        (breed_val, sex_val),
        growth_params.get(("_default", "_default"), (-80.0, 120.0))
    )

    # Month-by-month projection
    proj_months  = list(range(0, months_fwd + 1))
    proj_weights = [project_weight(cur_weight, cur_age, m, a, b) for m in proj_months]
    proj_ages    = [cur_age + m for m in proj_months]
    target_w     = proj_weights[-1]
    weight_gain  = target_w - cur_weight

    # LightGBM predictions along trajectory
    val_preds, ppkg_preds = [], []
    if model:
        wx = load_weather_latest()
        mart_wx = wx.get(mart_val, {})
        today = datetime.date.today()
        for m, wt, ag in zip(proj_months, proj_weights, proj_ages):
            future_month = (today.month + m - 1) % 12 + 1
            inp = {
                "weight": wt, "log_weight": np.log1p(wt),
                "weight_per_month": wt / max(ag, 1),
                "age_months": ag,
                "days_in_herd": 300, "no_of_owners": 1,
                "icbf_cbv_num": np.nan, "icbf_replacement_num": np.nan,
                "icbf_ebi_num": np.nan, "icbf_stars": 0, "icbf_has_data": 0,
                "has_genomic": 0, "quality_assured": 1, "bvd_ok": 1, "export_score": 2,
                "temp_max_c": mart_wx.get("temp_max_c", 15.0),
                "temp_min_c": mart_wx.get("temp_min_c", 8.0),
                "precipitation_mm": mart_wx.get("precipitation_mm", 2.0),
                "wind_speed_kmh": mart_wx.get("wind_speed_kmh", 20.0),
                "sale_month": future_month,
                "breed_grp": breed_val, "sex_clean": sex_val,
                "mart": mart_val, "dam_breed_grp": dam_val,
                "breed_sex": f"{breed_val}_{sex_val}",
                "source": "martbids",
            }
            ppkg = float(model.predict(pd.DataFrame([inp]))[0])
            ppkg_preds.append(ppkg)
            val_preds.append(ppkg * wt)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Growth Rate (sqrt model)", f"{b:.1f} kg/√month",
              help="Higher = faster-maturing breed from our mart data")
    r2.metric(f"Projected Weight (+{months_fwd}m)", f"{target_w:.0f} kg",
              delta=f"+{weight_gain:.0f} kg")
    if ppkg_preds:
        r3.metric("Est. €/kg at target", f"€{ppkg_preds[-1]:.2f}")
        r4.metric("Est. Value at target", f"€{val_preds[-1]:,.0f}",
                  delta=f"vs now €{val_preds[0]:,.0f}")

    st.divider()

    # ── Growth chart with sqrt population curve ────────────────────────────────
    comp_all = df[(df["breed_grp"] == breed_val) & (df["sex_clean"] == sex_val)] \
                 .dropna(subset=["age_months", "weight"])
    comp_all = comp_all[(comp_all["weight"] > 0) & (comp_all["age_months"] > 0)]
    fig = go.Figure()
    if len(comp_all) > 0:
        sample = comp_all.sample(min(400, len(comp_all)), random_state=42)
        fig.add_trace(go.Scatter(
            x=sample["age_months"], y=sample["weight"], mode="markers",
            marker=dict(color="rgba(24,119,242,0.18)", size=5),
            name=f"{breed_val} {sex_val} — sold",
            hovertemplate="Age: %{x:.0f}m  Weight: %{y:.0f}kg<extra></extra>",
        ))
        age_range = np.linspace(comp_all["age_months"].min(), comp_all["age_months"].max(), 80)
        fig.add_trace(go.Scatter(
            x=age_range, y=a + b * np.sqrt(age_range), mode="lines",
            line=dict(color=FB_BLUE, width=1.5, dash="dot"),
            name="Sqrt fit (population)",
        ))
    fig.add_trace(go.Scatter(
        x=proj_ages, y=proj_weights, mode="lines+markers",
        line=dict(color=FB_GREEN, width=3), marker=dict(size=7, color=FB_GREEN),
        name="Your animal", customdata=proj_months,
        hovertemplate="Month +%{customdata}: %{y:.0f}kg (age %{x:.0f}m)<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[cur_age], y=[cur_weight], mode="markers",
        marker=dict(color=FB_BLUE, size=16, symbol="circle",
                    line=dict(color="white", width=3)), name="Now"))
    fig.add_trace(go.Scatter(x=[proj_ages[-1]], y=[target_w], mode="markers",
        marker=dict(color=FB_GREEN, size=16, symbol="star",
                    line=dict(color="white", width=2)), name=f"Target (+{months_fwd}m)"))
    fig.update_layout(
        title=f"{breed_val} {sex_val} — Sqrt-Anchored Growth Projection",
        xaxis_title="Age (months)", yaxis_title="Weight (kg)",
        height=480, legend=dict(orientation="h", y=1.06, font=dict(color=FB_DARK)), **_chart_layout(),
    )
    fig.update_xaxes(gridcolor="#F0F2F5")
    fig.update_yaxes(gridcolor="#F0F2F5")
    _show_chart(fig)

    if val_preds:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fv = go.Figure(go.Scatter(
                x=proj_months, y=val_preds, fill="tozeroy",
                fillcolor="rgba(66,183,42,0.12)", line=dict(color=FB_GREEN, width=2.5),
                mode="lines+markers", marker=dict(size=6, color=FB_GREEN),
                hovertemplate="Month +%{x}: €%{y:,.0f}<extra></extra>",
            ))
            fv.update_layout(title="Estimated Market Value", xaxis_title="Months from now",
                             yaxis_title="€", height=300, **_chart_layout())
            fv.update_xaxes(gridcolor="#F0F2F5", tickvals=proj_months)
            fv.update_yaxes(gridcolor="#F0F2F5")
            _show_chart(fv)
        with col_v2:
            fp = go.Figure(go.Scatter(
                x=proj_months, y=ppkg_preds, line=dict(color=FB_BLUE, width=2.5),
                mode="lines+markers", marker=dict(size=6, color=FB_BLUE),
                hovertemplate="Month +%{x}: €%{y:.2f}/kg<extra></extra>",
            ))
            fp.update_layout(title="Est. Price per KG", xaxis_title="Months from now",
                             yaxis_title="€/kg", height=300, **_chart_layout())
            fp.update_xaxes(gridcolor="#F0F2F5", tickvals=proj_months)
            fp.update_yaxes(gridcolor="#F0F2F5")
            _show_chart(fp)

        st.subheader("Projection Summary")
        st.dataframe(pd.DataFrame({
            "Month":          [f"+{m}m" for m in proj_months],
            "Age":            [f"{ag}m" for ag in proj_ages],
            "Weight (kg)":    [f"{w:.0f}" for w in proj_weights],
            "Est. €/kg":      [f"€{p:.2f}" for p in ppkg_preds],
            "Est. Value (€)": [f"€{v:,.0f}" for v in val_preds],
        }), use_container_width=True, hide_index=True)

    # ── Best Time to Sell ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📅 Best Time to Sell")
    bts_data = df[(df["breed_grp"] == breed_val) & (df["sex_clean"] == sex_val)] \
                 .dropna(subset=["price_per_kg_num", "weight", "age_months"]).copy()

    if len(bts_data) >= 20:
        bts_data["wt_bucket"]  = (bts_data["weight"] // 50 * 50).astype(int)
        bts_data["age_bucket"] = (bts_data["age_months"] // 6 * 6).astype(int)
        pivot = bts_data.pivot_table(values="price_per_kg_num", index="wt_bucket",
                                     columns="age_bucket", aggfunc="median", observed=True)
        counts = bts_data.pivot_table(values="price_per_kg_num", index="wt_bucket",
                                      columns="age_bucket", aggfunc="count", observed=True)
        pivot = pivot.where(counts >= 5)
        if not pivot.empty:
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"{int(c)}m" for c in pivot.columns],
                y=[f"{int(r)}kg" for r in pivot.index],
                colorscale="YlGn", colorbar=dict(title="€/kg"),
                hovertemplate="Weight: %{y}  Age: %{x}<br>Median: €%{z:.2f}/kg<extra></extra>",
                zmin=bts_data["price_per_kg_num"].quantile(0.05),
                zmax=bts_data["price_per_kg_num"].quantile(0.95),
            ))
            fig_hm.add_trace(go.Scatter(
                x=[f"{int(cur_age // 6 * 6)}m"], y=[f"{int(cur_weight // 50 * 50)}kg"],
                mode="markers",
                marker=dict(symbol="star", size=18, color=FB_RED,
                            line=dict(color="white", width=2)),
                name="Your animal now",
            ))
            fig_hm.update_layout(
                title=f"{breed_val} {sex_val} — Median €/kg by Weight & Age",
                xaxis_title="Age bracket", yaxis_title="Weight bracket",
                height=400, **_chart_layout(),
            )
            _show_chart(fig_hm)

        # Model-simulated monthly price (holding weight/age constant at target)
        if model and ppkg_preds:
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            month_ppkg = []
            base = {"weight": target_w,
                    "log_weight": np.log1p(target_w),
                    "weight_per_month": target_w / max(float(proj_ages[-1]), 1),
                    "age_months": float(proj_ages[-1]),
                    "days_in_herd": 300, "no_of_owners": 1,
                    "icbf_cbv_num": np.nan, "icbf_replacement_num": np.nan,
                    "icbf_ebi_num": np.nan, "icbf_stars": 0, "icbf_has_data": 0,
                    "has_genomic": 0, "quality_assured": 1, "bvd_ok": 1, "export_score": 2,
                    "temp_max_c": 15.0, "temp_min_c": 8.0,
                    "precipitation_mm": 2.0, "wind_speed_kmh": 20.0,
                    "breed_grp": breed_val, "sex_clean": sex_val,
                    "mart": mart_val, "dam_breed_grp": dam_val,
                    "breed_sex": f"{breed_val}_{sex_val}",
                    "source": "martbids"}
            for mo in range(1, 13):
                month_ppkg.append(float(model.predict(pd.DataFrame([{**base, "sale_month": mo}]))[0]))
            fig_s = go.Figure(go.Bar(
                x=month_names, y=month_ppkg,
                marker_color=[FB_GREEN if v == max(month_ppkg) else FB_BLUE for v in month_ppkg],
                hovertemplate="%{x}: €%{y:.2f}/kg<extra></extra>",
            ))
            fig_s.update_layout(title="Predicted €/kg by Sale Month (at target weight & age)",
                                xaxis_title="Month", yaxis_title="Predicted €/kg",
                                height=280, **_chart_layout())
            _show_chart(fig_s)
    else:
        st.info(f"Not enough data for {breed_val} {sex_val} to build price heatmap "
                f"({len(bts_data)} rows, need ≥ 20).")

    # ── Comparable Recent Sales ───────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Comparable Recent Sales")
    st.caption(f"{breed_val} {sex_val}, weight {cur_weight*0.8:.0f}–{cur_weight*1.2:.0f} kg, "
               f"age {max(0,cur_age-4):.0f}–{cur_age+4:.0f} months")
    comp = df[
        (df["breed_grp"] == breed_val) & (df["sex_clean"] == sex_val) &
        df["weight"].between(cur_weight * 0.8, cur_weight * 1.2) &
        df["age_months"].between(max(0, cur_age - 4), cur_age + 4)
    ].copy().sort_values("sale_date", ascending=False)

    if len(comp) > 0:
        s1, s2, s3 = st.columns(3)
        s1.metric("Comparables Found", len(comp))
        s2.metric("Avg Price", f"€{comp['price_num'].mean():,.0f}")
        s3.metric("Avg €/kg",  f"€{comp['price_per_kg_num'].mean():.2f}")
        show = comp[["mart","breed","sex_clean","age_months","weight",
                     "price_num","price_per_kg_num","sale_date"]].head(50).copy()
        show = show.rename(columns={"sex_clean":"Sex","age_months":"Age (m)",
                                    "price_num":"Price (€)","price_per_kg_num":"€/kg",
                                    "sale_date":"Date"})
        show["Price (€)"] = show["Price (€)"].map("€{:,.0f}".format)
        show["€/kg"]      = show["€/kg"].map("€{:.2f}".format)
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        wider = df[(df["breed_grp"] == breed_val) &
                   df["weight"].between(cur_weight * 0.7, cur_weight * 1.3)]
        st.info(f"No close matches. Showing {len(wider)} similar {breed_val} animals (±30% weight, any sex).")
        if len(wider) > 0:
            st.dataframe(wider[["mart","breed","sex_clean","age_months","weight",
                                 "price_num","price_per_kg_num"]]
                           .sort_values("price_per_kg_num", ascending=False)
                           .head(20), use_container_width=True, hide_index=True)

    # ── Mart Comparison ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("🏪 Mart Comparison")
    st.caption(f"Predicted value for {breed_val} {sex_val} at {target_w:.0f}kg / "
               f"age {proj_ages[-1]}m across all marts")
    if model:
        today = datetime.date.today()
        future_month = (today.month + months_fwd - 1) % 12 + 1
        wx = load_weather_latest()
        mart_rows = []
        for m_name in all_marts:
            mwx = wx.get(m_name, {})
            inp = {"weight": target_w,
                   "log_weight": np.log1p(target_w),
                   "weight_per_month": target_w / max(float(proj_ages[-1]), 1),
                   "age_months": float(proj_ages[-1]),
                   "days_in_herd": 300, "no_of_owners": 1,
                   "icbf_cbv_num": np.nan, "icbf_replacement_num": np.nan,
                   "icbf_ebi_num": np.nan, "icbf_stars": 0, "icbf_has_data": 0,
                   "has_genomic": 0, "quality_assured": 1, "bvd_ok": 1, "export_score": 2,
                   "temp_max_c": mwx.get("temp_max_c", 15.0),
                   "temp_min_c": mwx.get("temp_min_c", 8.0),
                   "precipitation_mm": mwx.get("precipitation_mm", 2.0),
                   "wind_speed_kmh": mwx.get("wind_speed_kmh", 20.0),
                   "sale_month": future_month, "breed_grp": breed_val,
                   "sex_clean": sex_val, "mart": m_name,
                   "dam_breed_grp": dam_val, "breed_sex": f"{breed_val}_{sex_val}",
                   "source": "martbids"}
            ppkg_m = float(model.predict(pd.DataFrame([inp]))[0])
            mart_rows.append({"Mart": m_name, "Pred. €/kg": ppkg_m,
                              "Pred. Value": ppkg_m * target_w, "selected": m_name == mart_val})
        mart_df = pd.DataFrame(mart_rows).sort_values("Pred. €/kg", ascending=False)
        best_mart = mart_df.iloc[0]["Mart"]
        best_ppkg = mart_df.iloc[0]["Pred. €/kg"]
        colors = [FB_GREEN if r["selected"] else
                  ("#F7B928" if r["Mart"] == best_mart else FB_BLUE)
                  for _, r in mart_df.iterrows()]
        fig_m = go.Figure(go.Bar(
            x=mart_df["Mart"], y=mart_df["Pred. €/kg"],
            marker_color=colors,
            hovertemplate="%{x}: €%{y:.2f}/kg<extra></extra>",
        ))
        sel_ppkg = mart_df[mart_df["Mart"] == mart_val]["Pred. €/kg"].values[0]
        fig_m.add_hline(y=sel_ppkg, line_dash="dash", line_color=FB_GREEN,
                        annotation_text=f"Your mart ({mart_val})",
                        annotation_position="top left")
        fig_m.update_layout(title="Predicted €/kg at Target Weight Across All Marts",
                            xaxis_title="Mart", yaxis_title="Predicted €/kg",
                            xaxis_tickangle=-40, height=400, **_chart_layout())
        _show_chart(fig_m)
        if best_mart != mart_val:
            sel_val = mart_df[mart_df["Mart"] == mart_val]["Pred. Value"].values[0]
            best_val = mart_df.iloc[0]["Pred. Value"]
            st.info(f"Best predicted mart: **{best_mart}** at €{best_ppkg:.2f}/kg "
                    f"(€{best_val:,.0f} total) — €{best_val - sel_val:,.0f} more than {mart_val}.")

    # ── Break-Even Calculator ─────────────────────────────────────────────────
    st.divider()
    st.subheader("💰 Break-Even Calculator")
    be1, be2 = st.columns(2)
    purchase_cost    = be1.number_input("Purchase / total cost to date (€)", 0, 20000, 1200, step=50)
    ongoing_monthly  = be2.number_input("Ongoing monthly costs (€/month)", 0, 500, 50, step=10)

    if val_preds and purchase_cost > 0:
        total_costs = [purchase_cost + ongoing_monthly * m for m in proj_months]
        net_profit  = [v - c for v, c in zip(val_preds, total_costs)]
        be_month = next((m for m, p in zip(proj_months, net_profit) if p >= 0), None)

        if be_month == 0:
            st.success(f"Already profitable — current value €{val_preds[0]:,.0f} > cost €{purchase_cost:,.0f}.")
        elif be_month is not None:
            st.success(f"Break-even at **+{be_month} months** — age {cur_age+be_month}m, "
                       f"est. {proj_weights[be_month]:.0f}kg, value €{val_preds[be_month]:,.0f}.")
        else:
            st.warning(f"No break-even within {months_fwd} months. "
                       f"At +{months_fwd}m: value €{val_preds[-1]:,.0f} vs cost €{total_costs[-1]:,.0f}.")

        fig_be = go.Figure()
        fig_be.add_trace(go.Scatter(x=proj_months, y=val_preds, mode="lines+markers",
            name="Est. Value", line=dict(color=FB_GREEN, width=2.5),
            marker=dict(size=6), hovertemplate="Month +%{x}: €%{y:,.0f}<extra></extra>"))
        fig_be.add_trace(go.Scatter(x=proj_months, y=total_costs, mode="lines+markers",
            name="Total Cost", line=dict(color=FB_RED, width=2.5, dash="dash"),
            marker=dict(size=6), hovertemplate="Month +%{x}: cost €%{y:,.0f}<extra></extra>"))
        if be_month and be_month > 0:
            fig_be.add_vline(x=be_month, line_dash="dot", line_color=FB_GREEN,
                             annotation_text=f"Break-even +{be_month}m",
                             annotation_position="top")
        fig_be.update_layout(title="Market Value vs Total Cost",
                             xaxis_title="Months from now", yaxis_title="€",
                             height=320, legend=dict(orientation="h", y=1.06, font=dict(color=FB_DARK)),
                             **_chart_layout())
        _show_chart(fig_be)
        st.dataframe(pd.DataFrame({
            "Month":      [f"+{m}m" for m in proj_months],
            "Age":        [f"{ag}m" for ag in proj_ages],
            "Weight":     [f"{w:.0f}kg" for w in proj_weights],
            "Est. Value": [f"€{v:,.0f}" for v in val_preds],
            "Total Cost": [f"€{c:,.0f}" for c in total_costs],
            "Net Profit": [f"€{p:+,.0f}" for p in net_profit],
        }), use_container_width=True, hide_index=True)
    else:
        st.info("Enter a purchase cost above to see break-even analysis.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Factory Prices
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_factory_prices():
    path = BASE / "factory_prices_clean.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["report_date"], low_memory=False)
    df = df[df["price_euro_per_kg"].between(3, 12)].copy()
    return df


def tab_factory(fp: pd.DataFrame):
    if fp.empty:
        st.warning("factory_prices_clean.csv not found. Run prepare_factory_prices.py first.")
        return

    bpw    = fp[fp["source"] == "BeefPriceWatch"].copy()
    latest = bpw["report_date"].max()
    prev   = bpw[bpw["report_date"] < latest]["report_date"].max()

    # Reference: R3 Steer headline (most common classification)
    ref_latest = bpw[bpw["is_headline"] & (bpw["category"] == "Steer") & (bpw["report_date"] == latest)]
    ref_prev   = bpw[bpw["is_headline"] & (bpw["category"] == "Steer") & (bpw["report_date"] == prev)]
    ref_price  = ref_latest["price_euro_per_kg"].mean()
    ref_prev_p = ref_prev["price_euro_per_kg"].mean()
    ref_delta  = ref_price - ref_prev_p

    ref_heifer = bpw[bpw["is_headline"] & (bpw["category"] == "Heifer") & (bpw["report_date"] == latest)]["price_euro_per_kg"].mean()
    ref_cow    = bpw[bpw["is_headline"] & (bpw["category"] == "Cow")    & (bpw["report_date"] == latest)]["price_euro_per_kg"].mean()

    top_factory = ref_latest.loc[ref_latest["price_euro_per_kg"].idxmax(), "factory"] if not ref_latest.empty else "—"
    top_price   = ref_latest["price_euro_per_kg"].max()

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{FB_BLUE};border-radius:12px;padding:14px 20px;margin-bottom:18px;">
      <div style="color:rgba(255,255,255,0.7);font-size:0.75rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.8px;">
        Factory Prices — week ending {latest.strftime('%d %b %Y')}
      </div>
      <div style="color:white;font-size:0.82rem;margin-top:4px;opacity:0.85;">
        R3 Steer &middot; R3 Heifer &middot; O4 Cow &middot; reference grades &middot; BeefPriceWatch (DAFM)
      </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R3 Steer (avg)", f"€{ref_price:.2f}/kg",
              delta=f"{ref_delta:+.3f} vs prev week")
    k2.metric("R3 Heifer (avg)", f"€{ref_heifer:.2f}/kg")
    k3.metric("Cow (avg)",       f"€{ref_cow:.2f}/kg")
    k4.metric("Top paying factory", top_factory,
              delta=f"€{top_price:.2f}/kg", delta_color="normal")

    st.divider()

    # ── Section 1: Reference price trend ─────────────────────────────────────
    st.subheader("Reference Bullock Price Trend")
    st.caption("R3 Steer & R3 Heifer — national avg across all factories (headline prices)")

    weekly = (
        bpw[bpw["is_headline"] & bpw["category"].isin(["Steer", "Heifer", "Cow"])]
        .groupby(["report_date", "category"])["price_euro_per_kg"]
        .agg(mean="mean", lo="min", hi="max")
        .reset_index()
    )

    def _hex_rgba(hex_col: str, alpha: float = 0.12) -> str:
        h = hex_col.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig_trend = go.Figure()
    cat_colours = {"Steer": FB_BLUE, "Heifer": GOLD, "Cow": FB_RED}
    for cat, colour in cat_colours.items():
        g = weekly[weekly["category"] == cat].sort_values("report_date")
        if g.empty:
            continue
        fig_trend.add_trace(go.Scatter(
            x=pd.concat([g["report_date"], g["report_date"][::-1]]),
            y=pd.concat([g["hi"], g["lo"][::-1]]),
            fill="toself",
            fillcolor=_hex_rgba(colour),
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig_trend.add_trace(go.Scatter(
            x=g["report_date"], y=g["mean"],
            mode="lines+markers",
            line=dict(color=colour, width=2.5),
            marker=dict(size=6, color=colour),
            name=cat,
            hovertemplate=f"<b>{cat}</b><br>%{{x|%d %b}}: €%{{y:.3f}}/kg<extra></extra>",
        ))

    fig_trend.update_layout(
        height=320,
        legend=dict(orientation="h", y=1.08, font=dict(color=FB_DARK)),
        yaxis_title="€/kg",
        hovermode="x unified",
        **_chart_layout(),
    )
    _show_chart(fig_trend)

    st.divider()

    # ── Section 2: Factory price comparison ──────────────────────────────────
    # One specific reference grade per category (as reported by BPW):
    # Steer → R3, Heifer → R3, Cow → O4, Young Bull → U3, Bull → O2
    REF_GRADE = {"Steer": "R3", "Heifer": "R3", "Cow": "O4", "Young Bull": "U3", "Bull": "O2"}

    cat_sel = st.selectbox(
        "Category", ["Steer", "Heifer", "Cow", "Young Bull", "Bull"],
        key="factory_cat_sel"
    )
    ref_grade = REF_GRADE.get(cat_sel, "")

    st.subheader(f"Factory Price Comparison — {ref_grade} {cat_sel} · Latest Week")
    st.caption(f"Each bar is the {ref_grade} headline price reported by that factory for week ending {latest.strftime('%d %b %Y')}")

    league = (
        ref_latest if cat_sel == "Steer"
        else bpw[bpw["is_headline"] & (bpw["category"] == cat_sel) & (bpw["report_date"] == latest)]
    ).sort_values("price_euro_per_kg", ascending=True)

    if league.empty:
        st.info(f"No headline data for {cat_sel} in latest week.")
    else:
        nat_avg = league["price_euro_per_kg"].mean()
        colours = [
            FB_GREEN if p >= nat_avg else FB_RED
            for p in league["price_euro_per_kg"]
        ]
        fig_league = go.Figure(go.Bar(
            x=league["price_euro_per_kg"],
            y=league["factory"],
            orientation="h",
            marker_color=colours,
            text=[f"€{p:.3f}" for p in league["price_euro_per_kg"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>€%{x:.3f}/kg<extra></extra>",
        ))
        fig_league.add_vline(
            x=nat_avg, line_dash="dot", line_color=FB_DARK, line_width=1.5,
            annotation_text=f"Avg €{nat_avg:.3f}",
            annotation_font_color=FB_DARK,
        )
        x_min = max(0, league["price_euro_per_kg"].min() - 1.0)
        x_max = league["price_euro_per_kg"].max() + 0.15
        fig_league.update_layout(
            height=max(380, len(league) * 28),
            xaxis=dict(title="€/kg", range=[x_min, x_max]),
            margin=dict(l=160, r=80),
            **_chart_layout(),
        )
        _show_chart(fig_league)

    # ── Boxplot: price distribution by factory ────────────────────────────────
    st.subheader("Price Distribution by Factory")
    st.caption("All detail-grade prices for selected category · latest 4 weeks")

    last4 = sorted(bpw["report_date"].unique())[-4:]
    box_df = bpw[
        ~bpw["is_headline"] &
        (bpw["category"] == cat_sel) &
        (bpw["report_date"].isin(last4))
    ].copy()

    if not box_df.empty:
        # Sort factories by median price descending
        order = (
            box_df.groupby("factory")["price_euro_per_kg"]
            .median().sort_values(ascending=False).index.tolist()
        )
        fig_box = go.Figure()
        for fac in order:
            fac_prices = box_df[box_df["factory"] == fac]["price_euro_per_kg"]
            fig_box.add_trace(go.Box(
                y=fac_prices,
                name=fac,
                marker_color=FB_BLUE,
                line_color=FB_DARK,
                boxmean=True,
                hovertemplate=f"<b>{fac}</b><br>€%{{y:.3f}}/kg<extra></extra>",
            ))
        fig_box.update_layout(
            showlegend=False,
            yaxis_title="€/kg",
            height=400,
            **_chart_layout(),
        )
        _show_chart(fig_box)

    st.divider()

    # ── Section 3: Conformation × Fat heatmap ─────────────────────────────────
    st.subheader("Grade Breakdown — Conformation × Fat Class")
    st.caption("Avg €/kg by conformation and fat score across all factories · latest 4 weeks")

    col_cat, col_fac = st.columns(2)
    hm_cat = col_cat.selectbox(
        "Category", ["Steer", "Heifer", "Cow", "Young Bull", "Bull"],
        key="hm_cat"
    )
    all_factories = ["All"] + sorted(bpw["factory"].dropna().unique().tolist())
    hm_fac = col_fac.selectbox("Factory", all_factories, key="hm_fac")

    last4 = sorted(bpw["report_date"].unique())[-4:]
    hm_df = bpw[
        ~bpw["is_headline"] &
        (bpw["category"] == hm_cat) &
        (bpw["report_date"].isin(last4))
    ].copy()
    if hm_fac != "All":
        hm_df = hm_df[hm_df["factory"] == hm_fac]

    CONF_ORDER = ["E+","E=","E-","U+","U=","U-","R+","R=","R-","O+","O=","O-","P+","P=","P-"]
    FAT_ORDER  = ["1-","1=","1+","2-","2=","2+","3-","3=","3+","4-","4=","4+","5-","5=","5+"]

    if hm_df.empty:
        st.info("Not enough detail data for this selection.")
    else:
        pivot = (
            hm_df.groupby(["conformation", "fat_class"])["price_euro_per_kg"]
            .mean()
            .unstack(fill_value=None)
        )
        conf_idx = [c for c in CONF_ORDER if c in pivot.index]
        fat_cols = [f for f in FAT_ORDER  if f in pivot.columns]
        pivot = pivot.reindex(index=conf_idx, columns=fat_cols)

        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, FB_RED], [0.5, "#F5F0E8"], [1, FB_GREEN]],
            text=[[f"€{v:.2f}" if v == v else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="Conf: %{y}  Fat: %{x}<br>Avg €/kg: %{z:.3f}<extra></extra>",
            colorbar=dict(title="€/kg", tickfont=dict(color=FB_DARK),
                          title_font=dict(color=FB_DARK)),
        ))
        fig_hm.update_layout(
            xaxis_title="Fat Class",
            yaxis_title="Conformation",
            height=420,
            **_chart_layout(),
        )
        _show_chart(fig_hm)

    st.divider()

    # ── Section 4: All categories latest week side-by-side ───────────────────
    st.subheader("All Categories — Latest Week")

    all_cats = (
        bpw[bpw["is_headline"] & (bpw["report_date"] == latest)]
        .groupby("category")["price_euro_per_kg"]
        .agg(mean="mean", lo="min", hi="max")
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    fig_cats = go.Figure()
    for _, row in all_cats.iterrows():
        fig_cats.add_trace(go.Bar(
            name=row["category"],
            x=[row["category"]],
            y=[row["mean"]],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["hi"] - row["mean"]],
                arrayminus=[row["mean"] - row["lo"]],
                color=FB_GREY,
            ),
            marker_color=cat_colours.get(row["category"], FB_BLUE),
            text=f"€{row['mean']:.3f}",
            textposition="outside",
            hovertemplate=(
                f"<b>{row['category']}</b><br>"
                f"Avg: €{row['mean']:.3f}/kg<br>"
                f"Range: €{row['lo']:.3f} – €{row['hi']:.3f}<extra></extra>"
            ),
        ))

    fig_cats.update_layout(
        showlegend=False,
        yaxis_title="€/kg",
        height=320,
        **_chart_layout(),
    )
    _show_chart(fig_cats)



# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(f"""
    <div style="padding:8px 0 20px;">
      <div style="font-size:1.8rem;font-weight:900;color:{FB_DARK};
                  letter-spacing:-0.5px;line-height:1.1;">MartIndex</div>
      <div style="font-size:0.85rem;color:{FB_GREY};font-weight:500;margin-top:2px;">
          Irish Cattle Market Intelligence &middot; Live data from martbids.ie
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_full = load_data()
    df      = sidebar_filters(df_full)

    if df.empty:
        st.warning("No data matches the current filters. Try widening your selection.")
        return

    fp = load_factory_prices()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Overview",
        "🔍 Price Explorer",
        "🐄 Breed & Mart",
        "📈 Price Tracker",
        "🧮 Growth Calculator",
        "🏭 Factory Prices",
    ])

    with tab1: tab_overview(df)
    with tab2: tab_explorer(df)
    with tab3: tab_breed_mart(df)
    with tab4: tab_tracker(df)
    with tab5: tab_calculator(df)
    with tab6: tab_factory(fp)


if __name__ == "__main__":
    main()
