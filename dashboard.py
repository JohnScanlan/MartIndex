"""
MartBids Cattle Price Dashboard
================================
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
    page_title="MartBids",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

# ── Facebook colour palette ───────────────────────────────────────────────────
FB_BLUE   = "#1877F2"
FB_DARK   = "#1C1E21"
FB_BG     = "#F0F2F5"
FB_CARD   = "#FFFFFF"
FB_GREY   = "#4B4F56"   # darkened for readability
FB_GREEN  = "#42B72A"
FB_RED    = "#FA383E"
FB_BORDER = "#CED0D4"

PALETTE = [FB_BLUE, FB_GREEN, "#F7B928", FB_RED, "#9B59B6",
           "#00BCD4", "#FF9800", "#E91E63", "#3F51B5", "#009688"]

st.markdown(f"""
<style>
    /* ── App background ──────────────────────────────────────────────────── */
    .stApp {{ background-color: {FB_BG}; }}

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background-color: {FB_CARD};
        border-right: 1px solid {FB_BORDER};
    }}

    /* ── General text (scoped — not global div/span to avoid killing inputs) */
    h1 {{ color: {FB_DARK} !important; font-size:1.75rem !important;
          font-weight:800 !important; letter-spacing:-0.5px; }}
    h2, h3 {{ color: {FB_DARK} !important; font-weight:700 !important; }}
    p, li {{ color: {FB_DARK} !important; }}
    label {{ color: {FB_DARK} !important; font-weight: 500; }}
    .stMarkdown p {{ color: {FB_DARK} !important; font-size:0.95rem; }}
    .stCaptionContainer p {{ color: {FB_GREY} !important; font-size:0.83rem !important; }}

    /* ── Metric cards ────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {FB_CARD};
        border: 1px solid {FB_BORDER};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }}
    [data-testid="stMetricLabel"] p {{
        color: {FB_GREY} !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    [data-testid="stMetricValue"] {{
        color: {FB_DARK} !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
    }}

    /* ── Tab bar ─────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {FB_CARD};
        border-radius: 10px;
        padding: 6px;
        gap: 4px;
        border: 1px solid {FB_BORDER};
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        font-weight: 600;
        color: {FB_DARK} !important;
        padding: 8px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {FB_BLUE} !important;
        color: white !important;
    }}

    /* ── All input containers — WHITE background, DARK text ─────────────── */
    /* Selectbox, multiselect, number input, text input */
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="input"],
    [data-baseweb="input"] > div {{
        background-color: {FB_CARD} !important;
        border-color: {FB_BORDER} !important;
    }}
    /* The visible text inside any select/input */
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="select"] div[class*="placeholder"],
    [data-baseweb="input"] input {{
        color: {FB_DARK} !important;
        background-color: transparent !important;
    }}

    /* ── Multiselect tags (selected items) ───────────────────────────────── */
    [data-baseweb="tag"] {{
        background-color: #E7F3FF !important;
        border: 1px solid {FB_BLUE} !important;
        border-radius: 6px !important;
    }}
    [data-baseweb="tag"] span {{
        color: {FB_BLUE} !important;
        font-weight: 600;
    }}
    /* X button on tags */
    [data-baseweb="tag"] [role="presentation"] span {{
        color: {FB_BLUE} !important;
    }}

    /* ── Dropdown menu (the open list) ───────────────────────────────────── */
    [data-baseweb="menu"],
    [data-baseweb="popover"] > div {{
        background-color: {FB_CARD} !important;
        border: 1px solid {FB_BORDER} !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
    }}
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"] {{
        color: {FB_DARK} !important;
        background-color: {FB_CARD} !important;
    }}
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [aria-selected="true"] {{
        background-color: #E7F3FF !important;
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
        color: {FB_GREY} !important;
    }}

    /* ── Checkboxes ──────────────────────────────────────────────────────── */
    [data-testid="stCheckbox"] span {{ color: {FB_DARK} !important; }}

    /* ── Radio buttons ───────────────────────────────────────────────────── */
    [data-testid="stRadio"] label span {{ color: {FB_DARK} !important; }}

    /* ── Plotly chart panels ─────────────────────────────────────────────── */
    [data-testid="stPlotlyChart"] {{
        background: {FB_CARD};
        border: 1px solid {FB_BORDER};
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}

    /* ── Primary button ──────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {{
        background: {FB_BLUE};
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 10px 28px;
        font-size: 1rem;
        transition: background 0.15s;
    }}
    .stButton > button[kind="primary"]:hover {{ background: #166FE5; }}

    /* ── Secondary / normal buttons ──────────────────────────────────────── */
    .stButton > button {{
        color: {FB_DARK} !important;
        background: {FB_CARD} !important;
        border: 1px solid {FB_BORDER} !important;
        border-radius: 8px;
    }}

    /* ── Alert / info / success boxes ───────────────────────────────────── */
    div[data-testid="stAlert"] {{ border-radius: 8px; }}
    div[data-testid="stAlert"] p {{ color: inherit !important; }}

    /* ── Divider ─────────────────────────────────────────────────────────── */
    hr {{ border-color: {FB_BORDER} !important; }}
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
    "has_genomic", "quality_assured", "bvd_ok", "export_score",
    "temp_max_c", "temp_min_c", "precipitation_mm", "wind_speed_kmh",
    "sale_month",
]
CATEGORICAL_FEATURES = ["breed_grp", "sex_clean", "mart", "dam_breed_grp", "breed_sex"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _chart_layout(**kw):
    """Common Plotly layout defaults matching the FB theme."""
    base = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=FB_DARK, family="system-ui, -apple-system, sans-serif"),
    )
    base.update(kw)
    return base


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "sold_lots.csv")
    df["price_num"]            = df["price"].apply(parse_eur)
    df["weight"]               = pd.to_numeric(df["weight"], errors="coerce")
    df["age_months"]           = pd.to_numeric(df["age_months"], errors="coerce")
    df["days_in_herd"]         = pd.to_numeric(df["days_in_herd"], errors="coerce")
    df["no_of_owners"]         = pd.to_numeric(df["no_of_owners"], errors="coerce")
    df["price_per_kg_num"]     = df["price_num"] / df["weight"].replace(0, np.nan)
    df["icbf_cbv_num"]         = df["icbf_cbv"].apply(parse_eur)
    df["icbf_replacement_num"] = df["icbf_replacement_index"].apply(parse_eur)
    df["icbf_ebi_num"]         = df["icbf_ebi"].apply(parse_eur)
    df["icbf_stars"]           = df["icbf_across_breed"].apply(count_stars)
    df["has_genomic"]          = (df["icbf_genomic_eval"] == "Yes").astype(int)
    df["quality_assured"]      = (df["quality_assurance"] == "Yes").astype(int)
    df["bvd_ok"]               = (df["bvd_tested"] == "Yes").astype(int)
    df["export_score"]         = df["export_status"].apply(export_score_fn)
    df["sex_clean"]            = df["sex"].map({"M": "M", "F": "F", "B": "B"}).fillna("Unknown")
    sale_dt                    = pd.to_datetime(df["scraped_date"], errors="coerce")
    df["sale_date"]            = sale_dt
    df["sale_month"]           = sale_dt.dt.month.fillna(0).astype(int)

    top_breeds = df["breed"].value_counts().head(20).index
    df["breed_grp"] = df["breed"].where(df["breed"].isin(top_breeds), "Other")
    df["breed_sex"] = df["breed_grp"] + "_" + df["sex_clean"]

    top_dam = df["dam_breed"].value_counts().head(15).index
    df["dam_breed_grp"] = (df["dam_breed"]
                           .where(df["dam_breed"].isin(top_dam), "Other")
                           .fillna("Unknown"))

    df = df[df["price_num"] > 0].dropna(subset=["price_num", "weight"]).copy()

    wx_path = BASE / "weather_cache.csv"
    if wx_path.exists():
        wx = pd.read_csv(wx_path)
        wx["date"] = pd.to_datetime(wx["date"]).dt.strftime("%Y-%m-%d")
        df["sale_date_str"] = df["sale_date"].dt.strftime("%Y-%m-%d")
        df = df.merge(wx.rename(columns={"date": "sale_date_str"}),
                      on=["mart", "sale_date_str"], how="left")
    else:
        df["temp_max_c"] = np.nan
        df["temp_min_c"] = np.nan
        df["precipitation_mm"] = np.nan
        df["wind_speed_kmh"] = np.nan

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
      <div style="background:{FB_BLUE};border-radius:10px;width:38px;height:38px;
                  display:flex;align-items:center;justify-content:center;font-size:1.3rem;">🐄</div>
      <div>
        <div style="font-size:1.1rem;font-weight:800;color:{FB_DARK};">MartBids</div>
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig2, width="stretch")

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
        st.plotly_chart(fig3, width="stretch")

    with col_d:
        top_breed_vol = df["breed"].value_counts().head(12).reset_index()
        top_breed_vol.columns = ["breed", "count"]
        fig4 = px.pie(
            top_breed_vol, names="breed", values="count",
            title="Lot Volume by Breed (Top 12)",
            color_discrete_sequence=PALETTE, hole=0.38,
        )
        fig4.update_layout(height=360, **_chart_layout())
        st.plotly_chart(fig4, width="stretch")


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
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01),
                      **_chart_layout())
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = px.histogram(
            df, x="price_num", nbins=60, color="sex_clean", barmode="overlay",
            opacity=0.7, title="Price Distribution by Sex",
            labels={"price_num": "Price (€)", "sex_clean": "Sex"},
            color_discrete_sequence=PALETTE,
        )
        fig2.update_layout(height=380, **_chart_layout())
        st.plotly_chart(fig2, width="stretch")

    with col_b:
        fig3 = px.histogram(
            df, x="price_per_kg_num", nbins=60, color="sex_clean", barmode="overlay",
            opacity=0.7, title="Price per KG Distribution by Sex",
            labels={"price_per_kg_num": "Price per KG (€)", "sex_clean": "Sex"},
            color_discrete_sequence=PALETTE,
        )
        fig3.update_layout(height=380, **_chart_layout())
        st.plotly_chart(fig3, width="stretch")

    st.subheader("Price per KG vs Age")
    plot_df2 = df.dropna(subset=["age_months", "price_per_kg_num"])
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
    st.plotly_chart(fig4, width="stretch")


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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig2, width="stretch")

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
    st.plotly_chart(fig3, width="stretch")

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
    st.plotly_chart(fig4, width="stretch")


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
        legend=dict(orientation="h", y=1.08, x=0),
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
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig_vol, width="stretch")

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
    fig_bt.update_layout(legend=dict(orientation="h", y=1.08), **_chart_layout())
    fig_bt.update_xaxes(gridcolor="#F0F2F5")
    fig_bt.update_yaxes(gridcolor="#F0F2F5")
    st.plotly_chart(fig_bt, width="stretch")

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
            st.plotly_chart(fig_wx1, width="stretch")
        with col_wx2:
            fig_wx2 = px.scatter(wx_ts, x="avg_rain", y="avg_ppkg", trendline="ols",
                                 title="Rainfall vs Avg €/kg",
                                 labels={"avg_rain": "Precipitation (mm)", "avg_ppkg": "Avg €/kg"},
                                 height=350, color_discrete_sequence=[FB_GREEN])
            fig_wx2.update_layout(**_chart_layout())
            st.plotly_chart(fig_wx2, width="stretch")


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
                "weight": wt, "age_months": ag,
                "days_in_herd": 300, "no_of_owners": 1,
                "icbf_cbv_num": np.nan, "icbf_replacement_num": np.nan,
                "icbf_ebi_num": np.nan, "icbf_stars": 0,
                "has_genomic": 0, "quality_assured": 1, "bvd_ok": 1, "export_score": 2,
                "temp_max_c": mart_wx.get("temp_max_c", 15.0),
                "temp_min_c": mart_wx.get("temp_min_c", 8.0),
                "precipitation_mm": mart_wx.get("precipitation_mm", 2.0),
                "wind_speed_kmh": mart_wx.get("wind_speed_kmh", 20.0),
                "sale_month": future_month,
                "breed_grp": breed_val, "sex_clean": sex_val,
                "mart": mart_val, "dam_breed_grp": dam_val,
                "breed_sex": f"{breed_val}_{sex_val}",
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
        height=480, legend=dict(orientation="h", y=1.06), **_chart_layout(),
    )
    fig.update_xaxes(gridcolor="#F0F2F5")
    fig.update_yaxes(gridcolor="#F0F2F5")
    st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fv, use_container_width=True)
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
            st.plotly_chart(fp, use_container_width=True)

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
            st.plotly_chart(fig_hm, use_container_width=True)

        # Model-simulated monthly price (holding weight/age constant at target)
        if model and ppkg_preds:
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            month_ppkg = []
            base = {"weight": target_w, "age_months": float(proj_ages[-1]),
                    "days_in_herd": 300, "no_of_owners": 1,
                    "icbf_cbv_num": np.nan, "icbf_replacement_num": np.nan,
                    "icbf_ebi_num": np.nan, "icbf_stars": 0,
                    "has_genomic": 0, "quality_assured": 1, "bvd_ok": 1, "export_score": 2,
                    "temp_max_c": 15.0, "temp_min_c": 8.0,
                    "precipitation_mm": 2.0, "wind_speed_kmh": 20.0,
                    "breed_grp": breed_val, "sex_clean": sex_val,
                    "mart": mart_val, "dam_breed_grp": dam_val,
                    "breed_sex": f"{breed_val}_{sex_val}"}
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
            st.plotly_chart(fig_s, use_container_width=True)
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
            inp = {"weight": target_w, "age_months": float(proj_ages[-1]),
                   "days_in_herd": 300, "no_of_owners": 1,
                   "icbf_cbv_num": np.nan, "icbf_replacement_num": np.nan,
                   "icbf_ebi_num": np.nan, "icbf_stars": 0,
                   "has_genomic": 0, "quality_assured": 1, "bvd_ok": 1, "export_score": 2,
                   "temp_max_c": mwx.get("temp_max_c", 15.0),
                   "temp_min_c": mwx.get("temp_min_c", 8.0),
                   "precipitation_mm": mwx.get("precipitation_mm", 2.0),
                   "wind_speed_kmh": mwx.get("wind_speed_kmh", 20.0),
                   "sale_month": future_month, "breed_grp": breed_val,
                   "sex_clean": sex_val, "mart": m_name,
                   "dam_breed_grp": dam_val, "breed_sex": f"{breed_val}_{sex_val}"}
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
        st.plotly_chart(fig_m, use_container_width=True)
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
                             height=320, legend=dict(orientation="h", y=1.06),
                             **_chart_layout())
        st.plotly_chart(fig_be, use_container_width=True)
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
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;padding:8px 0 20px;">
      <div style="background:{FB_BLUE};border-radius:14px;width:52px;height:52px;
                  display:flex;align-items:center;justify-content:center;font-size:1.8rem;
                  box-shadow:0 2px 12px rgba(24,119,242,0.35);">🐄</div>
      <div>
        <div style="font-size:1.8rem;font-weight:900;color:{FB_DARK};
                    letter-spacing:-0.5px;line-height:1.1;">MartBids</div>
        <div style="font-size:0.85rem;color:{FB_GREY};font-weight:500;">
            Irish Cattle Market Intelligence · Live data from martbids.ie
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_full = load_data()
    df      = sidebar_filters(df_full)

    if df.empty:
        st.warning("No data matches the current filters. Try widening your selection.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview",
        "🔍 Price Explorer",
        "🐄 Breed & Mart",
        "📈 Price Tracker",
        "🧮 Growth Calculator",
    ])

    with tab1: tab_overview(df)
    with tab2: tab_explorer(df)
    with tab3: tab_breed_mart(df)
    with tab4: tab_tracker(df)
    with tab5: tab_calculator(df)


if __name__ == "__main__":
    main()
