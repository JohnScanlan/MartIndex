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
FB_GREY   = "#65676B"
FB_GREEN  = "#42B72A"
FB_RED    = "#FA383E"
FB_BORDER = "#CED0D4"

PALETTE = [FB_BLUE, FB_GREEN, "#F7B928", FB_RED, "#9B59B6",
           "#00BCD4", "#FF9800", "#E91E63", "#3F51B5", "#009688"]

st.markdown(f"""
<style>
    /* Background */
    .stApp {{ background-color: {FB_BG}; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {FB_CARD};
        border-right: 1px solid {FB_BORDER};
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: {FB_CARD};
        border: 1px solid {FB_BORDER};
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }}
    [data-testid="stMetricLabel"] {{
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

    /* Tab bar */
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
        color: {FB_GREY};
        padding: 8px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {FB_BLUE} !important;
        color: white !important;
    }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: {FB_BLUE};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 10px 28px;
        font-size: 1rem;
        transition: background 0.15s;
    }}
    .stButton > button[kind="primary"]:hover {{ background: #166FE5; }}

    /* Headings */
    h1 {{
        color: {FB_DARK} !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }}
    h2, h3 {{
        color: {FB_DARK} !important;
        font-weight: 700 !important;
    }}

    /* Plotly panels */
    [data-testid="stPlotlyChart"] {{
        background: {FB_CARD};
        border: 1px solid {FB_BORDER};
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}

    hr {{ border-color: {FB_BORDER} !important; }}

    /* Success/info boxes */
    .stSuccess {{
        background: #E7F3FF !important;
        border-left: 4px solid {FB_BLUE} !important;
        border-radius: 6px;
    }}
    div[data-testid="stAlert"] {{
        border-radius: 8px;
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
def compute_growth_rates(df_hash):
    """
    Fit weight ~ age_months per breed_grp × sex_clean.
    Returns dict (breed, sex) → kg_per_month.
    """
    df = load_data()
    grp = df.dropna(subset=["age_months", "weight"])
    rates = {}
    for (breed, sex), g in grp.groupby(["breed_grp", "sex_clean"]):
        if len(g) < 15:
            continue
        X = g["age_months"].values.reshape(-1, 1)
        y = g["weight"].values
        coef = LinearRegression().fit(X, y).coef_[0]
        rates[(breed, sex)] = max(float(coef), 0.3)
    # overall fallback
    X_all = grp["age_months"].values.reshape(-1, 1)
    y_all = grp["weight"].values
    rates[("_default", "_default")] = max(float(LinearRegression().fit(X_all, y_all).coef_[0]), 0.3)
    return rates


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
        Enter your animal's details to project its weight and estimated market value.
        Growth rates are calculated from our actual mart data.
      </div>
    </div>
    """, unsafe_allow_html=True)

    growth_rates = compute_growth_rates(str(len(df)))
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
    mart_val   = c2.selectbox("Target Mart", all_marts, key="calc_mart")
    cur_weight = c3.number_input("Current Weight (kg)", 50, 800, 330, step=5)
    cur_age    = c3.number_input("Current Age (months)", 0, 60, 8, step=1)

    months_fwd = st.slider("Months ahead to project", 1, 24, 6,
                            help="How many months from now you plan to sell")

    # Growth rate
    rate = growth_rates.get((breed_val, sex_val),
            growth_rates.get(("_default", "_default"), 5.0))

    # Month-by-month projection
    proj_months  = list(range(0, months_fwd + 1))
    proj_weights = [cur_weight + rate * m for m in proj_months]
    proj_ages    = [cur_age + m for m in proj_months]
    target_w     = proj_weights[-1]
    weight_gain  = target_w - cur_weight

    # Model predictions along the trajectory
    val_preds    = []
    ppkg_preds   = []
    if model:
        for m, wt, ag in zip(proj_months, proj_weights, proj_ages):
            future_month = (datetime.date.today().month + m - 1) % 12 + 1
            inp = {
                "weight":               wt,
                "age_months":           ag,
                "days_in_herd":         300,
                "no_of_owners":         1,
                "icbf_cbv_num":         np.nan,
                "icbf_replacement_num": np.nan,
                "icbf_ebi_num":         np.nan,
                "icbf_stars":           0,
                "has_genomic":          0,
                "quality_assured":      1,
                "bvd_ok":               1,
                "export_score":         2,
                "temp_max_c":           15.0,
                "temp_min_c":            8.0,
                "precipitation_mm":      2.0,
                "wind_speed_kmh":       20.0,
                "sale_month":           future_month,
                "breed_grp":            breed_val,
                "sex_clean":            sex_val,
                "mart":                 mart_val,
                "dam_breed_grp":        dam_val,
                "breed_sex":            f"{breed_val}_{sex_val}",
            }
            ppkg = model.predict(pd.DataFrame([inp]))[0]
            ppkg_preds.append(ppkg)
            val_preds.append(ppkg * wt)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Data-derived Growth Rate",
              f"{rate:.1f} kg/month",
              help=f"Linear regression on {breed_val} {sex_val} animals in our dataset")
    r2.metric(f"Projected Weight (+{months_fwd}m)",
              f"{target_w:.0f} kg",
              delta=f"+{weight_gain:.0f} kg")
    if ppkg_preds:
        r3.metric("Estimated €/kg at target",  f"€{ppkg_preds[-1]:.2f}")
        r4.metric("Estimated Value at target",
                  f"€{val_preds[-1]:,.0f}",
                  delta=f"vs now €{val_preds[0]:,.0f}" if val_preds else None)

    st.divider()

    # ── Growth chart ──────────────────────────────────────────────────────────
    comp = df[(df["breed_grp"] == breed_val) &
              (df["sex_clean"] == sex_val)].dropna(subset=["age_months", "weight"])

    fig = go.Figure()

    # Background scatter of comparable sold animals
    if len(comp) > 0:
        sample = comp.sample(min(400, len(comp)), random_state=42)
        fig.add_trace(go.Scatter(
            x=sample["age_months"], y=sample["weight"],
            mode="markers",
            marker=dict(color=f"rgba(24,119,242,0.20)", size=5, symbol="circle"),
            name=f"{breed_val} {sex_val} — sold animals",
            hovertemplate="Age: %{x:.0f}m  Weight: %{y:.0f}kg<extra></extra>",
        ))

    # Projected path
    fig.add_trace(go.Scatter(
        x=proj_ages, y=proj_weights,
        mode="lines+markers",
        line=dict(color=FB_GREEN, width=3),
        marker=dict(size=7, color=FB_GREEN),
        name="Your animal's projected path",
        customdata=proj_months,
        hovertemplate="Month +%{customdata}: %{y:.0f} kg  (age %{x:.0f}m)<extra></extra>",
    ))

    # Current position
    fig.add_trace(go.Scatter(
        x=[cur_age], y=[cur_weight],
        mode="markers",
        marker=dict(color=FB_BLUE, size=16, symbol="circle",
                    line=dict(color="white", width=3)),
        name="Current",
        hovertemplate=f"Now: {cur_weight}kg at {cur_age}m<extra></extra>",
    ))

    # Target
    fig.add_trace(go.Scatter(
        x=[proj_ages[-1]], y=[target_w],
        mode="markers",
        marker=dict(color=FB_GREEN, size=16, symbol="star",
                    line=dict(color="white", width=2)),
        name=f"Target (+{months_fwd}m)",
        hovertemplate=f"Target: {target_w:.0f}kg at {proj_ages[-1]}m<extra></extra>",
    ))

    fig.update_layout(
        title=f"{breed_val} {sex_val} — Growth Projection ({months_fwd} months)",
        xaxis_title="Age (months)",
        yaxis_title="Weight (kg)",
        height=500,
        legend=dict(orientation="h", y=1.06),
        **_chart_layout(),
    )
    fig.update_xaxes(gridcolor="#F0F2F5")
    fig.update_yaxes(gridcolor="#F0F2F5")
    st.plotly_chart(fig, width="stretch")

    # ── Value trajectory ──────────────────────────────────────────────────────
    if val_preds:
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(
                x=proj_months, y=val_preds,
                fill="tozeroy",
                fillcolor="rgba(66, 183, 42, 0.12)",
                line=dict(color=FB_GREEN, width=2.5),
                mode="lines+markers",
                marker=dict(size=6, color=FB_GREEN),
                hovertemplate="Month +%{x}: €%{y:,.0f}<extra></extra>",
            ))
            fig_val.update_layout(
                title="Estimated Market Value Over Time",
                xaxis_title="Months from now",
                yaxis_title="Estimated Value (€)",
                height=320,
                **_chart_layout(),
            )
            fig_val.update_xaxes(gridcolor="#F0F2F5", tickvals=proj_months)
            fig_val.update_yaxes(gridcolor="#F0F2F5")
            st.plotly_chart(fig_val, width="stretch")

        with col_v2:
            fig_ppkg = go.Figure()
            fig_ppkg.add_trace(go.Scatter(
                x=proj_months, y=ppkg_preds,
                line=dict(color=FB_BLUE, width=2.5),
                mode="lines+markers",
                marker=dict(size=6, color=FB_BLUE),
                hovertemplate="Month +%{x}: €%{y:.2f}/kg<extra></extra>",
            ))
            fig_ppkg.update_layout(
                title="Estimated Price per KG Over Time",
                xaxis_title="Months from now",
                yaxis_title="€/kg",
                height=320,
                **_chart_layout(),
            )
            fig_ppkg.update_xaxes(gridcolor="#F0F2F5", tickvals=proj_months)
            fig_ppkg.update_yaxes(gridcolor="#F0F2F5")
            st.plotly_chart(fig_ppkg, width="stretch")

        # Summary table
        st.subheader("Projection Summary")
        summary = pd.DataFrame({
            "Month":          [f"+{m}m" for m in proj_months],
            "Age":            [f"{a}m" for a in proj_ages],
            "Weight (kg)":    [f"{w:.0f}" for w in proj_weights],
            "Est. €/kg":      [f"€{p:.2f}" for p in ppkg_preds],
            "Est. Value (€)": [f"€{v:,.0f}" for v in val_preds],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ML Model
# ═══════════════════════════════════════════════════════════════════════════════

def _shap_beeswarm_fig(shap_values, feature_names):
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values.values,
            base_values=shap_values.base_values,
            data=shap_values.data,
            feature_names=feature_names,
        ),
        max_display=16, show=False, plot_size=None,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close("all")
    return data


def _shap_waterfall_fig(shap_exp_row, feature_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_exp_row.values,
            base_values=float(shap_exp_row.base_values),
            data=shap_exp_row.data,
            feature_names=feature_names,
        ),
        max_display=14, show=False,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close("all")
    return data


def tab_model(df):
    model       = load_model()
    meta        = load_meta()
    preds       = load_test_preds()
    shap_vals, shap_bg = load_shap_objects()
    wx_latest   = load_weather_latest()

    if meta:
        st.subheader("Model Performance  (LightGBM — target: €/kg)")
        tm = meta.get("test_metrics", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Test R²",   f"{tm.get('R2', 0):.4f}")
        m2.metric("Test MAE",  f"€{tm.get('MAE_eur_kg', 0):.4f}/kg")
        m3.metric("Test RMSE", f"€{tm.get('RMSE_eur_kg', 0):.4f}/kg")
        m4.metric("MAPE",      f"{tm.get('MAPE_%', 0):.1f}%")
        m5.metric("CV MAE",    f"€{meta.get('cv_mae_eur_kg', 0):.4f}/kg ± €{meta.get('cv_mae_std', 0):.4f}")

        within_items = {
            "Within 5%":  tm.get("within_5pct", 0),
            "Within 10%": tm.get("within_10pct", 0),
            "Within 20%": tm.get("within_20pct", 0),
        }
        wcols = st.columns(len(within_items))
        for col, (label, val) in zip(wcols, within_items.items()):
            col.metric(label, f"{val:.1f}%")

        st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        if meta.get("feature_importances"):
            imp = (pd.Series(meta["feature_importances"])
                     .sort_values(ascending=True)
                     .tail(15))
            fig = px.bar(
                x=imp.values, y=imp.index, orientation="h",
                color=imp.values,
                color_continuous_scale=[[0, "#E7F3FF"], [1, FB_BLUE]],
                title="Feature Importances (Top 15)",
                labels={"x": "Importance", "y": "Feature"},
            )
            fig.update_layout(coloraxis_showscale=False, height=460,
                              margin=dict(l=10), **_chart_layout())
            st.plotly_chart(fig, width="stretch")

    with col_b:
        if not preds.empty:
            act_col  = "actual_ppkg"    if "actual_ppkg"    in preds.columns else "actual"
            pred_col = "predicted_ppkg" if "predicted_ppkg" in preds.columns else "predicted"
            fig2 = px.scatter(
                preds, x=act_col, y=pred_col,
                color="breed", opacity=0.6,
                hover_data={"mart": True, "sex": True, "weight": True},
                title="Actual vs Predicted (€/kg)",
                labels={act_col: "Actual €/kg", pred_col: "Predicted €/kg"},
                height=460, color_discrete_sequence=PALETTE,
            )
            lo = min(preds[act_col].min(), preds[pred_col].min())
            hi = max(preds[act_col].max(), preds[pred_col].max())
            fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                      name="Perfect",
                                      line=dict(color=FB_RED, dash="dash", width=1.5)))
            fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01),
                               **_chart_layout())
            st.plotly_chart(fig2, width="stretch")

    if not preds.empty:
        act_col  = "actual_ppkg"    if "actual_ppkg"    in preds.columns else "actual"
        pred_col = "predicted_ppkg" if "predicted_ppkg" in preds.columns else "predicted"
        preds["residual"] = preds[pred_col] - preds[act_col]
        fig3 = px.histogram(
            preds, x="residual", nbins=60,
            color_discrete_sequence=[FB_BLUE],
            title="Residuals (Predicted − Actual, €/kg)",
            labels={"residual": "Residual (€/kg)"}, height=300,
        )
        fig3.add_vline(x=0, line_dash="dash", line_color=FB_RED)
        fig3.update_layout(**_chart_layout())
        st.plotly_chart(fig3, width="stretch")

    # SHAP Beeswarm
    if shap_vals is not None:
        st.divider()
        st.subheader("SHAP Feature Explanations (Global)")
        st.caption("Each dot is a test-set prediction. Red = high feature value, blue = low.")
        try:
            st.image(_shap_beeswarm_fig(shap_vals, ALL_FEATURES), use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP beeswarm unavailable: {e}")

    # Price Predictor
    if model:
        st.divider()
        st.subheader("Price Predictor")
        st.caption("Fill in the animal details to get a predicted price (€/kg and €total).")

        all_breeds = sorted(df["breed_grp"].dropna().unique())
        all_marts  = sorted(df["mart"].dropna().unique())
        all_dam    = sorted(df["dam_breed_grp"].dropna().unique())

        c1, c2, c3, c4 = st.columns(4)
        weight_val = c1.number_input("Weight (kg)",   50,  1000, 500, step=5)
        age_val    = c1.number_input("Age (months)",   0,   120,  24, step=1)
        days_herd  = c2.number_input("Days in Herd",   0,  2000, 300, step=10)
        n_owners   = c2.number_input("No. of Owners",  1,    10,   1, step=1)
        breed_val  = c3.selectbox("Breed",  all_breeds,
                                   index=all_breeds.index("LMX") if "LMX" in all_breeds else 0)
        sex_val    = c3.selectbox("Sex",    ["M", "F", "B", "Unknown"])
        mart_val   = c4.selectbox("Mart",   all_marts)
        dam_breed  = c4.selectbox("Dam Breed", all_dam)

        st.markdown("**ICBF / Health**")
        h1, h2, h3, h4, h5 = st.columns(5)
        qa_val      = h1.checkbox("Quality Assured", value=True)
        bvd_val     = h2.checkbox("BVD Tested",      value=True)
        genomic_val = h3.checkbox("Genomic Eval",    value=False)
        export_val  = h4.selectbox("Export Status",  ["Yes", "ReTest", "No"], index=0)
        icbf_cbv_on = h5.checkbox("Include ICBF CBV", value=False)
        icbf_cbv_val = h5.number_input("ICBF CBV (€)", -500, 1000, 0, step=5) if icbf_cbv_on else None

        i1, _ = st.columns(2)
        icbf_rep_on  = i1.checkbox("Include ICBF Replacement Index", value=False)
        icbf_rep_val = i1.number_input("ICBF Replacement Index (€)", -500, 1000, 0, step=5,
                                        disabled=not icbf_rep_on)

        st.markdown("**Weather (on day of sale)**")
        wx_def = wx_latest.get(mart_val, {})
        wx1, wx2, wx3, wx4 = st.columns(4)
        temp_max  = wx1.number_input("Max Temp (°C)",     -10.0,  40.0, float(wx_def.get("temp_max_c",   15.0)), step=0.5)
        temp_min  = wx2.number_input("Min Temp (°C)",     -15.0,  35.0, float(wx_def.get("temp_min_c",    8.0)), step=0.5)
        precip    = wx3.number_input("Precipitation (mm)",  0.0,  80.0, float(wx_def.get("precipitation_mm", 2.0)), step=0.5)
        wind_spd  = wx4.number_input("Wind Speed (km/h)",   0.0, 120.0, float(wx_def.get("wind_speed_kmh", 20.0)), step=1.0)

        if st.button("Predict Price", type="primary", width="content"):
            input_row = {
                "weight":               weight_val,
                "age_months":           age_val,
                "days_in_herd":         days_herd,
                "no_of_owners":         n_owners,
                "icbf_cbv_num":         float(icbf_cbv_val) if icbf_cbv_on and icbf_cbv_val is not None else np.nan,
                "icbf_replacement_num": float(icbf_rep_val) if icbf_rep_on else np.nan,
                "icbf_ebi_num":         np.nan,
                "icbf_stars":           0,
                "has_genomic":          int(genomic_val),
                "quality_assured":      int(qa_val),
                "bvd_ok":               int(bvd_val),
                "export_score":         {"Yes": 2, "ReTest": 1, "No": 0}.get(export_val, np.nan),
                "temp_max_c":           temp_max,
                "temp_min_c":           temp_min,
                "precipitation_mm":     precip,
                "wind_speed_kmh":       wind_spd,
                "sale_month":           datetime.date.today().month,
                "breed_grp":            breed_val,
                "sex_clean":            sex_val,
                "mart":                 mart_val,
                "dam_breed_grp":        dam_breed,
                "breed_sex":            f"{breed_val}_{sex_val}",
            }
            input_df  = pd.DataFrame([input_row])
            ppkg_pred = model.predict(input_df)[0]
            total_pred = ppkg_pred * weight_val

            r1, r2 = st.columns(2)
            r1.success(f"**Predicted Price/kg: €{ppkg_pred:.2f}/kg**")
            r2.success(f"**Predicted Total:  €{total_pred:,.0f}**")

            if shap_bg is not None:
                try:
                    prep        = model.named_steps["prep"]
                    lgb_step    = model.named_steps["model"]
                    input_t     = prep.transform(input_df)
                    explainer_w = shap.TreeExplainer(lgb_step, data=shap_bg)
                    sv_single   = explainer_w(input_t)
                    st.subheader("SHAP Explanation for this Prediction")
                    st.caption("How each feature pushed the price above or below average.")
                    st.image(_shap_waterfall_fig(sv_single[0], ALL_FEATURES),
                             use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP waterfall unavailable: {e}")


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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Overview",
        "🔍 Price Explorer",
        "🐄 Breed & Mart",
        "📈 Price Tracker",
        "🧮 Growth Calculator",
        "🤖 ML Model",
    ])

    with tab1: tab_overview(df)
    with tab2: tab_explorer(df)
    with tab3: tab_breed_mart(df)
    with tab4: tab_tracker(df)
    with tab5: tab_calculator(df)
    with tab6: tab_model(df)


if __name__ == "__main__":
    main()
