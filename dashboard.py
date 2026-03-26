"""
MartBids Cattle Price Dashboard
================================
Run with:  streamlit run dashboard.py
"""

import io
import json
import warnings
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

warnings.filterwarnings("ignore")

# streamlit run dashboard.py
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MartBids Dashboard",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

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

    # Merge weather
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
    """Returns (shap_values Explanation, X_background array) or (None, None)."""
    sv_path = BASE / "shap_values.pkl"
    bg_path = BASE / "shap_background.pkl"
    if sv_path.exists() and bg_path.exists():
        sv = joblib.load(sv_path)
        bg = joblib.load(bg_path)
        return sv, bg
    return None, None


@st.cache_data
def load_weather_latest():
    """Returns a dict of latest weather per mart for predictor defaults."""
    wx_path = BASE / "weather_cache.csv"
    if not wx_path.exists():
        return {}
    wx = pd.read_csv(wx_path)
    wx["date"] = pd.to_datetime(wx["date"])
    latest = wx.sort_values("date").groupby("mart").last().reset_index()
    return latest.set_index("mart").to_dict("index")


# ── Sidebar filters ───────────────────────────────────────────────────────────

def sidebar_filters(df):
    st.sidebar.header("🔎 Filters")

    marts = sorted(df["mart"].unique())
    sel_marts = st.sidebar.multiselect("Mart", marts, default=marts,
                                        placeholder="All marts")

    breeds = sorted(df["breed"].dropna().unique())
    sel_breeds = st.sidebar.multiselect("Breed", breeds, default=[],
                                         placeholder="All breeds")

    sexes = sorted(df["sex_clean"].unique())
    sel_sex = st.sidebar.multiselect("Sex", sexes, default=sexes,
                                      placeholder="All")

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


PALETTE = px.colors.qualitative.Vivid


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
            color="avg_price", color_continuous_scale="Teal",
            title="Average Sold Price by Mart",
            labels={"avg_price": "Avg Price (€)", "mart": ""},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=520, margin=dict(l=10, r=80))
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
        )
        fig2.update_layout(height=520, showlegend=False)
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
        fig3.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig3, width="stretch")

    with col_d:
        top_breed_vol = df["breed"].value_counts().head(12).reset_index()
        top_breed_vol.columns = ["breed", "count"]
        fig4 = px.pie(
            top_breed_vol, names="breed", values="count",
            title="Lot Volume by Breed (Top 12)",
            color_discrete_sequence=PALETTE, hole=0.35,
        )
        fig4.update_layout(height=360)
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
                    "sex_clean": True, "price_per_kg_num": ":.2f",
                    "export_status": True, "quality_assurance": True},
        trendline="ols" if trend else None,
        trendline_scope="overall",
        opacity=0.65,
        title=f"Weight vs Price — coloured by {colour_by}",
        labels={"weight": "Weight (kg)", "price_num": "Price (€)",
                "sex_clean": "Sex", "price_per_kg_num": "€/kg"},
        color_discrete_sequence=PALETTE,
        height=560,
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01))
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = px.histogram(
            df, x="price_num", nbins=60, color="sex_clean", barmode="overlay",
            opacity=0.7, title="Price Distribution by Sex",
            labels={"price_num": "Price (€)", "sex_clean": "Sex"},
            color_discrete_sequence=PALETTE,
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, width="stretch")

    with col_b:
        fig3 = px.histogram(
            df, x="price_per_kg_num", nbins=60, color="sex_clean", barmode="overlay",
            opacity=0.7, title="Price per KG Distribution by Sex",
            labels={"price_per_kg_num": "Price per KG (€)", "sex_clean": "Sex"},
            color_discrete_sequence=PALETTE,
        )
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, width="stretch")

    st.subheader("Price per KG vs Age")
    plot_df2 = df.dropna(subset=["age_months", "price_per_kg_num"])
    fig4 = px.scatter(
        plot_df2, x="age_months", y="price_per_kg_num",
        color="sex_clean", hover_name="breed",
        hover_data={"mart": True, "weight": True, "price_num": ":,.0f"},
        opacity=0.55, trendline="lowess", trendline_scope="trace",
        title="How Price per KG changes with Age",
        labels={"age_months": "Age (months)", "price_per_kg_num": "€/kg",
                "sex_clean": "Sex"},
        color_discrete_sequence=PALETTE, height=440,
    )
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
        fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    with col_b:
        ppkg = (df[df["breed"].isin(top_b)]
                  .groupby("breed")["price_per_kg_num"]
                  .median()
                  .sort_values(ascending=False)
                  .reset_index())
        fig2 = px.bar(
            ppkg, x="breed", y="price_per_kg_num",
            color="price_per_kg_num", color_continuous_scale="Tealgrn",
            text=ppkg["price_per_kg_num"].map("€{:.2f}".format),
            title="Median Price per KG by Breed",
            labels={"price_per_kg_num": "Median €/kg", "breed": "Breed"},
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(coloraxis_showscale=False, height=460, xaxis_tickangle=-30)
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
    fig3.update_layout(showlegend=False, xaxis_tickangle=-35)
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Breed × Sex — Average Price Heatmap")
    pivot_breeds = df["breed"].value_counts().head(15).index
    heat_df = (df[df["breed"].isin(pivot_breeds)]
                 .groupby(["breed", "sex_clean"])["price_num"]
                 .mean()
                 .unstack(fill_value=np.nan))
    fig4 = px.imshow(
        heat_df, text_auto=".0f", color_continuous_scale="Teal",
        aspect="auto",
        title="Average Price (€) — Breed × Sex",
        labels={"x": "Sex", "y": "Breed", "color": "Avg Price (€)"},
        height=500,
    )
    st.plotly_chart(fig4, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Time Series
# ═══════════════════════════════════════════════════════════════════════════════

def tab_timeseries(df):
    df2 = df.dropna(subset=["sale_date"]).copy()
    if df2.empty:
        st.info("No date information available yet. Data will appear here as daily scrapes accumulate.")
        return

    df2["week"] = df2["sale_date"].dt.to_period("W").apply(lambda p: p.start_time)
    df2["month"] = df2["sale_date"].dt.to_period("M").apply(lambda p: p.start_time)

    res_choice = st.radio("Resolution", ["Daily", "Weekly", "Monthly"],
                          horizontal=True, index=1)
    if res_choice == "Daily":
        grp_col = df2["sale_date"].dt.date
    elif res_choice == "Weekly":
        grp_col = df2["week"]
    else:
        grp_col = df2["month"]

    # Overall price trend
    ts = (df2.groupby(grp_col)
              .agg(avg_price=("price_num", "mean"),
                   median_price=("price_num", "median"),
                   avg_ppkg=("price_per_kg_num", "mean"),
                   lots=("price_num", "count"))
              .reset_index())
    ts.columns = ["date", "avg_price", "median_price", "avg_ppkg", "lots"]

    col_a, col_b = st.columns(2)

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["date"], y=ts["avg_price"],
                                  mode="lines+markers", name="Avg Price",
                                  line=dict(color="#2196F3", width=2)))
        fig.add_trace(go.Scatter(x=ts["date"], y=ts["median_price"],
                                  mode="lines", name="Median Price",
                                  line=dict(color="#FF9800", dash="dash")))
        fig.update_layout(title="Average & Median Price Over Time",
                          xaxis_title="Date", yaxis_title="Price (€)",
                          height=380, legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts["date"], y=ts["avg_ppkg"],
                                   mode="lines+markers", name="Avg €/kg",
                                   line=dict(color="#4CAF50", width=2)))
        fig2.update_layout(title="Average Price per kg Over Time",
                           xaxis_title="Date", yaxis_title="€/kg",
                           height=380)
        st.plotly_chart(fig2, width="stretch")

    # Volume over time
    fig3 = px.bar(
        ts, x="date", y="lots", color_discrete_sequence=["#9C27B0"],
        title="Lots Sold Over Time",
        labels={"date": "Date", "lots": "Lots Sold"},
        height=300,
    )
    st.plotly_chart(fig3, width="stretch")

    # Trend by breed over time
    st.subheader("Price per kg by Breed Over Time")
    top_breeds = df2["breed"].value_counts().head(6).index.tolist()
    breed_ts = (df2[df2["breed"].isin(top_breeds)]
                  .groupby([grp_col, "breed"])["price_per_kg_num"]
                  .mean()
                  .reset_index())
    breed_ts.columns = ["date", "breed", "avg_ppkg"]
    fig4 = px.line(
        breed_ts, x="date", y="avg_ppkg", color="breed",
        markers=True,
        title="Average €/kg by Breed Over Time (Top 6)",
        labels={"date": "Date", "avg_ppkg": "Avg €/kg", "breed": "Breed"},
        color_discrete_sequence=PALETTE,
        height=420,
    )
    fig4.update_layout(legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig4, width="stretch")

    # Weather overlay if available
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
            fig_wx1 = px.scatter(
                wx_ts, x="avg_temp", y="avg_ppkg",
                trendline="ols",
                title="Avg Temp vs Avg €/kg",
                labels={"avg_temp": "Avg Max Temp (°C)", "avg_ppkg": "Avg €/kg"},
                height=350,
            )
            st.plotly_chart(fig_wx1, width="stretch")
        with col_wx2:
            fig_wx2 = px.scatter(
                wx_ts, x="avg_rain", y="avg_ppkg",
                trendline="ols",
                title="Rainfall vs Avg €/kg",
                labels={"avg_rain": "Precipitation (mm)", "avg_ppkg": "Avg €/kg"},
                height=350,
            )
            st.plotly_chart(fig_wx2, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML Model
# ═══════════════════════════════════════════════════════════════════════════════

def _shap_beeswarm_fig(shap_values, feature_names):
    """Render SHAP beeswarm and return as bytes for st.image."""
    # Attach feature names to the Explanation object
    sv = shap_values
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.beeswarm(
        shap.Explanation(
            values=sv.values,
            base_values=sv.base_values,
            data=sv.data,
            feature_names=feature_names,
        ),
        max_display=16,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close("all")
    return data


def _shap_waterfall_fig(shap_exp_row, feature_names):
    """Render SHAP waterfall for a single row and return as bytes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_exp_row.values,
            base_values=float(shap_exp_row.base_values),
            data=shap_exp_row.data,
            feature_names=feature_names,
        ),
        max_display=14,
        show=False,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close("all")
    return data


def tab_model(df):
    model     = load_model()
    meta      = load_meta()
    preds     = load_test_preds()
    shap_vals, shap_bg = load_shap_objects()
    wx_latest = load_weather_latest()

    # ── Metrics ────────────────────────────────────────────────────────────────
    if meta:
        st.subheader("📊 Model Performance  (LightGBM — target: €/kg)")
        tm = meta.get("test_metrics", {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Test R²",       f"{tm.get('R2', 0):.4f}")
        m2.metric("Test MAE",      f"€{tm.get('MAE_eur_kg', tm.get('MAE_€', 0)):.4f}/kg")
        m3.metric("Test RMSE",     f"€{tm.get('RMSE_eur_kg', tm.get('RMSE_€', 0)):.4f}/kg")
        m4.metric("MAPE",          f"{tm.get('MAPE_%', 0):.1f}%")
        cv_key = "cv_mae_eur_kg" if "cv_mae_eur_kg" in meta else "cv_mae_eur"
        m5.metric("CV MAE",        f"€{meta.get(cv_key, 0):.4f}/kg ± €{meta.get('cv_mae_std', 0):.4f}")

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

    # Feature importance
    with col_a:
        if meta.get("feature_importances"):
            imp = (pd.Series(meta["feature_importances"])
                     .sort_values(ascending=True)
                     .tail(15))
            fig = px.bar(
                x=imp.values, y=imp.index, orientation="h",
                color=imp.values, color_continuous_scale="Teal",
                title="Feature Importances (Top 15)",
                labels={"x": "Importance", "y": "Feature"},
            )
            fig.update_layout(coloraxis_showscale=False, height=460, margin=dict(l=10))
            st.plotly_chart(fig, width="stretch")

    # Actual vs Predicted (€/kg)
    with col_b:
        if not preds.empty:
            act_col  = "actual_ppkg"  if "actual_ppkg"  in preds.columns else "actual"
            pred_col = "predicted_ppkg" if "predicted_ppkg" in preds.columns else "predicted"
            fig2 = px.scatter(
                preds, x=act_col, y=pred_col,
                color="breed", opacity=0.6,
                hover_data={"mart": True, "sex": True, "weight": True},
                title="Actual vs Predicted (€/kg)",
                labels={act_col: "Actual €/kg", pred_col: "Predicted €/kg"},
                height=460,
            )
            lo = min(preds[act_col].min(), preds[pred_col].min())
            hi = max(preds[act_col].max(), preds[pred_col].max())
            fig2.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi],
                                      mode="lines", name="Perfect",
                                      line=dict(color="red", dash="dash", width=1.5)))
            fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01))
            st.plotly_chart(fig2, width="stretch")

    # Residuals
    if not preds.empty:
        act_col  = "actual_ppkg"  if "actual_ppkg"  in preds.columns else "actual"
        pred_col = "predicted_ppkg" if "predicted_ppkg" in preds.columns else "predicted"
        preds["residual"] = preds[pred_col] - preds[act_col]
        fig3 = px.histogram(
            preds, x="residual", nbins=60,
            color_discrete_sequence=["steelblue"],
            title="Residuals (Predicted − Actual, €/kg)",
            labels={"residual": "Residual (€/kg)"},
            height=300,
        )
        fig3.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig3, width="stretch")

    # ── SHAP Beeswarm ──────────────────────────────────────────────────────────
    if shap_vals is not None:
        st.divider()
        st.subheader("🔍 SHAP Feature Explanations (Global)")
        st.caption("Each dot is a test-set prediction. Red = high feature value, blue = low. "
                   "Horizontal position shows the feature's impact on price.")
        try:
            img_bytes = _shap_beeswarm_fig(shap_vals, ALL_FEATURES)
            st.image(img_bytes, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP beeswarm unavailable: {e}")

    # ── Interactive Price Predictor ────────────────────────────────────────────
    if model:
        st.divider()
        st.subheader("🔮 Price Predictor")
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

        # Weather inputs with defaults from cache
        st.markdown("**Weather (on day of sale)**")
        wx_def = wx_latest.get(mart_val, {})
        wx1, wx2, wx3, wx4 = st.columns(4)
        temp_max  = wx1.number_input("Max Temp (°C)",      -10.0, 40.0,
                                     float(wx_def.get("temp_max_c",   15.0)), step=0.5)
        temp_min  = wx2.number_input("Min Temp (°C)",      -15.0, 35.0,
                                     float(wx_def.get("temp_min_c",    8.0)), step=0.5)
        precip    = wx3.number_input("Precipitation (mm)",   0.0, 80.0,
                                     float(wx_def.get("precipitation_mm", 2.0)), step=0.5)
        wind_spd  = wx4.number_input("Wind Speed (km/h)",    0.0, 120.0,
                                     float(wx_def.get("wind_speed_kmh", 20.0)), step=1.0)

        if st.button("🐄 Predict Price", type="primary", width="content"):
            import datetime
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
            input_df = pd.DataFrame([input_row])
            ppkg_pred  = model.predict(input_df)[0]
            total_pred = ppkg_pred * weight_val

            r1, r2 = st.columns(2)
            r1.success(f"**Predicted Price/kg: €{ppkg_pred:.2f}/kg**")
            r2.success(f"**Predicted Total:  €{total_pred:,.0f}**")

            # SHAP waterfall for this prediction
            if shap_bg is not None:
                try:
                    prep        = model.named_steps["prep"]
                    lgb_step    = model.named_steps["model"]
                    input_t     = prep.transform(input_df)
                    explainer_w = shap.TreeExplainer(lgb_step, data=shap_bg)
                    sv_single   = explainer_w(input_t)
                    st.subheader("SHAP Explanation for this Prediction")
                    st.caption("How each feature pushed the price above or below the average.")
                    wf_bytes = _shap_waterfall_fig(sv_single[0], ALL_FEATURES)
                    st.image(wf_bytes, use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP waterfall unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🐄 MartBids — Irish Cattle Sales Dashboard")
    st.caption("Live data scraped from martbids.ie · Filtered data shown throughout")

    df_full = load_data()
    df      = sidebar_filters(df_full)

    if df.empty:
        st.warning("No data matches the current filters. Try widening your selection.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview",
        "🔍 Price Explorer",
        "🐄 Breed & Mart Deep Dive",
        "📈 Time Series",
        "🤖 ML Model",
    ])

    with tab1:
        tab_overview(df)
    with tab2:
        tab_explorer(df)
    with tab3:
        tab_breed_mart(df)
    with tab4:
        tab_timeseries(df)
    with tab5:
        tab_model(df)


if __name__ == "__main__":
    main()
