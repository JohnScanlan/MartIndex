"""
MartBids — Banking & Lending Intelligence Dashboard
====================================================
Tailored for agricultural lenders: breed × weight cohort pricing,
regional differentials, rolling trends, and collateral reference tables.

Run with:  streamlit run bank_dashboard.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import load_data_safe

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent

# ── Finance colour palette ─────────────────────────────────────────────────────
NAVY    = "#1B2A4A"
NAVY_LT = "#2C3E60"
GOLD    = "#C9A84C"
CREAM   = "#F8F6F0"
WHITE   = "#FFFFFF"
LGREY   = "#F4F5F7"
DGREY   = "#4A5568"
GREEN   = "#276749"
RED     = "#9B2335"
BORDER  = "#D1D5DB"
PALETTE = [GOLD, "#4A90D9", GREEN, RED, "#8E44AD", "#E67E22",
           "#2ECC71", "#E74C3C", "#3498DB", "#F39C12"]

st.set_page_config(
    page_title="MartBids — Lending Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {CREAM}; }}
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
        border-right: none;
    }}
    section[data-testid="stSidebar"] * {{ color: #E8EAF0 !important; }}
    section[data-testid="stSidebar"] label {{ color: {GOLD} !important;
        font-weight: 600; font-size: 0.82rem; text-transform: uppercase;
        letter-spacing: 0.6px; }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"] > div {{
        background: {NAVY_LT} !important; border-color: #3D5278 !important; }}
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] input {{
        color: #E8EAF0 !important; }}
    h1 {{ color: {NAVY} !important; font-size:1.6rem !important;
          font-weight:800 !important; letter-spacing:-0.5px; }}
    h2, h3 {{ color: {NAVY} !important; font-weight:700 !important; }}
    p, li {{ color: {DGREY} !important; }}
    label {{ color: {NAVY} !important; font-weight:500; }}
    .stMarkdown p {{ color: {DGREY} !important; font-size:0.93rem; }}
    [data-testid="stMetric"] {{
        background: {WHITE}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 16px 20px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        border-top: 3px solid {GOLD};
    }}
    [data-testid="stMetricLabel"] p {{
        color: {DGREY} !important; font-size:0.76rem !important;
        font-weight:700 !important; text-transform:uppercase;
        letter-spacing:0.6px;
    }}
    [data-testid="stMetricValue"] {{
        color: {NAVY} !important; font-size:1.5rem !important;
        font-weight:800 !important;
    }}
    [data-testid="stPlotlyChart"] {{
        background: {WHITE}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: {WHITE}; border-radius: 8px; padding: 4px;
        border: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {DGREY} !important; font-weight:600; border-radius:6px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {NAVY} !important; color: {WHITE} !important;
    }}
    [data-baseweb="select"], [data-baseweb="select"] > div,
    [data-baseweb="input"], [data-baseweb="input"] > div {{
        background-color: {WHITE} !important; border-color: {BORDER} !important;
    }}
    [data-baseweb="select"] span, [data-baseweb="select"] input,
    [data-baseweb="input"] input {{ color: {NAVY} !important; }}
    [data-baseweb="menu"], [data-baseweb="popover"] > div {{
        background: {WHITE} !important; border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-baseweb="menu"] li {{ color: {NAVY} !important; background: {WHITE} !important; }}
    [data-baseweb="menu"] li:hover {{ background: {LGREY} !important; color: {NAVY} !important; }}
    div[data-testid="stAlert"] {{ border-radius: 8px; }}
    hr {{ border-color: {BORDER} !important; }}
    .stDownloadButton > button {{
        background: {NAVY} !important; color: {WHITE} !important;
        border: none !important; border-radius: 6px;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)


# ── Reference tables ───────────────────────────────────────────────────────────

REGION_MAP = {
    # Connacht (Galway, Mayo, Roscommon, Sligo, Leitrim)
    "Athenry":          "Connacht",
    "Balla":            "Connacht",
    "Ballinasloe":      "Connacht",
    "Ballinrobe":       "Connacht",
    "Ballymote":        "Connacht",
    "Carrigallen":      "Connacht",
    "Castlerea":        "Connacht",
    "Drumshanbo":       "Connacht",
    "Elphin":           "Connacht",
    "Headford":         "Connacht",
    "Loughrea":         "Connacht",
    "Mohill":           "Connacht",
    "Portumna":         "Connacht",
    "Roscommon":        "Connacht",
    "Tuam":             "Connacht",
    # Munster (Cork, Kerry, Tipperary, Limerick, Clare, Waterford)
    "Cashel":           "Munster",
    "Corrin":           "Munster",
    "Ennis":            "Munster",
    "Iveragh":          "Munster",
    "Kilfenora":        "Munster",
    "Kilrush":          "Munster",
    "Mid Tipp Mart":    "Munster",
    "Nenagh":           "Munster",
    "Roscrea":          "Munster",
    "Scarriff":         "Munster",
    "Templemore":       "Munster",
    # Leinster (Wicklow, Kilkenny, Offaly, Longford, Westmeath)
    "Baltinglass":      "Leinster",
    "Ballymahon LWFM":  "Leinster",
    "Birr":             "Leinster",
    "Carnew":           "Leinster",
    "Granard":          "Leinster",
    "Kilkenny":         "Leinster",
    "Midland and Western": "Leinster",
    # Ulster (Tyrone, Fermanagh, Down, Donegal)
    "Clogher":          "Ulster",
    "Donegal":          "Ulster",
    "Lisnaskea":        "Ulster",
    "Raphoe":           "Ulster",
    "Rathfriland":      "Ulster",
}

BREED_GROUP_MAP = {
    "LMX": "Continental ×", "LIM": "Continental ×", "LM": "Continental ×",
    "CHX": "Continental ×", "CH":  "Continental ×",
    "SIX": "Continental ×", "SI":  "Continental ×",
    "AUX": "Continental ×", "SAX": "Continental ×", "SHX": "Continental ×",
    "BBX": "Belgian Blue ×", "BB": "Belgian Blue ×",
    "AAX": "Angus ×",        "AA":  "Angus ×",
    "HEX": "Hereford ×",     "HER": "Hereford ×",
    "FR":  "Friesian / Dairy", "FRX": "Friesian / Dairy", "HOL": "Friesian / Dairy",
}

WEIGHT_BINS   = [0,   200,     300,     400,     500,     650,  9999]
WEIGHT_LABELS = ["<200 kg", "200–300 kg", "300–400 kg",
                 "400–500 kg", "500–650 kg", ">650 kg"]

SEX_LABELS = {"M": "Male", "F": "Female", "B": "Bull"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_eur(s):
    if pd.isna(s) or str(s).strip() == "":
        return np.nan
    return pd.to_numeric(str(s).replace("€", "").replace(",", "").strip(),
                         errors="coerce")

def _layout(**kw):
    base = dict(
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(color=NAVY, family="system-ui, -apple-system, sans-serif", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    base.update(kw)
    return base


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data():
    # MartBids (use parquet for faster loading)
    mb = load_data_safe(BASE / "sold_lots.csv", BASE / "sold_lots.parquet")
    mb["source"]    = "MartBids"
    mb["price_num"] = mb["price"].apply(parse_eur)
    mb["weight"]    = pd.to_numeric(mb["weight"], errors="coerce")
    mb["age_months"]= pd.to_numeric(mb["age_months"], errors="coerce")
    mb["sale_date"] = pd.to_datetime(mb["scraped_date"], errors="coerce")

    frames = [mb]

    # Livestock-Live
    lsl_path = BASE / "lsl_lots.csv"
    if lsl_path.exists():
        lsl = pd.read_csv(lsl_path)
        lsl["source"]    = "Livestock-Live"
        lsl["price_num"] = pd.to_numeric(lsl["price"], errors="coerce")
        lsl["weight"]    = pd.to_numeric(lsl["weight"], errors="coerce")
        lsl["age_months"]= pd.to_numeric(lsl["age_months"], errors="coerce")
        lsl["sale_date"] = pd.to_datetime(lsl["sale_date"], errors="coerce")
        lsl["scraped_date"] = lsl["sale_date"].dt.strftime("%Y-%m-%d")
        lsl["dam_breed"] = np.nan
        frames.append(lsl)

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["price_num"] > 0].dropna(subset=["price_num", "weight"]).copy()

    df["price_per_kg"] = df["price_num"] / df["weight"].replace(0, np.nan)
    df = df[(df["price_per_kg"] >= 0.5) & (df["price_per_kg"] <= 20)].copy()
    df = df[df["weight"] >= 50].copy()

    df["sex_clean"]    = df["sex"].map({"M": "Male", "F": "Female", "B": "Bull"}).fillna("Unknown")
    df["breed_group"]  = df["breed"].map(BREED_GROUP_MAP).fillna("Other")
    df["region"]       = df["mart"].map(REGION_MAP).fillna("Other")
    df["weight_band"]  = pd.cut(df["weight"], bins=WEIGHT_BINS,
                                labels=WEIGHT_LABELS, right=False)
    df["iso_week"]     = df["sale_date"].dt.isocalendar().week.astype(int)
    df["year_week"]    = (df["sale_date"].dt.isocalendar().year.astype(str)
                         + "-W" + df["iso_week"].astype(str).str.zfill(2))
    df["week_start"]   = df["sale_date"] - pd.to_timedelta(
                             df["sale_date"].dt.dayofweek, unit="D")
    df["week_start"]   = df["week_start"].dt.normalize()

    return df


@st.cache_data(ttl=3600)
def load_factory_prices():
    """Load factory reference prices (R3 headline steer)."""
    try:
        fp = load_data_safe(BASE / "factory_prices_clean.csv",
                           BASE / "factory_prices_clean.parquet")
        fp = fp[fp["is_headline"] == True].copy()  # Only headline prices
        fp["report_date"] = pd.to_datetime(fp["report_date"], errors="coerce")
        return fp
    except:
        return pd.DataFrame()




def section_kpis(df):
    lots      = len(df)
    avg_ppkg  = df["price_per_kg"].mean()
    avg_lot   = df["price_num"].mean()
    total_val = df["price_num"].sum()

    # Week-on-week change (compare most recent week to prior week)
    weeks = sorted(df["week_start"].dropna().unique())
    wow_delta = None
    if len(weeks) >= 2:
        cur = df[df["week_start"] == weeks[-1]]["price_per_kg"].mean()
        prv = df[df["week_start"] == weeks[-2]]["price_per_kg"].mean()
        wow_delta = cur - prv

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lots Analysed", f"{lots:,}")
    c2.metric("National Avg €/kg", f"€{avg_ppkg:.2f}",
              delta=f"{wow_delta:+.2f} vs prior wk" if wow_delta is not None else None)
    c3.metric("Avg Lot Value", f"€{avg_lot:,.0f}")
    c4.metric("Total Market Value", f"€{total_val/1e6:.2f}M")
    c5.metric("Marts Represented", str(df["mart"].nunique()))


# ── Section: Breed × Weight Cohort Matrix ─────────────────────────────────────

def section_cohort_matrix(df):
    st.subheader("Breed × Weight Cohort — Average €/kg")
    st.caption("Collateral reference matrix. Cells show median €/kg | (n lots). "
               "Min 5 lots per cell.")

    # Filter to meaningful breed groups
    top_groups = (df.groupby("breed_group")["price_per_kg"]
                  .count().sort_values(ascending=False)
                  .head(7).index.tolist())
    sub = df[df["breed_group"].isin(top_groups)].copy()

    pivot_val = (sub.groupby(["breed_group", "weight_band"])["price_per_kg"]
                 .agg(["median", "count"]).reset_index())
    pivot_val.columns = ["breed_group", "weight_band", "median_ppkg", "count"]
    pivot_val = pivot_val[pivot_val["count"] >= 5]

    # Heatmap values
    heat = pivot_val.pivot(index="breed_group", columns="weight_band",
                           values="median_ppkg")
    heat = heat.reindex(columns=[l for l in WEIGHT_LABELS if l in heat.columns])

    # Annotation: "€3.42 (n=47)"
    cnt_piv = pivot_val.pivot(index="breed_group", columns="weight_band",
                              values="count")
    cnt_piv = cnt_piv.reindex(columns=heat.columns)

    text = []
    for breed in heat.index:
        row = []
        for band in heat.columns:
            v = heat.loc[breed, band]
            n = cnt_piv.loc[breed, band] if breed in cnt_piv.index and band in cnt_piv.columns else np.nan
            row.append(f"€{v:.2f}<br>n={int(n)}" if not np.isnan(v) else "")
        text.append(row)

    fig = go.Figure(go.Heatmap(
        z=heat.values,
        x=list(heat.columns),
        y=list(heat.index),
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale=[[0, "#1B2A4A"], [0.5, "#C9A84C"], [1, "#276749"]],
        hoverongaps=False,
        colorbar=dict(title="€/kg", thickness=12),
    ))
    fig.update_layout(
        height=320,
        xaxis=dict(title="Weight Band", side="bottom"),
        yaxis=dict(title=""),
        **_layout(margin=dict(l=10, r=10, t=10, b=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Also show as downloadable table
    dl_rows = []
    for _, row in pivot_val.iterrows():
        # add percentile ranges
        sub2 = sub[(sub["breed_group"] == row["breed_group"]) &
                   (sub["weight_band"] == row["weight_band"])]["price_per_kg"]
        dl_rows.append({
            "Breed Group":       row["breed_group"],
            "Weight Band":       str(row["weight_band"]),
            "Lots":              int(row["count"]),
            "Median €/kg":       round(row["median_ppkg"], 2),
            "P25 €/kg":          round(sub2.quantile(0.25), 2),
            "P75 €/kg":          round(sub2.quantile(0.75), 2),
            "Median Lot Value":  round(sub[(sub["breed_group"] == row["breed_group"]) &
                                          (sub["weight_band"] == row["weight_band"])]["price_num"].median(), 0),
        })
    dl_df = pd.DataFrame(dl_rows).sort_values(["Breed Group","Weight Band"])
    st.download_button(
        "Download Cohort Table (CSV)",
        data=dl_df.to_csv(index=False),
        file_name="cohort_matrix.csv",
        mime="text/csv",
    )


# ── Section: Regional Analysis ────────────────────────────────────────────────

def section_regional(df):
    st.subheader("Regional Price Analysis")

    regions = ["Connacht", "Munster", "Leinster", "Ulster"]
    reg_df = df[df["region"].isin(regions)].copy()

    col1, col2 = st.columns([3, 2])

    with col1:
        # Regional avg €/kg bar with national benchmark line
        reg_agg = (reg_df.groupby("region")["price_per_kg"]
                   .agg(["mean", "count", "median"])
                   .reset_index()
                   .rename(columns={"mean": "avg_ppkg", "count": "lots",
                                    "median": "med_ppkg"}))
        national_avg = df["price_per_kg"].mean()

        fig = go.Figure()
        fig.add_bar(
            x=reg_agg["region"],
            y=reg_agg["avg_ppkg"].round(2),
            marker_color=[GOLD if r == reg_agg.loc[reg_agg["avg_ppkg"].idxmax(), "region"]
                          else NAVY for r in reg_agg["region"]],
            text=[f"€{v:.2f}<br>{int(n)} lots"
                  for v, n in zip(reg_agg["avg_ppkg"], reg_agg["lots"])],
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="%{x}: €%{y:.2f}/kg<extra></extra>",
        )
        fig.add_hline(y=national_avg, line_dash="dash", line_color=RED, line_width=1.5,
                      annotation_text=f"National avg €{national_avg:.2f}",
                      annotation_position="right")
        fig.update_layout(
            title="Average €/kg by Province",
            yaxis=dict(title="€/kg", range=[
                reg_agg["avg_ppkg"].min() * 0.95,
                reg_agg["avg_ppkg"].max() * 1.08,
            ]),
            height=300, showlegend=False,
            **_layout(),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Regional differential table
        st.markdown("**Regional vs National Differential**")
        reg_agg["vs_national"] = reg_agg["avg_ppkg"] - national_avg
        reg_agg["vs_national_%"] = (reg_agg["vs_national"] / national_avg * 100)

        tbl = reg_agg[["region", "avg_ppkg", "vs_national", "vs_national_%", "lots"]].copy()
        tbl.columns = ["Region", "Avg €/kg", "Diff €/kg", "Diff %", "Lots"]
        tbl["Avg €/kg"]  = tbl["Avg €/kg"].apply(lambda x: f"€{x:.2f}")
        tbl["Diff €/kg"] = tbl["Diff €/kg"].apply(lambda x: f"{x:+.2f}")
        tbl["Diff %"]    = tbl["Diff %"].apply(lambda x: f"{x:+.1f}%")
        tbl["Lots"]      = tbl["Lots"].astype(int)
        st.dataframe(tbl.set_index("Region"), use_container_width=True)

    # Breed × region breakdown (which breeds dominate where)
    st.markdown("**Breed Composition by Region**")
    top_groups = (df.groupby("breed_group")["price_per_kg"]
                  .count().sort_values(ascending=False).head(5).index.tolist())
    breed_reg = (reg_df[reg_df["breed_group"].isin(top_groups)]
                 .groupby(["region", "breed_group"])
                 .size().reset_index(name="lots"))
    fig2 = px.bar(breed_reg, x="region", y="lots", color="breed_group",
                  barmode="group", color_discrete_sequence=PALETTE,
                  labels={"lots": "Lots Sold", "region": "", "breed_group": "Breed Group"},
                  height=280)
    fig2.update_layout(**_layout())
    st.plotly_chart(fig2, use_container_width=True)


# ── Section: Rolling Trend ────────────────────────────────────────────────────

def section_trends(df):
    st.subheader("Price Trend — Rolling Averages")

    col1, col2 = st.columns([2, 1])
    with col1:
        granularity = st.radio("View by", ["Week", "Day"], horizontal=True, key="trend_gran")
    with col2:
        trend_region = st.selectbox(
            "Filter region", ["All Ireland"] + ["Connacht", "Munster", "Leinster", "Ulster"],
            key="trend_region"
        )

    tdf = df.copy()
    if trend_region != "All Ireland":
        tdf = tdf[tdf["region"] == trend_region]

    if granularity == "Week":
        grp = (tdf.groupby("week_start")["price_per_kg"]
               .agg(["mean", "count", "median"]).reset_index()
               .rename(columns={"mean": "avg", "count": "lots", "median": "med"}))
        grp = grp.sort_values("week_start")
        x_col = "week_start"
        x_label = "Week"
    else:
        grp = (tdf.groupby(tdf["sale_date"].dt.normalize())["price_per_kg"]
               .agg(["mean", "count", "median"]).reset_index()
               .rename(columns={"sale_date": "day", "mean": "avg", "count": "lots", "median": "med"}))
        grp = grp.sort_values("day")
        x_col = "day"
        x_label = "Date"

    # Rolling average (min 3 periods for weekly, 5 for daily)
    min_p = 3 if granularity == "Week" else 5
    grp["roll_13"] = grp["avg"].rolling(13, min_periods=min_p).mean()
    grp["roll_4"]  = grp["avg"].rolling(4,  min_periods=min_p).mean()

    fig = go.Figure()
    fig.add_scatter(
        x=grp[x_col], y=grp["avg"].round(2),
        mode="markers+lines", name="Avg €/kg",
        line=dict(color=NAVY, width=1.5, dash="dot"),
        marker=dict(size=6),
        hovertemplate=f"{x_label}: %{{x}}<br>Avg: €%{{y:.2f}}/kg<extra></extra>",
    )
    if grp["roll_4"].notna().sum() >= 2:
        fig.add_scatter(
            x=grp[x_col], y=grp["roll_4"].round(2),
            mode="lines", name="4-period MA",
            line=dict(color=GOLD, width=2.5),
            hovertemplate="4-period MA: €%{y:.2f}/kg<extra></extra>",
        )
    if grp["roll_13"].notna().sum() >= 2:
        fig.add_scatter(
            x=grp[x_col], y=grp["roll_13"].round(2),
            mode="lines", name="13-period MA",
            line=dict(color=RED, width=2.5),
            hovertemplate="13-period MA: €%{y:.2f}/kg<extra></extra>",
        )

    fig.update_layout(
        title=f"Average €/kg — {trend_region}",
        yaxis_title="€/kg",
        xaxis_title=x_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320,
        **_layout(margin=dict(l=10, r=10, t=50, b=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(grp) < 13:
        weeks_needed = 13 - len(grp)
        st.info(f"13-period rolling average will be fully populated after "
                f"~{weeks_needed} more {'weeks' if granularity=='Week' else 'days'} of data. "
                f"Currently showing {len(grp)} {granularity.lower()}(s).")

    # Volume bar below
    fig2 = go.Figure(go.Bar(
        x=grp[x_col], y=grp["lots"],
        marker_color=NAVY, opacity=0.6,
        hovertemplate=f"{x_label}: %{{x}}<br>Lots: %{{y}}<extra></extra>",
    ))
    fig2.update_layout(title=f"Lots Sold per {granularity}", yaxis_title="Lots",
                       height=180, **_layout(margin=dict(l=10, r=10, t=40, b=10)))
    st.plotly_chart(fig2, use_container_width=True)


# ── Section: Mart-Level Comparison ────────────────────────────────────────────

def section_mart_comparison(df):
    st.subheader("Mart-Level Cross-Comparison")

    weeks = sorted(df["year_week"].dropna().unique(), reverse=True)
    if not weeks:
        st.info("No weekly data available.")
        return

    selected_week = st.selectbox("Select week", weeks, key="mart_week")
    wdf = df[df["year_week"] == selected_week]

    col1, col2 = st.columns([1, 1])
    with col1:
        weight_filter = st.selectbox(
            "Filter by weight band", ["All"] + WEIGHT_LABELS, key="mart_wt")
    with col2:
        breed_filter = st.selectbox(
            "Filter by breed group",
            ["All"] + sorted(df["breed_group"].unique().tolist()),
            key="mart_breed"
        )

    if weight_filter != "All":
        wdf = wdf[wdf["weight_band"].astype(str) == weight_filter]
    if breed_filter != "All":
        wdf = wdf[wdf["breed_group"] == breed_filter]

    mart_agg = (wdf.groupby("mart")
                .agg(avg_ppkg=("price_per_kg", "mean"),
                     med_ppkg=("price_per_kg", "median"),
                     lots=("price_per_kg", "count"),
                     total_val=("price_num", "sum"))
                .reset_index()
                .query("lots >= 3")
                .sort_values("avg_ppkg", ascending=True))

    if mart_agg.empty:
        st.info("Not enough data for the selected filters.")
        return

    nat_avg = wdf["price_per_kg"].mean()

    fig = go.Figure(go.Bar(
        y=mart_agg["mart"],
        x=mart_agg["avg_ppkg"].round(2),
        orientation="h",
        marker_color=[GOLD if v >= nat_avg else NAVY for v in mart_agg["avg_ppkg"]],
        text=[f"€{v:.2f} ({int(n)} lots)" for v, n in zip(mart_agg["avg_ppkg"], mart_agg["lots"])],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{y}: €%{x:.2f}/kg<extra></extra>",
    ))
    fig.add_vline(x=nat_avg, line_dash="dash", line_color=RED, line_width=1.5,
                  annotation_text=f"Wk avg €{nat_avg:.2f}",
                  annotation_position="top right")
    fig.update_layout(
        title=f"Avg €/kg by Mart — {selected_week}",
        xaxis_title="€/kg",
        height=max(350, len(mart_agg) * 28),
        **_layout(margin=dict(l=10, r=80, t=50, b=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Spread table
    st.markdown("**Mart Summary Table**")
    mart_agg["region"] = mart_agg["mart"].map(REGION_MAP).fillna("Other")
    mart_agg["vs_nat"] = mart_agg["avg_ppkg"] - nat_avg
    tbl = mart_agg[["mart", "region", "lots", "avg_ppkg", "med_ppkg",
                     "vs_nat", "total_val"]].copy()
    tbl.columns = ["Mart", "Region", "Lots", "Avg €/kg", "Med €/kg",
                   "vs Week Avg", "Total Value €"]
    tbl["Avg €/kg"]      = tbl["Avg €/kg"].apply(lambda x: f"€{x:.2f}")
    tbl["Med €/kg"]      = tbl["Med €/kg"].apply(lambda x: f"€{x:.2f}")
    tbl["vs Week Avg"]   = tbl["vs Week Avg"].apply(lambda x: f"{x:+.2f}")
    tbl["Total Value €"] = tbl["Total Value €"].apply(lambda x: f"€{x:,.0f}")
    tbl = tbl.sort_values("Mart")
    st.dataframe(tbl.set_index("Mart"), use_container_width=True)


# ── Section: Collateral Reference Table ───────────────────────────────────────

def section_collateral_ref(df):
    st.subheader("Collateral Value Reference")
    st.caption("Expected market value ranges by breed group and weight band. "
               "Use P25–P75 range for loan-to-value stress testing.")

    # Build reference table
    top_groups = (df.groupby("breed_group")["price_per_kg"]
                  .count().sort_values(ascending=False)
                  .head(7).index.tolist())
    sub = df[df["breed_group"].isin(top_groups)].copy()

    rows = []
    for bg in top_groups:
        for wb in WEIGHT_LABELS:
            cell = sub[(sub["breed_group"] == bg) & (sub["weight_band"].astype(str) == wb)]
            if len(cell) < 5:
                continue
            mid_wt = cell["weight"].median()
            rows.append({
                "Breed Group":      bg,
                "Weight Band":      wb,
                "Lots":             len(cell),
                "Med Weight (kg)":  round(mid_wt, 0),
                "P25 €/kg":         round(cell["price_per_kg"].quantile(0.25), 2),
                "Med €/kg":         round(cell["price_per_kg"].median(), 2),
                "P75 €/kg":         round(cell["price_per_kg"].quantile(0.75), 2),
                "P25 Lot Value":    round(cell["price_num"].quantile(0.25), 0),
                "Med Lot Value":    round(cell["price_num"].median(), 0),
                "P75 Lot Value":    round(cell["price_num"].quantile(0.75), 0),
            })

    ref_df = pd.DataFrame(rows)
    if ref_df.empty:
        st.info("Insufficient data for collateral reference table.")
        return

    # Format for display
    disp = ref_df.copy()
    for col in ["P25 Lot Value", "Med Lot Value", "P75 Lot Value"]:
        disp[col] = disp[col].apply(lambda x: f"€{x:,.0f}")

    st.dataframe(disp.set_index(["Breed Group", "Weight Band"]), use_container_width=True)

    st.download_button(
        "Download Reference Table (CSV)",
        data=ref_df.to_csv(index=False),
        file_name="collateral_reference.csv",
        mime="text/csv",
        key="dl_ref",
    )


# ── Section: Sex & Age Breakdown ──────────────────────────────────────────────

def section_sex_age(df):
    st.subheader("Sex & Age Profile")

    col1, col2 = st.columns(2)

    with col1:
        sex_agg = (df.groupby("sex_clean")
                   .agg(avg_ppkg=("price_per_kg", "mean"),
                        lots=("price_per_kg", "count"))
                   .reset_index().query("lots >= 5"))
        fig = go.Figure(go.Bar(
            x=sex_agg["sex_clean"], y=sex_agg["avg_ppkg"].round(2),
            marker_color=[GOLD, NAVY, DGREY][:len(sex_agg)],
            text=[f"€{v:.2f}<br>n={int(n)}" for v, n in zip(sex_agg["avg_ppkg"], sex_agg["lots"])],
            textposition="outside",
        ))
        fig.update_layout(title="Avg €/kg by Sex", height=280,
                          yaxis_title="€/kg", **_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Age bands
        df2 = df.dropna(subset=["age_months"]).copy()
        df2["age_band"] = pd.cut(df2["age_months"],
                                 bins=[0, 6, 12, 18, 24, 36, 999],
                                 labels=["0–6m", "6–12m", "12–18m", "18–24m", "24–36m", "36m+"])
        age_agg = (df2.groupby("age_band")
                   .agg(avg_ppkg=("price_per_kg", "mean"),
                        lots=("price_per_kg", "count"))
                   .reset_index().query("lots >= 5"))
        fig2 = go.Figure(go.Bar(
            x=age_agg["age_band"].astype(str),
            y=age_agg["avg_ppkg"].round(2),
            marker_color=NAVY,
            text=[f"€{v:.2f}<br>n={int(n)}" for v, n in zip(age_agg["avg_ppkg"], age_agg["lots"])],
            textposition="outside",
        ))
        fig2.update_layout(title="Avg €/kg by Age Band", height=280,
                           yaxis_title="€/kg", **_layout())
        st.plotly_chart(fig2, use_container_width=True)


# ── Section: Factory Reference Prices ─────────────────────────────────────────

def section_factory_reference(fp):
    """Factory reference pricing for collateral comparison."""
    st.subheader("Factory Reference Prices (R3 Headline Steer)")
    st.caption("National benchmark factory prices. Use to validate mar pricing.")

    if fp.empty:
        st.info("Factory price data not available.")
        return

    # Latest week prices by factory
    latest_week = fp["report_date"].max()
    fp_week = fp[fp["report_date"] == latest_week].copy()
    fp_week["price_euro_per_kg"] = pd.to_numeric(fp_week["price_euro_per_kg"], errors="coerce")

    # Factory average
    factories = (fp_week.groupby("factory")["price_euro_per_kg"]
                 .mean().reset_index()
                 .sort_values("price_euro_per_kg", ascending=False))

    if factories.empty:
        st.info("No factory prices for the latest week.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Factory comparison bar
        nat_avg = factories["price_euro_per_kg"].mean()
        fig = go.Figure(go.Bar(
            x=factories["factory"],
            y=factories["price_euro_per_kg"].round(2),
            marker_color=[GOLD if v >= nat_avg else NAVY for v in factories["price_euro_per_kg"]],
            text=[f"€{v:.2f}" for v in factories["price_euro_per_kg"]],
            textposition="outside",
        ))
        fig.add_hline(y=nat_avg, line_dash="dash", line_color=RED,
                      annotation_text=f"Avg €{nat_avg:.2f}")
        fig.update_layout(title=f"Factory €/kg — Week of {latest_week.strftime('%d %b')}",
                         yaxis_title="€/kg", height=280, **_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("National Factory Avg", f"€{nat_avg:.2f}")
        st.metric("High", f"€{factories['price_euro_per_kg'].max():.2f}")
    with col3:
        st.metric("Low", f"€{factories['price_euro_per_kg'].min():.2f}")
        st.metric("Spread", f"€{factories['price_euro_per_kg'].max() - factories['price_euro_per_kg'].min():.2f}")

    # 12-week trend
    fp_trend = (fp[fp["factory"] == "National"]
                .sort_values("report_date")
                .tail(12).copy())
    if not fp_trend.empty:
        fp_trend["price_euro_per_kg"] = pd.to_numeric(fp_trend["price_euro_per_kg"], errors="coerce")
        fig2 = go.Figure(go.Scatter(
            x=fp_trend["report_date"],
            y=fp_trend["price_euro_per_kg"].round(2),
            mode="lines+markers", name="National Avg",
            line=dict(color=GOLD, width=3),
            marker=dict(size=8),
        ))
        fig2.update_layout(title="National Factory Price — 12-Week Trend",
                          xaxis_title="Week", yaxis_title="€/kg",
                          height=280, **_layout())
        st.plotly_chart(fig2, use_container_width=True)



def main():
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:20px 0 10px 0">
            <div style="font-size:1.25rem;font-weight:800;color:{GOLD};
                        letter-spacing:-0.3px;">MartBids</div>
            <div style="font-size:0.8rem;color:#9BA8C0;margin-top:2px;">
                Lending Intelligence Platform
            </div>
        </div>
        <hr style="border-color:#2C3E60;margin:0 0 16px 0"/>
        """, unsafe_allow_html=True)

        df_raw = load_data()

        # Date range
        min_d = df_raw["sale_date"].min().date()
        max_d = df_raw["sale_date"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)

        # Region
        all_regions = ["All Ireland", "Connacht", "Munster", "Leinster", "Ulster"]
        region_sel = st.selectbox("Province", all_regions)

        # Sex
        sex_sel = st.multiselect("Sex", ["Male", "Female", "Bull"],
                                 default=["Male", "Female", "Bull"])

        # Min weight
        min_wt = st.slider("Min weight (kg)", 0, 400, 0, step=50)

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:0.75rem;color:#6B7A99;line-height:1.6">
            <b style="color:{GOLD}">Data sources</b><br>
            MartBids.ie — {df_raw[df_raw['source']=='MartBids'].shape[0]:,} lots<br>
            Livestock-Live.com — {df_raw[df_raw['source']=='Livestock-Live'].shape[0]:,} lots<br>
            Factory (DAFM/BPW) — daily prices<br><br>
            <b style="color:{GOLD}">Updated</b><br>
            {df_raw['sale_date'].max().strftime('%d %b %Y')}
        </div>
        """, unsafe_allow_html=True)

    # ── Apply filters ─────────────────────────────────────────────────────────
    df = df_raw.copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        df = df[(df["sale_date"].dt.date >= date_range[0]) &
                (df["sale_date"].dt.date <= date_range[1])]
    if region_sel != "All Ireland":
        df = df[df["region"] == region_sel]
    if sex_sel:
        df = df[df["sex_clean"].isin(sex_sel)]
    if min_wt > 0:
        df = df[df["weight"] >= min_wt]

    if df.empty:
        st.warning("No data for the selected filters.")
        return

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">
        <h1 style="margin:0">Irish Cattle Mart Intelligence</h1>
        <span style="font-size:0.85rem;color:{DGREY};font-weight:500">
            Agricultural Lending Reference — {df['sale_date'].max().strftime('%d %b %Y')}
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Breed-level pricing, regional differentials, and collateral reference data for agricultural lenders.")
    st.divider()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    section_kpis(df)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    fp = load_factory_prices()
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Cohort Matrix", "Regional Analysis", "Price Trends",
        "Mart Comparison", "Collateral Reference", "Factory Reference"
    ])

    with tab1:
        section_cohort_matrix(df)
        st.divider()
        section_sex_age(df)

    with tab2:
        section_regional(df)

    with tab3:
        section_trends(df)

    with tab4:
        section_mart_comparison(df)

    with tab5:
        section_collateral_ref(df)

    with tab6:
        section_factory_reference(fp)


if __name__ == "__main__":
    main()
