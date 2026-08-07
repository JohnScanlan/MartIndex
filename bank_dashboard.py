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
from mart_coords import LSL_MART_REGIONS
import agri_credit as ac
import herd_store as hs
from viz_theme import layout as vt_layout
# Cohort definitions live in agri_credit so the valuation tools and the cohort
# matrix band animals identically — if these drifted apart the same animal
# would be worth two different amounts depending on which tab you opened.
from agri_credit import WEIGHT_BINS, WEIGHT_LABELS

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

# ── Chart colours ─────────────────────────────────────────────────────────────
# Defined in viz_theme so this dashboard and dashboard.py cannot drift apart.
# See that module for why each value is what it is.
from viz_theme import (GOLD_MARK, INK, INK_SOFT, PALETTE, SEQ_NAVY,
                       TEXT_FLIP as SEQ_TEXT_FLIP)

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
    /* The label sits inside a <p>, which the global `p {{ color: DGREY
       !important }}` rule above would otherwise win — leaving the selected
       tab navy-on-navy and unreadable. */
    .stTabs [aria-selected="true"] p {{ color: {WHITE} !important; }}
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
    .stDownloadButton > button, .stButton > button {{
        background: {NAVY} !important; color: {WHITE} !important;
        border: none !important; border-radius: 6px;
        font-weight: 600;
    }}
    /* The button label is a nested <p>/<div>, which the global
       `p, li {{ color: DGREY !important }}` rule above would otherwise win —
       leaving dark-grey text on a navy button at 1.9:1. */
    .stDownloadButton > button *, .stButton > button * {{
        color: {WHITE} !important;
    }}

    /* Streamlit's default multiselect chip is salmon (#FF4B4B) with pale text —
       2.75:1, and it clashes with the navy/gold scheme. */
    span[data-baseweb="tag"] {{
        background-color: {NAVY_LT} !important;
        border: 1px solid {GOLD} !important;
    }}
    span[data-baseweb="tag"] span, span[data-baseweb="tag"] div {{
        color: {WHITE} !important;
    }}
    /* Sidebar date-range input renders its value as placeholder-weight text,
       which lands near 2:1 on the navy ground. */
    section[data-testid="stSidebar"] input {{
        color: #F2F4F8 !important; -webkit-text-fill-color: #F2F4F8 !important;
    }}
    section[data-testid="stSidebar"] input::placeholder {{
        color: #B9C4D8 !important; opacity: 1 !important;
    }}
    /* Widget accents come from .streamlit/config.toml (primaryColor = gold).
       The slider's value badge paints text over that gold, so it needs dark
       ink rather than the default near-white (2.75:1). */
    /* The badge paints on the gold accent, and the sidebar's blanket
       `* {{ color: #E8EAF0 }}` rule would otherwise leave it at 1.9:1.
       Navy on gold is 6.2:1. */
    /* The value sits in a nested <p>, so the selector has to reach it AND
       outrank `section[data-testid="stSidebar"] *` — both carry !important,
       so specificity decides. */
    section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] *,
    [data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * {{
        color: {NAVY} !important; font-weight: 700;
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
    # Livestock-Live marts (their own naming — see mart_coords.py)
    **LSL_MART_REGIONS,
}

# WEIGHT_BINS / WEIGHT_LABELS / SEX_LABELS are imported
# from agri_credit above — single source of truth for cohort definitions.


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_eur(s):
    if pd.isna(s) or str(s).strip() == "":
        return np.nan
    return pd.to_numeric(str(s).replace("€", "").replace(",", "").strip(),
                         errors="coerce")

def _layout(**kw):
    """Shared Plotly layout. Styling comes from viz_theme's template."""
    return vt_layout(**kw)


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
    df = df[(df["weight"] >= 50) & (df["weight"] <= 1_250)].copy()

    df["sex_clean"]    = df["sex"].map({"M": "Male", "F": "Female", "B": "Bull"}).fillna("Unknown")
    # Raw mart codes (LMX, AAX, CHX …), the same vocabulary the model uses, so a
    # cohort price and a model prediction refer to the same population.
    df["breed_group"]  = ac.to_breed_group(df["breed"], ac.breed_levels(df))
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

    fig = go.Figure(go.Heatmap(
        z=heat.values,
        x=list(heat.columns),
        y=list(heat.index),
        colorscale=[[i / (len(SEQ_NAVY) - 1), c] for i, c in enumerate(SEQ_NAVY)],
        hoverongaps=False,
        hovertemplate="%{y} · %{x}<br>€%{z:.2f}/kg<extra></extra>",
        colorbar=dict(title="€/kg", thickness=12,
                      tickfont=dict(color=INK_SOFT, size=11)),
    ))

    # Cell labels are annotations rather than Heatmap text, because Plotly only
    # accepts a single textfont colour for the whole trace — which is what put
    # white text on the pale cells at 1:1 before. Each label picks its own ink
    # from where its cell sits on the ramp.
    zmin, zmax = np.nanmin(heat.values), np.nanmax(heat.values)
    span = (zmax - zmin) or 1
    for breed in heat.index:
        for band in heat.columns:
            v = heat.loc[breed, band]
            if np.isnan(v):
                continue
            n = cnt_piv.loc[breed, band] if breed in cnt_piv.index and band in cnt_piv.columns else np.nan
            frac = (v - zmin) / span
            fig.add_annotation(
                x=band, y=breed, showarrow=False,
                text=f"<b>€{v:.2f}</b><br>n={int(n)}" if not np.isnan(n) else f"<b>€{v:.2f}</b>",
                font=dict(size=11,
                          color=WHITE if frac >= SEQ_TEXT_FLIP else INK),
            )

    fig.update_layout(
        height=380,
        **_layout(margin=dict(l=10, r=10, t=10, b=40),
                  xaxis=dict(title="Weight Band", side="bottom"),
                  yaxis=dict(title="")),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # Also show as downloadable table
    dl_rows = []
    for _, row in pivot_val.iterrows():
        # add percentile ranges
        sub2 = sub[(sub["breed_group"] == row["breed_group"]) &
                   (sub["weight_band"] == row["weight_band"])]["price_per_kg"]
        dl_rows.append({
            "Breed":             row["breed_group"],
            "Weight Band":       str(row["weight_band"]),
            "Lots":              int(row["count"]),
            "Median €/kg":       round(row["median_ppkg"], 2),
            "P25 €/kg":          round(sub2.quantile(0.25), 2),
            "P75 €/kg":          round(sub2.quantile(0.75), 2),
            "Median Lot Value":  round(sub[(sub["breed_group"] == row["breed_group"]) &
                                          (sub["weight_band"] == row["weight_band"])]["price_num"].median(), 0),
        })
    dl_df = pd.DataFrame(dl_rows).sort_values(["Breed","Weight Band"])
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
            marker_color=[GOLD_MARK if r == reg_agg.loc[reg_agg["avg_ppkg"].idxmax(), "region"]
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
            height=300, showlegend=False,
            **_layout(yaxis=dict(title="€/kg", range=[
                reg_agg["avg_ppkg"].min() * 0.95,
                reg_agg["avg_ppkg"].max() * 1.08,
            ])),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

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
                  labels={"lots": "Lots Sold", "region": "", "breed_group": "Breed"},
                  height=280)
    fig2.update_layout(**_layout())
    st.plotly_chart(fig2, use_container_width=True, theme=None)


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
            line=dict(color=GOLD_MARK, width=2.5),
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
    st.plotly_chart(fig, use_container_width=True, theme=None)

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
    st.plotly_chart(fig2, use_container_width=True, theme=None)


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
        marker_color=[GOLD_MARK if v >= nat_avg else NAVY for v in mart_agg["avg_ppkg"]],
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
    st.plotly_chart(fig, use_container_width=True, theme=None)

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
                "Breed":            bg,
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

    st.dataframe(disp.set_index(["Breed", "Weight Band"]), use_container_width=True)

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
            marker_color=[GOLD_MARK, NAVY, INK_SOFT][:len(sex_agg)],
            text=[f"€{v:.2f}<br>n={int(n)}" for v, n in zip(sex_agg["avg_ppkg"], sex_agg["lots"])],
            textposition="outside",
        ))
        fig.update_layout(title="Avg €/kg by Sex", height=280,
                          yaxis_title="€/kg", **_layout())
        st.plotly_chart(fig, use_container_width=True, theme=None)

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
        st.plotly_chart(fig2, use_container_width=True, theme=None)


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
            marker_color=[GOLD_MARK if v >= nat_avg else NAVY for v in factories["price_euro_per_kg"]],
            text=[f"€{v:.2f}" for v in factories["price_euro_per_kg"]],
            textposition="outside",
        ))
        fig.add_hline(y=nat_avg, line_dash="dash", line_color=RED,
                      annotation_text=f"Avg €{nat_avg:.2f}")
        # Factory names are long ("Euro Farm Foods, Duleek") and the last tick
        # label overruns the plot's right edge on the default r=10 margin.
        fig.update_layout(title=f"Factory €/kg — Week of {latest_week.strftime('%d %b')}",
                          yaxis_title="€/kg", height=300,
                          **_layout(margin=dict(l=10, r=45, t=40, b=10)))
        st.plotly_chart(fig, use_container_width=True, theme=None)

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
            line=dict(color=GOLD_MARK, width=3),
            marker=dict(size=8),
        ))
        fig2.update_layout(title="National Factory Price — 12-Week Trend",
                          xaxis_title="Week", yaxis_title="€/kg",
                          height=280, **_layout())
        st.plotly_chart(fig2, use_container_width=True, theme=None)



# ══════════════════════════════════════════════════════════════════════════════
# Credit tools — herd valuation and deal underwriting
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def _breed_choices():
    """Breed vocabulary, taken from the trained model so the two agree."""
    return ac.breed_levels(load_data())


BREED_CHOICES = _breed_choices()
SEX_CHOICES   = ["Male", "Female", "Bull"]
REGION_CHOICES = ["All Ireland", "Connacht", "Munster", "Leinster", "Ulster"]


@st.cache_resource
def _growth_params(n_rows: int):
    """Fitted once per dataset size — refits when new lots land."""
    return ac.fit_growth_curves(load_data())


def _confidence_chip(cp) -> str:
    colour = {"High": GREEN, "Medium": "#B0761F", "Low": RED, "Very low": RED}
    c = colour.get(cp.confidence, DGREY)
    return (f"<span style='background:{c}18;color:{c};padding:2px 8px;"
            f"border-radius:3px;font-size:0.72rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.05em'>{cp.confidence}</span>")


def section_herd_valuation(df_all):
    """Mark a customer's herd to market and track it against their loan."""
    st.subheader("Herd Valuation & LTV Tracker")
    st.caption("Mark-to-market a customer's herd against the same cohort pricing "
               "used across this dashboard. LTV is calculated on the P25 "
               "(conservative) valuation, not the median.")

    existing = hs.herd_ids()
    c1, c2 = st.columns([1, 3])
    choice = c1.selectbox("Herd", ["+ New herd"] + existing, key="hv_pick")

    saved = hs.load_herds()
    is_new = choice == "+ New herd"
    rec = saved[saved["herd_id"] == choice] if not is_new else pd.DataFrame()

    # ── Herd header ──────────────────────────────────────────────────────────
    d1, d2, d3 = st.columns(3)
    herd_id  = d1.text_input("Herd ID", value="" if is_new else choice,
                             placeholder="e.g. OMALLEY-01", key="hv_id")
    customer = d1.text_input("Customer", value="" if rec.empty else rec.iloc[0]["customer"],
                             key="hv_cust")
    loan_ref = d2.text_input("Loan reference", value="" if rec.empty else rec.iloc[0]["loan_ref"],
                             key="hv_ref")
    region   = d2.selectbox("Region", REGION_CHOICES,
                            index=0 if rec.empty else
                            (REGION_CHOICES.index(rec.iloc[0]["region"])
                             if rec.iloc[0]["region"] in REGION_CHOICES else 0),
                            key="hv_region")
    loan_bal = d3.number_input("Loan balance (€)", 0.0, 5_000_000.0,
                               value=0.0 if rec.empty else float(rec.iloc[0]["loan_balance"]),
                               step=1000.0, key="hv_bal")
    max_ltv  = d3.number_input("Max LTV covenant (%)", 10.0, 100.0,
                               value=70.0 if rec.empty else float(rec.iloc[0]["max_ltv_pct"]),
                               step=5.0, key="hv_ltv")

    # ── Herd composition ─────────────────────────────────────────────────────
    st.markdown("**Herd composition** — one row per group of similar animals")
    if rec.empty:
        seed = pd.DataFrame([{"breed_group": "LMX", "sex": "Male",
                              "head": 20, "avg_weight_kg": 420.0}])
    else:
        seed = rec[["breed_group", "sex", "head", "avg_weight_kg"]].reset_index(drop=True)

    lines_df = st.data_editor(
        seed, num_rows="dynamic", use_container_width=True, key="hv_lines",
        column_config={
            "breed_group":   st.column_config.SelectboxColumn("Breed group", options=BREED_CHOICES, required=True),
            "sex":           st.column_config.SelectboxColumn("Sex", options=SEX_CHOICES, required=True),
            "head":          st.column_config.NumberColumn("Head", min_value=1, step=1, required=True),
            "avg_weight_kg": st.column_config.NumberColumn("Avg weight (kg)", min_value=50.0,
                                                           max_value=1250.0, step=10.0, format="%.0f"),
        })

    lines = [ac.HerdLine(r["breed_group"], r["sex"], int(r["head"]), float(r["avg_weight_kg"]))
             for _, r in lines_df.dropna(subset=["head", "avg_weight_kg"]).iterrows()]
    if not lines:
        st.info("Add at least one line to value this herd.")
        return

    hv = ac.value_herd(df_all, lines, region=None if region == "All Ireland" else region)

    # ── Actions ──────────────────────────────────────────────────────────────
    a1, a2, a3, _ = st.columns([1, 1, 1, 3])
    if a1.button("Save herd", use_container_width=True):
        if not herd_id.strip():
            st.error("Give the herd an ID before saving.")
        else:
            hs.save_herd(herd_id.strip(), customer, loan_ref,
                         lines_df.to_dict("records"), loan_bal, max_ltv, region)
            st.success(f"Saved {herd_id}.")
    if a2.button("Record valuation", use_container_width=True,
                 help="Appends today's valuation to the tracking history"):
        if not herd_id.strip():
            st.error("Give the herd an ID first.")
        else:
            n = hs.record_valuation(herd_id.strip(), customer, loan_ref, hv, loan_bal, max_ltv)
            st.success("Valuation recorded." if n else "Already recorded for this date.")
    if not is_new and a3.button("Delete herd", use_container_width=True):
        hs.delete_herd(choice)
        st.warning(f"{choice} marked deleted. History retained.")

    st.divider()

    # ── Valuation ────────────────────────────────────────────────────────────
    ltv      = hv.ltv(loan_bal) if loan_bal else None
    headroom = hv.headroom(loan_bal, max_ltv) if loan_bal else None

    k = st.columns(5)
    k[0].metric("Head", f"{hv.total_head:,}")
    k[1].metric("Total liveweight", f"{hv.total_kg:,.0f} kg")
    k[2].metric("Value — P25", f"€{hv.value_p25:,.0f}",
                help="Conservative / forced-sale estimate. LTV is based on this.")
    k[3].metric("Value — median", f"€{hv.value_median:,.0f}",
                help=f"Fair value. P75 upside €{hv.value_p75:,.0f}")
    if ltv is not None:
        k[4].metric("LTV (on P25)", f"{ltv:.1f}%",
                    delta=f"{ltv - max_ltv:+.1f} pts vs covenant",
                    delta_color="inverse")
    else:
        k[4].metric("LTV (on P25)", "—", help="Enter a loan balance")

    if ltv is not None:
        if ltv > max_ltv:
            st.error(f"**Covenant breach** — LTV {ltv:.1f}% is above the {max_ltv:.0f}% limit. "
                     f"Security shortfall €{abs(headroom):,.0f}.")
        elif ltv > max_ltv * 0.9:
            st.warning(f"**Close to covenant** — LTV {ltv:.1f}% against a {max_ltv:.0f}% limit. "
                       f"Headroom €{headroom:,.0f}.")
        else:
            st.success(f"Within covenant — LTV {ltv:.1f}%, headroom €{headroom:,.0f}.")

    # ── Depreciation warning ─────────────────────────────────────────────────
    drift = hv.drift_per_month_eur
    if drift is not None:
        mtb = hv.months_to_breach(loan_bal, max_ltv) if loan_bal else None
        pct = drift / hv.value_median * 100 if hv.value_median else 0
        if drift < 0:
            msg = (f"**Collateral is depreciating at €{abs(drift):,.0f}/month** "
                   f"({pct:.1f}%/mo) at current cohort momentum.")
            if mtb is not None:
                msg += f" At this rate the covenant breaks in **{mtb:.1f} months**."
            (st.error if (mtb is not None and mtb < 6) else st.warning)(msg)
        else:
            st.info(f"Cohort momentum is positive: +€{drift:,.0f}/month.")

    # ── Per-line detail ──────────────────────────────────────────────────────
    st.markdown("**Line detail**")
    rows = []
    for lv in hv.lines:
        rows.append({
            "Breed group": lv.line.breed_group, "Sex": lv.line.sex,
            "Head": lv.line.head, "Avg kg": f"{lv.line.avg_weight:.0f}",
            "Band": lv.price.band,
            "€/kg P25": f"{lv.price.ppkg_p25:.2f}",
            "€/kg med": f"{lv.price.ppkg_median:.2f}",
            "Value P25": f"€{lv.value_p25:,.0f}",
            "Value med": f"€{lv.value_median:,.0f}",
            "Drift €/kg/mo": f"{lv.price.drift_per_month:+.3f}" if lv.price.drift_per_month is not None else "—",
            "Comparables": f"{lv.price.n:,}",
            "Basis": lv.price.basis,
            "Confidence": lv.price.confidence,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"Priced from sales to {hv.as_of}. Weakest basis used across lines: "
               f"**{ac.FALLBACK_LABELS.get(hv.weakest_basis, hv.weakest_basis)}**.")

    # ── History ──────────────────────────────────────────────────────────────
    if herd_id.strip():
        hist = hs.load_history(herd_id.strip())
        if len(hist) > 1:
            st.divider()
            st.markdown("**Valuation history**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["valuation_date"], y=hist["value_p75"],
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist["valuation_date"], y=hist["value_p25"],
                                     fill="tonexty", fillcolor="rgba(201,168,76,0.18)",
                                     line=dict(width=0), name="P25–P75 range"))
            fig.add_trace(go.Scatter(x=hist["valuation_date"], y=hist["value_median"],
                                     line=dict(color=NAVY, width=2.5), name="Herd value (median)"))
            fig.add_trace(go.Scatter(x=hist["valuation_date"], y=hist["loan_balance"],
                                     line=dict(color=RED, width=2, dash="dash"), name="Loan balance"))
            fig.update_layout(height=340, yaxis_title="€", **_layout())
            st.plotly_chart(fig, use_container_width=True, theme=None)
        elif len(hist) == 1:
            st.caption("One valuation recorded. Record again on a later date to build a trend.")


def section_underwrite(df_all):
    """Underwrite a proposed store-to-finish purchase."""
    st.subheader("Quick Underwrite — proposed purchase")
    st.caption("Projects growth, prices the exit in its destination cohort, and "
               "nets off keep, mortality and finance costs.")

    growth = _growth_params(len(df_all))

    st.markdown("**The purchase**")
    p = st.columns(6)
    breed = p[0].selectbox("Breed group", BREED_CHOICES,
                           index=BREED_CHOICES.index("LMX") if "LMX" in BREED_CHOICES else 0,
                           key="uw_breed")
    sex   = p[1].selectbox("Sex", SEX_CHOICES, key="uw_sex")
    head  = p[2].number_input("Head", 1, 2000, 40, step=5, key="uw_head")
    bwt   = p[3].number_input("Weight (kg)", 50.0, 900.0, 320.0, step=10.0, key="uw_wt")
    bage  = p[4].number_input("Age (months)", 1.0, 48.0, 9.0, step=1.0, key="uw_age")
    bppkg = p[5].number_input("Price €/kg", 0.0, 15.0, 0.0, step=0.05, key="uw_ppkg",
                              help="Leave at 0.00 to use the current market median")

    st.markdown("**The plan**")
    q = st.columns(6)
    months = q[0].number_input("Finish (months)", 1.0, 24.0, 8.0, step=1.0, key="uw_months")
    adg_on = q[1].checkbox("Set daily gain", value=False, key="uw_adg_on",
                           help="Override the fitted growth curve with a known ADG")
    b_fit  = ac.growth_for(growth, breed, sex)
    fitted_adg = (ac.project_weight(bwt, bage, months, b_fit) - bwt) / (months * 30.44)
    adg = q[2].number_input("kg/head/day", 0.1, 2.5, round(float(fitted_adg), 2),
                            step=0.05, key="uw_adg", disabled=not adg_on)
    region = q[3].selectbox("Region", REGION_CHOICES, key="uw_region")
    feed   = q[4].number_input("Keep €/hd/day", 0.0, 8.0, 1.60, step=0.10, key="uw_feed",
                               help="Blended. At grass ≈ €1.20, indoor on silage + meal ≈ €2.20")
    other  = q[5].number_input("Other €/head", 0.0, 500.0, 90.0, step=10.0, key="uw_other",
                               help="Vet, dosing, transport, mart commission, levies")

    st.markdown("**Finance**")
    f = st.columns(6)
    loan  = f[0].number_input("Loan (€)", 0.0, 2_000_000.0, 45_000.0, step=1000.0, key="uw_loan")
    rate  = f[1].number_input("Rate (%)", 0.0, 25.0, 7.5, step=0.25, key="uw_rate")
    mort  = f[2].number_input("Mortality (%)", 0.0, 10.0, 1.5, step=0.5, key="uw_mort")

    deal = ac.DealInputs(
        breed_group=breed, sex=sex, head=int(head),
        buy_weight=bwt, buy_age=bage, finish_months=months,
        daily_gain_override=adg if adg_on else None,
        buy_ppkg=bppkg if bppkg > 0 else None,
        feed_per_head_day=feed, other_cost_per_head=other, mortality_pct=mort,
        loan_amount=loan, interest_rate=rate,
        region=None if region == "All Ireland" else region,
    )
    r = ac.underwrite(df_all, growth, deal)

    st.divider()

    # ── Verdict ──────────────────────────────────────────────────────────────
    if r.viable:
        st.success(f"**Viable** — net margin €{r.net_margin_total:,.0f} "
                   f"(€{r.net_margin_per_head:,.0f}/head), {r.roi_pct:.1f}% ROI over "
                   f"{months:.0f} months. Price can fall {r.margin_of_safety_pct:.1f}% "
                   f"before break-even.")
    else:
        st.error(f"**Does not wash its face** — net loss €{abs(r.net_margin_total):,.0f} "
                 f"(€{abs(r.net_margin_per_head):,.0f}/head). Sale price would need to reach "
                 f"€{r.breakeven_ppkg:.2f}/kg to break even, against €{r.sale_ppkg:.2f} projected.")

    # Deltas must lead with the sign — Streamlit reads the first character to
    # pick the arrow and colour, so "€-111/head" would render as a green rise.
    m = st.columns(5)
    m[0].metric("Net margin", f"€{r.net_margin_total:,.0f}",
                f"{r.net_margin_per_head:+,.0f} €/head")
    m[1].metric("ROI", f"{r.roi_pct:.1f}%", f"{r.annualised_roi_pct:+.1f}% annualised")
    m[2].metric("Projected sale weight", f"{r.sale_weight:.0f} kg",
                f"{r.sale_weight - bwt:+.0f} kg")
    m[3].metric("Break-even €/kg", f"€{r.breakeven_ppkg:.2f}",
                f"{r.margin_of_safety_pct:+.1f}% safety margin")
    m[4].metric("Finance cover", f"{r.finance_coverage:.1f}×" if r.finance_coverage else "—",
                help="Net margin ÷ interest cost. Below 1.0× the deal does not "
                     "cover its own interest.")

    # ── Cost waterfall ───────────────────────────────────────────────────────
    st.markdown("**Where the money goes**")
    wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total", "relative", "total"],
        x=["Purchase", "Keep", "Other", "Finance", "Total cost", "Sale proceeds", "Net margin"],
        y=[r.buy_cost_total, r.feed_total, r.other_total, r.finance_cost,
           0, r.sale_value_total, 0],
        text=[f"€{v:,.0f}" for v in
              [r.buy_cost_total, r.feed_total, r.other_total, r.finance_cost,
               r.total_cost, r.sale_value_total, r.net_margin_total]],
        textposition="outside",
        connector=dict(line=dict(color=BORDER)),
        increasing=dict(marker=dict(color=RED)),
        decreasing=dict(marker=dict(color=GREEN)),
        totals=dict(marker=dict(color=NAVY)),
    ))
    wf.update_layout(height=380, yaxis_title="€", showlegend=False, **_layout())
    st.plotly_chart(wf, use_container_width=True, theme=None)

    # ── The taper ────────────────────────────────────────────────────────────
    st.markdown("**Entry vs exit cohort**")
    naive = r.entry_price.ppkg_median * r.sale_weight * r.head_at_sale
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            f"Buying into **{r.entry_price.basis}** at €{r.buy_ppkg:.2f}/kg "
            f"({r.buy_price_source}) &nbsp; {_confidence_chip(r.entry_price)}<br>"
            f"<span style='color:{DGREY};font-size:0.85rem'>{r.entry_price.n:,} comparable lots</span>",
            unsafe_allow_html=True)
    with t2:
        st.markdown(
            f"Selling into **{r.exit_price.basis}** at €{r.sale_ppkg:.2f}/kg "
            f"&nbsp; {_confidence_chip(r.exit_price)}<br>"
            f"<span style='color:{DGREY};font-size:0.85rem'>{r.exit_price.n:,} comparable lots</span>",
            unsafe_allow_html=True)

    taper = r.entry_price.ppkg_median - r.exit_price.ppkg_median
    if taper > 0:
        st.info(
            f"€/kg falls **€{taper:.2f}** ({taper/r.entry_price.ppkg_median*100:.0f}%) moving from "
            f"{r.entry_price.band} to {r.exit_price.band} — the profit is in the extra kilos, not the "
            f"price. Valuing the sale at the purchase price would overstate proceeds by "
            f"€{naive - r.sale_value_total:,.0f}.")

    # ── Scenarios ────────────────────────────────────────────────────────────
    st.markdown("**Price scenarios**")
    sc = ac.scenario_table(df_all, growth, deal)
    st.dataframe(
        sc.style.format({"Sale value": "€{:,.0f}", "Total cost": "€{:,.0f}",
                         "Net margin": "€{:,.0f}", "€/head": "€{:,.0f}",
                         "Sale €/kg": "€{:.2f}", "ROI %": "{:.1f}%"}),
        use_container_width=True, hide_index=True)

    st.caption(
        "Exit price is today's market rate for the destination cohort. No seasonal "
        "forecast is applied — there are only 21 weeks of price history, which is enough "
        "to measure a cohort's current momentum but not to project 6–12 months ahead. "
        "Use the scenarios to apply your own view.")


def section_trajectory(df_all):
    """Where an animal's weight and value are likely to be at a future age."""
    st.subheader("Value Trajectory")
    st.caption("What comparable animals of this breed and sex weigh at each age, "
               "priced at today's market. Not a forecast — see the note below.")

    c = st.columns(5)
    breed = c[0].selectbox("Breed", BREED_CHOICES,
                           index=BREED_CHOICES.index("LMX") if "LMX" in BREED_CHOICES else 0,
                           key="tj_breed")
    sex   = c[1].selectbox("Sex", ["Male", "Female", "Bull"], key="tj_sex")
    age   = c[2].number_input("Current age (months)", 1.0, 38.0, 10.0, step=1.0, key="tj_age")
    horizon = c[3].slider("Look ahead (months)", 2, 24, 12, step=2, key="tj_h")
    head  = c[4].number_input("Head", 1, 2000, 1, step=1, key="tj_head")

    known = st.checkbox(
        "I know this animal's current weight", value=False, key="tj_known",
        help="Optional. Without it you get the pure cohort range, which is fully "
             "observed. With it, the animal's advantage over its cohort is carried "
             "forward — a reasonable but unverifiable assumption.")
    cur_w = None
    if known:
        cw = st.columns(4)
        cur_w = cw[0].number_input("Current weight (kg)", 50.0, 900.0, 340.0,
                                   step=10.0, key="tj_cw")
        ret = cw[1].slider("Advantage retained per year", 0.0, 1.0,
                           ac.ANCHOR_RETENTION_PER_YEAR, step=0.1, key="tj_ret",
                           help="1.0 = a heavy animal stays heavy. 0.0 = every animal "
                                "reverts to the cohort median (measurably wrong: that "
                                "runs +77 kg on light animals and -107 kg on heavy).")
    else:
        ret = ac.ANCHOR_RETENTION_PER_YEAR

    if age + horizon > ac.MAX_AGE_MONTHS:
        st.warning(f"Capped at {ac.MAX_AGE_MONTHS} months — comparables thin out "
                   f"badly beyond that.")

    traj = ac.weight_trajectory(df_all, breed, sex, age, months_ahead=horizon,
                                current_weight=cur_w, retention=ret)
    if not traj or not traj[-1].kg_median:
        st.info("Not enough comparable animals for this profile.")
        return

    # Value each point at today's price for the weight band it lands in
    rows = []
    for e in traj:
        cp = ac.price_cohort(df_all, breed, sex, e.kg_median)
        rows.append(dict(
            month=e.age_months - age, age=e.age_months,
            kg_p25=e.kg_p25, kg=e.kg_median, kg_p75=e.kg_p75,
            ppkg=cp.ppkg_median, band=cp.band,
            val_p25=e.kg_p25 * cp.ppkg_p25, val=e.kg_median * cp.ppkg_median,
            val_p75=e.kg_p75 * cp.ppkg_p75,
            n_w=e.n, n_p=cp.n, basis=e.basis, conf=e.confidence))
    T = pd.DataFrame(rows)
    first, last = T.iloc[0], T.iloc[-1]

    k = st.columns(5)
    k[0].metric(f"Weight at {last.age:.0f} mo", f"{last.kg:.0f} kg",
                f"{last.kg - first.kg:+.0f} kg")
    k[1].metric("Likely range", f"{last.kg_p25:.0f}–{last.kg_p75:.0f} kg",
                help="Middle half of comparable animals. Half fall outside it.")
    k[2].metric("€/kg at that weight", f"€{last.ppkg:.2f}",
                help=f"Today's market for the {last.band} band")
    k[3].metric(f"Value × {head:,}", f"€{last.val * head:,.0f}",
                f"{(last.val - first.val) * head:+,.0f}")
    k[4].metric("Value range", f"€{last.val_p25 * head:,.0f}–{last.val_p75 * head:,.0f}")

    # ── Fan charts ───────────────────────────────────────────────────────────
    g1, g2 = st.columns(2)
    for col, lo, mid, hi, title, unit in [
            (g1, "kg_p25", "kg", "kg_p75", "Weight by age", "kg"),
            (g2, "val_p25", "val", "val_p75", f"Value by age (× {head:,})", "€")]:
        scale = head if unit == "€" else 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T.age, y=T[hi] * scale, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=T.age, y=T[lo] * scale, mode="lines", fill="tonexty",
                                 fillcolor="rgba(42,111,191,0.16)", line=dict(width=0),
                                 name="middle half of comparables"))
        fig.add_trace(go.Scatter(x=T.age, y=T[mid] * scale, mode="lines+markers",
                                 line=dict(color=NAVY, width=2.5), name="expected"))
        fig.update_layout(height=340, title=title,
                          legend=dict(orientation="h", y=-0.22, x=0),
                          **_layout(margin=dict(l=10, r=10, t=44, b=10),
                                    xaxis=dict(title="Age (months)"),
                                    yaxis=dict(title=unit)))
        col.plotly_chart(fig, use_container_width=True, theme=None)

    disp = T[["age", "kg_p25", "kg", "kg_p75", "ppkg", "val", "n_w", "conf"]].copy()
    disp.columns = ["Age (mo)", "kg P25", "kg est", "kg P75", "€/kg",
                    "Value/head", "Comparables", "Confidence"]
    st.dataframe(disp.style.format({"Age (mo)": "{:.0f}", "kg P25": "{:.0f}",
                                    "kg est": "{:.0f}", "kg P75": "{:.0f}",
                                    "€/kg": "€{:.2f}", "Value/head": "€{:,.0f}",
                                    "Comparables": "{:,.0f}"}),
                 use_container_width=True, hide_index=True)

    st.caption(
        f"**Weights** come from {T.n_w.min():,}–{T.n_w.max():,} comparable animals per "
        f"age step, not from a growth model — no animal in this data is ever weighed "
        f"twice, so an individual growth curve could not be validated. Measured on "
        f"held-out sales: **±55 kg (16% MAPE), no bias**, with 54% of animals inside "
        f"the P25–P75 band against an ideal 50% — the range is slightly wide, which "
        f"errs the safe way. That accuracy assumes the scrapers are current; on a "
        f"6-week-stale dataset the same test drifts **31 kg light**. "
        f"**Values use today's €/kg** for whatever weight band the animal reaches; no "
        f"price forecast is applied, because 21 weeks of price history cannot support "
        f"one. Treat the value line as *what an animal like this is worth today at that "
        f"weight*, not as a prediction of the future market.")


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
    tab1, tab2, tab9, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Herd Valuation", "Quick Underwrite", "Value Trajectory",
        "Cohort Matrix", "Regional Analysis", "Price Trends",
        "Mart Comparison", "Collateral Reference", "Factory Reference"
    ])

    # The credit tools price against the full dataset, not the sidebar-filtered
    # view — a date or region filter set for browsing must not silently change
    # what a customer's herd is deemed to be worth.
    with tab1:
        section_herd_valuation(df_raw)

    with tab2:
        section_underwrite(df_raw)

    with tab9:
        section_trajectory(df_raw)

    with tab3:
        section_cohort_matrix(df)
        st.divider()
        section_sex_age(df)

    with tab4:
        section_regional(df)

    with tab5:
        section_trends(df)

    with tab6:
        section_mart_comparison(df)

    with tab7:
        section_collateral_ref(df)

    with tab8:
        section_factory_reference(fp)


if __name__ == "__main__":
    main()
