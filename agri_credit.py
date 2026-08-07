"""
Agri-lending credit tools — pricing engine, herd valuation, deal underwriting.

Pure functions over the DataFrame that bank_dashboard.load_data() produces.
No Streamlit here, so the maths can be tested and reused on its own.

The one rule that holds this together: every €/kg figure in either tool comes
from price_cohort(). A herd valuation and an underwrite must never disagree
about what the same animal is worth.

See docs/agri-credit-spec.md for the reasoning behind the design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ── Cohort definitions ────────────────────────────────────────────────────────
# bank_dashboard.py imports these so the two can never drift apart — a valuation
# that banded animals differently from the cohort matrix would be silently wrong.

WEIGHT_BINS   = [0, 200, 300, 400, 500, 650, 9999]
WEIGHT_LABELS = ["<200 kg", "200–300 kg", "300–400 kg",
                 "400–500 kg", "500–650 kg", ">650 kg"]

# Breeds are the raw mart codes (LMX, AAX, CHX …), not rolled-up groups.
# The model uses the same codes, so a cohort price and a model prediction refer
# to the same population — the previous "Continental ×" style grouping meant the
# dashboards and the model spoke about different animals.
TOP_BREEDS = 20          # keep the N most common; everything else is "Other"

# Sheep codes that turn up in the cattle feed (Ennis lists ewes at ~83 kg).
# train_model.py drops these before fitting, so the dashboards must not offer
# them either — the model has no category for them.
NON_CATTLE_BREEDS_UI = {"EWE", "LB", "L1", "L2", "L3", "L4", "LAMB", "RAM", "HOG"}
OTHER_BREED = "Other"


def breed_levels(df: pd.DataFrame) -> list[str]:
    """
    The canonical breed vocabulary, read from the trained model where possible
    so the dashboards and the model always agree on which codes are first-class
    and which collapse into "Other".
    """
    meta = Path(__file__).parent / "model_metadata.json"
    try:
        levels = json.loads(meta.read_text()).get("breed_levels")
        if levels:
            return list(levels)
    except (OSError, ValueError):
        pass
    top = df["breed"].value_counts().head(TOP_BREEDS).index.tolist()
    return sorted(top) + [OTHER_BREED]


def to_breed_group(breed: pd.Series, levels: list[str]) -> pd.Series:
    """Map raw breed codes onto the canonical vocabulary."""
    known = set(levels) - {OTHER_BREED}
    return breed.where(breed.isin(known), OTHER_BREED).fillna(OTHER_BREED)

SEX_LABELS = {"M": "Male", "F": "Female", "B": "Bull"}

# Defaults
LOOKBACK_WEEKS   = 8     # pricing window — recent enough to be mark-to-market
MOMENTUM_WEEKS   = 16    # trend window — needs more points to fit a slope
MIN_LOTS         = 20    # below this, fall down the ladder

FALLBACK_LABELS = {
    "exact":      "breed × sex × weight × region",
    "no_region":  "breed × sex × weight (all Ireland)",
    "no_sex":     "breed × weight (all Ireland)",
    "band_only":  "weight band only (all breeds)",
    "national":   "national average (no cohort match)",
}

# How much to trust each rung, shown in the UI as a confidence hint
FALLBACK_CONFIDENCE = {
    "exact": "High", "no_region": "High", "no_sex": "Medium",
    "band_only": "Low", "national": "Very low",
}


def _breed_col(df: pd.DataFrame) -> str:
    """
    Resolve the breed column. bank_dashboard.py calls it `breed_group`;
    dashboard.py calls it `breed_grp`. Rather than force one app to rename a
    column used in dozens of places, resolve it here — these helpers are shared
    and must work against either frame.
    """
    for c in ("breed_group", "breed_grp"):
        if c in df.columns:
            return c
    raise KeyError("no breed column found (looked for breed_group / breed_grp)")


def band_for_weight(weight_kg: float) -> str:
    """Weight → band label, matching pd.cut(bins=WEIGHT_BINS, right=False)."""
    for lo, hi, label in zip(WEIGHT_BINS[:-1], WEIGHT_BINS[1:], WEIGHT_LABELS):
        if lo <= weight_kg < hi:
            return label
    return WEIGHT_LABELS[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Cohort pricing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CohortPrice:
    """What a cohort of cattle is worth per kg, and how much to trust it."""
    ppkg_p25:    float
    ppkg_median: float
    ppkg_p75:    float
    n:           int
    band:        str
    fallback:    str                 # key into FALLBACK_LABELS
    basis:       str                 # human-readable description of the match
    drift_per_month: float | None    # €/kg/month within this cohort, None if unmeasurable
    as_of:       date | None

    @property
    def confidence(self) -> str:
        return FALLBACK_CONFIDENCE.get(self.fallback, "Unknown")

    @property
    def spread_pct(self) -> float:
        """P25→P75 spread as % of median — a rough liquidity/dispersion signal."""
        if not self.ppkg_median:
            return 0.0
        return (self.ppkg_p75 - self.ppkg_p25) / self.ppkg_median * 100


def _cohort_momentum(sub: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    """
    €/kg per month drift within a cohort, from weekly medians over
    MOMENTUM_WEEKS. Returns None when there aren't enough weeks to fit a line.

    Computed per cohort rather than nationally because the drift is strongly
    weight-dependent: store cattle have been falling ~0.4 €/kg/month while
    finished cattle are flat. A single national figure would mislead on both.
    """
    window = sub[sub["sale_date"] >= as_of - pd.Timedelta(weeks=MOMENTUM_WEEKS)]
    if window.empty:
        return None

    weekly = (window
              .groupby(window["sale_date"].dt.to_period("W").dt.start_time)["price_per_kg"]
              .agg(median="median", n="size"))
    weekly = weekly[weekly["n"] >= 15]
    if len(weekly) < 6:
        return None

    slope_per_week = float(np.polyfit(np.arange(len(weekly)), weekly["median"].values, 1)[0])
    return slope_per_week * 4.345          # weeks → months


def price_cohort(df: pd.DataFrame,
                 breed_group: str,
                 sex: str,
                 weight_kg: float,
                 region: str | None = None,
                 weeks: int = LOOKBACK_WEEKS,
                 min_lots: int = MIN_LOTS) -> CohortPrice:
    """
    Price one animal profile from recent comparable sales.

    Walks a fallback ladder until a cell has `min_lots` comparables, and reports
    which rung it landed on. A banker needs to know whether a number rests on
    900 comparable lots or on the national average.
    """
    band = band_for_weight(weight_kg)

    work = df.dropna(subset=["price_per_kg", "sale_date"])
    if work.empty:
        return CohortPrice(0, 0, 0, 0, band, "national",
                           "no data available", None, None)

    as_of  = work["sale_date"].max()
    recent = work[work["sale_date"] >= as_of - pd.Timedelta(weeks=weeks)]
    if recent.empty:
        recent = work

    in_band = recent["weight_band"].astype(str) == band
    is_breed = recent[_breed_col(recent)] == breed_group
    is_sex   = recent["sex_clean"] == sex

    # Ladder: most specific first. Each entry is (key, mask, description).
    rungs: list[tuple[str, pd.Series, str]] = []
    if region and region not in (None, "", "All Ireland"):
        rungs.append(("exact",
                      is_breed & is_sex & in_band & (recent["region"] == region),
                      f"{breed_group} · {sex} · {band} · {region}"))
    rungs += [
        ("no_region", is_breed & is_sex & in_band, f"{breed_group} · {sex} · {band}"),
        ("no_sex",    is_breed & in_band,          f"{breed_group} · {band}"),
        ("band_only", in_band,                     f"all breeds · {band}"),
        ("national",  pd.Series(True, index=recent.index), "national, all cattle"),
    ]

    for key, mask, basis in rungs:
        sub = recent[mask]
        if len(sub) < min_lots and key != "national":
            continue
        p = sub["price_per_kg"]
        # Momentum uses the same cohort but a longer window, so re-derive it
        # from the full frame rather than the 8-week slice.
        drift = _cohort_momentum(work[_rebuild_mask(work, key, breed_group, sex, band, region)], as_of)
        return CohortPrice(
            ppkg_p25=float(p.quantile(0.25)),
            ppkg_median=float(p.median()),
            ppkg_p75=float(p.quantile(0.75)),
            n=int(len(sub)),
            band=band,
            fallback=key,
            basis=basis,
            drift_per_month=drift,
            as_of=as_of.date() if pd.notna(as_of) else None,
        )

    # Unreachable — the national rung always returns.
    return CohortPrice(0, 0, 0, 0, band, "national", "no data", None, None)


def _rebuild_mask(work: pd.DataFrame, key: str, breed_group: str,
                  sex: str, band: str, region: str | None) -> pd.Series:
    """Rebuild a ladder rung's mask against the full (unwindowed) frame."""
    in_band  = work["weight_band"].astype(str) == band
    is_breed = work[_breed_col(work)] == breed_group
    is_sex   = work["sex_clean"] == sex
    if key == "exact":
        return is_breed & is_sex & in_band & (work["region"] == region)
    if key == "no_region":
        return is_breed & is_sex & in_band
    if key == "no_sex":
        return is_breed & in_band
    if key == "band_only":
        return in_band
    return pd.Series(True, index=work.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Growth model
# ═══════════════════════════════════════════════════════════════════════════════

def fit_growth_curves(df: pd.DataFrame, min_lots: int = 40) -> dict:
    """
    Fit weight = a + b·√(age_months) per breed_group × sex.

    The sqrt form captures the natural deceleration of growth as animals mature;
    a straight line badly overshoots at the finishing end. Linear in √age, so
    it solves exactly with polyfit — no iterative optimiser to fail.

    Returns {(breed_group, sex): (a, b)} plus a ("_default","_default") entry.
    """
    grp = df.dropna(subset=["age_months", "weight"])
    grp = grp[(grp["age_months"] > 0) & (grp["weight"] > 0)]
    if grp.empty:
        return {("_default", "_default"): (0.0, 60.0)}

    def _fit(ages: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        b, a = np.polyfit(np.sqrt(ages), weights, 1)
        return float(a), max(float(b), 1.0)      # never allow a shrinking animal

    params = {}
    for (breed, sex), g in grp.groupby([_breed_col(grp), "sex_clean"]):
        if len(g) < min_lots:
            continue
        params[(breed, sex)] = _fit(g["age_months"].values, g["weight"].values)

    params[("_default", "_default")] = _fit(grp["age_months"].values, grp["weight"].values)
    return params


def project_weight(cur_weight: float, cur_age: float,
                   months_ahead: float, b: float) -> float:
    """
    Project forward, anchoring the curve to this animal's actual weight.

    Only the slope is used — the fitted intercept is discarded so the curve
    passes through (cur_age, cur_weight). Otherwise a good animal gets dragged
    back to the population average the moment you project it.

    Floored at 0.5 kg/day-equivalent so a mature animal still shows some gain.
    """
    future_age = cur_age + months_ahead
    gain = b * (np.sqrt(max(future_age, 0.01)) - np.sqrt(max(cur_age, 0.01)))
    return float(cur_weight + max(gain, 0.5 * months_ahead))


def growth_for(params: dict, breed_group: str, sex: str) -> float:
    """Slope b for a breed×sex, falling back to the population fit."""
    a_b = params.get((breed_group, sex)) or params.get(("_default", "_default"))
    return a_b[1] if a_b else 60.0


# ═══════════════════════════════════════════════════════════════════════════════
# Weight at a future age
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is a LOOKUP over comparable animals, not a growth model, and that is a
# deliberate choice. A growth model would claim to know what *this* animal will
# weigh later; nothing in the data can validate such a claim, because no animal
# is ever measured twice. (Checked directly: 17,454 (dob, breed, sex) keys
# recur across sale dates, but ~35% of the implied growth rates are negative and
# 18% exceed 2 kg/day no matter how strictly the match is tightened — they are
# siblings sharing a birth date, not the same animal weighed again.)
#
# So we answer the question the data *can* answer: "what do animals of this
# breed and sex weigh at that age?" That is checkable on held-out sales, and it
# yields a distribution rather than a false point estimate.

MAX_AGE_MONTHS   = 40    # beyond this the comparables thin out badly
AGE_WINDOW       = 3     # ± months for the primary lookup
AGE_WINDOW_WIDE  = 6     # widened when the tight window is too thin
MIN_WEIGHT_LOTS  = 30

# How much of an animal's above/below-average weight persists per year, when a
# current weight is supplied. Cannot be measured without repeat observations, so
# it is an assumption, deliberately mid-way between the two failure modes:
#   1.0 — a heavy animal stays heavy forever (over-optimistic)
#   0.0 — every animal reverts to the cohort median. Measurably wrong: the old
#         sqrt curve did this and ran +77 kg on light animals, -107 kg on heavy.
ANCHOR_RETENTION_PER_YEAR = 0.5

# Real Irish cattle gain roughly 0.4–1.4 kg/day. The cohort curve can imply less
# than that, because animals sold at 8 months and animals sold at 20 months are
# different populations rather than the same animals grown up — farmers move
# lighter stores on early and keep the thrivers, so the cross-sectional slope
# understates individual growth. Measured implied gains over 8→14 months run
# 0.25–0.33 kg/day for AAX/LMX/CHX. Callers should surface this rather than
# present the number as a growth expectation.
PLAUSIBLE_ADG = (0.4, 1.4)


def adg_is_plausible(kg_per_day: float) -> bool:
    lo, hi = PLAUSIBLE_ADG
    return lo <= kg_per_day <= hi


@dataclass
class WeightEstimate:
    """What animals of this profile weigh at a given age."""
    age_months:  float
    kg_p25:      float
    kg_median:   float
    kg_p75:      float
    n:           int
    basis:       str
    fallback:    str
    anchored:    bool = False      # adjusted toward a known current weight

    @property
    def confidence(self) -> str:
        return FALLBACK_CONFIDENCE.get(self.fallback, "Unknown")


def weight_at_age(df: pd.DataFrame,
                  breed_group: str,
                  sex: str,
                  age_months: float,
                  min_lots: int = MIN_WEIGHT_LOTS) -> WeightEstimate:
    """
    Expected weight and range at `age_months`, from comparable sold animals.

    Walks the same style of fallback ladder as price_cohort() and reports which
    rung it used, so a thin cell is never presented as a confident answer.
    """
    age = float(np.clip(age_months, 1, MAX_AGE_MONTHS))
    work = df.dropna(subset=["weight", "age_months"])
    work = work[(work["age_months"] > 0) & (work["weight"] > 0)]
    if work.empty:
        return WeightEstimate(age, 0, 0, 0, 0, "no data", "national")

    bcol = _breed_col(work)

    def window(w):
        return work["age_months"].between(age - w, age + w)

    rungs = [
        ("exact",     (work[bcol] == breed_group) & (work["sex_clean"] == sex) & window(AGE_WINDOW),
         f"{breed_group} · {sex} · {age:.0f}±{AGE_WINDOW} mo"),
        ("no_region", (work[bcol] == breed_group) & (work["sex_clean"] == sex) & window(AGE_WINDOW_WIDE),
         f"{breed_group} · {sex} · {age:.0f}±{AGE_WINDOW_WIDE} mo"),
        ("no_sex",    (work[bcol] == breed_group) & window(AGE_WINDOW_WIDE),
         f"{breed_group} · {age:.0f}±{AGE_WINDOW_WIDE} mo"),
        ("band_only", (work["sex_clean"] == sex) & window(AGE_WINDOW_WIDE),
         f"all breeds · {sex} · {age:.0f}±{AGE_WINDOW_WIDE} mo"),
        ("national",  window(AGE_WINDOW_WIDE),
         f"all cattle · {age:.0f}±{AGE_WINDOW_WIDE} mo"),
    ]
    for key, mask, basis in rungs:
        sub = work[mask]
        if len(sub) < min_lots and key != "national":
            continue
        w = sub["weight"]
        return WeightEstimate(age, float(w.quantile(0.25)), float(w.median()),
                              float(w.quantile(0.75)), int(len(sub)), basis, key)

    w = work["weight"]
    return WeightEstimate(age, float(w.quantile(0.25)), float(w.median()),
                          float(w.quantile(0.75)), int(len(work)),
                          "all cattle (no age match)", "national")


def weight_trajectory(df: pd.DataFrame,
                      breed_group: str,
                      sex: str,
                      current_age: float,
                      months_ahead: int = 24,
                      current_weight: float | None = None,
                      retention: float = ANCHOR_RETENTION_PER_YEAR,
                      step: int = 2) -> list[WeightEstimate]:
    """
    Weight at each future age, as a distribution.

    With `current_weight` supplied, the animal's deviation from its own cohort
    today is carried forward, decaying by `retention` per year. Without it, the
    pure cohort range is returned — fully observed and free of assumptions,
    which is why current weight is optional.
    """
    out = []
    offset0 = None
    if current_weight:
        now = weight_at_age(df, breed_group, sex, current_age)
        if now.kg_median:
            offset0 = current_weight - now.kg_median

    for m in range(0, months_ahead + 1, step):
        est = weight_at_age(df, breed_group, sex, current_age + m)
        if offset0 is not None:
            keep = retention ** (m / 12)
            shift = offset0 * keep
            est = WeightEstimate(est.age_months,
                                 est.kg_p25 + shift, est.kg_median + shift,
                                 est.kg_p75 + shift, est.n, est.basis,
                                 est.fallback, anchored=True)
        out.append(est)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Herd valuation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HerdLine:
    """One homogeneous group within a herd."""
    breed_group: str
    sex:         str
    head:        int
    avg_weight:  float


@dataclass
class LineValuation:
    line:   HerdLine
    price:  CohortPrice
    value_p25:    float
    value_median: float
    value_p75:    float
    kg_total:     float


@dataclass
class HerdValuation:
    lines:        list[LineValuation] = field(default_factory=list)
    total_head:   int   = 0
    total_kg:     float = 0.0
    value_p25:    float = 0.0
    value_median: float = 0.0
    value_p75:    float = 0.0
    as_of:        date | None = None

    # ── Loan metrics ─────────────────────────────────────────────────────────
    def ltv(self, loan_balance: float) -> float | None:
        """LTV against the P25 (conservative / forced-sale) valuation."""
        if not self.value_p25:
            return None
        return loan_balance / self.value_p25 * 100

    def headroom(self, loan_balance: float, max_ltv: float) -> float:
        """€ the herd can lose before breaching the covenant."""
        floor = loan_balance / (max_ltv / 100) if max_ltv else 0
        return self.value_p25 - floor

    @property
    def drift_per_month_eur(self) -> float | None:
        """
        Blended € change per month across the herd, weighted by each line's kg.

        This is the number that says how fast the security is depreciating —
        light store cattle erode far quicker than finished stock.
        """
        contribs = [lv.price.drift_per_month * lv.kg_total
                    for lv in self.lines if lv.price.drift_per_month is not None]
        return float(sum(contribs)) if contribs else None

    def months_to_breach(self, loan_balance: float, max_ltv: float) -> float | None:
        """At the current drift, months until the LTV covenant breaks."""
        drift = self.drift_per_month_eur
        if not drift or drift >= 0:
            return None
        room = self.headroom(loan_balance, max_ltv)
        if room <= 0:
            return 0.0
        return room / abs(drift)

    @property
    def weakest_basis(self) -> str:
        """The least reliable rung used across all lines — the herd's true confidence."""
        order = ["exact", "no_region", "no_sex", "band_only", "national"]
        if not self.lines:
            return "national"
        return max((lv.price.fallback for lv in self.lines),
                   key=lambda k: order.index(k) if k in order else 99)


def value_herd(df: pd.DataFrame,
               lines: list[HerdLine],
               region: str | None = None,
               weeks: int = LOOKBACK_WEEKS) -> HerdValuation:
    """Mark a herd to market, line by line."""
    hv = HerdValuation()
    for line in lines:
        if line.head <= 0 or line.avg_weight <= 0:
            continue
        cp = price_cohort(df, line.breed_group, line.sex, line.avg_weight,
                          region=region, weeks=weeks)
        kg = line.head * line.avg_weight
        lv = LineValuation(
            line=line, price=cp, kg_total=kg,
            value_p25=kg * cp.ppkg_p25,
            value_median=kg * cp.ppkg_median,
            value_p75=kg * cp.ppkg_p75,
        )
        hv.lines.append(lv)
        hv.total_head   += line.head
        hv.total_kg     += kg
        hv.value_p25    += lv.value_p25
        hv.value_median += lv.value_median
        hv.value_p75    += lv.value_p75
        hv.as_of = cp.as_of or hv.as_of
    return hv


# ═══════════════════════════════════════════════════════════════════════════════
# Deal underwriting
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DealInputs:
    breed_group:  str
    sex:          str
    head:         int
    buy_weight:   float
    buy_age:      float
    finish_months: float
    # Growth — None uses the fitted curve; set to override with a known ADG
    daily_gain_override: float | None = None   # kg/head/day
    # Costs
    buy_ppkg:            float | None = None   # None → use the market price
    feed_per_head_day:   float = 1.60          # €/head/day, blended grass/indoor
    other_cost_per_head: float = 90.0          # vet, transport, commission, levies
    mortality_pct:       float = 1.5
    # Finance
    loan_amount:   float = 0.0
    interest_rate: float = 7.5                 # annual %
    # Scenario
    price_adj_pct: float = 0.0                 # applied to the exit €/kg
    region:        str | None = None


@dataclass
class DealResult:
    # Entry
    buy_price_source: str
    buy_ppkg:   float
    buy_cost_per_head:  float
    buy_cost_total:     float
    entry_price: CohortPrice
    # Exit
    sale_weight:  float
    sale_ppkg:    float
    exit_price:   CohortPrice
    head_at_sale: float
    sale_value_per_head: float
    sale_value_total:    float
    # Costs
    feed_total:    float
    other_total:   float
    finance_cost:  float
    mortality_cost: float
    total_cost:    float
    # Result
    net_margin_total:    float
    net_margin_per_head: float
    roi_pct:             float
    annualised_roi_pct:  float
    breakeven_ppkg:      float
    margin_of_safety_pct: float
    finance_coverage:     float | None

    @property
    def viable(self) -> bool:
        return self.net_margin_total > 0


def underwrite(df: pd.DataFrame,
               growth_params: dict,
               deal: DealInputs) -> DealResult:
    """
    Underwrite a store-to-finish purchase.

    The critical step is pricing the exit in the *exit* cohort. €/kg falls
    steeply with weight (Continental × runs 5.51 → 3.72 from <200 kg to
    >650 kg), so valuing the sale at the purchase cohort's rate would overstate
    margin by roughly a third and make losing deals look profitable.
    """
    # ── Entry ────────────────────────────────────────────────────────────────
    entry = price_cohort(df, deal.breed_group, deal.sex, deal.buy_weight,
                         region=deal.region)
    if deal.buy_ppkg is not None and deal.buy_ppkg > 0:
        buy_ppkg, source = deal.buy_ppkg, "entered"
    else:
        buy_ppkg, source = entry.ppkg_median, "market median"

    buy_cost_per_head = buy_ppkg * deal.buy_weight
    buy_cost_total    = buy_cost_per_head * deal.head

    # ── Grow ─────────────────────────────────────────────────────────────────
    # The fitted curve is cross-sectional: it compares what a 9-month animal
    # weighs against what a 17-month animal weighs, across different animals.
    # That is a fair population average but it understates a well-fed individual,
    # so the banker can override it with a known average daily gain.
    if deal.daily_gain_override and deal.daily_gain_override > 0:
        sale_weight = deal.buy_weight + deal.daily_gain_override * deal.finish_months * 30.44
    else:
        b = growth_for(growth_params, deal.breed_group, deal.sex)
        sale_weight = project_weight(deal.buy_weight, deal.buy_age, deal.finish_months, b)

    # ── Exit, priced in the destination cohort ───────────────────────────────
    exit_cp = price_cohort(df, deal.breed_group, deal.sex, sale_weight,
                           region=deal.region)
    sale_ppkg = exit_cp.ppkg_median * (1 + deal.price_adj_pct / 100)

    head_at_sale        = deal.head * (1 - deal.mortality_pct / 100)
    sale_value_per_head = sale_ppkg * sale_weight
    sale_value_total    = sale_value_per_head * head_at_sale

    # ── Costs ────────────────────────────────────────────────────────────────
    days           = deal.finish_months * 30.44
    feed_total     = deal.feed_per_head_day * days * deal.head
    other_total    = deal.other_cost_per_head * deal.head
    finance_cost   = deal.loan_amount * (deal.interest_rate / 100) * (deal.finish_months / 12)
    mortality_cost = buy_cost_per_head * (deal.head - head_at_sale)
    total_cost     = buy_cost_total + feed_total + other_total + finance_cost

    # ── Result ───────────────────────────────────────────────────────────────
    net_total    = sale_value_total - total_cost
    net_per_head = net_total / deal.head if deal.head else 0.0
    roi          = (net_total / total_cost * 100) if total_cost else 0.0
    annualised   = roi * (12 / deal.finish_months) if deal.finish_months else 0.0

    # Sale €/kg at which the deal exactly breaks even
    denom = sale_weight * head_at_sale
    breakeven = total_cost / denom if denom else 0.0
    mos = ((sale_ppkg - breakeven) / sale_ppkg * 100) if sale_ppkg else 0.0

    coverage = (net_total / finance_cost) if finance_cost > 0 else None

    return DealResult(
        buy_price_source=source, buy_ppkg=buy_ppkg,
        buy_cost_per_head=buy_cost_per_head, buy_cost_total=buy_cost_total,
        entry_price=entry,
        sale_weight=sale_weight, sale_ppkg=sale_ppkg, exit_price=exit_cp,
        head_at_sale=head_at_sale,
        sale_value_per_head=sale_value_per_head, sale_value_total=sale_value_total,
        feed_total=feed_total, other_total=other_total,
        finance_cost=finance_cost, mortality_cost=mortality_cost,
        total_cost=total_cost,
        net_margin_total=net_total, net_margin_per_head=net_per_head,
        roi_pct=roi, annualised_roi_pct=annualised,
        breakeven_ppkg=breakeven, margin_of_safety_pct=mos,
        finance_coverage=coverage,
    )


def scenario_table(df: pd.DataFrame, growth_params: dict,
                   deal: DealInputs,
                   adjustments=(0.0, -10.0, -20.0)) -> pd.DataFrame:
    """Run the deal at several exit-price adjustments for the credit committee."""
    rows = []
    for adj in adjustments:
        d = DealInputs(**{**deal.__dict__, "price_adj_pct": adj})
        r = underwrite(df, growth_params, d)
        rows.append({
            "Scenario":       "Base" if adj == 0 else f"{adj:+.0f}% price",
            "Sale €/kg":      round(r.sale_ppkg, 2),
            "Sale value":     round(r.sale_value_total, 0),
            "Total cost":     round(r.total_cost, 0),
            "Net margin":     round(r.net_margin_total, 0),
            "€/head":         round(r.net_margin_per_head, 0),
            "ROI %":          round(r.roi_pct, 1),
            "Viable":         "Yes" if r.net_margin_total > 0 else "No",
        })
    return pd.DataFrame(rows)
