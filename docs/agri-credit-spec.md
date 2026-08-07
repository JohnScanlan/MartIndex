# Herd Valuation Tracker & Quick Underwrite — specification

Two credit-decision tools for `bank_dashboard.py`, built on the existing mart
data. Both share one pricing engine so a herd valuation and an underwrite can
never disagree about what an animal is worth.

Status: implemented 7 Aug 2026. Module `agri_credit.py`, store `herd_store.py`,
UI in `bank_dashboard.py` tabs "Herd Valuation" and "Quick Underwrite".

---

## 1. Shared pricing engine (`agri_credit.price_cohort`)

The single source of truth for "what is this animal worth today".

**Input:** breed group, sex, weight, optional region, lookback window.
**Output:** P25 / median / P75 €/kg, comparable count, the basis used, and the
cohort's recent price momentum.

### Fallback ladder

Cells thin out fast once region is included, and a banker must never be shown a
confident number derived from four lots. The engine walks down this ladder until
a cell has `min_lots` (default 20) comparables, and **reports which rung it
used**:

| Rung | Match on | Typical n (8 wk) |
|---|---|---|
| `exact` | breed × sex × band × region | 100–900 |
| `no_region` | breed × sex × band | 400–3,100 |
| `no_sex` | breed × band | 900–5,000 |
| `band_only` | band | 5,000–15,000 |
| `national` | everything in window | 47,000 |

Measured density: in the last 8 weeks all 78 breed×sex×band cells have ≥20 lots,
and 100% of lots sit in cells with ≥20 comparables. The ladder is a safety net,
not the normal path — but the rung is always displayed, because a valuation
resting on `band_only` is a materially weaker number than one resting on `exact`.

### Why P25 is the headline for lending

Median is the fair-value estimate. **P25 is what the LTV is calculated on** — it
approximates a realistic forced-sale outcome, which is the number that matters
when the security is actually called on. Both are shown; the ratio between them
is itself a liquidity signal.

### Cohort momentum

Weekly medians within the *selected cohort* over a 16-week window, linear fit,
reported as €/kg/month. This exists because the drift is strongly
weight-dependent and a single national trend line would mislead:

| Cohort | Drift €/kg/month |
|---|---|
| <200 kg store | −0.31 to −0.46 |
| 300–500 kg | −0.07 to −0.13 |
| >650 kg finished | −0.03 to +0.01 |

Only ~28% of the national decline is mix shift, so this is a real market signal,
not an artefact. Surfaced as a per-cohort figure, never extrapolated forward.

---

## 2. Herd Valuation Tracker

Mark-to-market a customer's herd against their loan, and track it over time.

### Model

A **herd** belongs to a customer/loan and is made of **lines** — a line is
`(breed group, sex, head count, average weight)`. Real herds are heterogeneous;
valuing them as a single blob would be meaningless.

Per line: `head × avg_weight × ppkg` at P25 / median / P75. Summed to herd level.

### Outputs

- Herd value at P25 / median / P75, per line and in total
- **LTV** = loan balance ÷ P25 value, with a RAG band against a covenant threshold
- **Headroom** — how far value can fall in € and % before breaching the covenant
- **Weighted cohort momentum** — the herd's blended depreciation rate, in €/month
  and as months-to-breach if the current drift continues
- **Valuation history** — every valuation is appended to a time series and
  charted against the loan balance

### Persistence

Two files, with deliberately different write strategies:

| File | Strategy | Why |
|---|---|---|
| `herds.csv` | atomic read-modify-write, timestamped backup on every change | small user-managed registry that must be editable |
| `herd_valuation_history.csv` | append-only via `safe_append_csv` | the tracking time series; must never lose a point |

Neither touches scraped data. Deleting a herd writes `status=deleted` rather
than removing rows, so the valuation history stays interpretable.

---

## 3. Quick Underwrite

Answer, at the point of sale: *does this purchase pay for itself?*

### Flow

1. **Buy** — breed, sex, head, weight, age; price paid, or take the live market
   price from the same engine
2. **Grow** — `weight = a + b·√(age)` fitted per breed×sex from the mart data.
   Anchored to the animal's actual weight so the curve passes through
   `(current age, current weight)`; only the slope drives the projection
3. **Sell** — price the animal *in its exit cohort*, not its entry cohort. This
   is the single most important calculation in the tool
4. **Net off** — feed/keep per head per day, other costs per head (vet,
   transport, commission), mortality allowance, finance cost
5. **Verdict** — net margin, break-even sale price, margin of safety

### Why the exit cohort matters

The €/kg taper is steep and monotonic — Continental × falls 5.51 → 3.72 €/kg
from <200 kg to >650 kg, a 32% decline. A store-to-finish deal *buys high and
sells low on a per-kg basis*; the profit comes entirely from the extra kilos.
Pricing the exit at the entry cohort's €/kg would overstate margin by roughly a
third and turn losing deals into apparently good ones.

### Outputs

- Projected sale weight, sale €/kg (exit cohort), gross sale value
- Full cost stack, per head and total
- Net margin per head and total, ROI, annualised ROI
- **Break-even sale €/kg** and **margin of safety** — how far price can fall
  before the deal loses money
- **Price scenarios** — base / −10% / −20% applied to the exit price, so the
  committee sees the downside without re-running anything
- Finance coverage: net margin ÷ finance cost

### What this deliberately does not do

**No time-series price forecast.** There are 21 usable weeks of history. That is
enough to measure a cohort's current momentum, nowhere near enough to project a
sale price 6–12 months out, and a fabricated seasonal curve presented to a credit
committee would be worse than no curve. The exit price is today's price for the
exit cohort, and the scenario sliders carry the banker's own view of the market.

Revisit once there are ≥2 years of history and a genuine seasonal index can be
fitted.

---

## 4. Non-goals for this build

Deliberately out of scope, in rough priority order for a future pass:

- Distress-selling detection (needs per-customer identity, which the mart feed
  does not carry)
- Cost-of-production index (needs a Teagasc/CSO feed)
- TB restriction roll-up (data exists in `sold_lots.csv`, but herd-level
  attribution does not)
- Portfolio-level stress test across all loans at once
- Factory-price exit channel as an alternative to mart exit
