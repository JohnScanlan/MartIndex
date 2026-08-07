# Weather & Cattle Price Analysis
**Irish Mart Data — MartBids + Open-Meteo**
*Analysis date: 2026-04-29*

---

## Dataset

| Metric | Value |
|---|---|
| Total lots analysed | 65,435 |
| Mart-day observations | 298 |
| Unique marts | 34 |
| Date range | 24 Mar 2026 – 27 Apr 2026 |
| Mean avg €/kg | €4.45 |
| Std dev €/kg | €1.26 |

Weather data sourced from Open-Meteo archive API, matched to each mart's GPS coordinates on the exact sale date. Analysis unit is **mart-day** (all lots sold at a given mart on a given day, aggregated to a single price average).

Weather conditions during this period were typical Irish spring: mean high of 12°C, average rainfall of 5.3mm/day, average wind speed of 30 km/h.

---

## Hypothesis

> Better weather leads to higher cattle prices at Irish marts — farmers are in a better mood, more willing to attend, and more competitive in bidding.

---

## Finding 1 — No Significant Weather Effect on Price

Raw Pearson correlations between mart-day average €/kg and weather variables:

| Weather Variable | r | p-value | Significant? |
|---|---|---|---|
| Max temperature (°C) | -0.024 | 0.686 | No |
| Min temperature (°C) | -0.042 | 0.470 | No |
| Precipitation (mm) | -0.002 | 0.969 | No |
| Wind speed (km/h) | +0.017 | 0.766 | No |

**After removing mart and month fixed effects** (to eliminate between-mart differences and seasonal trends):

| Weather Variable | Partial r | p-value | Significant? |
|---|---|---|---|
| Max temperature (°C) | -0.013 | 0.818 | No |
| Precipitation (mm) | -0.016 | 0.779 | No |
| Wind speed (km/h) | +0.032 | 0.583 | No |

The signal does not improve after controlling for confounders. Weather has essentially **zero explanatory power** over mart-day prices in this dataset.

---

## Finding 2 — Good vs Bad Days Look Identical

Splitting mart-days into dry vs wet, and warm vs cold:

| Condition | Avg €/kg | n |
|---|---|---|
| Dry days (< 1mm rain) | €4.38 | 112 |
| Wet days (> 10mm rain) | €4.40 | 55 |
| **Difference** | **€0.02** | — |

t-test: t = -0.17, p = 0.86 — not significant.

| Condition | Avg €/kg | n |
|---|---|---|
| Warm days (top 25% temp) | €4.42 | 69 |
| Cold days (bottom 25% temp) | €4.53 | 70 |
| **Difference** | **-€0.11** | — |

t-test: t = -0.92, p = 0.36 — not significant. Interestingly, cold days have *slightly* higher prices — the opposite direction to the hypothesis — though not meaningful statistically.

---

## Finding 3 — Previous Day's Weather Also Has No Effect

Testing whether the day *before* a mart (when farmers decide whether to attend) predicts prices:

| Lagged Variable | r | p-value |
|---|---|---|
| Previous day rainfall | -0.023 | 0.714 |
| Previous day max temp | +0.058 | 0.345 |

No lag effect detected.

---

## Finding 4 — Volume Is More Important Than Weather

The one statistically significant finding: **lot count correlates positively with price** (r = +0.127, p = 0.028), and this holds after removing mart and month effects.

Days where more lots are sold at a mart tend to have slightly higher average prices. This is a more plausible driver than weather — well-attended sales likely attract more competitive buyers, driving bids up. Weather may matter indirectly only insofar as it affects attendance, but this analysis suggests attendance effects (if any) are not showing up in the current data.

---

## Finding 5 — Inconsistent Per-Mart Patterns

Individual mart correlations between rainfall and price (marts with n ≥ 5):

| Direction | Marts |
|---|---|
| Rain → lower price (negative r) | Raphoe\*, Ballinrobe, Headford, Balla, Roscrea, Elphin, Loughrea, Carnew, Athenry, Corrin |
| Rain → higher price (positive r) | Cashel\*, Mid Tipp, Ennis, Kilkenny, Castlerea, Ballinasloe, Mohill, Portumna, Carrigallen |

\* Only Raphoe (r = -0.87, n=6) and Cashel (r = +0.88, n=6) reach statistical significance — and they go in *opposite directions*. Both have very small sample sizes and the significance is not meaningful.

The lack of consistency in direction across marts is strong evidence that there is no underlying weather signal — the pattern looks like noise.

---

## Conclusions

1. **The hypothesis is not supported by current data.** Weather (temperature, rainfall, wind) has no detectable effect on mart-day average €/kg prices — raw or after controlling for mart and seasonal effects.

2. **The direction of effects is inconsistent.** Some marts show rain correlating with higher prices, some with lower. If weather mattered, we'd expect a consistent direction.

3. **Volume matters more than weather.** Days with more lots sold tend to have slightly higher prices. This suggests buyer competition and attendance drive prices more than conditions outside.

4. **The dataset is still young.** 298 mart-day observations over a 5-week window is a short run. With a full year of data, seasonal patterns will be clearer and any genuine weather effect (e.g. heavy snowfall preventing mart attendance) would have more opportunity to show up. Effects like heatwaves, frost, or flooding that are absent from this 5-week spring window could still be real.

---

## Recommendation

Continue scraping through summer and autumn. Re-run this analysis in September with 6+ months of data before drawing a final conclusion. The analysis is clean and ready to rerun — just re-execute the script on the updated CSVs. A genuine effect (if one exists) is most likely to appear around **extreme weather events** — storms, frost, prolonged drought — rather than the mild day-to-day variation seen in this window.
