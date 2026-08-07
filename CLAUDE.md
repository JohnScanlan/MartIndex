# MartIndex — Irish cattle market intelligence

Collects Irish mart and factory cattle prices, serves them through two Streamlit
dashboards and a nightly emailed PDF, and trains a LightGBM €/kg price model.

Everything is flat files in this one directory. There is no database. Each
consumer re-reads the CSVs from disk, so there is no shared canonical dataset —
the same filter written two ways can give two answers.

## Pipeline

Four independent lanes, one source → one collector → one file:

| Source | Collector | Output |
|---|---|---|
| martbids.ie (REST, token auth) | `martbids_scraper.py` | `sold_lots.csv` |
| livestock-live.com (public HTML) | `lsl_scraper.py` | `lsl_lots.csv` |
| Open-Meteo archive (no key) | `fetch_weather.py` | `weather_cache.csv` |
| DAFM BeefPriceWatch (REST) | `scrape_factory_prices.py` | `factory_prices.csv` |

Derived: `prepare_factory_prices.py` → `factory_prices_clean.csv` (+ `.parquet`);
`train_model.py` → `cattle_model.pkl`, `model_metadata.json`, `shap_*.pkl`.

Consumers: `dashboard.py` (MartIndex, farmer-facing), `bank_dashboard.py`
(lending, has a `region` dimension), `generate_report.py` (4-page PDF → 4
recipients), `git_push.py` (→ GitHub → Streamlit Cloud), `backup_to_gdrive.sh`
(→ Drive, dated folders).

## Scheduling — launchd only, never cron

`run_scraper.sh` in this repo is the single source of truth for the nightly
chain. `~/Library/Scripts/MartBids/run_scraper.sh` is a one-line shim that
`exec`s it; same for `backup_to_gdrive.sh`. Do not put logic in the shims.

**Two rules that cost weeks of data when broken:**

1. **Use launchd, not cron.** A `StartCalendarInterval` job missed while the Mac
   sleeps runs on wake. A cron entry at the same time is simply skipped and the
   run is gone. Morning cron jobs here silently missed weeks — the factory feed
   fired 3 times in 3 months. All cron entries were removed 7 Aug 2026.
2. **Scripts launchd runs must live in `~/Library`.** launchd's `/bin/bash`
   cannot read `~/Documents` under Full Disk Access, which is what made the
   backup job exit 78 for weeks. Python *can* read it — that's why `git_push.py`
   exists instead of inlining git in the shell script. Hence the shim pattern.

Jobs: `com.johnscanlan.martbids` (daily 21:30, full chain),
`com.johnscanlan.martbids.backup` (Sundays 22:00). Log: `/tmp/martbids_run.log`.

Chain order matters — `fetch_weather.py` reads the mart × date pairs the two lot
scrapers just wrote, so it must follow them. `prepare_factory_prices.py` must
follow `scrape_factory_prices.py`.

## Data safety

Never overwrite a data file in place. Appends go through
`data_utils.safe_append_csv`: build combined frame → write temp in same dir →
re-read and verify row count → atomic rename. `csv_to_parquet` does the same.
A crashed run therefore adds nothing rather than corrupting anything.

Dedup keys make every scraper safe to re-run: `val_code` for both lot sources,
`(source, report_date, category, grade, factory)` for factory prices,
`(mart, date)` for weather. Re-running any scraper is free.

Pre-change snapshots live in `_pre_fix_backup_*/` (gitignored).

## Secrets — none belong in tracked files

| What | Where |
|---|---|
| MartBids login | `$MARTBIDS_EMAIL`/`$MARTBIDS_PASSWORD`, else `martbids_config.json` |
| GitHub PAT | `$GITHUB_TOKEN`, else `~/Library/Scripts/MartBids/.github_token` |
| Gmail SMTP | `$SMTP_USER`/`$SMTP_PASS`, else `email_config.json` |

All three config files are gitignored. `martbids_scraper.py` previously carried
the password as a literal and was pushed to GitHub — treat that credential as
exposed in git history.

## Gotchas

- **Livestock-Live mart names.** Every catalogue page's `<h1>` is the site name
  ("LSL Auctions"), not the mart's. Names must come from the locations page,
  where each `.locationitem` carries `location` and `country` attributes. Get
  this wrong and all 18 marts collapse into one, which silently drops them from
  every regional breakdown and the weather join.
- **LSL uses its own mart naming** ("Gortatlea Mart", not "Gortatlea"), so it
  needs separate lookups — `LSL_MART_COORDS` / `LSL_MART_REGIONS` in
  `mart_coords.py`. Both dashboards and the report import `LSL_MART_REGIONS`;
  add new LSL marts there, not in the per-file `REGION_MAP`s.
- **The site rate-limits bursts.** `lsl_scraper.py` runs 4 workers with
  exponential backoff on 429. A 429 without retry looks exactly like "no sale".
- **Heredoc `__file__` is `<stdin>`.** In `python3 - <<EOF`, `Path(__file__).parent`
  silently resolves to the cwd. Pass paths as `sys.argv` instead.
- **`sold_lots.csv` has no sale date** — `scraped_date` is used as the sale date.
  `lsl_lots.csv` does have a real `sale_date`.
- **The model is not retrained automatically.** Nothing schedules
  `train_model.py`; check `model_metadata.json` for when it last ran.

## Commands

```bash
.venv/bin/python martbids_scraper.py          # --reset wipes and re-scrapes all
.venv/bin/python lsl_scraper.py
.venv/bin/python scrape_factory_prices.py && .venv/bin/python prepare_factory_prices.py
.venv/bin/python generate_report.py           # builds + emails the PDF
.venv/bin/streamlit run dashboard.py
.venv/bin/streamlit run bank_dashboard.py
/bin/bash run_scraper.sh                      # the whole nightly chain
```

## Open

- MartBids password needs rotating (exposed in git history).
- Model last trained 17 Apr 2026 on 54k lots; ~241k are now available.

## Weight at a future age — why it is a lookup, not a model

`agri_credit.weight_at_age()` / `weight_trajectory()`, surfaced as the
"Value Trajectory" tab.

**There is no longitudinal data and there cannot be a validated growth model.**
This was checked properly, not assumed: 92% of `sold_lots.csv` rows carry a date
of birth, and 17,454 `(dob, breed, sex)` keys recur across different sale dates.
They are not the same animals. ~35% of the implied growth rates are negative and
18% exceed 2 kg/day, and — the decisive test — **tightening the match does not
reduce the negative rate at all** (35.5% → 35.3% after adding dam breed, an ICBF
fingerprint and an owner-count check). A rate that will not respond to a stricter
key is random pairing: siblings and contemporaries sharing a birth date. 18% of
"repeat" animals even show their owner count *decreasing*, which is impossible.

So the tool answers the question the data can answer — *what do animals of this
breed and sex weigh at that age?* — which is checkable on held-out sales, and
returns a distribution rather than a false point estimate.

Held-out accuracy, ages ≤40 months, measured two ways — the difference matters:

| Holdout | MAE | Bias | In P25–P75 |
|---|---|---|---|
| **Grouped, no time gap** (production condition) | **55 kg** | **+0 kg** | 54% |
| Temporal, 6-week gap | 67 kg | −31 kg | 48% |

Production reads contemporaneous sales, so **±55 kg unbiased is the real
number**. The second row is not a worse measurement of the same thing — it is a
measurement of a different thing, and a useful one: **if the scrapers stall for
six weeks, weight estimates drift about 31 kg light.** Worth remembering the next
time a feed goes quiet.

Calibration is the point of the whole design: at any age the middle half of
animals spans 110–130 kg, so a point estimate is meaningless. 54% coverage against
an ideal 50% means the band is very slightly wide, which errs the safe way for
lending.

Two deliberate limits:

- **Current weight is optional.** Without it you get the pure cohort range, which
  is fully observed. With it, the animal's deviation from its cohort is carried
  forward decaying at `ANCHOR_RETENTION_PER_YEAR` (0.5/yr). That decay cannot be
  measured without repeat observations — it is an assumption, exposed as a slider
  and labelled as one. It sits between two known failure modes: full persistence
  (a heavy animal stays heavy forever) and full reversion, which is what the old
  sqrt curve did and which ran +77 kg on light animals and −107 kg on heavy ones.
- **No price forecast.** Values use *today's* €/kg for whichever weight band the
  animal reaches. 21 weeks of price history cannot support a 12–24 month price
  projection, and the binding constraint on value forecasting is the price side,
  not the weight side.

`MAX_AGE_MONTHS = 40` — comparables thin out badly past that.

### The selection effect, and why the tool declares it

The cohort curve **understates individual growth**, and this is not fixable with
this data. Implied daily gain over 8→14 months:

| Breed (male) | 8→14 mo | 14→20 mo |
|---|---|---|
| AAX | 0.25 kg/day | 0.57 |
| LMX | 0.25 | 0.66 |
| CHX | 0.27 | 0.77 |
| FR | 0.76 | 0.45 |

Real Irish cattle run 0.4–1.4 kg/day, so the early figures are too low. The
cause is selection: animals sold at 8 months and animals sold at 20 months are
different populations, not the same animals grown up — farmers move lighter
stores on early and keep the thrivers, which flattens the cross-sectional slope.

Both tools therefore **flag it rather than hide it**: `adg_is_plausible()` /
`PLAUSIBLE_ADG` in `agri_credit`, and the Growth Calculator shows a callout when
the implied gain falls outside the biological range. Treat the weight line as
conservative in the store-age bands.

### Column-name compatibility

`bank_dashboard.py` calls the breed column `breed_group` and encodes sex as
`Male/Female/Bull`; `dashboard.py` uses `breed_grp` and `M/F/B`. The shared
helpers resolve the column via `_breed_col()` and take whatever sex encoding the
caller's own frame uses — do not translate sex at the call site, or the lookup
silently matches nothing and falls down the ladder.

## Model evaluation

`train_model.py`. The one rule: **never split a sale across train and test.**

A sale is ~150 lots clearing at very similar money, and a third of MartBids rows
share a base lot (35A/35B/35C — the same pen, sold seconds apart). The original
`train_test_split(random_state=42)` put **100% of test lots in a sale that was
also in training**, which inflated R² from 0.70 to 0.76. Everything is grouped on
`sale_key` = `mart | sale_id` now (LSL has no sale_id, so it uses mart | date).

Three numbers, because they answer different questions:

| Metric | Question | Used by |
|---|---|---|
| **temporal** (primary) — holdout is the last 6 weeks | how well does it price a week it has never seen? | the underwrite tool, which projects forward |
| **grouped random** — unseen whole sales | how well does it price an unseen sale from a known period? | herd valuation, which prices today |
| **rolling-origin CV** — expanding window | is that stable across cutoffs? | sanity |

Read **MAE, not R²**, when comparing across time windows: R² is flattered or
punished by whatever price variance the test window happens to have, and this
market's variance moves a lot. A temporal window with compressed prices can show
a *better* MAE and a much worse R² without the model changing at all.

`cohort_metrics()` breaks error down by breed × sex × weight band, so a good
national average cannot hide a segment priced badly.

**Promotion is guarded.** A new model is scored against the *incumbent on the
same holdout* (the old metadata's numbers came from a leaky split and compare to
nothing). It only replaces `cattle_model.pkl` if it wins; otherwise it lands as
`cattle_model_candidate.pkl` and the live model is untouched. The previous model
and metadata are archived to `_model_archive/` on every promotion.

### Breeds and categorical encoding

Breeds are the **raw mart codes** (LMX, AAX, CHX …), top-20 by frequency plus
`"Other"`. The dashboards read that vocabulary from `model_metadata.json`
(`breed_levels`) via `agri_credit.breed_levels()`, so they and the model always
group breeds identically. `bank_dashboard.py` previously used its own rolled-up
`BREED_GROUP_MAP` ("Continental ×", "Angus ×") — a cohort price and a model
prediction were describing different populations.

Categorical features are **LightGBM-native**: ordinal codes declared via
`categorical_feature` (see `CAT_IDX` / `FIT_KW`). Measured on the temporal
holdout, same params:

| Encoding | MAE €/kg | Columns |
|---|---|---|
| **native categorical** | **0.3906** | 26 |
| ordinal read as numeric (the old behaviour) | 0.3946 | 26 |
| one-hot | 0.3959 | 150 |

One-hot is *worse* than doing nothing here — it fragments trees across sparse
columns, while LightGBM's native handling partitions categories optimally. Every
`.fit()` must pass `**FIT_KW`; miss one and that model silently reverts to
reading breed codes as numbers.

`NON_CATTLE_BREEDS` drops sheep (EWE and friends) from the cattle feed — Ennis
lists ewes at ~83 kg, and they were being priced as a first-class cattle breed.

Stratifying the split by breed/age/sex was considered and rejected: with 183k
rows a random split already balances those to within 0.4 pp, and the 85 rare
breeds collapse into `"Other"` before the model sees them. Grouping was the
problem, not balance. Stratified *reporting* is what that instinct wanted.

## Chart theme

`viz_theme.py` holds the palette, the Plotly template and the render helper for
**both** dashboards. Two rules, both learned the hard way:

1. **Render through `viz_theme.show_chart()`, never `st.plotly_chart` directly.**
   Streamlit applies its own Plotly theme by default and silently discards the
   figure's template — which left every axis label at `#808495` (3.7:1 on white)
   in both apps. `show_chart` passes `theme=None`.
2. **Put styling in the template, not in a layout dict.** Returning `xaxis=`,
   `legend=` or `margin=` from a layout helper collides with the call sites that
   already pass those — a `TypeError` that blanks the tab, not a graceful
   fallback. A template merges.

Also: brand `GOLD` (#C9A84C) is 2.29:1 on white — fine as an accent on the navy
sidebar, invisible as a line or bar. Charts use `GOLD_MARK` (#B08A1E, 3.23:1).
Heatmap cell labels are **annotations**, not `texttemplate`, because Plotly takes
only one text colour per trace; `cell_ink()` / `diverging_cell_ink()` pick per
cell. Sequential ramps are single-hue; the old navy→gold→green scale had no
perceptual order and no legible text colour.

`.streamlit/config.toml` sets `primaryColor` so Streamlit's default salmon
(#FF4B4B, 2.75:1 behind slider and multiselect labels) is fixed at source for
both apps rather than chased with CSS.

Verify with a browser contrast audit rather than by eye — the failures were
almost all in places the stylesheet looked fine. Both apps currently pass at
0 exceptions / 0 clipped labels / 0 contrast failures across 14 tabs.

## Credit tools

`agri_credit.py` — pure functions, no Streamlit, so the maths is testable on its
own. Cohort pricing with a fallback ladder, sqrt growth curves, herd valuation,
deal underwriting. `herd_store.py` persists herds and valuation history.
Surfaced as the "Herd Valuation" and "Quick Underwrite" tabs in
`bank_dashboard.py`. Full reasoning in `docs/agri-credit-spec.md`.

Three rules that must not be broken:

1. **`price_cohort()` is the only source of €/kg.** A herd valuation and an
   underwrite must never disagree about what the same animal is worth.
2. **Cohort constants live in `agri_credit.py`** (`WEIGHT_BINS`,
   `BREED_GROUP_MAP`, …) and `bank_dashboard.py` imports them. If these were
   duplicated and drifted, the same animal would be banded two ways.
3. **The credit tabs read `df_raw`, not the filtered `df`.** A sidebar region or
   date filter set for browsing must not silently change a customer's valuation.

Two things the tools deliberately refuse to do, both for honesty reasons:

- **No seasonal price forecast.** 21 weeks of history is enough to measure a
  cohort's current momentum, nowhere near enough to project a sale 6–12 months
  out. The exit is priced at today's rate for the destination cohort, and
  scenario columns carry the banker's own view. Revisit at ≥2 years of history.
- **Exit is priced in the *destination* cohort, not the purchase cohort.** €/kg
  falls steeply with weight (Continental × runs 5.51 → 3.72 from <200 kg to
  >650 kg). Pricing the sale at the entry rate overstates proceeds by roughly a
  third and turns losing deals into apparently good ones.

Note the fitted growth curve is cross-sectional — it compares what a 9-month
animal weighs against a 17-month animal, across different animals. That is a
fair population average but understates a well-fed individual, so the underwrite
tab offers a daily-gain override.

`herds.csv` and `herd_valuation_history.csv` hold customer and loan data and are
gitignored. The registry is rewritten atomically with a backup into
`_herd_backups/`; the history is append-only via `safe_append_csv`.

## Diagrams

`docs/pipeline-data-in.png` and `docs/pipeline-data-out.png` — the current
architecture: sources → collector scripts → files → derive steps, and files →
consumer scripts → destinations. Row counts are as of 7 Aug 2026.

Regenerate after editing `docs/diagram-source.html` (hand-written SVG):

```bash
.venv/bin/python docs/render_diagrams.py
```

`docs/2026-08-07-system-audit.html` — an **as-found snapshot**, kept as a record
of the nine problems found on 7 Aug 2026; eight were fixed the same day (see
History), so its findings and status chips are deliberately historical. Also at
https://claude.ai/code/artifact/914bd272-9be5-4f7d-bd0d-b1c881c77ced

## History

- **7 Aug 2026** — GitHub push restored after a new PAT; the backlog was pushed
  and the remote's `.devcontainer` commit merged in. Note the push had *two*
  causes of failure stacked: an expired token, and then a non-fast-forward
  because a commit had been made on GitHub directly. `git_push.py` now
  distinguishes the two and auto-merges the second case, aborting cleanly if the
  merge conflicts.
