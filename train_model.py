#!/usr/bin/env python3
"""
Cattle Price-per-kg Prediction — Model Training
=================================================
Trains a LightGBM regression model on sold lot data from martbids.ie.
Target: price_per_kg (€/kg).

Outputs:
  cattle_model.pkl          — trained model + preprocessing pipeline
  model_metadata.json       — metrics, feature importances, feature list
  model_test_predictions.csv
  shap_values.pkl           — SHAP values array (test set)
  shap_background.pkl       — 200-row background sample for SHAP
"""

import json
import shutil
import sys
import warnings
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

DIR           = Path(__file__).parent
CSV_PATH      = DIR / "sold_lots.csv"
LSL_CSV       = DIR / "lsl_lots.csv"
WEATHER_CSV   = DIR / "weather_cache.csv"
MODEL_PATH    = DIR / "cattle_model.pkl"
META_PATH     = DIR / "model_metadata.json"
SHAP_VAL_PATH = DIR / "shap_values.pkl"
SHAP_BG_PATH  = DIR / "shap_background.pkl"
PARAMS_PATH   = DIR / "best_params.json"
ARCHIVE_DIR   = DIR / "_model_archive"
SHAP_SAMPLE   = 6000          # rows of the holdout to explain


# ── Feature helpers ───────────────────────────────────────────────────────────

def parse_eur(s):
    """'€107' → 107.0,  '' / NaN → NaN"""
    if pd.isna(s) or str(s).strip() == "":
        return np.nan
    return pd.to_numeric(str(s).replace("€", "").replace(",", "").strip(),
                         errors="coerce")

def count_stars(s):
    """'☆☆☆☆' → 4"""
    if pd.isna(s) or str(s).strip() == "":
        return 0
    return str(s).count("☆") + str(s).count("★")

def export_score(s):
    """Yes→2, ReTest→1, No→0, unknown→NaN"""
    return {"Yes": 2, "ReTest": 1, "No": 0}.get(str(s).strip(), np.nan)

TOP_BREEDS     = 20
TOP_DAM_BREEDS = 15

# Sheep codes that appear in the cattle feed (Ennis lists ewes).
NON_CATTLE_BREEDS = {"EWE", "LB", "L1", "L2", "L3", "L4", "LAMB", "RAM", "HOG"}

def load_combined() -> pd.DataFrame:
    """Stack MartBids + Livestock-Live into one DataFrame."""
    mb = pd.read_csv(CSV_PATH)
    mb["source"] = "martbids"

    if not LSL_CSV.exists():
        return mb

    lsl = pd.read_csv(LSL_CSV)
    lsl["source"] = "lsl"
    # Normalise price to €-string so parse_eur handles both sources uniformly
    lsl["price"] = lsl["price"].apply(
        lambda x: f"€{x}" if pd.notna(x) else x
    )
    # Map numeric icbf_stars → ☆ symbols so count_stars works unchanged
    lsl["icbf_across_breed"] = lsl["icbf_stars"].apply(
        lambda s: "☆" * int(float(s)) if pd.notna(s) and float(s) > 0 else ""
    )
    # Use actual sale_date for seasonality (more accurate than scraped_date)
    lsl["scraped_date"] = lsl["sale_date"]

    return pd.concat([mb, lsl], ignore_index=True, sort=False)


def load_and_engineer(csv_path: Path = None, df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(csv_path)

    # ── Target ────────────────────────────────────────────────────────────────
    df["price_num"] = df["price"].apply(parse_eur)
    df["weight_num"] = pd.to_numeric(df["weight"], errors="coerce")
    # Compute price_per_kg; filter out implausible values
    df["ppkg"] = df["price_num"] / df["weight_num"].replace(0, np.nan)
    df = df[(df["ppkg"] >= 0.5) & (df["ppkg"] <= 20)].copy()
    df = df.dropna(subset=["ppkg"])

    # ── Numeric coercions ─────────────────────────────────────────────────────
    df["age_months"]   = pd.to_numeric(df["age_months"],   errors="coerce")
    df["days_in_herd"] = pd.to_numeric(df["days_in_herd"], errors="coerce")
    df["no_of_owners"] = pd.to_numeric(df["no_of_owners"], errors="coerce")
    df["weight"]       = df["weight_num"]

    # ── ICBF numeric ──────────────────────────────────────────────────────────
    df["icbf_cbv_num"]         = df["icbf_cbv"].apply(parse_eur)
    df["icbf_replacement_num"] = df["icbf_replacement_index"].apply(parse_eur)
    df["icbf_ebi_num"]         = df["icbf_ebi"].apply(parse_eur)
    df["icbf_stars"]           = df["icbf_across_breed"].apply(count_stars)

    # ── Binary / ordinal ──────────────────────────────────────────────────────
    df["has_genomic"]     = (df["icbf_genomic_eval"] == "Yes").astype(int)
    df["quality_assured"] = (df["quality_assurance"] == "Yes").astype(int)
    df["bvd_ok"]          = (df["bvd_tested"] == "Yes").astype(int)
    df["export_score"]    = df["export_status"].apply(export_score)

    # ── Drop non-cattle ───────────────────────────────────────────────────────
    # Some marts list sheep in the same feed. EWE lots run ~83 kg with no age
    # and no sex, and were being priced by a cattle model as a first-class breed.
    n_before = len(df)
    df = df[~df["breed"].isin(NON_CATTLE_BREEDS)].copy()
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df):,} non-cattle lots "
              f"({', '.join(sorted(NON_CATTLE_BREEDS))})")

    # ── Categorical cleaning ──────────────────────────────────────────────────
    top_breeds  = df["breed"].value_counts().head(TOP_BREEDS).index
    df["breed_grp"] = df["breed"].where(df["breed"].isin(top_breeds), "Other")

    if "dam_breed" in df.columns:
        top_dam = df["dam_breed"].value_counts().head(TOP_DAM_BREEDS).index
        df["dam_breed_grp"] = (df["dam_breed"]
                               .where(df["dam_breed"].isin(top_dam), "Other")
                               .fillna("Unknown"))
    else:
        df["dam_breed_grp"] = "Unknown"

    df["source"] = df["source"].fillna("martbids") if "source" in df.columns else "martbids"

    df["sex_clean"] = df["sex"].map({"M": "M", "F": "F", "B": "B"}).fillna("Unknown")

    # ── Breed × Sex interaction ───────────────────────────────────────────────
    df["breed_sex"] = df["breed_grp"] + "_" + df["sex_clean"]

    # ── Derived numeric features ───────────────────────────────────────────────
    df["log_weight"]       = np.log1p(df["weight"])
    df["weight_per_month"] = df["weight"] / df["age_months"].clip(lower=1)
    df["icbf_has_data"]    = (
        df["icbf_cbv_num"].notna() |
        df["icbf_replacement_num"].notna() |
        df["icbf_ebi_num"].notna()
    ).astype(int)

    # ── Sale date & seasonality ───────────────────────────────────────────────
    sale_dt = pd.to_datetime(df["scraped_date"], errors="coerce")
    df["sale_date"]   = sale_dt.dt.strftime("%Y-%m-%d")
    df["sale_dt"]     = sale_dt                   # kept for the temporal split
    df["sale_month"]  = sale_dt.dt.month          # 1–12, strong seasonal signal
    df["sale_month"]  = df["sale_month"].fillna(0).astype(int)

    # ── Grouping key: one auction ─────────────────────────────────────────────
    # A single sale contains ~150 lots that clear at very similar money, and a
    # third of MartBids rows share a base lot (35A/35B/35C) — the same pen, sold
    # seconds apart. Splitting a sale across train and test lets the model
    # memorise that sale's price level, which inflated the old random-split
    # score from R² 0.70 to 0.76. Never split a sale.
    if "sale_id" in df.columns:
        sale_id = df["sale_id"].astype(str)
    else:
        sale_id = pd.Series("", index=df.index)
    # LSL carries no sale_id, so its mart + date identifies the auction
    sale_id = sale_id.where(sale_id.notna() & (sale_id != "") & (sale_id != "nan"),
                            df["sale_date"].astype(str))
    df["sale_key"] = df["mart"].astype(str) + "|" + sale_id

    # ── Merge weather ─────────────────────────────────────────────────────────
    if WEATHER_CSV.exists():
        wx = pd.read_csv(WEATHER_CSV)
        wx["date"] = wx["date"].astype(str)
        df = df.merge(wx.rename(columns={"date": "sale_date"}),
                      on=["mart", "sale_date"], how="left")
    else:
        df["temp_max_c"]       = np.nan
        df["temp_min_c"]       = np.nan
        df["precipitation_mm"] = np.nan
        df["wind_speed_kmh"]   = np.nan

    return df


# ── Feature lists ─────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "weight", "log_weight", "weight_per_month",
    "age_months", "days_in_herd", "no_of_owners",
    "icbf_cbv_num", "icbf_replacement_num", "icbf_ebi_num", "icbf_stars",
    "icbf_has_data",
    "has_genomic", "quality_assured", "bvd_ok", "export_score",
    "temp_max_c", "temp_min_c", "precipitation_mm", "wind_speed_kmh",
    "sale_month",
]

CATEGORICAL_FEATURES = ["breed_grp", "sex_clean", "mart", "dam_breed_grp", "breed_sex", "source"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET       = "ppkg"

# After the ColumnTransformer the columns are [numeric…, categorical…], so the
# categoricals sit at these indices. LightGBM has to be TOLD they are
# categorical — otherwise it reads the ordinal codes as numbers and splits on
# ranges like `breed_grp <= 7.5`, which lumps breeds together alphabetically.
# Measured on the temporal holdout: native categorical MAE €0.3906 vs €0.3946
# for ordinal-as-numeric and €0.3959 for one-hot (one-hot is worse here — it
# fragments the trees across 150 sparse columns).
CAT_IDX = list(range(len(NUMERIC_FEATURES), len(ALL_FEATURES)))
FIT_KW  = {"model__categorical_feature": CAT_IDX}


# ── Default hyperparameters (conservative to avoid overfitting) ───────────────

DEFAULT_PARAMS = dict(
    n_estimators      = 1000,
    learning_rate     = 0.03,
    num_leaves        = 63,
    max_depth         = -1,
    min_child_samples = 25,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.2,
    reg_lambda        = 0.2,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)


# ── Preprocessing + model pipeline ────────────────────────────────────────────

def build_pipeline(params: dict = None):
    if params is None:
        if PARAMS_PATH.exists():
            with open(PARAMS_PATH) as f:
                params = json.load(f)
            print(f"  Using saved Optuna params from {PARAMS_PATH.name}")
        else:
            params = DEFAULT_PARAMS

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("impute",  SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode",  OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")

    model = lgb.LGBMRegressor(**params)

    return Pipeline([
        ("prep",  preprocessor),
        ("model", model),
    ])


# ── Optuna hyperparameter search ──────────────────────────────────────────────

def tune_hyperparams(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                     n_trials: int = 80) -> dict:
    """
    Search LightGBM hyperparameters with Optuna. Returns best param dict.

    Scored with GroupKFold on the sale key, not KFold(shuffle=True) — tuning
    against a leaky score optimises for memorising sale price levels.
    """

    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 300, 1500),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 15, 127),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 80),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.0, 2.0),
            max_depth         = -1,
            random_state      = 42,
            n_jobs            = -1,
            verbose           = -1,
        )
        # Folds are run explicitly rather than through cross_val_score:
        # routing a fit param (categorical_feature) through it needs sklearn's
        # metadata routing switched on, which is easy to forget and fails loudly
        # only when someone finally runs --tune.
        maes = []
        for tr_i, te_i in GroupKFold(n_splits=5).split(X, y, groups):
            pipe = build_pipeline(params)
            pipe.fit(X.iloc[tr_i], y.iloc[tr_i], **FIT_KW)
            maes.append(mean_absolute_error(y.iloc[te_i], pipe.predict(X.iloc[te_i])))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({"max_depth": -1, "random_state": 42, "n_jobs": -1, "verbose": -1})
    with open(PARAMS_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\n  Best CV MAE: €{study.best_value:.4f}/kg")
    print(f"  Best params saved → {PARAMS_PATH.name}")
    return best


# ── Splits ────────────────────────────────────────────────────────────────────
#
# Two questions, two splits, both grouped so a sale never straddles the boundary:
#
#   temporal  — "how well does this price a week I have never seen?"  The number
#               that matters for the underwrite tool, which projects forward.
#   grouped   — "how well does this price an unseen sale from a period I know?"
#               The number that matters for herd valuation, which prices today.
#
# The old code used a plain random split, which answers neither: every test lot
# came from a sale that was also in training.

HOLDOUT_WEEKS = 6


def temporal_split(df: pd.DataFrame, weeks: int = HOLDOUT_WEEKS):
    """Hold out the most recent `weeks` of trading. Whole sales only."""
    cutoff = df["sale_dt"].max() - pd.Timedelta(weeks=weeks)
    test = df["sale_dt"] > cutoff
    return ~test, test, cutoff


def grouped_split(df: pd.DataFrame, frac: float = 0.2, seed: int = 42):
    """Random holdout of whole sales, so no sale appears on both sides."""
    sales = df["sale_key"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(sales)
    holdout = set(sales[: max(1, int(len(sales) * frac))])
    test = df["sale_key"].isin(holdout)
    return ~test, test


def rolling_origin_cv(X, y, dates, n_splits: int = 4, weeks: int = 3):
    """
    Expanding-window CV: train on everything before a cutoff, test on the next
    `weeks`, walk the cutoff forward.

    Replaces KFold(shuffle=True), which on time-series data trains on the future
    to predict the past and reports a score the model cannot reproduce in use.
    """
    end = dates.max()
    maes = []
    for i in range(n_splits, 0, -1):
        cut = end - pd.Timedelta(weeks=weeks * i)
        tr = dates <= cut
        te = (dates > cut) & (dates <= cut + pd.Timedelta(weeks=weeks))
        if tr.sum() < 5000 or te.sum() < 500:
            continue
        pipe = build_pipeline()
        pipe.fit(X[tr], y[tr], **FIT_KW)
        mae = mean_absolute_error(y[te], pipe.predict(X[te]))
        maes.append(mae)
        print(f"    fold to {cut.date()}:  train {tr.sum():>7,}  "
              f"test {te.sum():>6,}  MAE €{mae:.4f}")
    return maes


# ── Metrics ───────────────────────────────────────────────────────────────────

def cohort_metrics(df_test: pd.DataFrame, y_true, y_pred, min_n: int = 100) -> dict:
    """
    Error broken out by cohort, so a good national average cannot hide a
    segment the model prices badly. This is what the dashboards need in order
    to show a confidence hint next to a prediction.
    """
    work = df_test.copy()
    work["_abs_err"] = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    work["_ape"] = work["_abs_err"] / np.asarray(y_true).clip(min=0.01) * 100
    work["_wband"] = pd.cut(work["weight"], [0, 200, 300, 400, 500, 650, 9999],
                            labels=["<200", "200-300", "300-400",
                                    "400-500", "500-650", ">650"], right=False)

    out = {}
    for name, keys in [("by_breed", ["breed_grp"]),
                       ("by_sex", ["sex_clean"]),
                       ("by_weight_band", ["_wband"]),
                       ("by_source", ["source"]),
                       ("by_breed_sex_weight", ["breed_grp", "sex_clean", "_wband"])]:
        rows = {}
        for key, g in work.groupby(keys, observed=True):
            if len(g) < min_n:
                continue
            label = " · ".join(str(k) for k in key) if isinstance(key, tuple) else str(key)
            rows[label] = {
                "n": int(len(g)),
                "MAE_eur_kg": round(float(g["_abs_err"].mean()), 4),
                "MAPE_%": round(float(g["_ape"].mean()), 2),
            }
        out[name] = dict(sorted(rows.items(), key=lambda kv: -kv[1]["MAE_eur_kg"]))
    return out


def compute_metrics(y_true, y_pred, label=""):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true.clip(lower=0.01))) * 100
    within = {
        f"within_{t}pct": float(np.mean(np.abs((y_true - y_pred) / y_true.clip(lower=0.01)) <= t / 100))
        for t in [5, 10, 20]
    }
    d = {
        "MAE_eur_kg":  round(mae, 4),
        "RMSE_eur_kg": round(rmse, 4),
        "R2":          round(r2, 4),
        "MAPE_%":      round(mape, 2),
        **{k: round(v * 100, 1) for k, v in within.items()},
    }
    if label:
        print(f"\n  {'─'*40}")
        print(f"  {label}")
        for k, v in d.items():
            print(f"    {k:25s}: {v}")
    return d


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tune = "--tune" in sys.argv

    print("Loading data…")
    df = load_and_engineer(df=load_combined())
    print(f"  {len(df):,} rows after cleaning")
    if "source" in df.columns:
        for src, cnt in df["source"].value_counts().items():
            print(f"    {src}: {cnt:,}")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # ── Optuna tuning (optional) ───────────────────────────────────────────────
    if tune:
        print("\nRunning Optuna search (80 trials, grouped CV)…")
        best_params = tune_hyperparams(X, y, df["sale_key"], n_trials=80)
    else:
        best_params = None   # build_pipeline will load saved or use defaults

    # ── Primary split: grouped temporal ───────────────────────────────────────
    tr_mask, te_mask, cutoff = temporal_split(df)
    X_train, X_test = X[tr_mask], X[te_mask]
    y_train, y_test = y[tr_mask], y[te_mask]
    print(f"\nTemporal split — holding out the last {HOLDOUT_WEEKS} weeks "
          f"(sales after {cutoff.date()})")
    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"  Sales in both sides: "
          f"{len(set(df.loc[tr_mask,'sale_key']) & set(df.loc[te_mask,'sale_key']))} "
          f"(must be 0)")

    # ── Fit pipeline ──────────────────────────────────────────────────────────
    print("\nTraining LightGBM pipeline…")
    pipeline = build_pipeline(best_params)
    pipeline.fit(X_train, y_train, **FIT_KW)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    train_metrics = compute_metrics(y_train, pipeline.predict(X_train), "Train metrics")
    test_pred     = pipeline.predict(X_test)
    test_metrics  = compute_metrics(y_test, test_pred,
                                    f"Test metrics — TEMPORAL (last {HOLDOUT_WEEKS} wks)")

    # ── Secondary split: grouped random (the "price today" regime) ────────────
    gtr, gte = grouped_split(df)
    g_pipe = build_pipeline(best_params)
    g_pipe.fit(X[gtr], y[gtr], **FIT_KW)
    grouped_metrics = compute_metrics(y[gte], g_pipe.predict(X[gte]),
                                      "Test metrics — GROUPED RANDOM (unseen sales)")

    # ── Per-cohort error ──────────────────────────────────────────────────────
    print("\n  Worst cohorts by MAE (temporal holdout, n>=100):")
    cohorts = cohort_metrics(df[te_mask], y_test, test_pred)
    for label, m in list(cohorts["by_breed_sex_weight"].items())[:6]:
        print(f"    {label:44s} n={m['n']:>5,}  MAE €{m['MAE_eur_kg']:.3f}  "
              f"MAPE {m['MAPE_%']:.1f}%")
    print("  Best cohorts:")
    for label, m in list(cohorts["by_breed_sex_weight"].items())[-3:]:
        print(f"    {label:44s} n={m['n']:>5,}  MAE €{m['MAE_eur_kg']:.3f}  "
              f"MAPE {m['MAPE_%']:.1f}%")

    # ── Rolling-origin CV ─────────────────────────────────────────────────────
    print("\n  Rolling-origin cross-validation (expanding window):")
    cv_maes = rolling_origin_cv(X, y, df["sale_dt"])
    cv_mae = float(np.mean(cv_maes)) if cv_maes else float("nan")
    cv_std = float(np.std(cv_maes)) if cv_maes else float("nan")
    print(f"    mean MAE €{cv_mae:.4f}/kg ± €{cv_std:.4f}")

    # ── Is the new model actually better? ─────────────────────────────────────
    # Compare like-for-like: score the CURRENT live model on this same holdout.
    # The old metadata's numbers came from a leaky random split and are not
    # comparable to anything here.
    # The deployed model is always refit on ALL data, so once a model has been
    # promoted from this script it has the holdout inside its own training set.
    # Scoring that pickle on the holdout returns an in-sample number (~€0.29 vs
    # a true ~€0.39) and would block every future promotion forever. Only score
    # the pickle when it genuinely predates the holdout; otherwise compare the
    # previous run's *recorded* holdout metric, which is like-for-like.
    incumbent_mae, incumbent_basis = None, "none"
    prev_meta = {}
    if META_PATH.exists():
        try:
            prev_meta = json.loads(META_PATH.read_text())
        except ValueError:
            pass

    trained_through = (prev_meta.get("date_range") or [None, None])[1]
    saw_holdout = trained_through is not None and str(trained_through) > str(cutoff.date())

    if saw_holdout:
        incumbent_mae = (prev_meta.get("test_metrics") or {}).get("MAE_eur_kg")
        incumbent_basis = "previous run's recorded holdout MAE"
        if incumbent_mae is not None:
            print(f"\n  Incumbent was trained through {trained_through}, which includes this "
                  f"holdout — comparing recorded metrics instead of re-scoring it.")
    elif MODEL_PATH.exists():
        try:
            old = joblib.load(MODEL_PATH)
            incumbent_mae = float(mean_absolute_error(y_test, old.predict(X_test)))
            incumbent_basis = "incumbent scored on this holdout"
        except Exception as exc:
            print(f"\n  [WARN] Could not score the incumbent model: {exc}")

    if incumbent_mae is not None:
        delta = test_metrics["MAE_eur_kg"] - incumbent_mae
        print(f"\n  Incumbent ({incumbent_basis}): MAE €{incumbent_mae:.4f}")
        print(f"  New model:                     MAE €{test_metrics['MAE_eur_kg']:.4f}"
              f"  ({delta:+.4f})")

    # ── Feature importances ───────────────────────────────────────────────────
    lgb_model = pipeline.named_steps["model"]
    importances = pd.Series(
        lgb_model.feature_importances_,
        index=ALL_FEATURES,
    ).sort_values(ascending=False)

    print("\n  Top 10 feature importances:")
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp / importances.max() * 20)
        print(f"    {feat:25s} {bar} {imp:,}")

    # ── SHAP values (sample of the holdout) ───────────────────────────────────
    # Wrapped: nothing in the dashboards currently reads these artifacts, so an
    # explainability failure must never throw away a model that trained fine.
    # A previous run died here after 12 minutes of good work.
    print("\nComputing SHAP values…")
    try:
        prep      = pipeline.named_steps["prep"]
        X_test_t  = prep.transform(X_test)
        X_train_t = prep.transform(X_train)

        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_train_t), size=min(200, len(X_train_t)), replace=False)
        X_bg = X_train_t[bg_idx]

        # Sample the holdout — it is ~30k rows now, and the pickle scales with it.
        n_shap = min(SHAP_SAMPLE, len(X_test_t))
        s_idx  = rng.choice(len(X_test_t), size=n_shap, replace=False)

        # Path-dependent TreeExplainer (no `data=`). The interventional variant
        # this used to pass a background set to fails LightGBM's additivity
        # check — the SHAP values did not sum to the model output.
        explainer   = shap.TreeExplainer(lgb_model)
        shap_values = explainer(X_test_t[s_idx])

        joblib.dump(shap_values, SHAP_VAL_PATH)
        joblib.dump(X_bg,        SHAP_BG_PATH)
        print(f"  SHAP values saved → {SHAP_VAL_PATH.name} ({n_shap:,} rows)")
        print(f"  SHAP background saved → {SHAP_BG_PATH.name}")
    except Exception as exc:
        print(f"  [WARN] SHAP step failed, continuing: {exc}")

    # ── Deploy, but only if the new model actually wins ───────────────────────
    # Refit on everything including the holdout: the shipped model should see
    # the most recent market. The holdout was for measurement, not for saving.
    print("\nRefitting on full dataset for deployment…")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y, **FIT_KW)

    TOLERANCE = 0.005      # €/kg — ignore noise-level differences
    promote = True
    if incumbent_mae is not None and test_metrics["MAE_eur_kg"] > incumbent_mae + TOLERANCE:
        promote = False

    if promote:
        if MODEL_PATH.exists():
            ARCHIVE_DIR.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            shutil.copy2(MODEL_PATH, ARCHIVE_DIR / f"cattle_model_{stamp}.pkl")
            if META_PATH.exists():
                shutil.copy2(META_PATH, ARCHIVE_DIR / f"model_metadata_{stamp}.json")
            print(f"  Previous model archived → {ARCHIVE_DIR.name}/cattle_model_{stamp}.pkl")
        joblib.dump(final_pipeline, MODEL_PATH)
        print(f"  Model saved → {MODEL_PATH.name}")
    else:
        cand = DIR / "cattle_model_candidate.pkl"
        joblib.dump(final_pipeline, cand)
        print(f"  NOT PROMOTED — the new model is worse on the holdout "
              f"(€{test_metrics['MAE_eur_kg']:.4f} vs €{incumbent_mae:.4f}).")
        print(f"  Live model left untouched. Candidate saved → {cand.name}")

    # ── Save test predictions ─────────────────────────────────────────────────
    test_preds = pd.DataFrame({
        "actual_ppkg":    y_test.values,
        "predicted_ppkg": pipeline.predict(X_test),
        "actual_eur":     (y_test * X_test["weight"]).values,
        "predicted_eur":  pipeline.predict(X_test) * X_test["weight"].values,
        "breed":          X_test["breed_grp"].values,
        "mart":           X_test["mart"].values,
        "sex":            X_test["sex_clean"].values,
        "weight":         X_test["weight"].values,
    })
    test_preds.to_csv(DIR / "model_test_predictions.csv", index=False)

    meta = {
        "features":              ALL_FEATURES,
        # The dashboards read this so they group breeds exactly as the model does.
        "breed_levels":          sorted(df["breed_grp"].unique().tolist()),
        "categorical_encoding":  "LightGBM native (ordinal codes declared categorical)",
        "numeric_features":      NUMERIC_FEATURES,
        "categorical_features":  CATEGORICAL_FEATURES,
        "target":                TARGET,
        "trained_at":            datetime.now().isoformat(timespec="seconds"),
        "n_rows_total":          int(len(df)),
        "date_range":            [str(df["sale_dt"].min().date()),
                                  str(df["sale_dt"].max().date())],
        # How the model was evaluated. The previous version used a random split
        # in which 100% of test lots came from a sale that was also in training;
        # its scores are not comparable to these.
        "evaluation": {
            "primary":     f"grouped temporal — holdout = last {HOLDOUT_WEEKS} weeks",
            "secondary":   "grouped random — unseen whole sales",
            "cv":          "rolling-origin, expanding window",
            "grouping_key": "mart | sale_id",
            "holdout_cutoff": str(cutoff.date()),
        },
        "train_metrics":         train_metrics,
        "test_metrics":          test_metrics,            # temporal (primary)
        "test_metrics_grouped":  grouped_metrics,         # unseen sales
        "cohort_metrics":        cohorts,
        "cv_mae_eur_kg":         round(cv_mae, 4),
        "cv_mae_std":            round(cv_std, 4),
        "cv_fold_maes":          [round(m, 4) for m in cv_maes],
        "incumbent_mae": (round(incumbent_mae, 4) if incumbent_mae is not None else None),
        "incumbent_basis": incumbent_basis,
        "promoted":              bool(promote),
        "n_train":               int(len(X_train)),
        "n_test":                int(len(X_test)),
        "feature_importances":   importances.round(1).to_dict(),
    }
    # Metadata must describe the model that is actually deployed. Writing the
    # candidate's metadata over the live file while leaving the old pickle in
    # place desyncs them — and the dashboards read `breed_levels` from here, so
    # they would group breeds one way while the live model expects another.
    meta_target = META_PATH if promote else DIR / "model_metadata_candidate.json"
    with open(meta_target, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {meta_target.name}")
    print(f"\nDone. TEMPORAL holdout — R²={test_metrics['R2']:.4f}  "
          f"MAE=€{test_metrics['MAE_eur_kg']:.4f}/kg  |  promoted={promote}")


if __name__ == "__main__":
    main()
