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
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pathlib import Path

warnings.filterwarnings("ignore")

DIR        = Path(__file__).parent
CSV_PATH   = DIR / "sold_lots.csv"
WEATHER_CSV = DIR / "weather_cache.csv"
MODEL_PATH = DIR / "cattle_model.pkl"
META_PATH  = DIR / "model_metadata.json"
SHAP_VAL_PATH = DIR / "shap_values.pkl"
SHAP_BG_PATH  = DIR / "shap_background.pkl"


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

def load_and_engineer(csv_path: Path) -> pd.DataFrame:
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

    # ── Categorical cleaning ──────────────────────────────────────────────────
    top_breeds  = df["breed"].value_counts().head(TOP_BREEDS).index
    df["breed_grp"] = df["breed"].where(df["breed"].isin(top_breeds), "Other")

    top_dam = df["dam_breed"].value_counts().head(TOP_DAM_BREEDS).index
    df["dam_breed_grp"] = (df["dam_breed"]
                           .where(df["dam_breed"].isin(top_dam), "Other")
                           .fillna("Unknown"))

    df["sex_clean"] = df["sex"].map({"M": "M", "F": "F", "B": "B"}).fillna("Unknown")

    # ── Breed × Sex interaction ───────────────────────────────────────────────
    df["breed_sex"] = df["breed_grp"] + "_" + df["sex_clean"]

    # ── Sale date & seasonality ───────────────────────────────────────────────
    sale_dt = pd.to_datetime(df["scraped_date"], errors="coerce")
    df["sale_date"]   = sale_dt.dt.strftime("%Y-%m-%d")
    df["sale_month"]  = sale_dt.dt.month          # 1–12, strong seasonal signal
    df["sale_month"]  = df["sale_month"].fillna(0).astype(int)

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
    "weight", "age_months", "days_in_herd", "no_of_owners",
    "icbf_cbv_num", "icbf_replacement_num", "icbf_ebi_num", "icbf_stars",
    "has_genomic", "quality_assured", "bvd_ok", "export_score",
    "temp_max_c", "temp_min_c", "precipitation_mm", "wind_speed_kmh",
    "sale_month",
]

CATEGORICAL_FEATURES = ["breed_grp", "sex_clean", "mart", "dam_breed_grp", "breed_sex"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET       = "ppkg"


# ── Preprocessing + model pipeline ────────────────────────────────────────────

def build_pipeline():
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

    model = lgb.LGBMRegressor(
        n_estimators      = 800,
        learning_rate     = 0.04,
        num_leaves        = 127,
        max_depth         = -1,
        min_child_samples = 15,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )

    return Pipeline([
        ("prep",  preprocessor),
        ("model", model),
    ])


# ── Metrics ───────────────────────────────────────────────────────────────────

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
    print("Loading data…")
    df = load_and_engineer(CSV_PATH)
    print(f"  {len(df):,} rows after cleaning")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"\nTrain: {len(X_train):,}   Test: {len(X_test):,}")

    # ── Fit pipeline ──────────────────────────────────────────────────────────
    print("\nTraining LightGBM pipeline…")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    train_metrics = compute_metrics(y_train, pipeline.predict(X_train), "Train metrics")
    test_metrics  = compute_metrics(y_test,  pipeline.predict(X_test),  "Test metrics")

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    print("\n  Running 5-fold cross-validation…")
    cv_scores = cross_val_score(
        build_pipeline(), X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_mae = -cv_scores.mean()
    print(f"    CV MAE: €{cv_mae:.4f}/kg ± €{cv_scores.std():.4f}/kg")

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

    # ── SHAP values (test set) ────────────────────────────────────────────────
    print("\nComputing SHAP values…")
    prep      = pipeline.named_steps["prep"]
    X_test_t  = prep.transform(X_test)
    X_train_t = prep.transform(X_train)

    # Use a 200-row background sample for efficiency
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train_t), size=min(200, len(X_train_t)), replace=False)
    X_bg = X_train_t[bg_idx]

    explainer   = shap.TreeExplainer(lgb_model, data=X_bg)
    shap_values = explainer(X_test_t)

    joblib.dump(shap_values, SHAP_VAL_PATH)
    joblib.dump(X_bg,        SHAP_BG_PATH)
    print(f"  SHAP values saved → {SHAP_VAL_PATH.name}")
    print(f"  SHAP background saved → {SHAP_BG_PATH.name}")

    # ── Re-fit on ALL data for deployment ─────────────────────────────────────
    print("\nRefitting on full dataset for deployment…")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH.name}")

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
        "numeric_features":      NUMERIC_FEATURES,
        "categorical_features":  CATEGORICAL_FEATURES,
        "target":                TARGET,
        "train_metrics":         train_metrics,
        "test_metrics":          test_metrics,
        "cv_mae_eur_kg":         round(cv_mae, 4),
        "cv_mae_std":            round(float(cv_scores.std()), 4),
        "n_train":               len(X_train),
        "n_test":                len(X_test),
        "feature_importances":   importances.round(1).to_dict(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {META_PATH.name}")
    print(f"\nDone. Test R²={test_metrics['R2']:.4f}  MAE=€{test_metrics['MAE_eur_kg']:.4f}/kg")


if __name__ == "__main__":
    main()
