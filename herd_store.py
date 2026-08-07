"""
Persistence for the herd valuation tracker.

Two files, two deliberately different write strategies:

  herds.csv                    small user-managed registry. Atomic
                               read-modify-write, timestamped backup on every
                               change, because the user needs to edit it.

  herd_valuation_history.csv   the tracking time series. Append-only via
                               data_utils.safe_append_csv, because losing a
                               valuation point silently breaks the trend chart
                               and there is no way to recover it.

Neither touches scraped data. A "deleted" herd is marked, never removed, so its
valuation history stays interpretable.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_utils import safe_append_csv

DIR          = Path(__file__).parent
HERDS_CSV    = DIR / "herds.csv"
HISTORY_CSV  = DIR / "herd_valuation_history.csv"
BACKUP_DIR   = DIR / "_herd_backups"

HERD_FIELDS = [
    "herd_id", "customer", "loan_ref", "line_id",
    "breed_group", "sex", "head", "avg_weight_kg",
    "loan_balance", "max_ltv_pct", "region",
    "status", "updated_at",
]

HISTORY_FIELDS = [
    "valuation_date", "herd_id", "customer", "loan_ref",
    "total_head", "total_kg",
    "value_p25", "value_median", "value_p75",
    "loan_balance", "ltv_pct", "headroom_eur",
    "drift_eur_per_month", "basis",
]


# ── Registry ──────────────────────────────────────────────────────────────────

def load_herds(include_deleted: bool = False) -> pd.DataFrame:
    """All herd lines. One row per line; a herd is several rows sharing herd_id."""
    if not HERDS_CSV.exists():
        return pd.DataFrame(columns=HERD_FIELDS)
    df = pd.read_csv(HERDS_CSV, dtype={"herd_id": str, "line_id": str})
    for col in HERD_FIELDS:
        if col not in df.columns:
            df[col] = ""
    if not include_deleted:
        df = df[df["status"] != "deleted"]
    return df[HERD_FIELDS]


def _write_herds(df: pd.DataFrame) -> None:
    """
    Atomic write with a timestamped backup.

    The registry is small and must be editable, so unlike the scraped data it
    is rewritten rather than appended — but never in place, and never without
    keeping the previous version.
    """
    if HERDS_CSV.exists():
        BACKUP_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        shutil.copy2(HERDS_CSV, BACKUP_DIR / f"herds_{stamp}.csv")

    tmp = HERDS_CSV.with_suffix(".csv.tmp")
    df[HERD_FIELDS].to_csv(tmp, index=False)

    check = pd.read_csv(tmp)
    if len(check) != len(df):
        tmp.unlink(missing_ok=True)
        raise ValueError(f"herds.csv verify failed: {len(df)} → {len(check)}")

    tmp.replace(HERDS_CSV)


def save_herd(herd_id: str, customer: str, loan_ref: str,
              lines: list[dict], loan_balance: float,
              max_ltv_pct: float, region: str) -> None:
    """Insert or replace every line of one herd. Other herds are untouched."""
    existing = load_herds(include_deleted=True)
    others = existing[existing["herd_id"] != herd_id]

    now = datetime.now().isoformat(timespec="seconds")
    rows = [{
        "herd_id": herd_id, "customer": customer, "loan_ref": loan_ref,
        "line_id": f"{herd_id}-{i}",
        "breed_group": ln["breed_group"], "sex": ln["sex"],
        "head": int(ln["head"]), "avg_weight_kg": float(ln["avg_weight_kg"]),
        "loan_balance": float(loan_balance), "max_ltv_pct": float(max_ltv_pct),
        "region": region, "status": "active", "updated_at": now,
    } for i, ln in enumerate(lines, 1)]

    _write_herds(pd.concat([others, pd.DataFrame(rows)], ignore_index=True))


def delete_herd(herd_id: str) -> None:
    """Mark deleted rather than removing — keeps the valuation history readable."""
    df = load_herds(include_deleted=True)
    if df.empty:
        return
    df.loc[df["herd_id"] == herd_id, "status"] = "deleted"
    df.loc[df["herd_id"] == herd_id, "updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_herds(df)


def herd_ids() -> list[str]:
    df = load_herds()
    return sorted(df["herd_id"].unique().tolist()) if not df.empty else []


# ── Valuation history ─────────────────────────────────────────────────────────

def record_valuation(herd_id: str, customer: str, loan_ref: str,
                     hv, loan_balance: float, max_ltv_pct: float) -> int:
    """
    Append one valuation point. Append-only and deduplicated on
    (herd_id, valuation_date), so re-valuing twice in a day is a no-op rather
    than a duplicate point on the chart.
    """
    ltv      = hv.ltv(loan_balance)
    drift    = hv.drift_per_month_eur
    row = {
        "valuation_date": str(hv.as_of or datetime.now().date()),
        "herd_id": herd_id, "customer": customer, "loan_ref": loan_ref,
        "total_head": hv.total_head, "total_kg": round(hv.total_kg, 1),
        "value_p25": round(hv.value_p25, 2),
        "value_median": round(hv.value_median, 2),
        "value_p75": round(hv.value_p75, 2),
        "loan_balance": round(loan_balance, 2),
        "ltv_pct": round(ltv, 2) if ltv is not None else "",
        "headroom_eur": round(hv.headroom(loan_balance, max_ltv_pct), 2),
        "drift_eur_per_month": round(drift, 2) if drift is not None else "",
        "basis": hv.weakest_basis,
    }
    return safe_append_csv(HISTORY_CSV, [row], HISTORY_FIELDS,
                           dedup_key=("herd_id", "valuation_date"))


def load_history(herd_id: str | None = None) -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        return pd.DataFrame(columns=HISTORY_FIELDS)
    df = pd.read_csv(HISTORY_CSV, dtype={"herd_id": str})
    df["valuation_date"] = pd.to_datetime(df["valuation_date"], errors="coerce")
    if herd_id:
        df = df[df["herd_id"] == herd_id]
    return df.sort_values("valuation_date")
