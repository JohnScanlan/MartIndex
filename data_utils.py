"""
Safe data I/O utilities for MartBids project.
- Atomic writes (temp file → verify → move)
- Parquet/CSV conversion and loading
- Google Drive backup helpers
"""

import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

log = logging.getLogger(__name__)


def safe_append_csv(csv_path: Path, new_rows: list, fieldnames: list, dedup_key: tuple = None) -> int:
    """
    Safely append new rows to CSV with deduplication.
    Returns: number of rows added.

    Process:
    1. Load existing CSV
    2. Deduplicate against new rows
    3. Write to temp file
    4. Verify row count
    5. Atomic move
    """
    if not csv_path.exists():
        # New file — just write it
        df_new = pd.DataFrame(new_rows)
        df_new[fieldnames].to_csv(csv_path, index=False)
        log.info(f"Created {csv_path.name} with {len(df_new)} rows")
        return len(df_new)

    # Load existing
    df_existing = pd.read_csv(csv_path)
    existing_count = len(df_existing)

    if dedup_key:
        # Remove rows that already exist
        df_new = pd.DataFrame(new_rows)
        existing_keys = set(df_existing[list(dedup_key)].apply(tuple, axis=1))
        new_keys = df_new[list(dedup_key)].apply(tuple, axis=1)
        mask = ~new_keys.isin(existing_keys)
        df_new = df_new[mask]
        new_rows_filtered = df_new.to_dict('records')
    else:
        new_rows_filtered = new_rows

    if not new_rows_filtered:
        log.info(f"{csv_path.name}: no new rows")
        return 0

    # Combine
    df_combined = pd.concat(
        [df_existing, pd.DataFrame(new_rows_filtered)],
        ignore_index=True
    )

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir=csv_path.parent) as f:
        temp_path = Path(f.name)
        df_combined[fieldnames].to_csv(temp_path, index=False)

    # Verify
    df_verify = pd.read_csv(temp_path)
    expected_count = existing_count + len(new_rows_filtered)
    actual_count = len(df_verify)

    if actual_count != expected_count:
        temp_path.unlink()
        raise ValueError(
            f"Write verification failed: expected {expected_count} rows, got {actual_count}"
        )

    # Atomic move
    backup_path = csv_path.with_suffix('.csv.bak')
    if backup_path.exists():
        backup_path.unlink()
    csv_path.rename(backup_path)
    temp_path.rename(csv_path)
    backup_path.unlink()

    added = len(new_rows_filtered)
    log.info(f"{csv_path.name}: added {added} rows ({actual_count} total)")
    return added


def load_data_safe(csv_path: Path, parquet_path: Path = None) -> pd.DataFrame:
    """
    Load data from parquet if available and fresh, else CSV.
    Supports graceful fallback.
    """
    # Try parquet first if it exists and is newer than CSV
    if parquet_path and parquet_path.exists():
        try:
            pq_mtime = parquet_path.stat().st_mtime
            csv_mtime = csv_path.stat().st_mtime if csv_path.exists() else 0

            if pq_mtime >= csv_mtime:
                df = pd.read_parquet(parquet_path)
                log.debug(f"Loaded {parquet_path.name} ({len(df)} rows)")
                return df
        except Exception as e:
            log.warning(f"Failed to read {parquet_path.name}: {e}, falling back to CSV")

    # Fall back to CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        log.debug(f"Loaded {csv_path.name} ({len(df)} rows)")
        return df

    raise FileNotFoundError(f"No data file found: {csv_path} or {parquet_path}")


def csv_to_parquet(csv_path: Path, parquet_path: Path = None, force: bool = False) -> Path:
    """
    Convert CSV to parquet. Returns parquet path.
    If force=False, skips if parquet is already newer.
    """
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')

    if parquet_path.exists() and not force:
        pq_mtime = parquet_path.stat().st_mtime
        csv_mtime = csv_path.stat().st_mtime
        if pq_mtime >= csv_mtime:
            log.debug(f"{parquet_path.name} already up-to-date")
            return parquet_path

    df = pd.read_csv(csv_path)

    # Write to temp, verify, move
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False, dir=parquet_path.parent) as f:
        temp_path = Path(f.name)
        df.to_parquet(temp_path, index=False, compression='snappy')

    # Verify
    df_verify = pd.read_parquet(temp_path)
    if len(df_verify) != len(df):
        temp_path.unlink()
        raise ValueError(f"Parquet verification failed: {len(df)} → {len(df_verify)}")

    # Atomic move
    backup_path = parquet_path.with_suffix('.parquet.bak')
    if backup_path.exists():
        backup_path.unlink()
    if parquet_path.exists():
        parquet_path.rename(backup_path)
    temp_path.rename(parquet_path)
    if backup_path.exists():
        backup_path.unlink()

    log.info(f"Converted {csv_path.name} → {parquet_path.name} ({len(df)} rows)")
    return parquet_path


def get_backup_filename(csv_path: Path, dated: bool = True) -> str:
    """Generate backup filename (e.g., sold_lots_2026-04-24.csv)"""
    if dated:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return csv_path.stem + f"_{date_str}" + csv_path.suffix
    return csv_path.with_suffix(csv_path.suffix + ".bak").name
