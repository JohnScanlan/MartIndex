#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Weekly backup of MartIndex data to Google Drive.
#
# Invoked by launchd (com.johnscanlan.martbids.backup, Sundays 22:00) through
# the shim at ~/Library/Scripts/MartBids/backup_to_gdrive.sh. launchd's
# /bin/bash cannot read scripts under ~/Documents (Full Disk Access), which is
# what made this job exit 78 for weeks — hence the shim.
#
# Writes to a dated remote folder, so a previous day's backup is never
# overwritten.
# ─────────────────────────────────────────────────────────────────────────────
set -u

DIR="/Users/johnscanlan/Documents/Kaggle/grass"
LOG="$DIR/backup.log"
RCLONE="/opt/homebrew/bin/rclone"
TODAY="$(date '+%Y-%m-%d')"
REMOTE="drive:MartBids/backups/$TODAY"

echo "-------- BACKUP $(date '+%Y-%m-%d %H:%M:%S') → $REMOTE --------" >> "$LOG"

if [ ! -x "$RCLONE" ]; then
    echo "  FATAL: rclone not found at $RCLONE" >> "$LOG"
    exit 1
fi

# Convert CSVs to parquet for faster dashboard loading.
# Paths are passed in explicitly: this runs as a stdin heredoc, where
# __file__ is "<stdin>" and any Path(__file__).parent trick silently
# resolves to the current directory instead.
echo "Converting CSVs to parquet..." >> "$LOG"
"$DIR/.venv/bin/python3" - "$DIR" << 'PYTHON_EOF' >> "$LOG" 2>&1
import sys, logging
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from data_utils import csv_to_parquet

logging.basicConfig(level=logging.INFO, format="  %(message)s")
base = Path(sys.argv[1])
for csv in ["sold_lots.csv", "factory_prices_clean.csv"]:
    csv_path = base / csv
    if csv_path.exists():
        csv_to_parquet(csv_path, csv_path.with_suffix('.parquet'))
    else:
        print(f"  skipped (not on disk): {csv}")
PYTHON_EOF

# Backup to dated subfolder — never overwrites a previous day's backup
FILES=(
    sold_lots.csv
    sold_lots.parquet
    lsl_lots.csv          # 50,947 lots that were in no backup at all
    factory_prices.csv
    factory_prices_clean.csv
    factory_prices_clean.parquet
    cattle_model.pkl
    model_metadata.json
    model_test_predictions.csv
    weather_cache.csv
    shap_values.pkl
    shap_background.pkl
)

COPIED=0
FAILED=0
for f in "${FILES[@]}"; do
    if [ ! -f "$DIR/$f" ]; then
        echo "  missing, skipped: $f" >> "$LOG"
        continue
    fi
    if "$RCLONE" copy "$DIR/$f" "$REMOTE" --log-level INFO 2>> "$LOG"; then
        echo "  backed up: $f" >> "$LOG"
        COPIED=$((COPIED + 1))
    else
        echo "  FAILED: $f" >> "$LOG"
        FAILED=$((FAILED + 1))
    fi
done

echo "Backup complete → $REMOTE  ($COPIED copied, $FAILED failed)" >> "$LOG"
echo "" >> "$LOG"

[ "$FAILED" -eq 0 ]
