#!/bin/bash
# Daily backup of MartBids data to Google Drive
# Run daily via cron (e.g., 9pm)

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/backup.log"
TODAY="$(date '+%Y-%m-%d')"
REMOTE="drive:MartBids/backups/$TODAY"

echo "-------- BACKUP $(date '+%Y-%m-%d %H:%M:%S') → $REMOTE --------" >> "$LOG"

# Convert CSVs to parquet for faster loading
echo "Converting CSVs to parquet..." >> "$LOG"
"$DIR/.venv/bin/python3" << 'PYTHON_EOF' >> "$LOG" 2>&1
from pathlib import Path
from data_utils import csv_to_parquet

DIR = Path(__file__).parent
for csv in ["sold_lots.csv", "factory_prices_clean.csv"]:
    csv_path = DIR / csv
    if csv_path.exists():
        parquet_path = csv_path.with_suffix('.parquet')
        csv_to_parquet(csv_path, parquet_path)
PYTHON_EOF

# Backup to dated subfolder — never overwrites a previous day's backup
FILES=(
    sold_lots.csv
    sold_lots.parquet
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

for f in "${FILES[@]}"; do
    if [ -f "$DIR/$f" ]; then
        /opt/homebrew/bin/rclone copy "$DIR/$f" "$REMOTE" --log-level INFO 2>> "$LOG"
        echo "  backed up: $f" >> "$LOG"
    fi
done

echo "Backup complete → $REMOTE" >> "$LOG"
echo "" >> "$LOG"
