#!/bin/bash
# Daily backup of MartBids data to Google Drive
# Run daily via cron (e.g., 9pm)

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/scraper.log"
REMOTE="drive:MartBids"

echo "-------- BACKUP $(date '+%Y-%m-%d %H:%M:%S') --------" >> "$LOG"

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

# Backup to Google Drive (atomic files only)
/opt/homebrew/bin/rclone copy "$DIR/sold_lots.csv"              "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/sold_lots.parquet"          "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/factory_prices_clean.csv"   "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/factory_prices_clean.parquet" "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/cattle_model.pkl"           "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/model_metadata.json"        "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/model_test_predictions.csv" "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/weather_cache.csv"          "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/shap_values.pkl"            "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/shap_background.pkl"        "$REMOTE" --log-level INFO 2>> "$LOG"

echo "Backup complete → $REMOTE" >> "$LOG"
echo "" >> "$LOG"
