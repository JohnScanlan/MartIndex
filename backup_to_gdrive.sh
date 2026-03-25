#!/bin/bash
# Weekly backup of MartBids data to Google Drive

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/scraper.log"
REMOTE="drive:MartBids"

echo "-------- RETRAIN + BACKUP $(date '+%Y-%m-%d %H:%M:%S') --------" >> "$LOG"

# Retrain model on latest data
echo "Retraining model..." >> "$LOG"
"$DIR/.venv/bin/python3" "$DIR/train_model.py" >> "$LOG" 2>&1
echo "Retraining complete." >> "$LOG"

/opt/homebrew/bin/rclone copy "$DIR/sold_lots.csv"              "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/cattle_model.pkl"           "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/model_metadata.json"        "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/model_test_predictions.csv" "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/weather_cache.csv"          "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/shap_values.pkl"            "$REMOTE" --log-level INFO 2>> "$LOG"
/opt/homebrew/bin/rclone copy "$DIR/shap_background.pkl"        "$REMOTE" --log-level INFO 2>> "$LOG"

echo "Backup complete → $REMOTE" >> "$LOG"
echo "" >> "$LOG"
