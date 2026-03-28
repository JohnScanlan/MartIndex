#!/bin/bash
# Daily MartBids scraper runner
# Logs to scraper.log in the same directory

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python3"
LOG="$DIR/scraper.log"

echo "-------- SCRAPE $(date '+%Y-%m-%d %H:%M:%S') --------" >> "$LOG"

# 1. Scrape latest lots
"$PYTHON" "$DIR/martbids_scraper.py" >> "$LOG" 2>&1

# 1b. Scrape Livestock-Live.com last sale
echo "Scraping Livestock-Live.com..." >> "$LOG"
"$PYTHON" "$DIR/lsl_scraper.py" >> "$LOG" 2>&1

# 2. Fetch weather for today's sales
echo "Fetching weather data..." >> "$LOG"
"$PYTHON" "$DIR/fetch_weather.py" >> "$LOG" 2>&1

# 3. Generate and email daily PDF report
echo "Generating daily report..." >> "$LOG"
"$PYTHON" "$DIR/generate_report.py" >> "$LOG" 2>&1

# 4. Push updated data to GitHub so Streamlit Cloud stays in sync
echo "Pushing data to GitHub..." >> "$LOG"
cd "$DIR" && git add sold_lots.csv lsl_lots.csv model_test_predictions.csv weather_cache.csv model_metadata.json cattle_model.pkl shap_values.pkl shap_background.pkl >> "$LOG" 2>&1
git commit -m "Daily data update $(date '+%Y-%m-%d')" >> "$LOG" 2>&1
git push https://JohnScanlan:$(cat "$DIR/.github_token")@github.com/JohnScanlan/MartIndex.git main >> "$LOG" 2>&1
echo "GitHub push complete." >> "$LOG"

echo "" >> "$LOG"
