#!/bin/bash
# Daily MartBids scraper runner
# Logs to scraper.log in the same directory

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python3"
LOG="$DIR/scraper.log"

echo "-------- SCRAPE $(date '+%Y-%m-%d %H:%M:%S') --------" >> "$LOG"

# 1. Scrape latest lots
"$PYTHON" "$DIR/martbids_scraper.py" >> "$LOG" 2>&1

# 2. Fetch weather for today's sales
echo "Fetching weather data..." >> "$LOG"
"$PYTHON" "$DIR/fetch_weather.py" >> "$LOG" 2>&1

# 3. Generate and email daily PDF report
echo "Generating daily report..." >> "$LOG"
"$PYTHON" "$DIR/generate_report.py" >> "$LOG" 2>&1

echo "" >> "$LOG"
