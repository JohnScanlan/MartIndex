#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# MartIndex nightly pipeline — the single source of truth for what runs.
#
# Invoked by launchd (com.johnscanlan.martbids, daily 21:30) through the shim
# at ~/Library/Scripts/MartBids/run_scraper.sh. Do not duplicate this logic
# there — the shim only forwards to this file.
#
# launchd is used rather than cron on purpose: a StartCalendarInterval job
# missed while the Mac sleeps runs on wake, whereas a cron entry is simply
# skipped. The morning cron jobs this replaced silently missed weeks at a time.
#
# Steps run in dependency order and are individually fault-isolated: a failure
# is logged and the chain continues, because a broken weather API should not
# cost us the nightly report.
#
# Log: /tmp/martbids_run.log  (/tmp is always readable by launchd; the repo
# directory is not, under macOS Full Disk Access rules for /bin/bash)
# ─────────────────────────────────────────────────────────────────────────────
set -u

DIR="/Users/johnscanlan/Documents/Kaggle/grass"
PYTHON="$DIR/.venv/bin/python3"
LOG="/tmp/martbids_run.log"

FAILED=()

run_step() {
    local name="$1" script="$2"
    echo "" >> "$LOG"
    echo ">>> $name" >> "$LOG"
    if "$PYTHON" "$DIR/$script" >> "$LOG" 2>&1; then
        echo "[OK]   $name" >> "$LOG"
    else
        echo "[FAIL] $name (exit $?)" >> "$LOG"
        FAILED+=("$name")
    fi
}

echo "======== RUN $(date '+%Y-%m-%d %H:%M:%S') ========" >> "$LOG"

# ── Mart lots ────────────────────────────────────────────────────────────────
run_step "martbids"        "martbids_scraper.py"
run_step "livestock-live"  "lsl_scraper.py"

# Weather reads the mart × date pairs the two scrapers above just wrote,
# so it must follow them.
run_step "weather"         "fetch_weather.py"

# ── Factory prices ───────────────────────────────────────────────────────────
# Runs daily even though DAFM publishes weekly: the scrape is idempotent
# (5-key dedup, 3-week window) and costs ~6 requests, so a daily run picks up
# a new week the day it appears instead of waiting for a fixed weekday.
run_step "factory-scrape"  "scrape_factory_prices.py"
run_step "factory-prepare" "prepare_factory_prices.py"

# ── Outputs ──────────────────────────────────────────────────────────────────
run_step "report"          "generate_report.py"
run_step "git-push"        "git_push.py"

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" >> "$LOG"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "======== END $(date '+%Y-%m-%d %H:%M:%S') — all steps OK ========" >> "$LOG"
else
    echo "======== END $(date '+%Y-%m-%d %H:%M:%S') — FAILED: ${FAILED[*]} ========" >> "$LOG"
fi
echo "" >> "$LOG"

[ ${#FAILED[@]} -eq 0 ]
