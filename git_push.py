#!/usr/bin/env python3
"""Called from launchd launcher — runs git from Python which has Documents access."""
import os, subprocess, sys
from pathlib import Path

DIR = Path(__file__).parent
TOKEN_FILE = Path.home() / "Library/Scripts/MartBids/.github_token"

try:
    token = TOKEN_FILE.read_text().strip()
except Exception as e:
    print(f"[git_push] Could not read token: {e}"); sys.exit(1)

os.chdir(DIR)

files = [
    "sold_lots.csv", "lsl_lots.csv", "model_test_predictions.csv",
    "weather_cache.csv", "model_metadata.json", "cattle_model.pkl",
    "shap_values.pkl", "shap_background.pkl",
]

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout.strip(): print(r.stdout.strip())
    if r.stderr.strip(): print(r.stderr.strip())
    return r.returncode

import datetime
run(["git", "add"] + files)
run(["git", "commit", "-m", f"Daily data update {datetime.date.today()}"])
rc = run(["git", "push",
          f"https://JohnScanlan:{token}@github.com/JohnScanlan/MartIndex.git",
          "main"])
sys.exit(rc)
