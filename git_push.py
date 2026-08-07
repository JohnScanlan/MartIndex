#!/usr/bin/env python3
"""
Push the nightly data update to GitHub.

Called from run_scraper.sh (launchd). Runs git from Python because Python has
Full Disk Access to ~/Documents, which /bin/bash under launchd does not.

Credentials
-----------
Reads a GitHub Personal Access Token from, in order:
  1. $GITHUB_TOKEN
  2. ~/Library/Scripts/MartBids/.github_token

The ~/Library path is the single source of truth — launchd can always read it,
and it sits outside the repo so it can never be committed. Classic ``ghp_``
tokens expire; if the push starts failing with "Invalid username or token",
mint a new fine-grained PAT with Contents:read-write on JohnScanlan/MartIndex
and write it to that path.

Exit codes: 0 push succeeded, 1 no token, 2 push rejected, 3 git error.
"""
import datetime
import os
import subprocess
import sys
from pathlib import Path

DIR        = Path(__file__).parent
TOKEN_FILE = Path.home() / "Library/Scripts/MartBids/.github_token"
REPO       = "github.com/JohnScanlan/MartIndex.git"
USER       = "JohnScanlan"

# Everything the deployed dashboards need to render current figures.
FILES = [
    "sold_lots.csv",
    "lsl_lots.csv",
    "weather_cache.csv",
    "factory_prices_clean.csv",   # both dashboards read this — was missing
    "model_test_predictions.csv",
    "model_metadata.json",
    "cattle_model.pkl",
    "shap_values.pkl",
    "shap_background.pkl",
]


def run(cmd, secret=None):
    """Run a git command; echo output with the token redacted.

    Returns (returncode, combined_output).
    """
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=DIR)
    out = []
    for stream in (r.stdout, r.stderr):
        text = stream.strip()
        if not text:
            continue
        if secret:
            text = text.replace(secret, "***")
        print(text)
        out.append(text)
    return r.returncode, "\n".join(out)


def get_token() -> str | None:
    env = os.environ.get("GITHUB_TOKEN", "").strip()
    if env:
        return env
    try:
        return TOKEN_FILE.read_text().strip() or None
    except OSError:
        return None


def main() -> int:
    token = get_token()
    if not token:
        print("[git_push] FATAL: no GitHub token.")
        print(f"[git_push] Put a PAT in {TOKEN_FILE} or set $GITHUB_TOKEN.")
        return 1

    present = [f for f in FILES if (DIR / f).exists()]
    missing = [f for f in FILES if not (DIR / f).exists()]
    if missing:
        print(f"[git_push] WARNING: not on disk, skipping: {', '.join(missing)}")

    if run(["git", "add", "--"] + present)[0] != 0:
        print("[git_push] FATAL: git add failed.")
        return 3

    rc, _ = run(["git", "commit", "-m", f"Daily data update {datetime.date.today()}"])
    if rc != 0:
        print("[git_push] Nothing new to commit — pushing any existing backlog.")

    ahead = subprocess.run(
        ["git", "rev-list", "--count", "@{u}..HEAD"],
        capture_output=True, text=True, cwd=DIR,
    ).stdout.strip()
    if ahead and ahead != "0":
        print(f"[git_push] {ahead} commit(s) to push.")

    url = f"https://{USER}:{token}@{REPO}"
    rc, out = run(["git", "push", url, "main"], secret=token)
    if rc == 0:
        print("[git_push] Push complete.")
        return 0

    # Two very different failures look similar at a glance. Say which it is.
    if "Invalid username or token" in out or "Authentication failed" in out:
        print("[git_push] FATAL: the token is invalid or expired.")
        print(f"[git_push] Mint a new PAT and write it to {TOKEN_FILE}")
        return 2

    if "fetch first" in out or "non-fast-forward" in out or "rejected" in out:
        # Someone committed on GitHub directly (a web edit, another machine).
        # Merge it and retry once, rather than blocking the pipeline nightly.
        print("[git_push] Remote has commits we don't. Merging and retrying...")
        if run(["git", "fetch", url, "main"], secret=token)[0] != 0:
            print("[git_push] FATAL: fetch failed.")
            return 2

        if run(["git", "merge", "FETCH_HEAD", "--no-edit"])[0] != 0:
            # Never leave the repo mid-merge — the next run would fail on a
            # dirty tree and the cause would be far from obvious.
            run(["git", "merge", "--abort"])
            print("[git_push] FATAL: merge conflicts with the remote. "
                  "Resolve by hand, then re-run this script.")
            return 2

        if run(["git", "push", url, "main"], secret=token)[0] != 0:
            print("[git_push] FATAL: push still rejected after merging.")
            return 2

        print("[git_push] Push complete (after merging remote changes).")
        return 0

    print("[git_push] FATAL: push failed for an unrecognised reason (above).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
