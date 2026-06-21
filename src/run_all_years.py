"""
run_all_years.py
================
Manual/local-use script to backfill ALL historical seasons (2021-2025) at
once, producing per-year outputs under historical/{year}/. This is NOT
called by either GitHub Actions workflow (those only ever fetch/rank the
single season in config.YEAR) — run this yourself when you need to
(re)populate multiple years of history in one pass.

What it does
------------
1. Logs in to tennisreporting.com and stores the bearer token.
2. For each season in SEASONS:
   a. Crawls all school matches for that year.
   b. Writes  historical/{year}/all_matches.csv
   c. Writes  historical/{year}/all_matches_excluding_state.csv
   d. Writes  historical/{year}/school_meta.json
3. Fetches state tournament seeds for each year and writes
      historical/{year}/state_seeds.csv

Credentials
-----------
Set TENNIS_EMAIL / TENNIS_PASSWORD as environment variables before
running this script. Credentials are never hardcoded here — if you see
a hardcoded email/password in any version of this file, treat that
password as compromised and rotate it immediately.

    export TENNIS_EMAIL="you@example.com"
    export TENNIS_PASSWORD="your-password"
    python run_all_years.py

Output layout
-------------
historical/
  2021/
    all_matches.csv
    all_matches_excluding_state.csv
    school_meta.json
    state_seeds.csv
  2022/ ...
  2023/ ...
  2024/ ...
  2025/ ...

NOTE on dependencies
--------------------
This script imports fetch_state_seeds.py for state-tournament seed
enrichment (fetch_season_seeds, add_division_from_meta,
enrich_with_match_stats, save_season_output) and crawler.py imports
match_parser.py for response parsing. Neither of those two modules'
internals were available when this repo was assembled, so they are
NOT included here — bring your existing copies of match_parser.py and
fetch_state_seeds.py into src/ before running this script (main_fetch.py
and run_ranking.py, used by the GitHub Actions workflows, only need
match_parser.py, not fetch_state_seeds.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import aiohttp
import pandas as pd

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── seasons to fetch ──────────────────────────────────────────────────────────
SEASONS = [2021, 2022, 2023, 2024, 2025]

# Seed school used as the BFS starting point for each season crawl.
# Berrien Springs (3877) has matches in every season — safe default.
SEED_SCHOOL_ID = 3877

# Output root
OUTPUT_ROOT = Path(__file__).parent.parent / "historical"

# ── imports that depend on the local source modules ───────────────────────────
from api_fetcher import login
from crawler import crawl_school_matches
from config import MAX_SCHOOLS, STATE_EVENT_IDS

try:
    from fetch_state_seeds import (
        fetch_season_seeds,
        add_division_from_meta,
        enrich_with_match_stats,
        save_season_output,
    )
    _HAS_STATE_SEEDS = True
except ImportError:
    _HAS_STATE_SEEDS = False
    log.warning(
        "fetch_state_seeds.py not found — state_seeds.csv will be skipped "
        "for every season. Match crawling still works."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-year crawl
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_year(session: aiohttp.ClientSession, year: int) -> None:
    log.info("")
    log.info("=" * 62)
    log.info("SEASON %d — crawling matches", year)
    log.info("=" * 62)

    year_dir = OUTPUT_ROOT / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    # ── crawl ─────────────────────────────────────────────────────────────────
    matches, matches_no_state, school_meta = await crawl_school_matches(
        seed_id=SEED_SCHOOL_ID,
        max_schools=MAX_SCHOOLS,
        year=year,
    )

    if not matches:
        log.error("Season %d: no matches collected — skipping.", year)
        return

    # ── write match CSVs ──────────────────────────────────────────────────────
    all_path = year_dir / "all_matches.csv"
    no_state_path = year_dir / "all_matches_excluding_state.csv"

    pd.DataFrame(matches).to_csv(all_path, index=False)
    log.info("Saved  %s  (%d rows)", all_path, len(matches))

    pd.DataFrame(matches_no_state).to_csv(no_state_path, index=False)
    log.info("Saved  %s  (%d rows)", no_state_path, len(matches_no_state))

    # ── write school metadata ─────────────────────────────────────────────────
    meta_path = year_dir / "school_meta.json"
    existing_meta: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = json.load(f)
    existing_meta.update({str(k): v for k, v in school_meta.items()})
    with open(meta_path, "w") as f:
        json.dump(existing_meta, f, indent=2)
    log.info("Saved  %s  (%d schools)", meta_path, len(existing_meta))

    # ── state seeds ───────────────────────────────────────────────────────────
    if not _HAS_STATE_SEEDS:
        return

    event_id = STATE_EVENT_IDS.get(year)
    if event_id is None:
        log.warning("Season %d: no state event ID configured — skipping seeds.", year)
        return

    log.info("")
    log.info("Season %d — fetching state seeds (event %d)", year, event_id)

    seeds_df = await fetch_season_seeds(session, year, event_id)
    if seeds_df.empty:
        log.warning("Season %d: no seed rows returned.", year)
        return

    seeds_df = add_division_from_meta(seeds_df, year)

    matches_path = no_state_path if no_state_path.exists() else all_path
    seeds_df = enrich_with_match_stats(seeds_df, matches_path)

    save_season_output(seeds_df, year)

    log.info("Season %d complete.", year)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    email = os.environ.get("TENNIS_EMAIL")
    password = os.environ.get("TENNIS_PASSWORD")
    if not email or not password:
        log.error(
            "TENNIS_EMAIL / TENNIS_PASSWORD environment variables are not set. "
            "Export them before running this script."
        )
        sys.exit(1)

    log.info("=" * 62)
    log.info("Michigan HS Tennis — Historical Fetch  (%s)", ", ".join(map(str, SEASONS)))
    log.info("=" * 62)

    connector = aiohttp.TCPConnector(limit=16)
    async with aiohttp.ClientSession(connector=connector) as session:

        log.info("Logging in as %s …", email)
        token = await login(session, email, password)
        if not token:
            log.error("Login failed. Check TENNIS_EMAIL / TENNIS_PASSWORD and try again.")
            return
        log.info("Login successful.")

        for year in SEASONS:
            await fetch_year(session, year)

    log.info("")
    log.info("=" * 62)
    log.info("All seasons done.  Outputs are in:  %s/", OUTPUT_ROOT)
    log.info("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
