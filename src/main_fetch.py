# main_fetch.py — Phase 1: crawl → historical/{YEAR}/all_matches.csv
#                                  historical/{YEAR}/all_matches_excluding_state.csv
#                                  historical/{YEAR}/school_meta.json
#
# Run this first; its outputs are consumed by run_ranking.py.
#
# Credentials come from environment variables ONLY (TENNIS_EMAIL /
# TENNIS_PASSWORD), set as GitHub Secrets in CI or exported locally —
# never hardcode credentials in this file.

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import aiohttp
import pandas as pd

from api_fetcher import login
from config import MAX_SCHOOLS, TARGET_GENDER, YEAR
from crawler import crawl_school_matches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEED_SCHOOL_ID = 3877   # Berrien Springs — change to any valid school

OUTPUT_DIR = Path(__file__).parent.parent / "historical" / str(YEAR)


async def main() -> None:
    email = os.environ.get("TENNIS_EMAIL")
    password = os.environ.get("TENNIS_PASSWORD")
    if not email or not password:
        log.error(
            "TENNIS_EMAIL / TENNIS_PASSWORD environment variables are not set. "
            "Set them as GitHub Secrets (CI) or export them locally before running."
        )
        sys.exit(1)

    log.info("=" * 62)
    log.info(
        "Tennis Ranking System  —  Fetch Phase  —  season: %d  seed: %d",
        YEAR, SEED_SCHOOL_ID,
    )
    if TARGET_GENDER:
        log.info("Gender filter active: %s only", TARGET_GENDER)
    log.info("=" * 62)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=16)
    async with aiohttp.ClientSession(connector=connector) as session:
        log.info("Logging in as %s …", email)
        token = await login(session, email, password)
        if not token:
            log.error("Login failed. Check TENNIS_EMAIL / TENNIS_PASSWORD and try again.")
            sys.exit(1)
        log.info("Login successful.")

        # ── Phase 1: crawl ───────────────────────────────────────────────
        matches, matches_no_state, school_meta = await crawl_school_matches(
            seed_id=SEED_SCHOOL_ID,
            max_schools=MAX_SCHOOLS,
            year=YEAR,
        )

    if not matches:
        log.error("No matches collected — check seed ID, year, and network.")
        sys.exit(1)

    # ── Phase 2: persist raw data ────────────────────────────────────────
    all_path = OUTPUT_DIR / "all_matches.csv"
    no_state_path = OUTPUT_DIR / "all_matches_excluding_state.csv"
    meta_path = OUTPUT_DIR / "school_meta.json"

    pd.DataFrame(matches).to_csv(all_path, index=False)
    log.info("Saved  %s  (%d rows)", all_path, len(matches))

    pd.DataFrame(matches_no_state).to_csv(no_state_path, index=False)
    log.info("Saved  %s  (%d rows)", no_state_path, len(matches_no_state))

    # Merge with any existing meta so manual overrides aren't lost
    existing_meta: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing_meta = json.load(f)
    existing_meta.update({str(k): v for k, v in school_meta.items()})
    with open(meta_path, "w") as f:
        json.dump(existing_meta, f, indent=2)
    log.info("Saved  %s  (%d schools)", meta_path, len(existing_meta))

    log.info("=" * 62)
    log.info("Fetch phase done — run run_ranking.py next.")
    log.info("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
