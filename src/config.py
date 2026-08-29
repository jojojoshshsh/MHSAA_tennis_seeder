# config.py — parameter settings for the tennis ranking system.
#
# YEAR is the single knob that drives the whole pipeline:
#   - main_fetch.py / run_all_years.py crawl matches for this season
#   - run_ranking.py ranks this season's matches
#   - scripts/build_site.py labels the published site with this season
#
# Change YEAR, push, and re-run the "Rank + Publish" workflow (or the
# full "Fetch + Rank + Publish" workflow if you need fresh data) to
# regenerate everything for that season.

import datetime

today = datetime.date.today()
# Stay on the previous year until April 1 (adjust month as needed)
if today.month < 8:
    YEAR = today.year - 1
else:
    YEAR = today.year

IS_NOT_VARSITY = 0           # 0 = varsity only
TARGET_STATE   = "MI"        # or None for no filter
TARGET_GENDER  = "Boys"      # or "Girls" or None for both
MAX_SCHOOLS    = None        # optional crawl limit

# Minimum matches to appear in rankings
MIN_MATCHES = 5

# Division lookups (not needed for core logic; used in ranking output)
TARGET_DIVISION = None
TARGET_FLIGHT   = None
TARGET_POOL     = None

# Known state tournament event IDs — used to exclude state matches
# from all_matches_excluding_state.csv
STATE_EVENT_IDS: dict[int, int] = {
    2021: 240,
    2022: 320,
    2023: 472,
    2024: 577,
    2025: 688,
}
