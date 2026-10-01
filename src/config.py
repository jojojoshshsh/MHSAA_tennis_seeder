# config.py — parameter settings for the tennis ranking system.
#
# YEAR is the single knob that drives the whole pipeline:
#   - main_fetch.py / run_all_years.py crawl matches for this season
#   - run_ranking.py ranks this season's matches
#   - scripts/build_site.py labels the published site with this season
#
# YEAR is chosen in this order:
#   1. The YEAR_OVERRIDE environment variable, if set and non-blank
#      (the "Fetch + Rank + Publish" and "Rank + Publish" workflows set it
#      from their optional "year" input).
#   2. Otherwise, inferred from today's date (see below).
#
# So you can either run a workflow with a year typed in, or leave it blank
# to use the default.

import datetime
import os

today = datetime.date.today()
# Stay on the previous year until August 1 (adjust month as needed)
if today.month < 8:
    _DEFAULT_YEAR = today.year - 1
else:
    _DEFAULT_YEAR = today.year

_year_override = os.environ.get("YEAR_OVERRIDE", "").strip()
if _year_override:
    try:
        YEAR = int(_year_override)
    except ValueError:
        raise SystemExit(
            f"YEAR_OVERRIDE must be a 4-digit year like 2025, got {_year_override!r}"
        )
else:
    YEAR = _DEFAULT_YEAR

IS_NOT_VARSITY = 0           # 0 = varsity only
TARGET_STATE   = "MI"        # or None for no filter
TARGET_GENDER  = "Boys"      # or "Girls" or None for both
MAX_SCHOOLS    = None        # optional crawl limit

# Minimum matches to appear in rankings
MIN_MATCHES = 4

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
