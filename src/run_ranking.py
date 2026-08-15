"""
run_ranking.py — Phase 2: rank → src/rankings_by_division_flight/*.csv
                                  src/rankings_by_division_flight/team_*.csv
#
# Reads config.YEAR to find historical/{YEAR}/all_matches_excluding_state.csv
# (and the matching school_meta.json next to it), runs the full seeding
# pipeline, and writes output directly into the shape scripts/build_site.py
# expects.
#
# This is the ONLY script both workflows have in common:
#   - "Fetch + Rank + Publish" runs main_fetch.py (or run_all_years.py)
#     first to populate historical/{YEAR}/, then runs this.
#   - "Rank + Publish" assumes historical/{YEAR}/ already exists (checked
#     into the repo from a previous fetch) and just re-runs this — much
#     faster since no network calls happen.
"""
import logging
import os
import sys

import config as _config
import mhsaa_seeding_v2 as seeding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "rankings_by_division_flight")


def _clear_stale_csvs(path: str) -> int:
    """Remove all existing ranking CSVs in `path` before writing this
    run's output.

    Without this, a (division, gender, category) combo that produces
    zero rows in the current run — e.g. early in a new season before
    every division has played, or a division that's been dropped/
    renamed — would leave LAST SEASON's file untouched. build_site.py
    has no concept of "this file is from an old season"; it just globs
    every *.csv in this directory and renders it under the *current*
    season's header/timestamp. That means the site can look freshly
    updated while quietly showing stale data for whichever slots
    weren't re-ranked this run.

    Called only after seeding.run() has already succeeded, so a failed
    ranking run never wipes out a working (if stale) previous output.
    """
    if not os.path.isdir(path):
        return 0
    removed = 0
    for name in os.listdir(path):
        if name.endswith(".csv"):
            os.remove(os.path.join(path, name))
            removed += 1
    if removed:
        log.info("Cleared %d stale CSV(s) from %s before writing this run's output", removed, path)
    return removed


def main() -> None:
    year = getattr(_config, "YEAR", None)
    if year is None:
        log.error("config.YEAR is not set — cannot determine which season to rank.")
        sys.exit(1)

    csv_path = os.path.join("..", "historical", str(year), "all_matches_excluding_state.csv")
    if not os.path.exists(csv_path):
        # also try relative to repo root, in case this is invoked from
        # a different working directory
        alt = os.path.join("historical", str(year), "all_matches.csv")
        if os.path.exists(alt):
            csv_path = alt
        else:
            log.error(
                "Could not find match data for YEAR=%d. Looked for:\n  %s\n  %s\n"
                "Run the fetch phase first (main_fetch.py / run_all_years.py).",
                year, csv_path, alt,
            )
            sys.exit(1)

    log.info("=" * 62)
    log.info("Tennis Ranking System  —  Rank Phase  —  season: %d", year)
    log.info("Input:  %s", csv_path)
    log.info("Output: %s", OUTPUT_DIR)
    log.info("=" * 62)

    results = seeding.run(csv_path)
    seeding.print_results(results)

    # Only clear old output once ranking has actually succeeded — see
    # _clear_stale_csvs() docstring for why this matters on a season
    # rollover.
    _clear_stale_csvs(OUTPUT_DIR)

    written_div = seeding.write_division_csvs(results, OUTPUT_DIR)
    written_team = seeding.write_team_csvs(results, OUTPUT_DIR)

    log.info("=" * 62)
    log.info("Rank phase done — %d division CSV(s), %d team CSV(s) written to %s",
              len(written_div), len(written_team), OUTPUT_DIR)
    log.info("Next: run scripts/build_site.py to publish the website.")
    log.info("=" * 62)


if __name__ == "__main__":
    main()
