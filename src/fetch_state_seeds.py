"""
fetch_state_seeds.py
====================
Fetches official MHSAA state tournament seeds from the tennisreporting API
for each historical season and produces:

  historical/{year}/state_seeds.csv   — seeds matched to player IDs
  historical/{year}/school_meta.json  — school meta snapshot (if not present)

Uses the existing api_fetcher.py functions so auth/retry logic is shared.

Usage
-----
  # Fetch all known seasons
  python fetch_state_seeds.py

  # Fetch a specific year only
  python fetch_state_seeds.py --years 2025

  # Dry run — print what would be fetched, write nothing
  python fetch_state_seeds.py --dry-run

Known state tournament event IDs
---------------------------------
  2021 → 240
  2022 → 320
  2023 → 472
  2024 → 577
  2025 → 688
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import aiohttp
import pandas as pd

from api_fetcher import (
    fetch_event,
    fetch_seed_list,
    _get_headers,
    _TIMEOUT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Known event IDs — add future years here.
# Must stay in sync with config.STATE_EVENT_IDS.
# ─────────────────────────────────────────────────────────────────────────────

STATE_EVENT_IDS: dict[int, int] = {
    2021: 240,
    2026: 906
}

# Default flight counts when the event API doesn't specify
_DEFAULT_SINGLES_FLIGHTS = 4
_DEFAULT_DOUBLES_FLIGHTS = 4

# Gender ID → readable label
_GENDER_MAP = {1: "Boys", 2: "Girls"}

# Explicit division_id → division label per year.
# Keys come directly from the tennisreporting API event structure.
DIVISION_ID_MAP: dict[int, dict[int, str]] = {
    2021: {350: "1", 351: "2", 352: "3", 353: "4_other"},
    2026: {1453: "1", 1454: "2", 1455: "3", 1456:"4"}
}

OUTPUT_ROOT = Path("historical")


# ─────────────────────────────────────────────────────────────────────────────
# Seed-list parsing
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_seed_entries(seed_list) -> list:
    """Mirror match_parser.py — handles multiple possible wrapper shapes."""
    if isinstance(seed_list, list):
        return seed_list
    if isinstance(seed_list, dict):
        for key in ("seedList", "data", "items", "result", "configuration"):
            val = seed_list.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                nested = _unwrap_seed_entries(val)
                if nested:
                    return nested
    return []


def parse_seed_entries(
    seed_list_raw,
    event_id: int,
    year: int,
    gender_id: int,
    match_type: str,   # "Singles" | "Doubles"
    flight: int,
    division_id,
    host_id,
) -> list[dict]:
    """
    Extract one row per player/pair from a seed-list API response.

    Returns list of dicts with columns matching state_seeds.csv spec:
      player_id, pair_id, category, flight, division, seed,
      name, school, school_id, gender, year, event_id
    """
    entries = _unwrap_seed_entries(seed_list_raw)
    if not entries:
        log.warning(
            "  No seed entries — e=%s %s flight=%s div=%s host=%s",
            event_id, match_type, flight, division_id, host_id,
        )
        return []

    rows = []
    category = match_type.lower()   # "singles" | "doubles"

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        seed_num = entry.get("seed")
        if seed_num is None:
            continue  # unplayed / bye slots

        players_in_entry = entry.get("players", [])
        if not players_in_entry:
            continue

        pids, names, school_id, school_name = [], [], None, ""
        for pe in players_in_entry:
            player = pe.get("player", pe)   # some responses nest, some don't
            pid = player.get("id") or player.get("playerId")
            if pid is None:
                continue
            first = (player.get("firstName") or "").strip()
            last  = (player.get("lastName")  or "").strip()
            full  = f"{first} {last}".strip()
            pids.append(str(pid))
            if full:
                names.append(full)
            if school_id is None:
                school = player.get("school", {}) or {}
                school_id   = school.get("id")
                school_name = (school.get("name") or "").strip()

        if not pids:
            continue

        if category == "singles":
            player_id = pids[0]
            pair_id   = ""
        else:
            sorted_ids = sorted(pids)
            player_id  = sorted_ids[0]          # representative ID
            pair_id    = "|".join(sorted_ids)   # canonical doubles key

        rows.append({
            "year":        year,
            "event_id":    event_id,
            "gender":      _GENDER_MAP.get(gender_id, str(gender_id)),
            "category":    category,
            "flight":      str(flight),
            "division_id": str(division_id),
            "host_id":     str(host_id),
            "seed":        int(seed_num),
            "player_id":   player_id,
            "pair_id":     pair_id,
            "name":        " / ".join(names),
            "school":      school_name,
            "school_id":   str(school_id) if school_id else "",
        })

    log.info(
        "  e=%s %s flight=%s div=%s: %d seeds parsed",
        event_id, match_type, flight, division_id, len(rows),
    )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-season fetch
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_season_seeds(
    session: aiohttp.ClientSession,
    year: int,
    event_id: int,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Fetch all seed lists for a state tournament event and return a DataFrame.
    Supports 4 flights for both Singles and Doubles.
    """
    log.info("=" * 60)
    log.info("Season %d  event_id=%d", year, event_id)
    log.info("=" * 60)

    # ── 1. Fetch event metadata ───────────────────────────────────────────────
    event_data = await fetch_event(session, event_id)
    if not event_data:
        log.error("Could not fetch event %d — skipping year %d", event_id, year)
        return pd.DataFrame()

    gender_id = event_data.get("genderId", 1)
    n_singles = int(event_data.get("flightSinglesNumber") or _DEFAULT_SINGLES_FLIGHTS)
    n_doubles = int(event_data.get("flightDoublesNumber") or _DEFAULT_DOUBLES_FLIGHTS)
    divisions = event_data.get("divisions", [])

    log.info(
        "Event %d: gender=%s  singles_flights=%d  doubles_flights=%d  divisions=%d",
        event_id, _GENDER_MAP.get(gender_id, gender_id),
        n_singles, n_doubles, len(divisions),
    )

    if not divisions:
        log.warning("No divisions in event %d — will probe with division_id=None", event_id)
        divisions = [{"id": None, "hosts": [{"id": None}]}]

    if dry_run:
        log.info(
            "[DRY RUN] Would fetch %d divisions × flights — skipping actual requests.",
            len(divisions),
        )
        return pd.DataFrame()

    # ── 2. Build task list ────────────────────────────────────────────────────
    tasks = []
    for div in divisions:
        div_id = div.get("id")
        for host in div.get("hosts", []):
            host_id = host.get("id")
            for mt, n_fl in [("Singles", n_singles), ("Doubles", n_doubles)]:
                for fl in range(1, n_fl + 1):   # flights 1-4 (or more)
                    tasks.append((div_id, host_id, mt, fl))

    log.info("Fetching %d seed lists …", len(tasks))

    all_rows: list[dict] = []
    chunk_size = 6

    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]

        seed_coros = [
            fetch_seed_list(session, event_id, div_id, host_id, mt, fl)
            for div_id, host_id, mt, fl in chunk
        ]
        results = await asyncio.gather(*seed_coros)

        for (div_id, host_id, mt, fl), seed_raw in zip(chunk, results):
            if not seed_raw:
                continue
            rows = parse_seed_entries(
                seed_raw, event_id, year, gender_id,
                mt, fl, div_id, host_id,
            )
            all_rows.extend(rows)

        await asyncio.sleep(0.15)

    if not all_rows:
        log.warning("No seed rows extracted for event %d (year %d)", event_id, year)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    log.info("Season %d: %d total seed rows", year, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Match stats enrichment
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_match_stats(
    seeds_df: pd.DataFrame,
    matches_path: Path,
) -> pd.DataFrame:
    """
    Join seed rows with season match statistics so each seed row also carries:
      wins, losses, matches_played, win_pct
    from the pre-state all_matches.csv (all_matches_excluding_state.csv preferred).

    If the matches file is missing, returns seeds_df unchanged.
    """
    if not matches_path.exists():
        log.warning("  %s not found — skipping stat enrichment", matches_path)
        return seeds_df

    import ast
    try:
        mdf = pd.read_csv(matches_path, dtype=str).fillna("")
    except Exception as e:
        log.warning("  Could not read %s: %s", matches_path, e)
        return seeds_df

    for col in ("winner_player_ids", "loser_player_ids"):
        if col in mdf.columns:
            mdf[col] = mdf[col].apply(
                lambda v: ast.literal_eval(v)
                if isinstance(v, str) and v.startswith("[") else []
            )

    win_counts:  dict[str, int] = {}
    loss_counts: dict[str, int] = {}

    for _, row in mdf.iterrows():
        for pid in row.get("winner_player_ids", []):
            win_counts[str(pid)]  = win_counts.get(str(pid), 0) + 1
        for pid in row.get("loser_player_ids", []):
            loss_counts[str(pid)] = loss_counts.get(str(pid), 0) + 1

    def _stats(pid):
        w = win_counts.get(str(pid), 0)
        l = loss_counts.get(str(pid), 0)
        m = w + l
        return w, l, m, (w / m if m > 0 else 0.0)

    seeds_df = seeds_df.copy()
    seeds_df[["wins", "losses", "matches_played", "win_pct"]] = seeds_df[
        "player_id"
    ].apply(lambda pid: pd.Series(_stats(pid)))

    log.info("  Enriched %d seed rows with match stats", len(seeds_df))
    return seeds_df


# ─────────────────────────────────────────────────────────────────────────────
# Division resolution from school_meta
# ─────────────────────────────────────────────────────────────────────────────

def _load_meta_for_year(year: int) -> dict:
    """
    Load school_meta.json + division_overrides.json for a given year
    from historical/{year}/.
    """
    year_dir = OUTPUT_ROOT / str(year)
    meta_path = year_dir / "school_meta.json"
    if not meta_path.exists():
        log.warning("  No school_meta.json for %d — divisions will be empty", year)
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    overrides_path = year_dir / "division_overrides.json"
    if overrides_path.exists():
        with open(overrides_path) as f:
            overrides = json.load(f)
        for sid, fields in overrides.items():
            s = str(sid)
            if s not in meta:
                meta[s] = {}
            meta[s].update(fields)
        log.info("  Applied %d division overrides for %d", len(overrides), year)
    return meta


_DIV_MAP = {
    "d1": "1", "d2": "2", "d3": "3", "d4": "4", "d-1": "1",
    "aa": "1", "a": "1", "d": "4", "one": "1", "ii": "2", "iv": "4",
    "2a": "2", "3a": "4", "4a": "1", "klaa": "1", "ok red": "1",
    "mac bl": "1", "mac": "2", "oaa": "2", "lvc": "3", "sec wh": "3",
    "bwac": "3", "silver": "3", "gac": "4", "tennis": "4",
}


def normalize_division(raw) -> str:
    s = str(raw or "").strip().lower()
    if s in _DIV_MAP:
        return _DIV_MAP[s]
    if s in ("1", "div 1", "div1", "division 1", "division1"):               return "1"
    if s in ("2", "div 2", "div2", "division 2", "division2"):               return "2"
    if s in ("3", "div 3", "div3", "division 3", "division3"):               return "3"
    if s in ("4/other", "4 other", "4other", "other",
             "division 4", "division4"):                                      return "4_other"
    if s == "4":                                                              return "4"
    return "4_other"


def add_division_from_meta(seeds_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Resolve each seed row's division using the explicit DIVISION_ID_MAP first
    (keyed by the integer division_id from the event API), then falling back
    to school_meta, then normalizing the raw division_id string.

    The explicit map is authoritative.
    """
    year_map = DIVISION_ID_MAP.get(year, {})

    if year_map:
        def _div_from_id(raw_id):
            try:
                return year_map.get(int(raw_id), None)
            except (TypeError, ValueError):
                return None

        seeds_df = seeds_df.copy()
        seeds_df["division"] = seeds_df["division_id"].apply(_div_from_id)

        unmapped = seeds_df[seeds_df["division"].isna()]["division_id"].unique()
        if len(unmapped) > 0:
            log.warning(
                "Year %d: unmapped division_id(s): %s — "
                "falling back to school_meta for these rows",
                year, list(unmapped),
            )
            meta = _load_meta_for_year(year)

            def _div_fallback(row):
                if row["division"] is not None:
                    return row["division"]
                sid    = str(row.get("school_id", "")).split(".")[0]
                gender = row.get("gender", "Boys")
                school = meta.get(sid, {})
                raw    = school.get(
                    "division_boys" if gender == "Boys" else "division_girls", ""
                )
                return (
                    normalize_division(raw)
                    if raw
                    else normalize_division(row.get("division_id", ""))
                )

            seeds_df["division"] = seeds_df.apply(_div_fallback, axis=1)

        log.info(
            "Year %d: division breakdown:\n%s",
            year,
            seeds_df.groupby("division").size().rename("count").to_string(),
        )
        return seeds_df

    # No explicit map — fall back to school_meta
    log.warning(
        "Year %d not in DIVISION_ID_MAP — falling back to school_meta. "
        "Consider adding it to DIVISION_ID_MAP.",
        year,
    )
    meta = _load_meta_for_year(year)
    if not meta:
        seeds_df = seeds_df.copy()
        seeds_df["division"] = seeds_df["division_id"].apply(normalize_division)
        return seeds_df

    def _div_meta(row):
        sid    = str(row.get("school_id", "")).split(".")[0]
        gender = row.get("gender", "Boys")
        school = meta.get(sid, {})
        raw    = school.get(
            "division_boys" if gender == "Boys" else "division_girls", ""
        )
        return (
            normalize_division(raw)
            if raw
            else normalize_division(row.get("division_id", ""))
        )

    seeds_df = seeds_df.copy()
    seeds_df["division"] = seeds_df.apply(_div_meta, axis=1)
    return seeds_df


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

_SEED_CSV_COLS = [
    "year", "event_id", "gender", "category", "division", "flight",
    "seed", "player_id", "pair_id", "name", "school", "school_id",
    "wins", "losses", "matches_played", "win_pct",
    # keep raw IDs for debugging
    "division_id", "host_id",
]


def save_season_output(seeds_df: pd.DataFrame, year: int, dry_run: bool = False):
    year_dir = OUTPUT_ROOT / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    out_path = year_dir / "state_seeds.csv"

    cols   = [c for c in _SEED_CSV_COLS if c in seeds_df.columns]
    out_df = seeds_df[cols].sort_values(
        ["gender", "category", "division", "flight", "seed"],
        ignore_index=True,
    )

    if dry_run:
        log.info("[DRY RUN] Would write %d rows to %s", len(out_df), out_path)
        print(out_df.head(20).to_string(index=False))
        return

    out_df.to_csv(out_path, index=False)
    log.info("Wrote %s  (%d rows)", out_path, len(out_df))

    if not out_df.empty:
        log.info(
            "  Seeds by category:\n%s",
            out_df.groupby(["gender", "category", "division", "flight"])["seed"]
            .count()
            .rename("n_seeds")
            .reset_index()
            .to_string(index=False),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(years: list[int], dry_run: bool = False):
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        for year in sorted(years):
            event_id = STATE_EVENT_IDS.get(year)
            if event_id is None:
                log.error("No event_id configured for year %d — skipping.", year)
                continue

            seeds_df = await fetch_season_seeds(session, year, event_id, dry_run=dry_run)
            if seeds_df.empty:
                continue

            # Resolve divisions from year-specific mapping / school_meta
            seeds_df = add_division_from_meta(seeds_df, year)

            # Enrich with pre-state match stats.
            # Prefer all_matches_excluding_state.csv so state results don't
            # inflate regular-season win counts.
            year_dir   = OUTPUT_ROOT / str(year)
            no_state   = year_dir / "all_matches_excluding_state.csv"
            full_match = year_dir / "all_matches.csv"
            matches_path = no_state if no_state.exists() else full_match
            seeds_df = enrich_with_match_stats(seeds_df, matches_path)

            # Write output
            save_season_output(seeds_df, year, dry_run=dry_run)

    log.info("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fetch MHSAA state tournament seeds from tennisreporting API"
    )
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=sorted(STATE_EVENT_IDS.keys()),
        help="Which seasons to fetch (default: all known years)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without writing any files",
    )
    args = ap.parse_args()

    asyncio.run(main(years=args.years, dry_run=args.dry_run))
