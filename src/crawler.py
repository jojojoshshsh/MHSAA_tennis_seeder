# crawler.py
#
# crawl_school_matches() returns:
#   (all_matches, all_matches_excluding_state, school_meta)
#
# all_matches                   — every match including state tournament
# all_matches_excluding_state   — same minus any match whose source_event_id
#                                 is a known state tournament event ID
#
# Supports 4 flights for both Singles and Doubles.
# Accepts an optional `year` parameter so a multi-year runner can override
# config.YEAR without patching the module.

import asyncio
import logging
from collections import deque

import aiohttp

from api_fetcher import (fetch_bracket, fetch_event,
                          fetch_school_report, fetch_seed_list)
from config import MAX_SCHOOLS, STATE_EVENT_IDS, TARGET_GENDER, TARGET_STATE
from match_parser import (build_event_player_lookup, extract_school_meta,
                           match_key, parse_bracket_matches,
                           parse_school_matches)

_REQUEST_DELAY   = 0.15
_MAX_CONNECTIONS = 16
_BRACKET_CHUNK   = 6

# Flat set of all known state tournament event IDs for fast lookup
_STATE_EVENT_ID_SET: set[int] = set(STATE_EVENT_IDS.values())


def _school_state(data: dict) -> str:
    return (data.get("school", {})
                .get("city", {})
                .get("state", {})
                .get("abbr") or "")


def _gender_name(gender_id) -> str | None:
    return {1: "Boys", 2: "Girls"}.get(gender_id)


def _gender_ok(target: str | None, gender: str | None) -> bool:
    return target is None or gender == target


def _is_state_match(match: dict) -> bool:
    """Return True if this match belongs to a known state tournament event."""
    eid = match.get("source_event_id")
    if eid is None:
        return False
    try:
        return int(eid) in _STATE_EVENT_ID_SET
    except (TypeError, ValueError):
        return False


async def _fetch_event_matches(session: aiohttp.ClientSession,
                               event_id, seen_keys: set) -> list[dict]:
    event_data = await fetch_event(session, event_id)
    if not event_data:
        return []

    gender_id    = event_data.get("genderId")
    event_gender = _gender_name(gender_id)
    event_date   = event_data.get("dateEventStart", "")

    # Default to 4 flights if the event doesn't advertise counts
    n_singles = int(event_data.get("flightSinglesNumber") or 4)
    n_doubles = int(event_data.get("flightDoublesNumber") or 4)

    if not _gender_ok(TARGET_GENDER, event_gender):
        logging.debug("Event %s: gender=%s skipped (target=%s)",
                      event_id, event_gender, TARGET_GENDER)
        return []

    tasks = []
    for div in event_data.get("divisions", []):
        div_id = div.get("id")
        for host in div.get("hosts", []):
            host_id = host.get("id")
            for mt, n_fl in [("Singles", n_singles), ("Doubles", n_doubles)]:
                for fl in range(1, n_fl + 1):   # flights 1-4 (or more)
                    tasks.append((div_id, host_id, mt, fl))

    logging.info("Event %s: %d bracket slices to fetch", event_id, len(tasks))

    all_matches: list[dict] = []

    for i in range(0, len(tasks), _BRACKET_CHUNK):
        chunk = tasks[i:i + _BRACKET_CHUNK]

        seed_tasks = [
            fetch_seed_list(session, event_id, div_id, host_id, mt, fl)
            for div_id, host_id, mt, fl in chunk
        ]
        bracket_tasks = [
            fetch_bracket(session, event_id, host_id, div_id, mt, fl)
            for div_id, host_id, mt, fl in chunk
        ]

        seed_results, bracket_results = await asyncio.gather(
            asyncio.gather(*seed_tasks),
            asyncio.gather(*bracket_tasks),
        )

        for (div_id, host_id, mt, fl), seed_list_raw, bdata in zip(
            chunk, seed_results, bracket_results
        ):
            if not seed_list_raw or not bdata:
                continue

            player_lookup = build_event_player_lookup(seed_list_raw)
            if not player_lookup:
                logging.warning(
                    "Event %s %s flight %s: seed list returned no usable players",
                    event_id, mt, fl,
                )
                continue

            matches = parse_bracket_matches(
                bdata, event_id, event_date, gender_id, mt, fl, player_lookup
            )
            for m in matches:
                key = match_key(m)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_matches.append(m)

        await asyncio.sleep(_REQUEST_DELAY)

    logging.info("Event %s: %d new matches extracted.", event_id, len(all_matches))
    return all_matches


async def crawl_school_matches(
    seed_id: int,
    max_schools: int | None = MAX_SCHOOLS,
    year: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    Crawl all matches reachable from seed_id for the given year.

    Parameters
    ----------
    seed_id      : starting school ID for BFS crawl
    max_schools  : cap on schools visited (None = unlimited)
    year         : season year; defaults to config.YEAR if not supplied

    Returns
    -------
    all_matches                 : list[dict]  — includes state tournament
    all_matches_excluding_state : list[dict]  — state matches removed
    school_meta                 : dict        — {school_id: {...}}
    """
    processed:      set   = set()
    queue:          deque = deque([seed_id])
    all_matches:    list  = []
    seen_keys:      set   = set()
    seen_event_ids: set   = set()
    school_meta:    dict  = {}
    skipped_oos:    int   = 0

    connector = aiohttp.TCPConnector(limit=_MAX_CONNECTIONS)
    async with aiohttp.ClientSession(connector=connector) as session:

        while queue:
            school_id = queue.popleft()
            if school_id in processed:
                continue
            if max_schools is not None and len(processed) >= max_schools:
                logging.info("max_schools=%d reached.", max_schools)
                break

            processed.add(school_id)
            # Pass year through so the correct season URL is built
            data = await fetch_school_report(session, school_id,
                                             gender_id=1, year=year)
            if data is None:
                continue

            # ── state filter ───────────────────────────────────────────────
            if TARGET_STATE:
                state = _school_state(data)
                if state and state.upper() != TARGET_STATE.upper():
                    skipped_oos += 1
                    await asyncio.sleep(_REQUEST_DELAY)
                    continue

            meta = extract_school_meta(data)
            if meta["id"]:
                school_meta[meta["id"]] = meta

            meets = data.get("meets", [])
            logging.info("School %s: %d meets in response", school_id, len(meets))

            logging.info(
                "Processing school %s  [done=%d  queued=%d  matches=%d]",
                school_id, len(processed), len(queue), len(all_matches),
            )

            # ── regular-season matches ─────────────────────────────────────
            season_matches = parse_school_matches(data, source_school_id=school_id)
            logging.info(
                "School %s: parse_school_matches returned %d matches",
                school_id, len(season_matches),
            )

            for m in season_matches:
                key = match_key(m)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_matches.append(m)
                for sid in (m["winner_school_id"], m["loser_school_id"]):
                    if sid and sid not in processed:
                        queue.append(sid)

            # ── event/tournament bracket fetching ─────────────────────────
            for meet in data.get("meets", []):
                eid = meet.get("eventId")
                if eid and eid not in seen_event_ids:
                    seen_event_ids.add(eid)
                    event_matches = await _fetch_event_matches(
                        session, eid, seen_keys
                    )
                    for m in event_matches:
                        all_matches.append(m)
                        for sid in (m["winner_school_id"], m["loser_school_id"]):
                            if sid and sid not in processed:
                                queue.append(sid)

            await asyncio.sleep(_REQUEST_DELAY)

    # ── build the excluding-state list ────────────────────────────────────────
    all_matches_excluding_state = [
        m for m in all_matches if not _is_state_match(m)
    ]

    state_count = len(all_matches) - len(all_matches_excluding_state)
    logging.info(
        "Crawl complete — MI schools: %d  OOS skipped: %d  "
        "total matches: %d  state matches: %d  non-state matches: %d",
        len(processed) - skipped_oos,
        skipped_oos,
        len(all_matches),
        state_count,
        len(all_matches_excluding_state),
    )
    return all_matches, all_matches_excluding_state, school_meta
