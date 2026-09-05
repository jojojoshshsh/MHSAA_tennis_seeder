"""
MHSAA-Style Seeding/Ranking Engine — TRANSITIVITY-FIRST REWRITE
==================================================================

New ranking algorithm (replaces the old multi-rule cmp_to_key sort):

  STEP 1 — Cycle removal
      Build the winner->loser "beats" graph for the whole cross-division
      group. Repeatedly find ANY cycle (length >= 2, e.g. A beat B and B
      beat A is a 2-cycle) and remove the OLDEST match on that cycle.
      Repeat until the graph is fully acyclic (a DAG). This is the same
      "remove the oldest contradicting result" idea as the original
      resolve_cycles(), but it now runs on cycles of every length, not
      just length > 4, because the new algorithm needs a clean DAG to
      build a transitive closure from.

  STEP 2 — Transitivity-only seed order
      On the acyclic graph, compute each player's transitive-win set
      (everyone they reach by any chain of wins). Players are sorted by
      the SIZE of that reachable set, descending — i.e. "beats the most
      people, directly or transitively" ranks first. Ties (which will
      happen whenever two players aren't comparable in the DAG) are
      broken by recency, then by average score margin, then TrueSkill
      conservative rating, then — as an absolute last resort, purely to
      guarantee a single deterministic order — alphabetically by name.
      This produces one strict, tie-free order with no shortcuts taken
      on win/loss logic, and the same input always produces the same
      output.

  STEP 3 — Adjacent fix-up pass (cross-division, run BEFORE splitting)
      Walk the order top to bottom comparing each adjacent pair (i, i+1)
      using, in this strict order:
          1. head-to-head (direct result between the two)
          2. common opponents — decided first by WIN COUNT among shared
             opponents, then by TOTAL SETS PLAYED against those shared
             opponents (fewer sets = more efficient/dominant win, so it
             ranks higher), and only if that's ALSO tied, by cumulative
             signed score margin against those shared opponents
          3. dominance — multi-hop reachability in the same DAG used in
             step 2 (does one of them transitively beat the other?)
          4. TrueSkill conservative rating (last resort before random)
      If the lower-ranked player should outrank the one above them by
      any of these four checks, swap them. Re-run full top-to-bottom
      passes until one entire pass produces zero swaps (fully stable).

  STEP 4 — Division split
      Once the cross-division order is stable, split players into their
      own divisions, preserving relative order from the unified ranking.
      No new comparisons happen at this stage — division splitting is a
      pure partition of the already-final order.

  STEP 4b — Cross-division "general" ranking (NEW)
      In addition to the per-division seed lists from step 4, the same
      transitivity-only order from step 2 is ALSO run through the
      adjacent fix-up pass against the FULL cross-division roster (no
      division split at all). This produces one flight-wide ranking
      (e.g. "Boys Singles Flight 2, every division combined") using the
      exact same head-to-head / common-opponents / dominance / TrueSkill
      rules as the per-division fix-up. It's published as its own
      pseudo-division named "overall", on top of (not replacing) the
      normal per-division output.

DETERMINISM
-------------------------------
  This engine is fully deterministic: the same input CSV always produces
  the exact same output, run after run, regardless of PYTHONHASHSEED or
  process restarts. Two things previously broke that guarantee and have
  been fixed:

    1. `random.random()` was used as the very-last-resort tiebreak in
       the seed order. It is gone. The last-resort tiebreak is now
       alphabetical by player name — deterministic, and only ever
       reached when two players are DAG-incomparable AND tied on
       recency, margin, and TrueSkill mu (rare).

    2. Cycle detection iterated over `set[str]` children in whatever
       order Python's hash randomization handed back that run. Since
       which cycle is discovered first determines which "oldest match"
       gets deleted to break it, this could silently produce a
       different acyclic graph — and therefore different rankings — on
       two runs over identical input. Children are now visited in
       sorted order.

  Floating-point comparisons (score margins, TrueSkill mu diffs, TGRS
  ties) were already epsilon-guarded (1e-9) and are unaffected — they
  are deterministic as long as the match set feeding them is (which the
  two fixes above now guarantee).

RANKING-EXCLUDED MATCHES
-------------------------------
  Some matches are short-circuited results that shouldn't influence
  anyone's ranking but should still be visible in win/loss records:
    - matches that end after only ONE set (e.g. a retirement: the score
      string contains exactly one "W-L" set token),
    - matches whose score is literally "2-0 2-0" (a placeholder/forfeit-
      style short-set score, not a real two real sets played to a normal
      conclusion), and
    - matches where either side's name contains the word "default"
      (case-insensitive; e.g. a placeholder opponent name like "Smith
      Default" or, for doubles, "Jones/Default") — these are walkovers
      recorded as a "match" against a stand-in name rather than a real
      completed result.
  These matches are flagged at load time (is_ranking_excluded) and are
  then filtered out of every consideration the ranking core makes:
  cycle-detection, the beats graph, the transitive closure, head-to-head,
  common-opponent sets, recency, margin, and TrueSkill. They are NOT
  filtered out of `records` (matches/wins/losses) or CSV/console output —
  they still count there exactly as before.

RECORDS (WINS/LOSSES) — COUNTED BEFORE ANY REMOVAL
-------------------------------
  A player's `wins`/`losses`/`matches` record is computed from the FULL
  set of matches in their (gender, match_type, flight) bucket, exactly
  as loaded — before the MIN_MATCHES drop, before the per-school dedup
  pass, and across every division in that bucket (not just their own).
  This means a player's win/loss totals reflect everyone they actually
  played, including opponents from other divisions and opponents who
  were later dropped from ranking consideration for having too few
  matches. Nothing about the ranking core, seed order, or any other
  output column is affected by this — only the `wins`/`losses`/
  `matches`/`last_match_date` figures are sourced from the wider set.

  The new comeback stat (won_after_set1_loss) and the opponent-strength
  record (vs_weaker_opp / vs_mid_opp / vs_top_opp) are ALSO sourced from
  this same wide, pre-filter/pre-dedup snapshot — see section 9c below —
  for exactly the same reason: they should reflect everything a player
  actually played, not just the matches that survived MIN_MATCHES/dedup
  filtering.

TRUESKILL (UPDATED — now margin-aware)
-------------------------------
  Ratings are computed by trueskill_engine_v2.compute_trueskill_margin(),
  not the plain win/loss compute_trueskill(). Each ranking-eligible match
  now carries its set score into the rating update, so a 6-0 6-0 win
  moves ratings more than a 7-6 7-6 win, and a wildly inconsistent win
  like 7-6 0-6 7-6 is treated as a bare, non-dominant win rather than
  being credited (or penalized) based on raw game differential. See
  trueskill_engine_v2.py for the full reasoning. Everything downstream
  (reach/dominance, sos, quality_wins, TGRS, tiebreak rule 4) consumes
  whatever Rating objects come back, so it's agnostic to which engine
  produced them — only the import and the match_pairs construction
  changed.

OUTPUT LAYER (UPDATED)
-------------------------------
  Output now writes directly into the shape that scripts/build_site.py
  expects, instead of one CSV per (gender, match_type, division, flight)
  group under historical/{year}/seeds/:

    src/rankings_by_division_flight/
        singles_boys_division_1.csv        ...one per (category, gender, division),
        doubles_girls_division_3.csv       containing ALL flights for that combo,
        singles_boys_division_overall.csv  ...plus one per (category, gender)
        doubles_girls_division_overall.csv cross-division "general" ranking,
        ...                                 all with a "flight" column to subdivide.
        team_boys_division_1.csv      one per (gender, division), team-level
        team_girls_division_4.csv     aggregates built by team_aggregation.py
                                       (NOT built for the "overall" pseudo-
                                       division — see write_team_csvs)

  Each individual CSV row carries the column set build_site.py looks for:
    rank, name / pair_name, school, division, flight, wins, losses,
    TGRS, TGRS_scaled, ts_rating, ts_mu, local_ts_mu, ts_sigma,
    reachability, local_reachability, sos, local_sos, quality_wins,
    won_after_set1_loss, vs_weaker_opp, vs_mid_opp, vs_top_opp,
    last_match_date, reason_below

  "reason_below" is a human-readable explanation of why this player is
  seeded below the player immediately above them (e.g. "Lost head-to-head",
  "Fewer wins vs common opponents"). Empty for the #1 seed.

  "won_after_set1_loss" is a comeback-rate stat: of the matches where this
  player lost set 1, what fraction did they still go on to win overall
  (formatted "3/5 (60%)"). Blank if they never lost a first set.

  "vs_weaker_opp" / "vs_mid_opp" / "vs_top_opp" is a win-loss record
  against opponents grouped by TrueSkill conservative-rating percentile
  — see section 9c for exactly how the thresholds and buckets work.

  "local_*" columns are computed within just that player's division/
  flight group (the existing per-division ranking pass); the non-local
  versions (ts_rating / ts_mu / reachability / sos) are computed from the
  SAME underlying TrueSkill ratings and transitive-closure reach sets
  used by the ranking core, but reported pre-division-split (i.e. across
  the full cross-division group) so they're comparable across divisions
  too. TGRS ("Team/Group Ranking Score") is a single composite number —
  see _compute_tgrs() below for the exact formula.

  For the "overall" pseudo-division (step 4b), "local_*" columns are
  identical to their non-local counterparts, since there's no division
  scoping left to distinguish them from — the whole cross-division group
  IS the local pool at that point.

Everything in the ranking core (cycle removal, transitive closure,
adjacent fix-up, CSV loading, division normalisation, school lookup) is
unchanged from the previous version, aside from the determinism fixes
called out above. Section 9c (comeback stat + opponent-strength record),
section 12 (output helpers), run() (section 11), the TrueSkill
import/call in process_group (section 10), and the new STEP 4b
cross-division "overall" ranking (also in process_group) were
added/rewritten.
"""

import csv
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path

from trueskill_engine_v2 import compute_trueskill_margin
from team_aggregation import build_team_rankings

import config as _config
import logging
log = logging.getLogger(__name__)

# ============================================================================
# A.  Division normalisation  (unchanged)
# ============================================================================

_DIVISION_MAP: dict[str, str] = {
    "1": "1", "2": "2", "3": "3", "4": "4",
    "d1": "1", "d2": "2", "d3": "3", "d4": "4",
    "d-1": "1", "d-2": "2", "d-3": "3", "d-4": "4",
    "div1": "1", "div2": "2", "div3": "3", "div4": "4",
    "div 1": "1", "div 2": "2", "div 3": "3", "div 4": "4",
    "division 1": "1", "division 2": "2", "division 3": "3", "division 4": "4",
    "division1": "1", "division2": "2", "division3": "3", "division4": "4",
    "i": "1", "ii": "2", "iii": "3", "iv": "4",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "aa": "1", "a": "1", "4a": "1", "klaa": "1", "ok red": "1",
    "mac bl": "1", "mac blue": "1", "ok white": "1", "okac": "1",
    "mac": "2", "oaa": "2", "2a": "2",
    "lvc": "3", "sec wh": "3", "bwac": "3", "silver": "3", "3a": "3",
    "gac": "4", "tennis": "4", "d": "4",
}

VALID_DIVISIONS: frozenset[str] = frozenset({"1", "2", "3", "4"})


def normalize_player_name(name: str) -> str:
    """
    Doubles teams are written as "Player A/Player B". The two names can
    appear in either order in the source data ("Josh/Gio" vs "Gio/Josh")
    but represent the same team, so canonicalize by alphabetizing the
    slash-separated parts. Singles names (no "/") pass through unchanged.
    """
    parts = [p.strip() for p in name.split("/")]
    if len(parts) > 1:
        parts = sorted(parts)
    return "/".join(parts)


def normalize_division(raw) -> str | None:
    s = str(raw or "").strip().lower().strip(".")
    if not s:
        return None
    return _DIVISION_MAP.get(s)


# ============================================================================
# B.  school_meta.json lookup  (unchanged)
# ============================================================================

def load_school_meta(csv_path: str) -> dict:
    script_dir = Path(__file__).parent.resolve()
    csv_dir = Path(csv_path).parent.resolve()
    for candidate in [csv_dir / "school_meta.json", script_dir / "school_meta.json"]:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                meta = json.load(f)
            print(f"  school_meta loaded: {candidate} ({len(meta)} entries)")
            return meta
    return {}


def load_correct_divisions(csv_path: str) -> dict[str, str]:
    """
    Optional override table: a tab- or comma-separated file named
    "correct_divisions.csv" with two columns, "school" and "division",
    living next to the input CSV (or next to this script). When a match's
    winner_school or loser_school name is found in this table, that
    division WINS over anything inferred from the raw CSV division column
    or school_meta.json — this is the authoritative source of truth.
    """
    script_dir = Path(__file__).parent.resolve()
    csv_dir = Path(csv_path).parent.resolve()
    repo_root = script_dir.parent.resolve()
    for candidate in [
        csv_dir / "correct_divisions.csv",
        script_dir / "correct_divisions.csv",
        repo_root / "data" / "correct_divisions.csv",
    ]:
        if candidate.exists():
            overrides: dict[str, str] = {}
            with open(candidate, newline="", encoding="utf-8") as f:
                sample = f.read(2048)
                f.seek(0)
                delimiter = "\t" if "\t" in sample else ","
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
            start = 0
            if rows and rows[0] and rows[0][0].strip().lower() in ("school", ""):
                start = 1
            for row in rows[start:]:
                if len(row) < 2:
                    continue
                school = row[0].strip()
                division = normalize_division(row[1])
                if school and division:
                    overrides[school] = division
            print(f"  correct_divisions loaded: {candidate} ({len(overrides)} schools)")
            return overrides
    return {}


def division_from_school(school_id, gender: str, school_meta: dict) -> str | None:
    sid = str(school_id or "").split(".")[0].strip()
    if not sid or not school_meta:
        return None
    entry = school_meta.get(sid, {})
    key = "division_boys" if str(gender).lower() == "boys" else "division_girls"
    raw = entry.get(key, "")
    return normalize_division(raw) if raw else None


# ============================================================================
# C.  Ranking-excluded match detection
# ============================================================================

def _parse_set_tokens(score_str: str) -> list[tuple[int, int]]:
    """
    Parse a score string like "6-2 6-3" or "2-0 2-0" into a list of
    (won, lost) integer tuples, one per set token — from the MATCH
    WINNER's perspective (i.e. token[0] is games won by whoever won the
    overall match, token[1] is games won by whoever lost it). Non-numeric
    or malformed tokens are skipped, matching parse_score_margin's
    existing tolerance for messy data.
    """
    sets: list[tuple[int, int]] = []
    for part in score_str.split():
        halves = part.split("-")
        if len(halves) == 2:
            try:
                sets.append((int(halves[0]), int(halves[1])))
            except ValueError:
                pass
    return sets


def is_ranking_excluded_score(score_str: str) -> bool:
    """
    True for matches that should be KEPT in win/loss records but EXCLUDED
    from every ranking consideration (cycles, transitive closure, h2h,
    common opponents, recency, margin, TrueSkill). Two patterns trigger
    this:

      1. The match ended after only one set — the score string has
         exactly one parseable "W-L" set token (e.g. a retirement after
         set 1, score = "6-2").
      2. The score is literally two sets of "2-0" each (score = "2-0 2-0"),
         a short-set placeholder pattern rather than a normally completed
         match.

    Matches with no parseable sets at all (empty/garbage score strings)
    are left alone here — that's a separate data-quality issue, not what
    was asked for.
    """
    sets = _parse_set_tokens(score_str)
    if len(sets) == 1:
        return True
    if len(sets) == 2 and sets[0] == (2, 0) and sets[1] == (2, 0):
        return True
    return False


def _name_is_default(name: str) -> bool:
    """
    True if "default" appears anywhere in a player/pair name
    (case-insensitive) — e.g. "Smith Default", "Default", or a doubles
    pair like "Jones/Default" (the substring check naturally covers
    either half of a slash-joined pair name, so nothing pair-specific is
    needed here). Used to flag walkover/default "matches" so they can be
    excluded from ranking consideration while still counting toward
    win/loss records.
    """
    return "default" in (name or "").lower()


def is_default_match(winner: str, loser: str) -> bool:
    """
    True if either side of the match has "default" in their name. These
    are treated as ranking-excluded for the same reason as
    is_ranking_excluded_score: not a real completed result, but the
    outcome should still show up in wins/losses/matches/last_match_date
    for the real player involved.
    """
    return _name_is_default(winner) or _name_is_default(loser)


# ============================================================================
# 1.  CSV Loading  (adds is_ranking_excluded flag)
# ============================================================================

def load_matches(filepath: str, school_meta: dict | None = None,
                  correct_divisions: dict | None = None) -> list[dict]:
    if school_meta is None:
        school_meta = {}
    if correct_divisions is None:
        correct_divisions = {}

    matches = []
    skipped_div = 0
    skipped_flight = 0
    excluded_from_ranking = 0
    unknown_div_values: set[str] = set()
    unknown_flight_values: set[str] = set()

    with open(filepath, newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.DictReader(f, delimiter=delimiter)

        for row in reader:
            raw_date = row.get("match_updated_at", "") or row.get("date", "")
            try:
                match_date = datetime.fromisoformat(
                    raw_date.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except (ValueError, AttributeError):
                match_date = datetime.min

            winner = normalize_player_name(row.get("winner_names", row.get("winner", "")).strip())
            loser = normalize_player_name(row.get("loser_names", row.get("loser", "")).strip())
            if not winner or not loser:
                continue

            score = row.get("set_score", row.get("score", "")).strip()
            gender = row.get("gender", "").strip()
            match_type = row.get("match_type", "").strip()

            raw_flight = row.get("flight", "").strip()
            if raw_flight not in ("1", "2", "3", "4"):
                skipped_flight += 1
                if raw_flight:
                    unknown_flight_values.add(raw_flight)
                continue
            flight = raw_flight

            winner_school = (
                row.get("winner_school_name", "")
                or row.get("winner_school", "")
                or row.get("winner_team", "")
            ).strip()
            loser_school = (
                row.get("loser_school_name", "")
                or row.get("loser_school", "")
                or row.get("loser_team", "")
            ).strip()

            raw_div = row.get("division", "").strip()

            def _resolve_division(school_name: str, school_id) -> str | None:
                """
                Resolve a single player's division using ONLY that
                player's own school — never the opponent's. Priority:
                  1. correct_divisions.csv (by this player's school name)
                  2. the raw CSV "division" column
                  3. school_meta.json (by this player's school id)
                """
                if correct_divisions and school_name in correct_divisions:
                    return correct_divisions[school_name]
                if raw_div:
                    d = normalize_division(raw_div)
                    if d:
                        return d
                return division_from_school(school_id, gender, school_meta)

            winner_division = _resolve_division(winner_school, row.get("winner_school_id", ""))
            loser_division = _resolve_division(loser_school, row.get("loser_school_id", ""))

            if winner_division is None or loser_division is None:
                skipped_div += 1
                if raw_div:
                    unknown_div_values.add(raw_div)
                continue

            # A match is excluded from ranking consideration (but still
            # kept in win/loss records) if it's a short-circuited score
            # (single set / "2-0 2-0") OR if either side's name contains
            # "default" (a walkover recorded against a placeholder name).
            ranking_excluded = (
                is_ranking_excluded_score(score)
                or is_default_match(winner, loser)
            )
            if ranking_excluded:
                excluded_from_ranking += 1

            matches.append({
                "date": match_date,
                "winner": winner,
                "loser": loser,
                "score": score,
                "gender": gender,
                "match_type": match_type,
                "flight": flight,
                "winner_division": winner_division,
                "loser_division": loser_division,
                "winner_school": winner_school,
                "loser_school": loser_school,
                "is_ranking_excluded": ranking_excluded,
            })

    if skipped_flight:
        print(f"  WARNING: {skipped_flight} row(s) skipped — flight not in (1,2,3,4).")
        if unknown_flight_values:
            listed = ", ".join(f'"{v}"' for v in sorted(unknown_flight_values))
            print(f"  Unrecognised flight values found: {listed}")

    if skipped_div:
        print(f"  WARNING: {skipped_div} row(s) skipped — division could not be "
              f"resolved from CSV or school_meta.")
        if unknown_div_values:
            listed = ", ".join(f'"{v}"' for v in sorted(unknown_div_values))
            print(f"  Unrecognised division values found: {listed}")
            print("  Add these to _DIVISION_MAP or fix school_meta.json to include them.")

    if excluded_from_ranking:
        print(f"  NOTE: {excluded_from_ranking} match(es) end in a single set, "
              f"\"2-0 2-0\", or involve a 'default' placeholder name — kept in "
              f"records, excluded from ranking consideration.")

    return matches


# ============================================================================
# 2.  Grouping  (unchanged)
# ============================================================================

def bucket_matches(matches: list[dict]) -> dict[tuple, list[dict]]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for m in matches:
        key = (m["gender"], m["match_type"], m["flight"])
        buckets[key].append(m)
    return dict(buckets)


def players_in_group(matches: list[dict]) -> list[str]:
    """
    Every distinct player/pair name that appears in this group's matches
    — EXCEPT names containing "default" (see _name_is_default). Those
    are placeholder walkover names, not real players, so they're kept
    out of the player pool entirely here: they never get a records
    entry, never enter the ranking core, never occupy a seeded slot in
    output, and — critically — never get picked as a school's dedup
    representative (build_school_map/school_to_players in process_group
    both derive their candidate lists from this function). The real
    opponent in a default match still keeps that match on their own
    record via original_group_matches/original_results_idx, which is
    unaffected by this filter.
    """
    seen: set[str] = set()
    for m in matches:
        if not _name_is_default(m["winner"]):
            seen.add(m["winner"])
        if not _name_is_default(m["loser"]):
            seen.add(m["loser"])
    return sorted(seen)


def build_division_map(matches: list[dict]) -> dict[str, str]:
    """
    Each player's division now comes ONLY from data tied to their own
    school (correct_divisions.csv / school_meta / raw CSV), resolved
    independently for the winner and loser of every match — never
    inherited from the opponent. If a player's matches ever disagree
    (e.g. a stray school-name typo), the most frequently seen division
    for that player wins.

    NOTE: this intentionally uses ALL matches (including ranking-excluded
    ones) — division assignment is a data-quality lookup, not a ranking
    consideration, so excluded matches still count here.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in matches:
        counts[m["winner"]][m["winner_division"]] += 1
        counts[m["loser"]][m["loser_division"]] += 1

    div: dict[str, str] = {}
    for player, tally in counts.items():
        div[player] = max(tally.items(), key=lambda kv: kv[1])[0]
    return div


def build_school_map(matches: list[dict]) -> dict[str, str]:
    school: dict[str, str] = {}
    for m in matches:
        if m["winner_school"]:
            school[m["winner"]] = m["winner_school"]
        if m["loser_school"]:
            school[m["loser"]] = m["loser_school"]
    return school


# ============================================================================
# 3.  Head-to-head index  (built from ranking-eligible matches only)
# ============================================================================

def build_h2h_index(matches: list[dict]) -> dict[tuple, dict]:
    h2h: dict[tuple, dict] = {}
    for m in matches:
        w, l = m["winner"], m["loser"]
        key = (w, l) if w < l else (l, w)
        cur = h2h.get(key)
        if cur is None or m["date"] > cur["date"]:
            h2h[key] = m
    return h2h


def head_to_head_result(a: str, b: str, h2h: dict) -> str | None:
    m = h2h.get((a, b) if a < b else (b, a))
    if m is None:
        return None
    return "a" if m["winner"] == a else "b"


# ============================================================================
# 4.  Results index  (unchanged — used for both full records AND, when
#     given a pre-filtered match list, ranking-only indexing)
# ============================================================================

def build_results_index(matches: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        idx[m["winner"]].append(m)
        idx[m["loser"]].append(m)
    for lst in idx.values():
        lst.sort(key=lambda x: x["date"])
    return idx


def build_records(
    players: list[str],
    results_idx: dict[str, list[dict]],
) -> dict[str, dict]:
    """Total matches / wins / losses per player (used for the 'matches'
    and 'record' columns in the output CSVs).

    IMPORTANT: callers should pass a results_idx built from the FULL,
    unfiltered match set for the bucket (i.e. before the MIN_MATCHES
    drop and before per-school dedup pass, and including matches against
    players from every division in the bucket) — records intentionally
    still count single-set / "2-0 2-0" / "default"-name matches, matches
    against players who were later dropped from ranking, and matches
    against opponents in other divisions. This is purely a "how many
    matches did this player actually play, and what was the outcome"
    tally; it never feeds the ranking core."""
    records: dict[str, dict] = {}
    for p in players:
        ms = results_idx.get(p, [])
        wins = sum(1 for m in ms if m["winner"] == p)
        losses = len(ms) - wins
        last_date = max((m["date"] for m in ms), default=None)
        records[p] = {
            "matches": len(ms),
            "wins": wins,
            "losses": losses,
            "last_match_date": last_date,
        }
    return records


def build_opponent_sets(
    players: list[str],
    results_idx: dict[str, list[dict]],
) -> dict[str, set[str]]:
    opp: dict[str, set[str]] = {}
    for p in players:
        opp[p] = {
            m["loser"] if m["winner"] == p else m["winner"]
            for m in results_idx.get(p, [])
        }
    return opp


# ============================================================================
# 5.  Score margin parsing  (unchanged, cached)
# ============================================================================

_MARGIN_CACHE: dict[str, float] = {}


def parse_score_margin(score_str: str) -> float:
    cached = _MARGIN_CACHE.get(score_str)
    if cached is not None:
        return cached
    won = lost = 0
    for part in score_str.split():
        halves = part.split("-")
        if len(halves) == 2:
            try:
                won += int(halves[0])
                lost += int(halves[1])
            except ValueError:
                pass
    result = float(won - lost)
    _MARGIN_CACHE[score_str] = result
    return result


# ============================================================================
# 6.  Recency / margin precompute  (unchanged — callers pass a
#     ranking-eligible-only results_idx)
# ============================================================================

def precompute_recency(
    players: list[str],
    results_idx: dict[str, list[dict]],
) -> dict[str, float]:
    now = datetime.now()
    scores: dict[str, float] = {}
    for p in players:
        s = 0.0
        for m in results_idx.get(p, []):
            days_ago = max(1, (now - m["date"]).days)
            s += (1.0 / days_ago) if m["winner"] == p else (-0.5 / days_ago)
        scores[p] = s
    return scores


def precompute_margins(
    players: list[str],
    results_idx: dict[str, list[dict]],
) -> dict[str, float]:
    margins: dict[str, float] = {}
    for p in players:
        vals: list[float] = []
        for m in results_idx.get(p, []):
            raw = parse_score_margin(m["score"])
            vals.append(raw if m["winner"] == p else -raw)
        margins[p] = sum(vals) / len(vals) if vals else 0.0
    return margins


# ============================================================================
# 7.  STEP 1 — Cycle removal (ALL cycle lengths, not just > 4)
# ============================================================================

def _directed_beats_from_matches(active: list[dict]) -> dict[str, set[str]]:
    beats: dict[str, set[str]] = defaultdict(set)
    for m in active:
        beats[m["winner"]].add(m["loser"])
    return beats


def _find_any_cycle(
    beats: dict[str, set[str]],
    players: list[str],
) -> list[str] | None:
    """
    Iterative DFS with explicit backtracking (O(V+E), no path copying).
    Returns the first cycle found (any length >= 2), or None if the graph
    is already acyclic. Unlike the old engine, this does NOT ignore short
    cycles — every cycle must be broken before step 2 can build a clean
    transitive closure.

    DETERMINISM: `beats[node]` is a set[str], and Python's set iteration
    order for strings depends on hash randomization (PYTHONHASHSEED),
    which varies process-to-process. Since which cycle is discovered
    first determines which "oldest match" gets removed to break it, that
    would let two runs over identical input silently land on a different
    acyclic graph. We visit children in sorted() order instead, so the
    DFS traversal — and therefore the cycle found, and therefore the
    match removed — is always the same for the same input.
    """
    player_set = set(players)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {p: WHITE for p in players}

    path: list[str] = []
    path_idx: dict[str, int] = {}

    for start in players:
        if color[start] != WHITE:
            continue

        color[start] = GRAY
        path_idx[start] = len(path)
        path.append(start)
        stack = [(start, iter(sorted(beats.get(start, set()))))]

        while stack:
            node, children = stack[-1]
            pushed = False

            for nxt in children:
                if nxt not in player_set:
                    continue
                c = color[nxt]
                if c == GRAY:
                    idx = path_idx[nxt]
                    return path[idx:]
                elif c == WHITE:
                    color[nxt] = GRAY
                    path_idx[nxt] = len(path)
                    path.append(nxt)
                    stack.append((nxt, iter(sorted(beats.get(nxt, set())))))
                    pushed = True
                    break

            if not pushed:
                stack.pop()
                color[node] = BLACK
                path.pop()
                del path_idx[node]

    return None


def _oldest_match_on_cycle(cycle: list[str], active: list[dict]) -> dict | None:
    k = len(cycle)
    edges: set[tuple[str, str]] = {
        (cycle[i], cycle[(i + 1) % k]) for i in range(k)
    }
    oldest: dict | None = None
    for m in active:
        if (m["winner"], m["loser"]) in edges:
            if oldest is None or m["date"] < oldest["date"]:
                oldest = m
    return oldest


def resolve_all_cycles(matches: list[dict], players: list[str]) -> list[dict]:
    """
    Repeatedly find ANY cycle and drop the oldest match on it, until the
    beats graph is fully acyclic. This is the foundation step the rest of
    the algorithm depends on — everything downstream assumes a DAG.

    `matches` is expected to already be the ranking-eligible subset
    (ranking-excluded matches filtered out by the caller).
    """
    active = list(matches)
    for _ in range(len(matches) + 1):
        beats = _directed_beats_from_matches(active)
        cycle = _find_any_cycle(beats, players)
        if cycle is None:
            break
        oldest = _oldest_match_on_cycle(cycle, active)
        if oldest is None:
            break
        active = [m for m in active if m is not oldest]
    return active


# ============================================================================
# 8.  STEP 2 — Transitive closure + transitivity-only seed order
# ============================================================================

def build_transitive_closure(
    players: list[str],
    beats: dict[str, set[str]],
) -> dict[str, set[str]]:
    """
    `beats` must already be acyclic. Returns reach[p] = every player p
    beats, directly or through any chain of wins, computed via Kahn's
    topological sort + DP in reverse topo order (each node's reach =
    union of its direct beats' own reach sets, plus the direct beats
    themselves). O(V+E) topo sort, O(V*E) worst case for the unions —
    fine at typical roster sizes.
    """
    indeg = {p: 0 for p in players}
    for p in players:
        for q in beats.get(p, ()):
            if q in indeg:
                indeg[q] += 1

    queue = deque(p for p in players if indeg[p] == 0)
    topo_order: list[str] = []
    indeg_work = dict(indeg)
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for nxt in sorted(beats.get(node, ())):
            if nxt in indeg_work:
                indeg_work[nxt] -= 1
                if indeg_work[nxt] == 0:
                    queue.append(nxt)
    if len(topo_order) < len(players):
        seen = set(topo_order)
        topo_order.extend(p for p in players if p not in seen)

    reach: dict[str, set[str]] = {p: set() for p in players}
    for node in reversed(topo_order):
        s = set(beats.get(node, ()))
        for child in list(s):
            s |= reach.get(child, set())
        reach[node] = s
    return reach


def transitivity_seed_order(
    players: list[str],
    reach: dict[str, set[str]],
    recency: dict[str, float],
    margins: dict[str, float],
    trueskill_ratings: dict,
) -> list[str]:
    """
    STEP 2 ranking: sort by size of transitive-win set, descending.
    Ties (incomparable players in the DAG) fall back to recency, then
    average margin, then TrueSkill conservative rating, then — as an
    absolute last resort, purely to force a single deterministic
    ordering — alphabetically by player name. These ties get a real
    chance to be fixed by h2h / common opponents / dominance in the
    per-division fix-up pass; the name tiebreak just guarantees the sort
    itself is fully deterministic, run after run, for identical input.
    """
    return sorted(
        players,
        key=lambda p: (
            -len(reach.get(p, ())),
            -recency.get(p, 0.0),
            -margins.get(p, 0.0),
            -(trueskill_ratings[p].mu if p in trueskill_ratings else 0.0),
            p,
        ),
    )


# ============================================================================
# 9.  STEP 3 — Adjacent fix-up pass
# ============================================================================

def common_opponent_comparison(
    a: str, b: str,
    opp_sets: dict[str, set[str]],
    h2h: dict,
) -> tuple[str | None, str | None]:
    """
    Common-opponents comparison, decided in this order:
      1. win count among shared opponents
      2. total sets played against those shared opponents — fewer sets
         means a more efficient/dominant path through the same common
         opponents (e.g. beating a shared opponent in straight sets
         beats winning a 3-setter against them), so the player with the
         SMALLER total set count wins this tiebreak
      3. cumulative signed score margin against those shared opponents
         (only reached if both wins and total sets are tied)

    Returns (winner, sub_reason) where sub_reason is "wins", "sets", or
    "margin" for diagnostics, or (None, None) if no shared-opponent data
    exists.
    """
    common = (opp_sets.get(a, set()) & opp_sets.get(b, set())) - {a, b}
    if not common:
        return None, None

    a_wins = b_wins = 0
    a_sets = b_sets = 0
    a_margin = b_margin = 0.0
    counted = False
    for c in sorted(common):
        ac = h2h.get((a, c) if a < c else (c, a))
        bc = h2h.get((b, c) if b < c else (c, b))
        if ac is None or bc is None:
            continue
        counted = True
        if ac["winner"] == a:
            a_wins += 1
        if bc["winner"] == b:
            b_wins += 1

        a_sets += len(_parse_set_tokens(ac["score"]))
        b_sets += len(_parse_set_tokens(bc["score"]))

        raw_ac = parse_score_margin(ac["score"])
        raw_bc = parse_score_margin(bc["score"])
        a_margin += raw_ac if ac["winner"] == a else -raw_ac
        b_margin += raw_bc if bc["winner"] == b else -raw_bc

    if not counted:
        return None, None

    if a_wins != b_wins:
        return ("a" if a_wins > b_wins else "b"), "wins"

    if a_sets != b_sets:
        return ("a" if a_sets < b_sets else "b"), "sets"

    if abs(a_margin - b_margin) > 1e-9:
        return ("a" if a_margin > b_margin else "b"), "margin"

    return None, None


def dominance_comparison(
    a: str, b: str,
    reach: dict[str, set[str]],
) -> str | None:
    """
    Multi-hop dominance using the same transitive closure built in step 2.
    Since the underlying beats graph is now a true DAG, "a transitively
    beats b" and "b transitively beats a" can never both be true, so this
    is unambiguous.
    """
    if b in reach.get(a, set()):
        return "a"
    if a in reach.get(b, set()):
        return "b"
    return None


def compare_adjacent(
    a: str, b: str,
    h2h: dict,
    opp_sets: dict[str, set[str]],
    reach: dict[str, set[str]],
    trueskill_ratings: dict,
) -> tuple[str | None, str]:
    """
    The four-rule fix-up comparator, run strictly in this order:
      1. head-to-head
      2. common opponents (win count, then total sets played, then margin)
      3. dominance (multi-hop transitive beats)
      4. TrueSkill conservative rating (last resort)
    """
    r = head_to_head_result(a, b, h2h)
    if r is not None:
        return r, "head-to-head"

    r, sub = common_opponent_comparison(a, b, opp_sets, h2h)
    if r is not None:
        return r, f"common-opponents-{sub}"

    r = dominance_comparison(a, b, reach)
    if r is not None:
        return r, "dominance"

    # ── Rule 4: TrueSkill conservative rating (last resort) ──
    ra = trueskill_ratings.get(a)
    rb = trueskill_ratings.get(b)
    if ra is not None and rb is not None:
        diff = ra.mu - rb.mu
        if abs(diff) > 1e-9:
            return ("a" if diff > 0 else "b"), "trueskill"

    return None, "tied"


def adjacent_fixup(
    order: list[str],
    h2h: dict,
    opp_sets: dict[str, set[str]],
    reach: dict[str, set[str]],
    trueskill_ratings: dict,
) -> tuple[list[str], list[dict]]:
    """
    Repeated top-to-bottom adjacent-swap passes until a full pass makes
    zero swaps. Returns the final stable order plus a log of every swap
    made (for diagnostics).
    """
    cur = list(order)
    swap_log: list[dict] = []

    max_passes = max(1, len(cur)) ** 2
    for _ in range(max_passes):
        swapped = False
        i = 0
        while i < len(cur) - 1:
            a, b = cur[i], cur[i + 1]
            winner, rule = compare_adjacent(
                a, b, h2h, opp_sets, reach, trueskill_ratings
            )
            if winner == "b":
                cur[i], cur[i + 1] = b, a
                swap_log.append({
                    "moved_up": b, "moved_down": a, "decided_by": rule,
                })
                swapped = True
            i += 1
        if not swapped:
            break

    return cur, swap_log


def build_adjacency_explanations(
    order: list[str],
    h2h: dict,
    opp_sets: dict[str, set[str]],
    reach: dict[str, set[str]],
    trueskill_ratings: dict,
) -> list[dict]:
    """Re-derive the deciding rule for each adjacent pair in the FINAL
    stable order, for the seed_above/seed_below explain table."""
    out = []
    for k in range(len(order) - 1):
        a, b = order[k], order[k + 1]
        winner, rule = compare_adjacent(
            a, b, h2h, opp_sets, reach, trueskill_ratings
        )
        out.append({
            "seed_above": a,
            "seed_below": b,
            "decided_by": rule,
            "direction": winner,
        })
    return out


# ============================================================================
# 9b.  Strength-of-schedule / quality-wins / TGRS  (feeds build_site.py)
# ============================================================================

def precompute_sos(
    players: list[str],
    results_idx: dict[str, list[dict]],
    trueskill_ratings: dict,
) -> dict[str, float]:
    """
    Strength of schedule: average opponent TrueSkill mu across every
    ranking-eligible match a player played, in this group. Computed once
    on the full cross-division group (the "sos" column) and again on
    just a division/flight subset (the "local_sos" column) by passing a
    results_idx already scoped to that subset.
    """
    sos: dict[str, float] = {}
    for p in players:
        ms = results_idx.get(p, [])
        opp_mus = []
        for m in ms:
            opp = m["loser"] if m["winner"] == p else m["winner"]
            r = trueskill_ratings.get(opp)
            if r is not None:
                opp_mus.append(r.mu)
        sos[p] = sum(opp_mus) / len(opp_mus) if opp_mus else 0.0
    return sos


def precompute_quality_wins(
    players: list[str],
    results_idx: dict[str, list[dict]],
    trueskill_ratings: dict,
    threshold_mu: float = 25.0,
) -> dict[str, int]:
    """
    Count of wins against opponents whose TrueSkill mu is at or above
    threshold_mu (the starting mu — i.e. "at least average") at the time
    ratings were finalized. A simple, explainable quality-win count
    rather than a continuous score.
    """
    qw: dict[str, int] = {}
    for p in players:
        count = 0
        for m in results_idx.get(p, []):
            if m["winner"] != p:
                continue
            opp = m["loser"]
            r = trueskill_ratings.get(opp)
            if r is not None and r.mu >= threshold_mu:
                count += 1
        qw[p] = count
    return qw


def _compute_tgrs(
    reach_size: int,
    ts_conservative: float,
    sos: float,
    quality_wins: int,
    win_pct: float,
) -> float:
    """
    TGRS ("Group Ranking Score") — a single composite number for sorting/
    display purposes only. It is NOT used anywhere in the ranking core;
    the actual seed order always comes from the transitivity + adjacent
    fix-up pipeline above. TGRS exists purely so the website has one
    "at a glance" number to show and sort by.

    Formula (weights are intentionally simple/explainable, tune freely):
        TGRS = (2.0 * reach_size)
             + (5.0 * ts_conservative)
             + (0.5 * sos)
             + (1.5 * quality_wins)
             + (1.0 * win_pct)
    """
    return (
        2.0 * reach_size
        + 5.0 * ts_conservative
        + 0.5 * sos
        + 1.5 * quality_wins
        + 0.1 * win_pct
    )


def _scale_tgrs(raw_scores: dict[str, float]) -> dict[str, float]:
    """Min-max scale TGRS to a 0-100 band (TGRS_scaled) within whatever
    player set is passed in, for readability on the site. Falls back to
    a flat 50.0 for every player if everyone is tied (avoids div/0)."""
    if not raw_scores:
        return {}
    lo = min(raw_scores.values())
    hi = max(raw_scores.values())
    if hi - lo < 1e-9:
        return {p: 50.0 for p in raw_scores}
    return {
        p: round(100.0 * (v - lo) / (hi - lo), 1)
        for p, v in raw_scores.items()
    }


# ============================================================================
# 9c.  Comeback rate + opponent-strength record  (feeds build_site.py)
# ============================================================================
#
#   won_after_set1_loss
#   --------------------
#   For every match a player appears in, look at set token #1 (from the
#   MATCH WINNER's perspective, per _parse_set_tokens). If the player
#   lost that first set, this was a "comeback opportunity"; if they went
#   on to win the match anyway, it counts as a "comeback win". Reported
#   as "wins/opportunities (pct%)", e.g. "3/5 (60%)". Blank if the player
#   never lost a first set.
#
#   Sourced from the FULL, unfiltered bucket snapshot (same population as
#   `records`/wins/losses) — not the MIN_MATCHES/dedup-filtered set, and
#   not restricted to ranking-eligible matches — because this is a
#   descriptive stat about what actually happened on court, not an input
#   to the ranking core.
#
#   vs_weaker_opp / vs_mid_opp / vs_top_opp
#   -----------------------------------------
#   Per the request: instead of bucketing by opponent's raw win%, bucket
#   by opponent's TrueSkill conservative rating (mu - 3*sigma), which
#   already accounts for strength of schedule instead of just counting
#   wins against whoever they happened to play.
#
#   Thresholds (p50 / p75) are computed ONCE per division/flight pool,
#   from the ts_rating of that division/flight's own seeded roster only
#   (the same population `local_sos` is computed from) — this is what
#   "for each specific division and flight pool" means here. But once
#   the thresholds exist, a player's record against them counts EVERY
#   opponent they played, including ones from other divisions/flights,
#   exactly as requested ("the record would count against everybody").
# ============================================================================

def compute_won_after_set1_loss(
    player: str,
    results_idx: dict[str, list[dict]],
) -> tuple[int, int]:
    """
    Returns (wins, opportunities):
      opportunities = matches in which `player` lost set 1
      wins          = how many of those matches `player` still won overall
    """
    wins = 0
    opportunities = 0
    for m in results_idx.get(player, []):
        sets = _parse_set_tokens(m["score"])
        if not sets:
            continue
        w_games, l_games = sets[0]  # match-winner-perspective, set 1 only
        is_match_winner = (m["winner"] == player)
        player_games, opp_games = (w_games, l_games) if is_match_winner else (l_games, w_games)
        if player_games < opp_games:
            opportunities += 1
            if is_match_winner:
                wins += 1
    return wins, opportunities


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def compute_opponent_strength_thresholds(
    division_roster: list[str],
    trueskill_ratings: dict,
) -> tuple[float, float]:
    """
    p50 and p75 of TrueSkill conservative rating, computed over this
    division/flight's own seeded roster (`division_roster`) — the same
    population `local_sos`/`local_quality_wins` are scoped to.

    Returns (p50, p75). If the roster has no rated players, returns
    (0.0, 0.0), which will bucket everyone as "top" — harmless edge case
    for an empty/degenerate group.
    """
    vals = sorted(
        trueskill_ratings[p].conservative
        for p in division_roster
        if p in trueskill_ratings
    )
    return _percentile(vals, 0.50), _percentile(vals, 0.75)


def compute_opponent_strength_record(
    player: str,
    results_idx: dict[str, list[dict]],
    trueskill_ratings: dict,
    p50: float,
    p75: float,
) -> dict[str, list[int]]:
    """
    Win-loss record for `player`, bucketed by OPPONENT's TrueSkill
    conservative rating against the (p50, p75) thresholds:
        < p50        -> "weak"  (bottom 50% of the pool)
        p50 to <p75  -> "mid"   (50th-75th percentile)
        >= p75       -> "top"   (top quartile)

    Uses `results_idx` for ALL of the player's matches (any division,
    any flight, including matches later dropped from ranking) — only the
    thresholds themselves are scoped to the player's own division/flight
    pool. Opponents with no TrueSkill rating on record are skipped (can
    happen for players who never had enough matches to be rated at all).
    """
    buckets: dict[str, list[int]] = {"weak": [0, 0], "mid": [0, 0], "top": [0, 0]}
    for m in results_idx.get(player, []):
        opp = m["loser"] if m["winner"] == player else m["winner"]
        r = trueskill_ratings.get(opp)
        if r is None:
            continue
        rating = r.conservative
        if rating < p50:
            key = "weak"
        elif rating < p75:
            key = "mid"
        else:
            key = "top"
        buckets[key][0 if m["winner"] == player else 1] += 1
    return buckets


def _format_fraction(wins: int, opportunities: int) -> str:
    """"3/5 (60%)" style formatting; blank if there were no opportunities."""
    if opportunities == 0:
        return ""
    pct = round(100 * wins / opportunities)
    return f"{wins}/{opportunities} ({pct}%)"


def _format_record(wl: list[int]) -> str:
    """"5-0 (100%)" style formatting; blank if the bucket is empty."""
    w, l = wl
    total = w + l
    if total == 0:
        return ""
    pct = round(100 * w / total)
    return f"{w}-{l} ({pct}%)"


# ============================================================================
# 10.  Per-group pipeline
# ============================================================================

def process_group(key: tuple, group_matches: list[dict]) -> list[dict]:
    """
    Full pipeline for one (gender, match_type, flight) group, ranked
    across all divisions together, then split.
    """
    gender, match_type, flight = key

    # Snapshot of every match in this bucket exactly as loaded, before
    # the MIN_MATCHES drop or the per-school dedup pass below. Records
    # (wins/losses/matches/last_match_date), the comeback stat, and the
    # opponent-strength record are all computed from THIS full snapshot —
    # spanning every division in the bucket — so they reflect everyone the
    # player actually played, not just the opponents who survived
    # filtering. Nothing else in the pipeline uses this snapshot; ranking
    # still runs on the filtered set.
    original_group_matches = list(group_matches)

    # ── STEP 0a: iteratively drop players with fewer than MIN_MATCHES
    #    total matches in the group. Records still count ranking-excluded
    #    matches so a player isn't unfairly dropped just because some of
    #    their matches are single-set / "2-0 2-0" / "default"-name. ──────

    # ── STEP 0b: per-school deduplication within this slot.
    full_idx_for_dedup = build_results_index(group_matches)
    school_map_pre = build_school_map(group_matches)
    MIN_MATCHES = getattr(_config, "MIN_MATCHES", 5)

    school_to_players: dict[str, list[str]] = defaultdict(list)
    for p in players_in_group(group_matches):
        school = school_map_pre.get(p, "")
        if school:
            school_to_players[school].append(p)

    drop_players: set[str] = set()
    for school, candidates in school_to_players.items():
        if len(candidates) <= 1:
            continue

        def _match_count(p: str, _idx=full_idx_for_dedup) -> int:
            return len(_idx.get(p, []))

        def _last_date(p: str, _idx=full_idx_for_dedup) -> datetime:
            ms = _idx.get(p, [])
            return max((m["date"] for m in ms), default=datetime.min)

        # Only players who've cleared MIN_MATCHES are eligible to be
        # picked as the school's "current" rep on recency grounds — a
        # one-off substitute with a single late match shouldn't bump a
        # regular starter with a full season of matches just because
        # their one match happened to be more recent. If NO candidate
        # at this school clears MIN_MATCHES (e.g. an injury-shuffled
        # slot where nobody's played enough yet), fall back to picking
        # by recency among all candidates so the school isn't dropped
        # from this slot entirely.
        qualified = [p for p in candidates if _match_count(p) >= MIN_MATCHES]
        pool = qualified if qualified else candidates

        best = max(pool, key=_last_date)
        dropped = [p for p in candidates if p != best]
        drop_players.update(dropped)
        print(
            f"  Dedup ({gender} {match_type} flight={flight}): "
            f"school={school!r}  kept={best!r} ({_match_count(best)} matches)  "
            f"dropped={', '.join(repr(p) for p in dropped)}"
        )

    if drop_players:
        group_matches = [
            m for m in group_matches
            if m["winner"] not in drop_players and m["loser"] not in drop_players
        ]

    players = players_in_group(group_matches)
    school_map = build_school_map(group_matches)
    division_map = build_division_map(group_matches)

    # Records: built from the full pre-filter/pre-dedup bucket snapshot,
    # not from the filtered `group_matches`, so wins/losses/matches count
    # everything the player actually played (including opponents from
    # other divisions and opponents later dropped from ranking). The
    # comeback stat and opponent-strength record (section 9c) are also
    # sourced from this same snapshot's results index.
    original_results_idx = build_results_index(original_group_matches)
    records = build_records(players, original_results_idx)

    ranking_matches = [m for m in group_matches if not m.get("is_ranking_excluded")]

    h2h = build_h2h_index(ranking_matches)
    ranking_results_idx = build_results_index(ranking_matches)
    opp_sets = build_opponent_sets(players, ranking_results_idx)
    recency = precompute_recency(players, ranking_results_idx)
    margins = precompute_margins(players, ranking_results_idx)

    # NOTE: trueskill_engine_v2.compute_trueskill_margin() expects
    # (winner, loser, set_score) triples — not just (winner, loser)
    # pairs — so it can derive a per-match margin/epsilon from the
    # scoreline. The set score is already sitting on every match dict
    # as "score", so we just carry it through here.
    match_triples = [
        (m["winner"], m["loser"], m["score"])
        for m in sorted(ranking_matches, key=lambda x: x["date"])
    ]
    trueskill_ratings = compute_trueskill_margin(match_triples)

    sos_full = precompute_sos(players, ranking_results_idx, trueskill_ratings)
    quality_wins_full = precompute_quality_wins(players, ranking_results_idx, trueskill_ratings)

    # --- STEP 1: break every cycle, regardless of length ---
    acyclic_matches = resolve_all_cycles(ranking_matches, players)
    beats = _directed_beats_from_matches(acyclic_matches)

    # --- STEP 2: transitive closure + transitivity-only order ---
    reach = build_transitive_closure(players, beats)
    seed_order = transitivity_seed_order(
        players, reach, recency, margins, trueskill_ratings
    )

    # --- STEP 3: split into divisions ---
    div_players: dict[str, list[str]] = defaultdict(list)
    for player in seed_order:
        d = division_map.get(player, "")
        div_players[d].append(player)

    results = []
    for division in sorted(div_players):
        division_roster = div_players[division]

        # --- STEP 4: adjacent fix-up within division ---
        div_ranked, swap_log = adjacent_fixup(
            division_roster, h2h, opp_sets, reach, trueskill_ratings
        )
        div_explanations = build_adjacency_explanations(
            div_ranked, h2h, opp_sets, reach, trueskill_ratings
        )

        div_ranking_matches = [
            m for m in ranking_matches
            if m["winner"] in div_ranked and m["loser"] in div_ranked
        ]
        div_ranking_idx = build_results_index(div_ranking_matches)
        local_sos = precompute_sos(div_ranked, div_ranking_idx, trueskill_ratings)
        local_quality_wins = precompute_quality_wins(div_ranked, div_ranking_idx, trueskill_ratings)

        tgrs_raw: dict[str, float] = {}
        for p in div_ranked:
            rec = records.get(p, {"matches": 0, "wins": 0, "losses": 0})
            win_pct = rec["wins"] / rec["matches"] if rec["matches"] else 0.0
            r = trueskill_ratings.get(p)
            tgrs_raw[p] = _compute_tgrs(
                reach_size=len(reach.get(p, ())),
                ts_conservative=r.conservative if r else 0.0,
                sos=sos_full.get(p, 0.0),
                quality_wins=quality_wins_full.get(p, 0),
                win_pct=win_pct,
            )
        tgrs_scaled = _scale_tgrs(tgrs_raw)

        results.append({
            "group": (gender, match_type, division, flight),
            "gender": gender,
            "match_type": match_type,
            "division": division,
            "flight": flight,
            "seeds": div_ranked,
            "school_map": school_map,
            "explanations": div_explanations,
            "unified_rank": seed_order,
            "unified_explanations": None,
            "swap_log": swap_log,
            "records": records,
            "original_results_idx": original_results_idx,
            "reach": reach,
            "trueskill_ratings": trueskill_ratings,
            "sos": sos_full,
            "local_sos": local_sos,
            "quality_wins": quality_wins_full,
            "local_quality_wins": local_quality_wins,
            "tgrs_raw": tgrs_raw,
            "tgrs_scaled": tgrs_scaled,
        })

    # --- STEP 4b: cross-division "general" ranking (NEW) ---
    # Same transitivity-only order from STEP 2, but the adjacent fix-up
    # pass now runs against the FULL cross-division roster instead of
    # being scoped to one division at a time — producing a single
    # flight-wide ranking on top of (not instead of) the per-division
    # seed lists above. Published under the pseudo-division "overall".
    overall_ranked, overall_swap_log = adjacent_fixup(
        seed_order, h2h, opp_sets, reach, trueskill_ratings
    )
    overall_explanations = build_adjacency_explanations(
        overall_ranked, h2h, opp_sets, reach, trueskill_ratings
    )

    tgrs_raw_overall: dict[str, float] = {}
    for p in overall_ranked:
        rec = records.get(p, {"matches": 0, "wins": 0, "losses": 0})
        win_pct = rec["wins"] / rec["matches"] if rec["matches"] else 0.0
        r = trueskill_ratings.get(p)
        tgrs_raw_overall[p] = _compute_tgrs(
            reach_size=len(reach.get(p, ())),
            ts_conservative=r.conservative if r else 0.0,
            sos=sos_full.get(p, 0.0),
            quality_wins=quality_wins_full.get(p, 0),
            win_pct=win_pct,
        )
    tgrs_scaled_overall = _scale_tgrs(tgrs_raw_overall)

    results.append({
        "group": (gender, match_type, "overall", flight),
        "gender": gender,
        "match_type": match_type,
        "division": "overall",
        "flight": flight,
        "seeds": overall_ranked,
        "school_map": school_map,
        "explanations": overall_explanations,
        "unified_rank": seed_order,
        "unified_explanations": None,
        "swap_log": overall_swap_log,
        "records": records,
        "original_results_idx": original_results_idx,
        "reach": reach,
        "trueskill_ratings": trueskill_ratings,
        # "local" == "regular" here — there's no division scoping left to
        # distinguish them, the whole cross-division group IS local now.
        "sos": sos_full,
        "local_sos": sos_full,
        "quality_wins": quality_wins_full,
        "local_quality_wins": quality_wins_full,
        "tgrs_raw": tgrs_raw_overall,
        "tgrs_scaled": tgrs_scaled_overall,
    })

    return results


# ============================================================================
# 11.  Main orchestration
# ============================================================================

def run(csv_path: str) -> list[dict]:
    print(f"Loading: {csv_path}")
    school_meta = load_school_meta(csv_path)
    correct_divisions = load_correct_divisions(csv_path)
    all_matches = load_matches(csv_path, school_meta, correct_divisions)
    print(f"  {len(all_matches):,} matches loaded.")

    buckets = bucket_matches(all_matches)
    print(f"  {len(buckets)} cross-division groups found "
          f"(ranked together, then split by division).")

    for key in sorted(buckets):
        gender, mtype, flight = key
        print(
            f"    {gender:6}  {mtype:8}  "
            f"flight={flight or '-':3}  ({len(buckets[key])} matches)"
        )

    results = []
    for key in sorted(buckets):
        results.extend(process_group(key, buckets[key]))

    return results


# ============================================================================
# 12.  Output helpers
# ============================================================================

def _safe_filename(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", str(text or "")).strip("_") or "unknown"


# Human-readable labels for each raw reason_below value written by
# build_adjacency_explanations / compare_adjacent.
_REASON_LABELS: dict[str, str] = {
    "head-to-head":              "Lost head-to-head vs player above",
    "common-opponents-wins":     "Fewer wins vs common opponents",
    "common-opponents-sets":     "More sets needed vs common opponents",
    "common-opponents-margin":   "Worse score margin vs common opponents",
    "dominance":                 "Transitively beaten by player above",
    "trueskill":                 "Lower TrueSkill rating",
    "tied":                      "Tied — no deciding criterion found",
}


def _format_reason(raw: str) -> str:
    """Return a human-readable explanation for a raw decided_by string,
    falling back to the raw value itself if it's not in the table."""
    return _REASON_LABELS.get(raw, raw)


_INDIVIDUAL_FIELDS = [
    "rank", "name", "school", "division", "flight",
    "wins", "losses",
    "TGRS", "TGRS_scaled",
    "ts_rating", "ts_mu", "local_ts_mu", "ts_sigma",
    "reachability", "local_reachability",
    "sos", "local_sos", "quality_wins",
    "won_after_set1_loss",
    "vs_weaker_opp", "vs_mid_opp", "vs_top_opp",
    "last_match_date",
    "reason_below",   # ← why this player is ranked below the one above them
]

# Doubles rows use "pair_name" instead of "name" so build_site.py's
# preview_cols (which checks for either) picks the right one up.
_DOUBLES_FIELDS = [c if c != "name" else "pair_name" for c in _INDIVIDUAL_FIELDS]


def _category_filename_stem(match_type: str, gender: str) -> str:
    category = "singles" if match_type.strip().lower().startswith("single") else "doubles"
    gender_l = "boys" if gender.strip().lower().startswith("b") else "girls"
    return category, gender_l


def _result_rows_for_division(r: dict) -> list[dict]:
    """One row per seeded player/pair in this division/flight result,
    using the column names build_site.py looks for."""
    expl_raw = {e["seed_below"]: e["decided_by"] for e in r["explanations"]}
    school_map = r.get("school_map", {})
    records = r.get("records", {})
    original_results_idx = r.get("original_results_idx", {})
    reach = r.get("reach", {})
    trueskill_ratings = r.get("trueskill_ratings", {})
    sos = r.get("sos", {})
    local_sos = r.get("local_sos", {})
    quality_wins = r.get("quality_wins", {})
    tgrs_raw = r.get("tgrs_raw", {})
    tgrs_scaled = r.get("tgrs_scaled", {})

    # NEW: MIN_MATCHES is now applied only here, at output time — every
    # player/pair was fully part of ranking (cycles, transitive closure,
    # h2h, dominance, TrueSkill) regardless of match count. This just
    # decides who actually appears in the published seed list.
    MIN_MATCHES = getattr(_config, "MIN_MATCHES", 5)
    seeds = [
        p for p in r["seeds"]
        if records.get(p, {}).get("matches", 0) >= MIN_MATCHES
    ]

    p50, p75 = compute_opponent_strength_thresholds(r["seeds"], trueskill_ratings)
    _, gender_l = _category_filename_stem(r["match_type"], r["gender"])
    is_doubles = "/" in (seeds[0] if seeds else "")
    name_field = "pair_name" if is_doubles else "name"
    rows = []
    for seed, player in enumerate(seeds, start=1):
        rec = records.get(player, {"matches": 0, "wins": 0, "losses": 0, "last_match_date": None})
        ts = trueskill_ratings.get(player)
        last_dt = rec.get("last_match_date")

        # reason_below: empty for the #1 seed; human-readable label for everyone else.
        if seed == 1:
            reason_below = ""
        else:
            raw_reason = expl_raw.get(player, "")
            reason_below = _format_reason(raw_reason) if raw_reason else ""

        # Comeback stat + opponent-strength record — both sourced from the
        # full, unfiltered bucket snapshot (same population as wins/losses),
        # so they reflect everything the player actually played.
        wins_after_l1, opportunities_l1 = compute_won_after_set1_loss(player, original_results_idx)
        strength_record = compute_opponent_strength_record(
            player, original_results_idx, trueskill_ratings, p50, p75
        )

        row = {
            "rank": seed,
            name_field: player,
            "school": school_map.get(player, ""),
            "division": r["division"],
            "flight": r["flight"],
            "wins": rec["wins"],
            "losses": rec["losses"],
            "TGRS": round(tgrs_raw.get(player, 0.0), 2),
            "TGRS_scaled": tgrs_scaled.get(player, 0.0),
            "ts_rating": round(ts.conservative, 2) if ts else "",
            "ts_mu": round(ts.mu, 2) if ts else "",
            "local_ts_mu": round(ts.mu, 2) if ts else "",
            "ts_sigma": round(ts.sigma, 2) if ts else "",
            "reachability": len(reach.get(player, ())),
            "local_reachability": len(reach.get(player, ())),
            "sos": round(sos.get(player, 0.0), 2),
            "local_sos": round(local_sos.get(player, 0.0), 2),
            "quality_wins": quality_wins.get(player, 0),
            "won_after_set1_loss": _format_fraction(wins_after_l1, opportunities_l1),
            "vs_weaker_opp": _format_record(strength_record["weak"]),
            "vs_mid_opp": _format_record(strength_record["mid"]),
            "vs_top_opp": _format_record(strength_record["top"]),
            "last_match_date": last_dt.date().isoformat() if last_dt and last_dt.year > 1 else "",
            "reason_below": reason_below,
        }
        if not is_doubles:
            row.pop("pair_name", None)
        else:
            row.pop("name", None)
        rows.append(row)
    return rows


def write_division_csvs(results: list[dict], out_dir: str) -> list[str]:
    """
    Groups per-division results by (category, gender, division) — merging
    all 4 flights into ONE file each, e.g. singles_boys_division_1.csv —
    which is the shape scripts/build_site.py expects (it then filters by
    the "flight" column internally). This also includes the "overall"
    pseudo-division from STEP 4b (e.g. singles_boys_division_overall.csv),
    which build_site.py treats as a separate "General Rankings" section
    rather than a normal division. Returns the list of file paths written.
    """
    os.makedirs(out_dir, exist_ok=True)

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        category, gender_l = _category_filename_stem(r["match_type"], r["gender"])
        buckets[(category, gender_l, r["division"])].extend(_result_rows_for_division(r))

    written = []
    for (category, gender_l, division), rows in sorted(buckets.items()):
        fields = _DOUBLES_FIELDS if category == "doubles" else _INDIVIDUAL_FIELDS
        rows.sort(key=lambda row: (row["flight"], row["rank"]))

        filename = f"{category}_{gender_l}_division_{_safe_filename(division)}.csv"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        written.append(filepath)

    return written


def write_team_csvs(results: list[dict], out_dir: str) -> list[str]:
    """
    Builds team_{gender}_division_{division}.csv files by aggregating all
    seeded players/pairs for each school within a (gender, division),
    across every match_type and flight. Delegates the actual aggregation
    math to team_aggregation.build_team_rankings().

    The "overall" pseudo-division from STEP 4b is now INCLUDED here too,
    producing team_{gender}_division_overall.csv — a single cross-division
    team ranking built with the exact same aggregation logic
    (build_team_rankings), just fed the cross-division "overall" seed rows
    instead of one division's rows. This mirrors the individual "General
    Rankings" already published for singles/doubles: it's published
    alongside the per-division team tables, not instead of them.
    """
    os.makedirs(out_dir, exist_ok=True)

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        _, gender_l = _category_filename_stem(r["match_type"], r["gender"])
        for row in _result_rows_for_division(r):
            buckets[(gender_l, r["division"])].append(row)

    written = []
    for (gender_l, division), rows in sorted(buckets.items()):
        team_rows = build_team_rankings(rows)
        if not team_rows:
            continue
        filename = f"team_{gender_l}_division_{_safe_filename(division)}.csv"
        filepath = os.path.join(out_dir, filename)
        fieldnames = list(team_rows[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(team_rows)
        written.append(filepath)

    return written


def _print_seed_table(players: list[str], school_map: dict,
                       explanations: list[dict], label: str) -> None:
    name_w = max((len(p) for p in players), default=20) + 2
    school_w = max((len(school_map.get(p, "")) for p in players), default=20) + 2

    print(f"\n  {label}")
    print(f"  {'#':>3}  {'Player':<{name_w}}  {'School':<{school_w}}")
    print(f"  {'-'*3}  {'-'*name_w}  {'-'*school_w}")
    for seed, player in enumerate(players, start=1):
        print(f"  {seed:>3}.  {player:<{name_w}}  {school_map.get(player,''):<{school_w}}")

    if explanations:
        print()
        print("  Tiebreaker log:")
        for ex in explanations:
            print(f"    {ex['seed_above']}  >  {ex['seed_below']}  [{ex['decided_by']}]")


def print_results(results: list[dict]) -> None:
    from itertools import groupby
    keyfn = lambda r: (r["gender"], r["match_type"], r["flight"])
    for group_key, group_iter in groupby(sorted(results, key=keyfn), key=keyfn):
        group_list = list(group_iter)
        gender, match_type, flight = group_key
        fl_label = f"Flight {flight}" if flight else ""
        parts = [p for p in [gender, match_type, fl_label] if p]
        header = "  " + "  |  ".join(parts)
        school_map = group_list[0]["school_map"]

        print()
        print("=" * max(72, len(header) + 4))
        print(header)
        print("=" * max(72, len(header) + 4))

        unified = group_list[0].get("unified_rank", [])
        unified_expls = group_list[0].get("unified_explanations") or []
        if unified:
            _print_seed_table(unified, school_map, unified_expls,
                               "Transitivity-only order, pre fix-up (cross-division, "
                               "before the per-division head-to-head/common-opponents/"
                               "dominance pass)")

        print()
        print("  ── Per-division seeds (fixed up only within each division/flight) ──")
        for r in group_list:
            label = "Overall (General Ranking)" if r["division"] == "overall" else f"Division {r['division']}"
            _print_seed_table(
                r["seeds"], school_map, r["explanations"],
                label
            )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        year = getattr(_config, "YEAR", 2025)
        default_paths = [
            os.path.join("..", "historical", str(year), "all_matches.csv"),
            os.path.join("historical", str(year), "all_matches.csv"),
        ]
        csv_files = [p for p in default_paths if os.path.exists(p)]
        if not csv_files:
            print("Usage: python mhsaa_seeding_v2.py <csv_path> [output_dir]")
            sys.exit(1)
        jobs = [(csv_files[0], os.path.join("rankings_by_division_flight"))]
    else:
        csv_path = sys.argv[1]
        out_dir = (
            sys.argv[2] if len(sys.argv) > 2
            else os.path.join(os.path.dirname(csv_path) or ".", "rankings_by_division_flight")
        )
        jobs = [(csv_path, out_dir)]

    for csv_path, out_dir in jobs:
        print(f"\n{'#'*64}")
        print(f"# {csv_path}")
        print(f"{'#'*64}")
        try:
            t0 = time.perf_counter()
            results = run(csv_path)
            t1 = time.perf_counter()

            print_results(results)

            written_div = write_division_csvs(results, out_dir)
            written_team = write_team_csvs(results, out_dir)

            print(f"\n  {len(written_div)} division CSV(s)  → {out_dir}/")
            print(f"  {len(written_team)} team CSV(s)      → {out_dir}/")
            print(f"  Total time                 → {t1 - t0:.3f}s")

        except FileNotFoundError:
            print(f"  ERROR: File not found: {csv_path}")
        except Exception:
            import traceback
            traceback.print_exc()
