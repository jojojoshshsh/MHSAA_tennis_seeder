"""
team_aggregation.py
====================
Builds team-level rankings (team_*.csv) from the per-player/pair rows
that mhsaa_seeding_v2.py already produces for a (gender, division) group.

Scoring blends TWO independent systems, each on a 0-100 scale, weighted
50/50 into a single combined_score used for the final rank:

  1) LEGACY FLIGHT-FINISH POINTS (max 100 across 8 slots)
     - 8 lineup slots: Singles flights 1-4 and Doubles flights 1-4
     - One entry per school per slot (highest-ranked player/pair from
       that school)
     - Points by finishing position:
         1st         -> 12.5
         2nd         -> 10.0
         3rd-4th     -> 7.5
         5th-8th     -> 5.0
         9th-16th    -> 2.5
         17th-32nd   -> 1.0
         33rd+       -> 0.0
     - total_points = sum of points across all 8 slots (max 100)

  2) DEPTH SCORE (0-100)
     For a league of N teams (N = number of schools fielding at least
     one entrant anywhere in this gender/division group):

         flight_score = (N - rank + 1) / N * 100

     This converts each flight rank into a 0-100 scale where 1st place
     = 100 and last place (rank N) is approximately 0. If a school has
     NO entrant in a given flight, that flight's rank is treated as
     N + 1 (worse than last place) rather than being excluded from the
     average -- this prevents a team that only fielded a handful of
     flights from looking artificially strong.

         depth_score = average(flight_score across all 8 flights)

  combined_score = 0.5 * total_points + 0.5 * depth_score

Both total_points and depth_score are reported independently alongside
combined_score, and teams are ranked by combined_score.

TEAM STRENGTH OF SCHEDULE (team_sos / team_local_sos)
------------------------------------------------------
  Each per-player row already carries that player's own "sos" (strength
  of schedule, computed cross-division) and "local_sos" (computed within
  just that player's division/flight group) — see mhsaa_seeding_v2.py's
  precompute_sos(). For a team's 8 lineup slots (the same slots used for
  total_points/depth_score above), we take the SAME best-ranked
  player/pair per slot and average that player's sos (and, separately,
  local_sos) across every slot the school actually fielded an entrant
  in. Empty slots are simply excluded from the average (unlike
  depth_score, there's no "penalize for not fielding a flight" concept
  here — SOS only describes the schedule of the matches actually played).

    team_sos       = average(sos of the best entrant)       across filled slots
    team_local_sos = average(local_sos of the best entrant)  across filled slots

  If a school has no filled slots at all (shouldn't normally happen,
  since it wouldn't appear in school_slots), both default to 0.0.

  - reason_below column explains why each team ranks below the one above it
"""
from collections import defaultdict


# ── Point table by finishing position ────────────────────────────────────────
def _finish_points(rank: int) -> float:
    if rank == 1:
        return 12.5
    if rank == 2:
        return 10.0
    if rank <= 4:
        return 7.5
    if rank <= 8:
        return 5.0
    if rank <= 16:
        return 2.5
    if rank <= 32:
        return 1.0
    return 0.0


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th'][min(n % 10, 4)]}"


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ── Slot key: (category, flight) where category = "singles" or "doubles" ────
def _slot_key(row: dict) -> tuple:
    """Derive which of the 8 lineup slots this row belongs to."""
    # category is inferred from whether the row has pair_name or name
    if "pair_name" in row and row.get("pair_name", ""):
        category = "doubles"
    else:
        category = "singles"
    flight = str(row.get("flight", ""))
    return (category, flight)


SLOTS = [
    ("singles", "1"), ("singles", "2"), ("singles", "3"), ("singles", "4"),
    ("doubles", "1"), ("doubles", "2"), ("doubles", "3"), ("doubles", "4"),
]

SLOT_COL = {
    ("singles", "1"): "s1_pts",
    ("singles", "2"): "s2_pts",
    ("singles", "3"): "s3_pts",
    ("singles", "4"): "s4_pts",
    ("doubles", "1"): "d1_pts",
    ("doubles", "2"): "d2_pts",
    ("doubles", "3"): "d3_pts",
    ("doubles", "4"): "d4_pts",
}

SLOT_LABELS = {
    "s1_pts": "Singles F1", "s2_pts": "Singles F2",
    "s3_pts": "Singles F3", "s4_pts": "Singles F4",
    "d1_pts": "Doubles F1", "d2_pts": "Doubles F2",
    "d3_pts": "Doubles F3", "d4_pts": "Doubles F4",
}
SLOT_COLS_ORDERED = list(SLOT_LABELS.keys())

# Weights for combined_score. Kept as module-level constants so callers
# can tweak the blend without touching the aggregation logic.
LEGACY_WEIGHT = 0.5
DEPTH_WEIGHT = 0.5


def build_team_rankings(player_rows: list[dict]) -> list[dict]:
    """
    Parameters
    ----------
    player_rows : list of dicts, one per seeded player/pair, with at
        least the keys: school, rank, flight, sos, local_sos, and either
        name or pair_name. (This is exactly the row shape
        mhsaa_seeding_v2._result_rows_for_division() produces.) The
        "rank" field is assumed to already be the player/pair's rank
        WITHIN its flight (1 = best in that flight).

    Returns
    -------
    list of dicts, one per school, sorted by combined_score descending, with
    columns: rank, school, combined_score, total_points, depth_score,
             team_sos, team_local_sos,
             slots_counted,
             s1_pts, s2_pts, s3_pts, s4_pts,
             d1_pts, d2_pts, d3_pts, d4_pts,
             reason_below.

    Scoring logic
    -------------
    LEGACY (total_points):
      For each of the 8 slots (Singles/Doubles x Flights 1-4):
        - Find all rows for a given school in that slot.
        - Take only the one with the lowest rank number (best-ranked
          player/pair).
        - Award points based on that rank using _finish_points().
      total_points = sum of points across all slots (max 100).

    DEPTH (depth_score):
      N = number of schools fielding at least one entrant anywhere in
      this gender/division group.
      For each of the 8 flights:
        - If the school has an entrant, use that entrant's best rank
          within the flight.
        - If the school has NO entrant in that flight, use rank = N + 1
          (worse than last place) instead of skipping the flight.
        - flight_score = (N - rank + 1) / N * 100
      depth_score = average(flight_score across all 8 flights).

    TEAM SOS (team_sos / team_local_sos):
      For each of the 8 slots the school actually fielded an entrant in,
      take that best-ranked entrant's sos (resp. local_sos) and average
      across those filled slots only (unfilled slots are excluded, not
      penalized).

    combined_score = LEGACY_WEIGHT * total_points + DEPTH_WEIGHT * depth_score
    """
    # Group rows by school, then by slot
    # school -> slot_key -> list of rows
    school_slots: dict[str, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in player_rows:
        school = (row.get("school") or "").strip()
        if not school:
            continue
        slot = _slot_key(row)
        school_slots[school][slot].append(row)

    # N = league size = number of distinct schools fielding any entrant
    # anywhere in this gender/division group.
    N = len(school_slots)

    team_rows = []
    for school, slot_map in school_slots.items():
        total_points = 0.0
        slots_counted = 0
        slot_pts: dict[str, float] = {}
        slot_ranks: dict[str, int] = {}  # for reason_below
        flight_scores: list[float] = []  # for depth_score

        sos_sum = 0.0
        local_sos_sum = 0.0
        sos_filled = 0

        for slot in SLOTS:
            col = SLOT_COL[slot]
            rows_in_slot = slot_map.get(slot, [])

            if not rows_in_slot:
                # No entrant in this flight -> legacy points are 0, and
                # depth-score treats this as rank N+1 (worse than last).
                # Team SOS simply skips this slot (nothing was played).
                slot_pts[col] = 0.0
                slot_ranks[col] = 0
                no_entrant_rank = N + 1
                flight_score = (N - no_entrant_rank + 1) / N * 100 if N > 0 else 0.0
                flight_scores.append(flight_score)
                continue

            # Best-ranked entry = lowest rank number
            best_row = min(rows_in_slot, key=lambda r: _safe_int(r.get("rank"), 9999))
            best_rank = _safe_int(best_row.get("rank"), 9999)

            # Legacy points
            pts = _finish_points(best_rank)
            slot_pts[col] = pts
            slot_ranks[col] = best_rank
            total_points += pts
            if pts > 0:
                slots_counted += 1

            # Depth score for this flight
            flight_score = (N - best_rank + 1) / N * 100 if N > 0 else 0.0
            flight_scores.append(flight_score)

            # Team SOS for this flight (best entrant's own sos/local_sos)
            sos_sum += _safe_float(best_row.get("sos"))
            local_sos_sum += _safe_float(best_row.get("local_sos"))
            sos_filled += 1

        depth_score = sum(flight_scores) / len(flight_scores) if flight_scores else 0.0
        combined_score = LEGACY_WEIGHT * total_points + DEPTH_WEIGHT * depth_score

        team_sos = round(sos_sum / sos_filled, 2) if sos_filled else 0.0
        team_local_sos = round(local_sos_sum / sos_filled, 2) if sos_filled else 0.0

        team_rows.append({
            "school": school,
            "total_points": round(total_points, 1),
            "depth_score": round(depth_score, 1),
            "combined_score": round(combined_score, 1),
            "team_sos": team_sos,
            "team_local_sos": team_local_sos,
            "slots_counted": slots_counted,
            **slot_pts,
            "_slot_ranks": slot_ranks,   # internal, stripped before writing
        })

    # Sort by combined_score descending (ties broken by legacy total_points,
    # then depth_score, for stability)
    team_rows.sort(
        key=lambda r: (r["combined_score"], r["total_points"], r["depth_score"]),
        reverse=True,
    )

    # Assign ordinal ranks
    for i, r in enumerate(team_rows):
        r["rank"] = i + 1

    # reason_below: for each team, list its flight finishes (from the
    # legacy system) as before.
    for r in team_rows:
        entries = []
        for col in SLOT_COLS_ORDERED:
            rank = r["_slot_ranks"].get(col, 0)
            if rank > 0:
                label = SLOT_LABELS[col]
                entries.append(f"{label}: {_ordinal(rank)}")
        r["reason_below"] = "; ".join(entries) if entries else "—"

    # Strip internal fields and reorder keys
    ordered_rows = []
    for r in team_rows:
        r.pop("_slot_ranks", None)
        ordered_rows.append({
            "rank": _ordinal(r["rank"]),
            "school": r["school"],
            "combined_score": r["combined_score"],
            "total_points": r["total_points"],
            "depth_score": r["depth_score"],
            "team_sos": r["team_sos"],
            "team_local_sos": r["team_local_sos"],
            "slots_counted": r["slots_counted"],
            "s1_pts": r["s1_pts"],
            "s2_pts": r["s2_pts"],
            "s3_pts": r["s3_pts"],
            "s4_pts": r["s4_pts"],
            "d1_pts": r["d1_pts"],
            "d2_pts": r["d2_pts"],
            "d3_pts": r["d3_pts"],
            "d4_pts": r["d4_pts"],
            "reason_below": r["reason_below"],
        })

    return ordered_rows
