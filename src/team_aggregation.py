"""
team_aggregation.py
====================
Builds team-level rankings (team_*.csv) from the per-player/pair rows
that mhsaa_seeding_v2.py already produces for a (gender, division) group.

Scoring uses the MHSAA flight-finish point system:
  - 8 lineup slots: Singles flights 1-4 and Doubles flights 1-4
  - One entry per school per slot (highest-ranked player/pair from that school)
  - Points by finishing position:
      1st         → 12.5
      2nd         → 10.0
      3rd–4th     → 7.5
      5th–8th     → 5.0
      9th–16th    → 2.5
      17th–32nd   → 1.0
      33rd+       → 0.0
  - Team score = sum of points across all 8 slots
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


def build_team_rankings(player_rows: list[dict]) -> list[dict]:
    """
    Parameters
    ----------
    player_rows : list of dicts, one per seeded player/pair, with at
        least the keys: school, rank, flight, and either name or pair_name.
        (This is exactly the row shape mhsaa_seeding_v2._result_rows_for_division()
        produces.)

    Returns
    -------
    list of dicts, one per school, sorted by total_points descending, with
    columns: rank, school, total_points, slots_counted,
             s1_pts, s2_pts, s3_pts, s4_pts,
             d1_pts, d2_pts, d3_pts, d4_pts,
             reason_below.

    Scoring logic
    -------------
    For each of the 8 slots (Singles/Doubles × Flights 1–4):
      - Find all rows for a given school in that slot.
      - Take only the one with the lowest rank number (best-ranked player/pair).
      - Award points based on that rank using _finish_points().
    Team score = sum of points across all slots where the school has
    at least one ranked entry.
    """
    # Group rows by school, then by slot
    # school → slot_key → list of rows
    school_slots: dict[str, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in player_rows:
        school = (row.get("school") or "").strip()
        if not school:
            continue
        slot = _slot_key(row)
        school_slots[school][slot].append(row)

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

    team_rows = []
    for school, slot_map in school_slots.items():
        total_points = 0.0
        slots_counted = 0
        slot_pts: dict[str, float] = {}
        slot_ranks: dict[str, int] = {}  # for reason_below

        for slot in SLOTS:
            col = SLOT_COL[slot]
            rows_in_slot = slot_map.get(slot, [])
            if not rows_in_slot:
                slot_pts[col] = 0.0
                slot_ranks[col] = 0
                continue
            # Best-ranked entry = lowest rank number
            best_row = min(rows_in_slot, key=lambda r: _safe_int(r.get("rank"), 9999))
            best_rank = _safe_int(best_row.get("rank"), 9999)
            pts = _finish_points(best_rank)
            slot_pts[col] = pts
            slot_ranks[col] = best_rank
            total_points += pts
            if pts > 0:
                slots_counted += 1

        team_rows.append({
            "school": school,
            "total_points": round(total_points, 1),
            "slots_counted": slots_counted,
            **slot_pts,
            "_slot_ranks": slot_ranks,   # internal, stripped before writing
        })

    # Sort by total_points descending
    team_rows.sort(key=lambda r: r["total_points"], reverse=True)

    # Assign ordinal ranks and compute reason_below
    for i, r in enumerate(team_rows):
        r["rank"] = i + 1

    # reason_below: for each team (except last), explain why the team
    # ranked directly below it scores less
    SLOT_LABELS = {
        "s1_pts": "Singles F1", "s2_pts": "Singles F2",
        "s3_pts": "Singles F3", "s4_pts": "Singles F4",
        "d1_pts": "Doubles F1", "d2_pts": "Doubles F2",
        "d3_pts": "Doubles F3", "d4_pts": "Doubles F4",
    }
    SLOT_COLS_ORDERED = list(SLOT_LABELS.keys())

    for i, r in enumerate(team_rows):
        if i == 0 or i >= len(team_rows):
            r["reason_below"] = "—"
            continue
        above = team_rows[i - 1]
        diff = round(above["total_points"] - r["total_points"], 1)
        # Find the slot(s) where above outscores this team
        gaps = []
        for col in SLOT_COLS_ORDERED:
            a_pts = above.get(col, 0.0)
            b_pts = r.get(col, 0.0)
            if a_pts > b_pts:
                a_rank = above["_slot_ranks"].get(col, 0)
                b_rank = r["_slot_ranks"].get(col, 0)
                label = SLOT_LABELS[col]
                if a_rank > 0 and b_rank > 0:
                    gaps.append(f"{label}: {_ordinal(a_rank)} vs {_ordinal(b_rank)}")
                elif a_rank > 0:
                    gaps.append(f"{label}: {_ordinal(a_rank)} vs unranked")
                else:
                    gaps.append(label)
        if gaps:
            r["reason_below"] = f"−{diff} pts ({'; '.join(gaps[:3])})"
        else:
            r["reason_below"] = f"−{diff} pts"

    # Strip internal fields and reorder keys
    ordered_rows = []
    for r in team_rows:
        r.pop("_slot_ranks", None)
        ordered_rows.append({
            "rank": _ordinal(r["rank"]),
            "school": r["school"],
            "total_points": r["total_points"],
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
