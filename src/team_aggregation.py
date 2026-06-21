"""
team_aggregation.py
====================
Builds team-level rankings (team_*.csv) from the per-player/pair rows
that mhsaa_seeding_v2.py already produces for a (gender, division) group.

There is no existing team-ranking algorithm in this project, so this
module implements a simple, explainable default: aggregate each school's
seeded players within the group, weighting by depth (flight 1 counts
more than flight 4) the way MHSAA-style team scoring typically does.

This is intentionally a starting point, not a finished competition
format — tune _FLIGHT_WEIGHT and TEAM_TGRS below to match your league's
actual team-scoring rules if they differ.
"""

from collections import defaultdict

# Flight 1 (top court) contributes more to a team's strength than flight 4,
# matching the usual MHSAA team-scoring emphasis on the top of the lineup.
_FLIGHT_WEIGHT = {"1": 1.4, "2": 1.2, "3": 1.0, "4": 0.8}


def _flight_weight(flight) -> float:
    return _FLIGHT_WEIGHT.get(str(flight), 1.0)


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


def build_team_rankings(player_rows: list[dict]) -> list[dict]:
    """
    Parameters
    ----------
    player_rows : list of dicts, one per seeded player/pair, with at
        least the keys: school, wins, losses, TGRS, ts_rating, ts_mu,
        sos, quality_wins, flight. (This is exactly the row shape
        mhsaa_seeding_v2._result_rows_for_division() produces.)

    Returns
    -------
    list of dicts, one per school, sorted by TGRS descending, with
    columns: rank, school, players_counted, wins, losses, win_pct,
    TGRS, TGRS_scaled, ts_rating, sos, quality_wins.

    Aggregation formula
    --------------------
    For each school, every counted player/pair contributes:
        weight = _FLIGHT_WEIGHT[flight]
    Team TGRS = sum(player.TGRS * weight) / sum(weight)   [weighted mean]
    Team ts_rating / sos = same weighted-mean approach.
    wins / losses / quality_wins = simple sums across counted players.
    win_pct = wins / (wins + losses), 0.0 if no matches.

    Only one row per (school, flight) is counted — if a school has both
    a singles and doubles entry at the same flight, both count (they're
    different competitors), but duplicate rows for the same exact player
    are not expected here since input is already per-(division,flight).
    """
    by_school: dict[str, list[dict]] = defaultdict(list)
    for row in player_rows:
        school = (row.get("school") or "").strip()
        if not school:
            continue
        by_school[school].append(row)

    team_rows = []
    for school, rows in by_school.items():
        total_weight = 0.0
        weighted_tgrs = 0.0
        weighted_ts = 0.0
        weighted_sos = 0.0
        wins = 0
        losses = 0
        quality_wins = 0

        for row in rows:
            w = _flight_weight(row.get("flight"))
            total_weight += w
            weighted_tgrs += w * _safe_float(row.get("TGRS"))
            weighted_ts += w * _safe_float(row.get("ts_rating"))
            weighted_sos += w * _safe_float(row.get("sos"))
            wins += _safe_int(row.get("wins"))
            losses += _safe_int(row.get("losses"))
            quality_wins += _safe_int(row.get("quality_wins"))

        if total_weight <= 0:
            continue

        team_tgrs = weighted_tgrs / total_weight
        team_ts = weighted_ts / total_weight
        team_sos = weighted_sos / total_weight
        win_pct = wins / (wins + losses) if (wins + losses) else 0.0

        team_rows.append({
            "school": school,
            "players_counted": len(rows),
            "wins": wins,
            "losses": losses,
            "win_pct": round(win_pct, 3),
            "TGRS": round(team_tgrs, 2),
            "ts_rating": round(team_ts, 2),
            "sos": round(team_sos, 2),
            "quality_wins": quality_wins,
        })

    # Scale TGRS to 0-100 within this division for readability, then sort.
    if team_rows:
        lo = min(r["TGRS"] for r in team_rows)
        hi = max(r["TGRS"] for r in team_rows)
        for r in team_rows:
            if hi - lo < 1e-9:
                r["TGRS_scaled"] = 50.0
            else:
                r["TGRS_scaled"] = round(100.0 * (r["TGRS"] - lo) / (hi - lo), 1)

    team_rows.sort(key=lambda r: r["TGRS"], reverse=True)
    for rank, r in enumerate(team_rows, start=1):
        r["rank"] = rank

    # Reorder keys so "rank" leads, matching the individual CSVs' style.
    ordered_rows = []
    for r in team_rows:
        ordered_rows.append({
            "rank": r["rank"],
            "school": r["school"],
            "players_counted": r["players_counted"],
            "wins": r["wins"],
            "losses": r["losses"],
            "win_pct": r["win_pct"],
            "TGRS": r["TGRS"],
            "TGRS_scaled": r["TGRS_scaled"],
            "ts_rating": r["ts_rating"],
            "sos": r["sos"],
            "quality_wins": r["quality_wins"],
        })
    return ordered_rows
