# match_parser.py
# Two entry points:
#   parse_school_matches()  — season-report API  (meets -> matches)
#   parse_bracket_matches() — event bracket API  (bracketItems, NEW structure)
#
# Bracket API reality (confirmed from live response):
#   item["teams"]            list of 2 team dicts  (NOT "matchTeams")
#   team["items"]            list of {"id": player_id_int}  (bare IDs only)
#   team["isWinner"]         bool — which team won
#   item["score"]            ["6 - 4", "7 - 5"] from WINNER's perspective, or null
#   item["matchId"]          int, present only on played matches
#
# Player names/schools are NOT in bracket items.
# They are resolved via build_event_player_lookup() from the seed-list API.

from datetime import datetime
import logging


# ── date helper ───────────────────────────────────────────────────────────────

def _parse_dt(raw) -> datetime:
    if not raw:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


# ── school-match helpers (for parse_school_matches) ───────────────────────────

def _extract_players(team: dict) -> tuple[list, list, int | None, str]:
    """Parse a matchTeam entry from the SCHOOL report API."""
    pids, names = [], []
    school_id, school_name = None, ""
    for p in team.get("players", []):
        pid   = p.get("id") or p.get("playerId")
        first = (p.get("firstName") or "").strip()
        last  = (p.get("lastName")  or "").strip()
        full  = f"{first} {last}".strip()
        if pid is not None:
            pids.append(str(pid))
        if full:
            names.append(full)
        if school_id is None:
            school_id   = p.get("schoolId")
            school_name = ((p.get("school") or {}).get("name") or "").strip()
    return pids, names, school_id, school_name


def _parse_sets(sets_raw, winner_id: int, loser_id: int) -> str:
    """Convert set dicts {teamId: score} to '6-1 6-3'.  Null sets skipped."""
    if not sets_raw:
        return ""
    w_str, l_str = str(winner_id), str(loser_id)
    ordered = sorted(
        (s for s in sets_raw if isinstance(s, dict)),
        key=lambda s: s.get("number", 0),
    )
    parts = []
    for s in ordered:
        w = s.get(w_str)
        l = s.get(l_str)
        if w is None or l is None:
            continue
        parts.append(f"{w}-{l}")
    return " ".join(parts)


def _resolve_school_winner(teams, winner_team_id_val, sets_raw=None):
    if len(teams) != 2:
        return None, None
    t0, t1 = teams
    if winner_team_id_val:
        if t0.get("id") == winner_team_id_val:
            return t0, t1
        if t1.get("id") == winner_team_id_val:
            return t1, t0
    if t0.get("isWinner") and not t1.get("isWinner"):
        return t0, t1
    if t1.get("isWinner") and not t0.get("isWinner"):
        return t1, t0
    # fallback: infer from sets
    if sets_raw:
        a_str, b_str = str(t0.get("id", 0)), str(t1.get("id", 0))
        a_w = b_w = 0
        for s in sets_raw:
            if not isinstance(s, dict):
                continue
            a = s.get(a_str)
            b = s.get(b_str)
            if a is not None and b is not None:
                if a > b: a_w += 1
                elif b > a: b_w += 1
        if a_w > b_w: return t0, t1
        if b_w > a_w: return t1, t0
    return t0, t1   # last resort assumption


# ── bracket helpers (for parse_bracket_matches) ───────────────────────────────

def _parse_bracket_score(score_raw: list) -> str:
    """
    Convert ["6 - 4", "7 - 5"] → "6-4 7-5".
    Scores are already winner_games-loser_games (winner's perspective).
    """
    parts = []
    for s in (score_raw or []):
        if not isinstance(s, str):
            continue
        cleaned = s.strip().replace(" – ", "-").replace("–", "-").replace(" - ", "-").replace(" ", "")
        if "-" in cleaned:
            try:
                left, right = cleaned.split("-", 1)
                parts.append(f"{int(left)}-{int(right)}")
            except ValueError:
                continue
    return " ".join(parts)


def _unwrap_seed_entries(seed_list):
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


def build_event_player_lookup(seed_list) -> dict:
    lookup: dict = {}
    entries = _unwrap_seed_entries(seed_list)
    if not entries:
        return lookup

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for pe in entry.get("players", []):
            player = pe.get("player", {})
            pid = player.get("id")
            if pid is None:
                continue
            school = player.get("school", {})
            lookup[pid] = {
                "firstName":  (player.get("firstName") or "").strip(),
                "lastName":   (player.get("lastName")  or "").strip(),
                "schoolId":   school.get("id"),
                "schoolName": (school.get("name") or "").strip(),
                "genderId":   player.get("genderId"),
            }
    return lookup


def _resolve_bracket_players(team: dict, player_lookup: dict):
    """
    Resolve a bracket team to (player_id_strings, full_names, school_id, school_name).
    team["items"] = [{"id": int_player_id}, ...]
    """
    pids, names = [], []
    school_id, school_name = None, ""
    for item in team.get("items", []):
        pid = item.get("id")
        if pid is None:
            continue
        pids.append(str(pid))
        info = player_lookup.get(pid, {})
        first = info.get("firstName", "")
        last  = info.get("lastName", "")
        full  = f"{first} {last}".strip()
        if full:
            names.append(full)
        if school_id is None:
            school_id   = info.get("schoolId")
            school_name = info.get("schoolName", "")
    return pids, names, school_id, school_name


# ── public entry points ───────────────────────────────────────────────────────

def parse_school_matches(data: dict, source_school_id) -> list[dict]:
    """
    Walk data["meets"] → meet["matches"]["Singles"|"Doubles"].
    Uses meetDateTime as match timestamp.
    Supports all flights (1-4) for both Singles and Doubles.
    """
    results = []
    for meet in data.get("meets", []):
        meet_dt  = _parse_dt(meet.get("meetDateTime"))
        event_id = meet.get("eventId")
        matches_map = meet.get("matches", {})
        if not isinstance(matches_map, dict):
            continue
        for type_key in ("Singles", "Doubles"):
            for m in (matches_map.get(type_key) or []):
                if not isinstance(m, dict):
                    continue
                match_id  = m.get("id") or m.get("matchId")
                gender_id = m.get("genderId")
                gender    = "Boys" if gender_id == 1 else "Girls" if gender_id == 2 else None
                match_type = (m.get("matchType") or type_key).capitalize()
                flight    = str(m.get("flight") or "")
                teams     = m.get("matchTeams", [])
                sets_raw  = m.get("sets") or []

                winner_team, loser_team = _resolve_school_winner(
                    teams, m.get("winnerTeamId"), sets_raw
                )
                if winner_team is None:
                    continue

                w_ids, w_names, w_sid, w_school = _extract_players(winner_team)
                l_ids, l_names, l_sid, l_school = _extract_players(loser_team)
                if not w_ids or not l_ids:
                    continue

                set_score = _parse_sets(sets_raw,
                                        winner_id=winner_team.get("id", 0),
                                        loser_id=loser_team.get("id", 0))
                results.append({
                    "match_id":          match_id,
                    "gender":            gender,
                    "match_type":        match_type,
                    "flight":            flight,
                    "winner_names":      " / ".join(w_names),
                    "loser_names":       " / ".join(l_names),
                    "winner_school":     w_school,
                    "loser_school":      l_school,
                    "winner_school_id":  w_sid,
                    "loser_school_id":   l_sid,
                    "winner_player_ids": w_ids,
                    "loser_player_ids":  l_ids,
                    "set_score":         set_score,
                    "match_updated_at":  meet_dt.isoformat(),
                    "winner_team_id":    winner_team.get("id"),
                    "loser_team_id":     loser_team.get("id"),
                    "source_school_id":  source_school_id,
                    "source_event_id":   event_id,
                })
    return results


def parse_bracket_matches(bracket_data: dict, event_id, event_date: str,
                          gender_id: int | None, match_type: str, flight: int,
                          player_lookup: dict) -> list[dict]:
    """
    Parse event bracket API response.

    Bracket item structure (actual, confirmed):
      item["teams"]       — list of 2 teams
      team["isWinner"]    — bool
      team["items"]       — [{"id": player_id_int}]   ← bare IDs, no name/school
      item["score"]       — ["6 - 4", "7 - 5"] from winner's POV, or null/absent
      item["matchId"]     — int, present only when match is played

    player_lookup  {player_id_int -> {firstName, lastName, schoolId, schoolName}}
    is built from GET /event/{id}/seed_list_by_params and passed in by the crawler.

    Supports flights 1–4 for both Singles and Doubles.
    """
    if not bracket_data:
        return []

    config  = bracket_data.get("configuration", bracket_data)
    items   = config.get("bracketItems") or []
    event_dt = _parse_dt(event_date)
    gender   = "Boys" if gender_id == 1 else "Girls" if gender_id == 2 else None

    played = skipped_no_score = skipped_no_winner = skipped_no_players = 0
    results = []

    for item in items:
        if not isinstance(item, dict):
            continue

        score_raw = item.get("score")
        if not score_raw:
            skipped_no_score += 1
            continue

        teams = item.get("teams", [])
        if len(teams) != 2:
            skipped_no_winner += 1
            continue

        winner_team = next((t for t in teams if t.get("isWinner")), None)
        loser_team  = next((t for t in teams if not t.get("isWinner")), None)
        if winner_team is None or loser_team is None:
            skipped_no_winner += 1
            continue

        w_ids, w_names, w_sid, w_school = _resolve_bracket_players(winner_team, player_lookup)
        l_ids, l_names, l_sid, l_school = _resolve_bracket_players(loser_team,  player_lookup)

        if not w_ids or not l_ids:
            skipped_no_players += 1
            continue

        set_score = _parse_bracket_score(score_raw)
        match_id  = item.get("matchId") or item.get("id")

        results.append({
            "match_id":          match_id,
            "gender":            gender,
            "match_type":        match_type.capitalize(),
            "flight":            str(flight),
            "winner_names":      " / ".join(w_names),
            "loser_names":       " / ".join(l_names),
            "winner_school":     w_school,
            "loser_school":      l_school,
            "winner_school_id":  w_sid,
            "loser_school_id":   l_sid,
            "winner_player_ids": w_ids,
            "loser_player_ids":  l_ids,
            "set_score":         set_score,
            "match_updated_at":  event_dt.isoformat(),
            "winner_team_id":    None,
            "loser_team_id":     None,
            "source_school_id":  None,
            "source_event_id":   event_id,
        })
        played += 1

    if items:
        logging.info(
            "  Bracket e=%s %s flight=%s: %d items → %d played, "
            "%d no_score, %d no_winner, %d no_players",
            event_id, match_type, flight, len(items),
            played, skipped_no_score, skipped_no_winner, skipped_no_players,
        )
    return results


# ── school metadata ───────────────────────────────────────────────────────────

def extract_school_meta(data: dict) -> dict:
    school = data.get("school", {})
    div_boys = div_girls = ""
    for coach in data.get("coaches", []):
        g = coach.get("genderId")
        d = str(coach.get("division") or "").strip()
        if d:
            if g == 1 and not div_boys:    div_boys  = d
            elif g == 2 and not div_girls: div_girls = d
    return {
        "id":             school.get("id"),
        "name":           school.get("name", ""),
        "division_boys":  div_boys,
        "division_girls": div_girls,
    }


# ── deduplication key ─────────────────────────────────────────────────────────

def match_key(match: dict) -> str:
    mid = match.get("match_id")
    if mid:
        return f"id:{mid}"
    all_ids  = sorted(match["winner_player_ids"] + match["loser_player_ids"])
    date_pfx = (match.get("match_updated_at") or "")[:10]
    return f"players:{all_ids}-score:{match.get('set_score','')}-date:{date_pfx}"
