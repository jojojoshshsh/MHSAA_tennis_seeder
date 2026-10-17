#!/usr/bin/env python3
"""
Fetch every flight's bracket for the tennisreporting.com state finals,
resolve player ids to names/schools/seeds via the seed_list_by_params
endpoint, print each bracket, and write everything to a CSV.

Bracket structure (bracketType 32):
  Round 1: 16 matches, Round 2: 8, Round 3: 4, Round 4: 2, Round 5: 1
  R(r) position p is fed by R(r-1) positions 2p-1 and 2p.
  Round 1 positions 1-8 = teams[0] (top side), 9-16 = teams[1] (bottom side).
  A round 1 match with only one entrant is a bye; that entrant advances.

NOTE: the ids inside bracketItems are PLAYER ids (not team/school ids).

Usage:
  python fetch_brackets.py                 # all divisions/flights -> brackets.csv
  python fetch_brackets.py --singles       # singles only
  python fetch_brackets.py --out file.csv  # custom CSV name
  python fetch_brackets.py --save          # also dump raw JSON responses
"""
import csv
import json
import sys
import requests

EVENT_ID = 906
BRACKET_URL = "https://api.tennisreporting.com/event/{event}/host/{host}/bracket/get"
SEEDS_URL = "https://api.tennisreporting.com/event/{event}/seed_list_by_params"

# division number -> (division id, host id)
DIVISIONS = {
    1: (1453, 3971),
    2: (1454, 3972),
    3: (1455, 3973),
    4: (1456, 3974),
}
FLIGHTS = [1, 2, 3, 4]
MATCH_TYPES = ["Singles", "Doubles"]

HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Origin": "https://www.tennisreporting.com",
    "Referer": "https://www.tennisreporting.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36",
    "token": "undefined",
}

FIELDS = [
    "division", "match_type", "flight", "round", "position", "match_id",
    "team1_players", "team1_school", "team1_seed", "team1_player_ids",
    "team2_players", "team2_school", "team2_seed", "team2_player_ids",
    "bye", "note", "date",
]


def payload(division_id, host_id, match_type, flight):
    return {
        "division": division_id,
        "host": host_id,
        "matchType": match_type,
        "flight": flight,
        "isConsolation": False,
    }


def fetch_bracket(division_id, host_id, match_type, flight):
    r = requests.post(BRACKET_URL.format(event=EVENT_ID, host=host_id),
                      headers=HEADERS,
                      json=payload(division_id, host_id, match_type, flight),
                      timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_seeds(division_id, host_id, match_type, flight):
    r = requests.post(SEEDS_URL.format(event=EVENT_ID),
                      headers=HEADERS,
                      json=payload(division_id, host_id, match_type, flight),
                      timeout=30)
    r.raise_for_status()
    return r.json()


def clean(s):
    return " ".join((s or "").split())


def build_player_map(seed_list):
    """playerId -> {name, school, grade, seed}"""
    pmap = {}
    for entry in seed_list:
        for p in entry.get("players", []):
            pl = p.get("player") or {}
            pid = p.get("playerId") or pl.get("id")
            school = (pl.get("school") or {}).get("name")
            pmap[pid] = {
                "name": clean(f"{pl.get('firstName', '')} {pl.get('lastName', '')}"),
                "school": clean(school),
                "grade": pl.get("grade"),
                "seed": entry.get("seed"),
            }
    return pmap


# ---------- bracket logic ----------

def parse_bracket(data):
    rounds = {}
    for m in data["configuration"]["bracketItems"]:
        rounds.setdefault(m["round"], {})[m["position"]] = m
    return rounds


def team_ids(team):
    return [i["id"] for i in team.get("items", [])]


def resolve(rounds, rnd, pos, slot):
    """Player ids in a slot, [] if unknown/TBD, or "BYE"."""
    m = rounds[rnd][pos]
    ids = team_ids(m["teams"][slot])
    if ids:
        return ids
    if rnd == 1:
        return "BYE"
    feeder = rounds[rnd - 1].get(2 * pos - 1 + slot)
    if not feeder:
        return []
    for t in feeder["teams"]:
        if t.get("isWinner") and t.get("items"):
            return team_ids(t)
    if rnd - 1 == 1:  # feeder was a round-1 bye
        filled = [team_ids(t) for t in feeder["teams"] if t.get("items")]
        if len(filled) == 1:
            return filled[0]
    return []


def describe(ids, pmap):
    """-> (players, school, seed) strings for CSV/console."""
    if ids == "BYE":
        return "BYE", "", ""
    if not ids:
        return "TBD", "", ""
    players, schools, seeds = [], [], []
    for i in ids:
        info = pmap.get(i)
        if info:
            players.append(info["name"])
            if info["school"] and info["school"] not in schools:
                schools.append(info["school"])
            if info["seed"] is not None and str(info["seed"]) not in seeds:
                seeds.append(str(info["seed"]))
        else:
            players.append(f"#{i}")
    return " / ".join(players), " / ".join(schools), " / ".join(seeds)


def show(ids, pmap):
    p, s, sd = describe(ids, pmap)
    if ids == "BYE" or not ids:
        return p
    return f"{p} ({s}) [{sd}]"


def process_bracket(data, pmap, label, div, mt, flight, rows):
    rounds = parse_bracket(data)
    max_round = max(rounds)
    print("=" * 90)
    print(label)
    print("=" * 90)
    for rnd in sorted(rounds):
        print(f"\n--- {'Final' if rnd == max_round else f'Round {rnd}'} ---")
        for pos in sorted(rounds[rnd]):
            m = rounds[rnd][pos]
            a = resolve(rounds, rnd, pos, 0)
            b = resolve(rounds, rnd, pos, 1)
            bye = rnd == 1 and (a == "BYE" or b == "BYE")
            note = (m.get("note") or "").strip()
            date = m.get("date") or ""

            if bye:
                team = b if a == "BYE" else a
                line = f"{show(team, pmap)}  (bye)"
            else:
                line = f"{show(a, pmap)}  vs  {show(b, pmap)}"
            extra = " [" + ", ".join(x for x in (note, date[:16].replace('T', ' ')) if x) + "]" \
                if (note or date) else ""
            print(f"  R{rnd}-{pos:<2} {line}{extra}")

            p1, s1, sd1 = describe(a, pmap)
            p2, s2, sd2 = describe(b, pmap)
            rows.append({
                "division": div, "match_type": mt, "flight": flight,
                "round": rnd, "position": pos, "match_id": m["id"],
                "team1_players": p1, "team1_school": s1, "team1_seed": sd1,
                "team1_player_ids": "" if a in ("BYE", []) else ",".join(map(str, a)),
                "team2_players": p2, "team2_school": s2, "team2_seed": sd2,
                "team2_player_ids": "" if b in ("BYE", []) else ",".join(map(str, b)),
                "bye": bye, "note": note, "date": date,
            })


def main():
    args = sys.argv[1:]
    save = "--save" in args
    only_singles = "--singles" in args
    out = args[args.index("--out") + 1] if "--out" in args else "brackets.csv"
    rows = []

    for div, (div_id, host_id) in DIVISIONS.items():
        for mt in MATCH_TYPES:
            if only_singles and mt != "Singles":
                continue
            for flight in FLIGHTS:
                label = f"Division {div} - {mt} Flight {flight}"
                try:
                    data = fetch_bracket(div_id, host_id, mt, flight)
                    seeds = fetch_seeds(div_id, host_id, mt, flight)
                except Exception as e:
                    print(f"{label}: failed ({e})")
                    continue
                if "configuration" not in data:
                    print(f"{label}: no bracket published")
                    continue
                pmap = build_player_map(seeds)
                process_bracket(data, pmap, label, div, mt, flight, rows)
                if save:
                    with open(f"bracket_d{div}_{mt.lower()}_f{flight}.json", "w") as f:
                        json.dump({"bracket": data, "seeds": seeds}, f, indent=2)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()