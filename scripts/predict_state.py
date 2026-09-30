"""
predict_state.py
=================

Projects the MHSAA state tournament from the rankings written by
mhsaa_seeding_v2.py: real single-elimination 32-draw brackets (1v32, 16v17,
...), exact champion / finalist / semifinalist probabilities, a single "most
likely" run through every bracket with a realistic scoreline, and a team point
projection (+1 per predicted match win).

Run AFTER mhsaa_seeding_v2.py (and, optionally, build_site.py):

    python scripts/mhsaa_seeding_v2.py <matches.csv>
    python scripts/predict_state.py

THE MODEL IS NOW FIT DIRECTLY TO THE MATCH DATA
-----------------------------------------------
The old model derived everything from a theoretical point-by-point tennis
model and then needed hand-set "shape multipliers" to stop it from predicting
too many close matches. This version has no theory in the middle: it reads
src/predictor_calibration.json, written by scripts/calibrate_predictor.py,
which is fit to held-out real matches (ratings fit on 80% of each pool, scored
on the other 20%).

    rating gap (games)  ->  P(win)                    logistic, fit to results
    P(win of the winner) -> P(3rd set), P(7-6), P(7-5),
                            distribution of straight-set lines,
                            distribution of 3-set lines   kernel-fit to results

Everything about the score hangs off one number, the winning side's win
probability. A close match (small rating gap = small expected margin) has a
much higher chance of a 3rd set, a lopsided one is far more likely to be
6-0 6-0 / 6-0 6-1, and an upset winner gets the closer-looking scores that
upsets really have. Game margin is NOT matched line-by-line any more: it only
matters through the win probability and therefore the 3rd-set chance.

HOW THE PRINTED SCORELINE IS CHOSEN
-----------------------------------
For the predicted winner the possible outcomes are

    * every straight-set line, ORDER-FREE (6-1 6-3 counts the same as 6-3 6-1)
    * ONE combined "3 sets" outcome (all 3-set lines merged)

If the combined chance of a 3rd set beats the single most likely straight-set
line, the prediction is a 3-setter, and the 3-set line printed is one of the
most likely 3-set lines; otherwise the prediction is a straight-set line. A
little seeded randomness is mixed in (near-ties can go either way and the
likeliest lines are favored, not forced), so scores vary between matchups but
are identical on every rebuild. THREE_SET_BIAS in the calibration file scales
the 3-set side of that comparison if you want more or fewer predicted
3-setters.

No Monte Carlo is needed: the outcome table IS the long-run average a
simulation would converge to, so it is used directly and the only randomness
is the small seeded pick above.

Ratings are only comparable inside one pool (gender + singles/doubles +
flight). Every bracket here lives in exactly one pool.

OUTPUT
------
  - docs/csv/predictions/bracket_*.csv           (per-bracket seed odds)
  - docs/csv/predictions/team_predicted_*.csv    (per-division team points)
  - docs/csv/predictions/matches_*.csv           (every predicted match)
  - docs/prediction_of_state.html                (standalone report)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SRC_DIR = REPO_ROOT / "src" / "rankings_by_division_flight"
DOCS_DIR = REPO_ROOT / "docs"
PRED_CSV_DIR = DOCS_DIR / "csv" / "predictions"
PRED_HTML_PATH = DOCS_DIR / "prediction_of_state.html"
CALIBRATION_PATH = REPO_ROOT / "src" / "predictor_calibration.json"

MAX_BRACKET = 32
VALID_FLIGHTS = {"1", "2", "3", "4"}

# Must match mhsaa_seeding_v2.py (POWER_RATING_DEFAULT_RIDGE).
POWER_RIDGE = 0.5

FINISH_LABELS = {
    1: "Champion",
    2: "Runner-up",
    4: "Semifinalist",
    8: "Quarterfinalist",
    16: "Round of 16",
    32: "Round of 32",
}

BYE = object()  # sentinel for an empty bracket slot (small fields get first-round byes)
_EPS = 1e-9


# ============================================================================
# 1.  Data-fit match model (loaded from predictor_calibration.json)
# ============================================================================

def _load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise SystemExit(
            f"  Missing {CALIBRATION_PATH}.\n"
            f"  Run: python scripts/calibrate_predictor.py <matches.csv>"
        )
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


_CAL = _load_calibration()
_WIN = _CAL["win"]                       # {"k", "lam", "cap"}
_PICK = _CAL["pick"]                     # shape_noise, line_temp, three_set_bias
_ANCHORS = _CAL["anchors"]               # kernel-fit tables on a logit grid
_Z_LO, _Z_HI = _ANCHORS[0]["z"], _ANCHORS[-1]["z"]

# Fixed for the run: prebuilt {line: prob} dicts per anchor.
for _a in _ANCHORS:
    _a["_straight"] = {k: p for k, p in _a["straight"]}
    _a["_three"] = {k: p for k, p in _a["three"]}


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return default


def _to_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key)))
    except (TypeError, ValueError):
        return default


def _player_name(row: dict) -> str:
    return row.get("name") or row.get("pair_name") or "Unknown"


def _power(row: dict) -> float:
    """The row's power rating (0 = pool average if missing)."""
    return _to_float(row, "power_rating", 0.0)


def _se2(row: dict) -> float:
    """Squared standard error of the rating (same 1/(ridge + matches) the
    rating fit reports, rebuilt from the row's record)."""
    n = max(0, _to_int(row, "wins") + _to_int(row, "losses"))
    return 1.0 / (POWER_RIDGE + n)


def expected_margin(a: dict, b: dict) -> float:
    """Games A is expected to beat B by: rating gap, capped at +/-cap."""
    cap = _WIN["cap"]
    return max(-cap, min(cap, _power(a) - _power(b)))


def _fav_prob(a: dict, b: dict) -> float:
    """P(the higher-rated side wins), fit to held-out results."""
    ad = abs(expected_margin(a, b))
    s2 = _se2(a) + _se2(b)
    z = _WIN["k"] * ad / math.sqrt(1.0 + _WIN["lam"] * s2)
    return 1.0 / (1.0 + math.exp(-z))


def match_win_prob(a, b) -> float:
    """P(a beats b). A real player always beats a BYE."""
    if a is BYE and b is BYE:
        return 0.5
    if a is BYE:
        return 0.0
    if b is BYE:
        return 1.0
    pf = _fav_prob(a, b)
    d = _power(a) - _power(b)
    if d > 0:
        return pf
    if d < 0:
        return 1.0 - pf
    return 0.5


def _stats_at(pw: float) -> dict:
    """Score-model tables for a winner whose win probability was pw:
    linear interpolation between the two nearest logit anchors."""
    z = min(max(_logit(pw), _Z_LO), _Z_HI)
    hi = next((i for i, a in enumerate(_ANCHORS) if a["z"] >= z), len(_ANCHORS) - 1)
    lo = max(hi - 1, 0)
    a0, a1 = _ANCHORS[lo], _ANCHORS[hi]
    f = 0.0 if a1["z"] == a0["z"] else (z - a0["z"]) / (a1["z"] - a0["z"])

    def lerp(x, y):
        return x + f * (y - x)

    def blend(key):
        keys = set(a0[key]) | set(a1[key])
        return {k: lerp(a0[key].get(k, 0.0), a1[key].get(k, 0.0)) for k in keys}

    return {
        "p3": lerp(a0["p3"], a1["p3"]),
        "p_tb": lerp(a0["p_tb"], a1["p_tb"]),
        "p_75": lerp(a0["p_75"], a1["p_75"]),
        "straight": blend("_straight"),
        "three": blend("_three"),
    }


# ============================================================================
# ============================================================================
# 2.  Standard tournament bracket seeding (1v32, 16v17, 8v25, ... etc.)
# ============================================================================

def make_seed_order(size: int) -> list[int]:
    """
    Standard recursive bracket-seeding order: for size=4 -> [1,4,2,3]
    (1v4, 2v3); for size=8 -> [1,8,4,5,2,7,3,6] (1v8, 4v5, 2v7, 3v6); and
    so on. Keeps the top seeds maximally separated so #1 and #2 can only
    meet in the final, #1-#4 can only meet by the semifinal, etc.
    """
    order = [1]
    while len(order) < size:
        m = len(order) * 2
        new_order = []
        for s in order:
            new_order.append(s)
            new_order.append(m + 1 - s)
        order = new_order
    return order


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def build_bracket_players(rows: list[dict]) -> list:
    """
    Take up to MAX_BRACKET rows (already sorted by rank ascending), pad
    to the next power of two with BYEs (lower seeds get the byes), and
    lay them out in standard bracket order.
    """
    top = rows[:MAX_BRACKET]
    n = len(top)
    size = next_pow2(n) if n > 1 else 2
    order = make_seed_order(size)
    slot_to_row = {i + 1: top[i] for i in range(n)}
    return [slot_to_row.get(seed, BYE) for seed in order]


# ============================================================================
# 3.  Exact bracket probabilities (champion / finalist / semifinalist)
# ============================================================================

def compute_bracket_probabilities(players: list) -> dict[int, dict]:
    """
    Recursively computes, for every power-of-two sub-bracket size that
    appears while splitting `players` in half all the way down, the
    probability distribution over "who emerges from a sub-bracket of
    this size". Exact combinatorics over the known bracket tree -- no
    randomness involved. Uses match_win_prob() completely unmodified.
    """
    key_to_row: dict = {}

    def key_of(p):
        if p is BYE:
            return "BYE"
        k = id(p)
        key_to_row[k] = p
        return k

    captured: dict[int, dict] = {}

    def recurse(sub: list) -> dict:
        if len(sub) == 1:
            return {key_of(sub[0]): 1.0}
        half = len(sub) // 2
        left = recurse(sub[:half])
        right = recurse(sub[half:])
        combined: dict = defaultdict(float)
        for lk, lp in left.items():
            row_l = BYE if lk == "BYE" else key_to_row[lk]
            for rk, rp in right.items():
                row_r = BYE if rk == "BYE" else key_to_row[rk]
                joint = lp * rp
                if joint == 0.0:
                    continue
                p_l_wins = match_win_prob(row_l, row_r)
                combined[lk] += joint * p_l_wins
                combined[rk] += joint * (1.0 - p_l_wins)
        size = len(sub)
        captured.setdefault(size, {}).update(combined)
        return dict(combined)

    recurse(players)
    for size in list(captured):
        captured[size].pop("BYE", None)
    return captured




# ============================================================================
# 4.  Predicted scoreline (data-fit outcome table + small seeded pick)
# ============================================================================

def flip_score(s: str) -> str:
    """Reverse a set score string to the other side's perspective."""
    if s == "BYE":
        return s
    x, y = s.split("-")
    return f"{y}-{x}"


# Salt for the per-matchup score draw. Same players + same salt -> same
# predicted score on every rebuild; change it to reshuffle all the scorelines.
SCORE_SEED = 2026


def _hash32(text: str) -> int:
    """FNV-1a, 32-bit. Mirror in build_site.py (simHash32)."""
    h = 2166136261
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def _uniform_from_seed(seed: int) -> float:
    """First output of mulberry32(seed), in [0, 1). Mirror in build_site.py."""
    m = 0xFFFFFFFF
    a = (seed + 0x6D2B79F5) & m
    t = ((a ^ (a >> 15)) * (a | 1)) & m
    t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & m)) & m
    return ((t ^ (t >> 14)) & m) / 4294967296.0


def _matchup_seed(a: dict, b: dict, winner_is_a: bool) -> int:
    """Order-independent seed for a matchup (players' names + schools)."""
    ka = f"{_player_name(a)}|{a.get('school') or ''}"
    kb = f"{_player_name(b)}|{b.get('school') or ''}"
    w = ka if winner_is_a else kb
    return _hash32(f"{SCORE_SEED}~{'~'.join(sorted((ka, kb)))}~{w}")


def _u(seed: int, salt: int) -> float:
    return _uniform_from_seed((seed + salt) & 0xFFFFFFFF)


def _gumbel(u: float) -> float:
    u = min(max(u, 1e-12), 1.0 - 1e-12)
    return -math.log(-math.log(u))


def _pick_weighted(items: list[tuple[str, float]], temp: float, u: float) -> str:
    """Draw one key with probability proportional to p**(1/temp). temp < 1
    concentrates on the likeliest entries, temp = 1 samples the data exactly."""
    ws = [(k, p ** (1.0 / temp)) for k, p in items if p > 0.0]
    total = sum(w for _, w in ws)
    r = u * total
    acc = 0.0
    pick = ws[-1][0]
    for k, w in ws:
        acc += w
        if r < acc:
            pick = k
            break
    return pick


def _is_tb10(tok: str) -> bool:
    return int(tok.split("-")[0]) >= 10


def _deal_sets(line: str, seed: int) -> list[str]:
    """Turn a stored line (winner's games first in every set) into an actual
    set sequence. Straight sets are order-free, so the order is dealt by a
    seeded coin. A 3-set line is stored as (set lost, set won, decider): the
    winner loses set 1 or set 2 with equal odds and always wins the last set;
    when the decider is a full set the two sets they won can come in either
    order (a match tiebreak is always last)."""
    toks = line.split()
    if len(toks) == 2:
        return toks if _u(seed, 0x2545F491) < 0.5 else toks[::-1]
    lost, w1, w2 = toks
    if not _is_tb10(w2) and _u(seed, 0x3C6EF372) < 0.5:
        w1, w2 = w2, w1
    return [lost, w1, w2] if _u(seed, 0x9E3779B9) < 0.5 else [w1, lost, w2]


def predict_match_details(a: dict, b: dict, winner_is_a: bool) -> dict:
    """
    Everything about one matchup:

      - "score": one scoreline, first number in each set = predicted
        winner's games. Chosen from the data-fit outcome table for the
        winner's win probability (see the module docstring): the combined
        "3 sets" outcome competes with the single likeliest straight-set
        line, then a line is drawn from the winning group, favoring the
        likeliest lines, with a seed built from the two players.
      - "exp_margin": games the favorite (by rating) is expected to win by.
      - "prob_three_sets" / "prob_tiebreak" / "prob_75": chance the match
        goes to a 3rd set / contains a 7-6 set / contains a 7-5 set,
        averaged over who wins.
    """
    d = _power(a) - _power(b)
    pf = _fav_prob(a, b)
    a_is_fav = d >= 0
    winner_is_fav = (winner_is_a == a_is_fav)
    pw = pf if winner_is_fav else 1.0 - pf

    st = _stats_at(pw)
    p3 = st["p3"]
    seed = _matchup_seed(a, b, winner_is_a)

    # Outcomes for the predicted winner: each straight-set line (order-free)
    # and ONE merged "3 sets" outcome.
    straight = sorted(((k, (1.0 - p3) * q) for k, q in st["straight"].items()),
                      key=lambda kv: (-kv[1], kv[0]))
    top_straight = straight[0][1] if straight else 0.0
    bias = _PICK["three_set_bias"]
    noise = _PICK["shape_noise"]
    score3 = math.log(max(p3 * bias, 1e-12)) + noise * _gumbel(_u(seed, 0x1B873593))
    score2 = math.log(max(top_straight, 1e-12)) + noise * _gumbel(_u(seed, 0x85EBCA6B))
    three = bool(st["three"]) and score3 > score2

    if three:
        line = _pick_weighted(sorted(st["three"].items(), key=lambda kv: (-kv[1], kv[0])),
                              _PICK["line_temp"], _u(seed, 0xC2B2AE35))
    else:
        line = _pick_weighted(straight, _PICK["line_temp"], _u(seed, 0x27D4EB2F))
    score = _deal_sets(line, seed)

    # Reported shape odds: mix the winner-side tables by who wins.
    st_other = _stats_at(1.0 - pw)
    def mix(key):
        return pw * st[key] + (1.0 - pw) * st_other[key]

    return {
        "score": score,
        "exp_margin": abs(max(-_WIN["cap"], min(_WIN["cap"], d))),
        "prob_three_sets": mix("p3"),
        "prob_tiebreak": mix("p_tb"),
        "prob_75": mix("p_75"),
    }


# ============================================================================
# 5.  Deterministic single-path bracket run
# ============================================================================

def simulate_bracket(players: list) -> list[list[dict]]:
    """
    One deterministic run through the whole bracket: in every match the
    favorite (p >= 0.5) always advances (no coin flips -- keeps this
    path consistent with the exact probabilities in section 3), and each
    match's scoreline comes from predict_match_details() above. Bye
    matches are marked score=["BYE"].
    """
    current = list(players)
    rounds: list[list[dict]] = []
    while len(current) > 1:
        matches = []
        next_round = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            if a is BYE and b is BYE:
                next_round.append(BYE)
                continue
            if a is BYE or b is BYE:
                winner = b if a is BYE else a
                matches.append({"a": a, "b": b, "winner": winner, "loser": BYE,
                                 "score": ["BYE"], "p_fav": 1.0, "exp_margin": None,
                                 "prob_three_sets": None, "prob_tiebreak": None,
                                 "prob_75": None})
                next_round.append(winner)
                continue
            p = match_win_prob(a, b)
            winner_is_a = p >= 0.5
            winner, loser = (a, b) if winner_is_a else (b, a)
            p_fav = max(p, 1.0 - p)
            details = predict_match_details(a, b, winner_is_a)
            matches.append({
                "a": a, "b": b, "winner": winner, "loser": loser,
                "score": details["score"], "p_fav": p_fav,
                "exp_margin": details["exp_margin"],
                "prob_three_sets": details["prob_three_sets"],
                "prob_tiebreak": details["prob_tiebreak"],
                "prob_75": details["prob_75"],
            })
            next_round.append(winner)
        rounds.append(matches)
        current = next_round
    return rounds


def finish_round_reached(players: list, rounds: list[list[dict]]) -> dict:
    """
    Maps each real player -> the size of the bracket they were still
    alive for immediately before being eliminated (or 1 if they won it
    all), for human-readable finish labels via FINISH_LABELS.
    """
    finish: dict = {}
    alive_count = len(players)
    for rnd in rounds:
        next_alive = alive_count // 2
        for m in rnd:
            if m["score"] == ["BYE"]:
                continue
            finish[id(m["loser"])] = alive_count
        alive_count = next_alive
    if rounds:
        champion = rounds[-1][0]["winner"]
        if champion is not BYE:
            finish[id(champion)] = 1
    return finish



# ============================================================================
# 6.  Loading rankings
# ============================================================================

def load_groups() -> dict[tuple, list[dict]]:
    """
    Reads every per-(category,gender,division) CSV mhsaa_seeding_v2.py
    writes (skipping team_*.csv) and re-groups rows by
    (category, gender, division, flight), sorted by rank ascending --
    exactly the ordering a bracket needs.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    if not SRC_DIR.exists():
        return groups
    for path in sorted(SRC_DIR.glob("*.csv")):
        stem = path.stem
        if stem.startswith("team_"):
            continue
        category = "singles" if stem.startswith("singles") else "doubles"
        gender = "boys" if "_boys_" in stem else "girls"
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                flight = (row.get("flight") or "").strip()
                if flight not in VALID_FLIGHTS:
                    continue
                division = (row.get("division") or "").strip()
                if not division:
                    continue
                groups[(category, gender, division, flight)].append(row)

    for rows in groups.values():
        rows.sort(key=lambda r: int(r.get("rank") or 9999))

    n_rows = sum(len(v) for v in groups.values())
    n_missing = sum(1 for v in groups.values() for r in v if not (r.get("power_rating") or "").strip())
    if n_rows and n_missing:
        print(f"  WARNING: {n_missing}/{n_rows} ranking rows have no power_rating "
              f"(treated as pool-average 0.0). Re-run mhsaa_seeding_v2.py.")
    return groups


_CATEGORY_SORT_ORDER = {"singles": 0, "doubles": 1}


def _group_sort_key(key: tuple) -> tuple:
    """
    Ordering used everywhere a list of (category, gender, division,
    flight) groups gets iterated -- the console summary in run(), and
    (via all_results' insertion order) the HTML report's "Championship /
    Final / Semifinal Odds" and "Predicted Bracket Path" sections:
    division first, then singles before doubles, then gender, then
    flight. This replaces the tuple's natural field order, which sorted
    by category before division (putting every doubles bracket ahead of
    every singles one, and interleaving divisions within each).
    """
    category, gender, division, flight = key
    return (division, _CATEGORY_SORT_ORDER.get(category, 2), gender, flight)


def _team_group_sort_key(key: tuple) -> tuple:
    """Sort key for (gender, division) team-points groups: division
    first, matching _group_sort_key()'s division-first ordering."""
    gender, division = key
    return (division, gender)


# ============================================================================
# 7.  Per-group processing: probabilities + deterministic bracket + team pts
# ============================================================================

def process_group(key: tuple, rows: list[dict]) -> dict:
    category, gender, division, flight = key
    players = build_bracket_players(rows)
    bracket_size = len(players)

    probs = compute_bracket_probabilities(players)
    champion_probs = probs.get(bracket_size, {})
    finalist_probs = probs.get(bracket_size // 2, {})
    semifinalist_probs = probs.get(bracket_size // 4, {}) if bracket_size >= 4 else {}

    rounds = simulate_bracket(players)
    finish = finish_round_reached(players, rounds)

    player_rows = []
    for row in rows[:bracket_size]:
        k = id(row)
        player_rows.append({
            "seed": row.get("rank"),
            "name": _player_name(row),
            "school": row.get("school", ""),
            "p_champion": champion_probs.get(k, 0.0),
            "p_final": finalist_probs.get(k, 0.0),
            "p_semifinal": semifinalist_probs.get(k, champion_probs.get(k, 0.0) if bracket_size < 4 else 0.0),
            "predicted_finish": FINISH_LABELS.get(finish.get(k, bracket_size), f"Round of {finish.get(k, bracket_size)}"),
        })
    player_rows.sort(key=lambda r: -r["p_champion"])

    return {
        "key": key,
        "bracket_size": bracket_size,
        "players": player_rows,
        "rounds": rounds,
    }


def build_team_points(all_results: list[dict]) -> dict[tuple, dict[str, int]]:
    """
    +1 predicted team point per real (non-bye) match win in each
    deterministic bracket run, aggregated across every flight and
    match_type (singles + doubles) within a (gender, division).
    """
    team_points: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in all_results:
        _, gender, division, _flight = result["key"]
        for rnd in result["rounds"]:
            for m in rnd:
                if m["score"] == ["BYE"]:
                    continue
                winner = m["winner"]
                school = winner.get("school", "") if isinstance(winner, dict) else ""
                if school:
                    team_points[(gender, division)][school] += 1
    return team_points


# ============================================================================
# 8.  CSV output
# ============================================================================

def write_prediction_csvs(all_results: list[dict], team_points: dict) -> None:
    PRED_CSV_DIR.mkdir(parents=True, exist_ok=True)

    for result in all_results:
        category, gender, division, flight = result["key"]
        filename = f"bracket_{category}_{gender}_division_{division}_flight_{flight}.csv"
        with open(PRED_CSV_DIR / filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "name", "school", "p_champion", "p_final",
                "p_semifinal", "predicted_finish",
            ])
            writer.writeheader()
            for row in result["players"]:
                writer.writerow({
                    **row,
                    "p_champion": round(row["p_champion"] * 100, 2),
                    "p_final": round(row["p_final"] * 100, 2),
                    "p_semifinal": round(row["p_semifinal"] * 100, 2),
                })

    for (gender, division), schools in sorted(team_points.items(), key=lambda kv: _team_group_sort_key(kv[0])):
        filename = f"team_predicted_{gender}_division_{division}.csv"
        rows = sorted(schools.items(), key=lambda kv: -kv[1])
        with open(PRED_CSV_DIR / filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "school", "predicted_points"])
            for i, (school, pts) in enumerate(rows, start=1):
                writer.writerow([i, school, pts])

    for result in all_results:
        category, gender, division, flight = result["key"]
        filename = f"matches_{category}_{gender}_division_{division}_flight_{flight}.csv"
        with open(PRED_CSV_DIR / filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "player_a", "seed_a", "player_b", "seed_b",
                "predicted_winner", "predicted_score", "win_prob_pct",
                "exp_margin_games", "prob_three_sets_pct", "prob_7_6_tiebreak_pct",
                "prob_7_5_set_pct",
            ])
            for rnd_idx, rnd in enumerate(result["rounds"], start=1):
                for m in rnd:
                    if m["score"] == ["BYE"]:
                        continue
                    a_name = _player_name(m["a"]) if isinstance(m["a"], dict) else "BYE"
                    b_name = _player_name(m["b"]) if isinstance(m["b"], dict) else "BYE"
                    a_seed = m["a"].get("rank", "") if isinstance(m["a"], dict) else ""
                    b_seed = m["b"].get("rank", "") if isinstance(m["b"], dict) else ""
                    writer.writerow([
                        rnd_idx, a_name, a_seed, b_name, b_seed,
                        _player_name(m["winner"]), " ".join(m["score"]),
                        round(m["p_fav"] * 100, 1),
                        round(m.get("exp_margin") or 0.0, 1),
                        round((m.get("prob_three_sets") or 0.0) * 100, 1),
                        round((m.get("prob_tiebreak") or 0.0) * 100, 1),
                        round((m.get("prob_75") or 0.0) * 100, 1),
                    ])


# ============================================================================
# 9.  Standalone HTML report (no longer injected into docs/index.html)
# ============================================================================

def _esc(v) -> str:
    return (str(v).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;"))


def _bracket_path_html(result: dict) -> str:
    category, gender, division, flight = result["key"]
    label = f"{gender.title()} {category.title()} · Division {division} · Flight {flight}"
    rows_html = ""
    for rnd_idx, rnd in enumerate(result["rounds"], start=1):
        for m in rnd:
            if m["score"] == ["BYE"]:
                continue
            a_name = _player_name(m["a"]) if isinstance(m["a"], dict) else "BYE"
            b_name = _player_name(m["b"]) if isinstance(m["b"], dict) else "BYE"
            a_seed = m["a"].get("rank", "") if isinstance(m["a"], dict) else ""
            b_seed = m["b"].get("rank", "") if isinstance(m["b"], dict) else ""
            winner_name = _player_name(m["winner"])
            score_str = " ".join(m["score"])
            em = m.get("exp_margin")
            em_str = f"{em:.1f}" if em is not None else "--"
            p3 = m.get("prob_three_sets")
            ptb = m.get("prob_tiebreak")
            p75 = m.get("prob_75")
            p3_str = f"{p3*100:.1f}%" if p3 is not None else "--"
            ptb_str = f"{ptb*100:.1f}%" if ptb is not None else "--"
            p75_str = f"{p75*100:.1f}%" if p75 is not None else "--"
            rows_html += (
                f"<tr><td>R{rnd_idx}</td>"
                f"<td>{_esc(a_name)} (#{_esc(a_seed)}) vs {_esc(b_name)} (#{_esc(b_seed)})</td>"
                f"<td><b>{_esc(winner_name)}</b></td><td>{_esc(score_str)}</td>"
                f"<td>{m['p_fav']*100:.0f}%</td>"
                f"<td>{em_str}</td>"
                f"<td>{p3_str}</td><td>{ptb_str}</td><td>{p75_str}</td></tr>"
            )
    return f"""
    <div class="pred-bracket">
      <h3>{_esc(label)}</h3>
      <table class="pred-table">
        <thead><tr>
          <th>Round</th><th>Matchup (seed #)</th><th>Predicted Winner</th>
          <th>Predicted Score</th><th>Win Prob.</th>
          <th>Fav. By (games)</th>
          <th>Goes to 3rd Set</th><th>Contains 7-6 TB</th><th>Contains 7-5 Set</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def _probability_table_html(result: dict) -> str:
    category, gender, division, flight = result["key"]
    label = f"{gender.title()} {category.title()} · Division {division} · Flight {flight}"
    top = result["players"][:8]
    rows_html = ""
    for r in top:
        rows_html += (
            f"<tr><td>{_esc(r['seed'])}</td><td>{_esc(r['name'])}</td>"
            f"<td>{_esc(r['school'])}</td>"
            f"<td>{r['p_champion']*100:.1f}%</td>"
            f"<td>{r['p_final']*100:.1f}%</td>"
            f"<td>{r['p_semifinal']*100:.1f}%</td>"
            f"<td>{_esc(r['predicted_finish'])}</td></tr>"
        )
    return f"""
    <div class="pred-probs">
      <h3>{_esc(label)}</h3>
      <table class="pred-table">
        <thead><tr><th>Seed</th><th>Name</th><th>School</th>
        <th>Win It All</th><th>Make Final</th><th>Make Semis</th><th>Predicted Finish</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def _team_table_html(gender: str, division: str, schools: dict[str, int]) -> str:
    rows = sorted(schools.items(), key=lambda kv: -kv[1])[:16]
    rows_html = "".join(
        f"<tr><td>{i}</td><td>{_esc(school)}</td><td>{pts}</td></tr>"
        for i, (school, pts) in enumerate(rows, start=1)
    )
    return f"""
    <div class="pred-team">
      <h3>{gender.title()} · Division {_esc(division)} — Projected Team Standings</h3>
      <table class="pred-table">
        <thead><tr><th>Rank</th><th>School</th><th>Predicted Points</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


_PAGE_CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 2rem 1.25rem 4rem; color: #1c1c1e; background: #fff; }
h1 { font-size: 1.7rem; margin-bottom: .25rem; }
h2 { font-size: 1.25rem; margin-top: 2.25rem; border-bottom: 2px solid #eee; padding-bottom: .35rem; }
h3 { font-size: 1.02rem; margin: 1.5rem 0 .5rem; color: #333; }
.back-link { display: inline-block; margin: .5rem 0 1.25rem; font-size: .88rem; }
.back-link a { color: #1a3a5c; text-decoration: none; border: 1px solid #c0d4e8; border-radius: 6px;
                padding: .3rem .7rem; }
.back-link a:hover { background: #e8f0f8; }
.intro-note { font-size: .88rem; color: #555; line-height: 1.5; max-width: 780px; }
.pred-table { border-collapse: collapse; width: 100%; margin-bottom: 1rem; font-size: .88rem; }
.pred-table th, .pred-table td { border: 1px solid #e2e2e2; padding: .4rem .55rem; text-align: left; }
.pred-table thead th { background: #f5f5f7; font-weight: 600; }
.pred-table tbody tr:nth-child(even) { background: #fafafa; }
.pred-bracket, .pred-probs, .pred-team { margin-bottom: 1.5rem; }
.generated-note { font-size: .78rem; color: #888; margin-top: 3rem; border-top: 1px solid #eee; padding-top: .75rem; }
"""


def build_full_html(all_results: list[dict], team_points: dict) -> str:
    team_html = "".join(
        _team_table_html(gender, division, schools)
        for (gender, division), schools in sorted(team_points.items(), key=lambda kv: _team_group_sort_key(kv[0]))
    )
    prob_html = "".join(_probability_table_html(r) for r in all_results)
    bracket_html = "".join(_bracket_path_html(r) for r in all_results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prediction of State</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
  <h1>Prediction of State</h1>
  <p class="back-link"><a href="index.html">&larr; Back to Rankings</a></p>
  <p class="intro-note">
    Every number below comes from each player's <b>power rating</b> and is
    fit to real match results (held-out matches, not the ones that built the
    ratings). The gap between two ratings is the expected game margin
    (12 = 6-0 6-0); it sets the win probability, and the win probability sets
    the score: close matches are much likelier to go three sets, lopsided ones
    to be 6-0 6-0 or 6-0 6-1, and upset wins look like upsets do in the data.
    Championship / final / semifinal odds are computed exactly over the real
    seeded 32-draw bracket (#1 and #2 can only meet in the final, etc.). The
    bracket path is the single most-likely outcome: the higher-rated side
    always advances. For each match the printed score picks between the
    single likeliest straight-set line (6-1 6-3 and 6-3 6-1 count as the same
    line) and "goes three sets" (all 3-set lines combined); when the combined
    chance of a 3rd set is bigger, a likely 3-set line is shown. A little
    seeded randomness keeps scorelines varied but identical on every rebuild.
    "Fav. By" is the expected game margin; the last three columns are the
    chance the match goes to a 3rd set, contains a 7-6 tiebreak set, or
    contains a 7-5 set.
  </p>

  <h2>Projected Team Standings</h2>
  {team_html}

  <h2>Championship / Final / Semifinal Odds (Top 8 Seeds)</h2>
  {prob_html}

  <h2>Predicted Bracket Path</h2>
  {bracket_html}

  <p class="back-link"><a href="index.html">&larr; Back to Rankings</a></p>
  <p class="generated-note">Generated by predict_state.py.</p>
</body>
</html>
"""


def write_html_report(all_results: list[dict], team_points: dict) -> Path:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    html = build_full_html(all_results, team_points)
    PRED_HTML_PATH.write_text(html, encoding="utf-8")
    return PRED_HTML_PATH


# ============================================================================
# 10.  Orchestration
# ============================================================================

def run() -> None:
    groups = load_groups()
    if not groups:
        print(f"  No ranking CSVs found under {SRC_DIR}. Run mhsaa_seeding_v2.py first.")
        return

    all_results = []
    for key in sorted(groups, key=_group_sort_key):
        result = process_group(key, groups[key])
        all_results.append(result)
        category, gender, division, flight = key
        champ = result["players"][0] if result["players"] else None
        champ_desc = f"{champ['name']} ({champ['p_champion']*100:.1f}%)" if champ else "n/a"
        print(f"  {gender:6} {category:8} div={division} flight={flight}  "
              f"bracket={result['bracket_size']:3}  predicted champion: {champ_desc}")

    team_points = build_team_points(all_results)

    write_prediction_csvs(all_results, team_points)
    html_path = write_html_report(all_results, team_points)

    print(f"\n  {len(all_results)} bracket(s) processed.")
    print(f"  Prediction CSVs written -> {PRED_CSV_DIR}/")
    print(f"  Standalone report written -> {html_path}")


if __name__ == "__main__":
    run()
