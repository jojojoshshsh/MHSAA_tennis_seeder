"""
predict_state.py
=================

Projects the MHSAA state tournament from the rankings written by
mhsaa_seeding_v2.py: real single-elimination 32-draw brackets (1v32,
16v17, ...), exact champion / finalist / semifinalist probabilities, a
single "most likely" run through every bracket with a realistic scoreline,
and a team point projection (+1 per predicted match win).

Run AFTER mhsaa_seeding_v2.py (and, optionally, build_site.py):

    python scripts/mhsaa_seeding_v2.py <matches.csv>
    python scripts/predict_state.py

WHAT CHANGED: THE POWER RATING IS NOW THE PREDICTION MODEL
------------------------------------------------------------
Previously every prediction came from TrueSkill (mu/sigma) blended with a
seed prior, and the scoreline was a Monte Carlo of a *different* quantity
(a dominance proxy built from win%, SOS and TGRS). Three separate signals
had to be glued together and tuned against each other.

The power rating already answers the question directly. Its contract is:

    rating_A - rating_B  =  how many GAMES A is expected to win by
                            (capped at 12 = 6-0 6-0)

so this file now uses that number as the single source of truth, and
derives *everything* from it in one chain, with no randomness:

    rating gap d  (expected game margin)
        |  invert: which per-POINT win probability p makes a best-of-3
        |  match (best of 3 full sets) have an expected game
        |  margin of exactly d?         [_solve_point_prob]
        v
    p  ->  game prob (deuce math) -> set-score distribution (6-x, 7-5,
           7-6 via a 7-pt tiebreak) -> match-score distribution
        |  all EXACT, computed by dynamic programming
        v
    P(win), P(3 sets), P(7-6 set), P(7-5 set), every exact scoreline

Because the model is built on the same margin convention the rating was
fit on (a match tiebreak counts as +/-1 game, i.e. an ordinary 7-6), a
rating gap of 12 maps to p -> 1 (6-0 6-0), a gap of 2 to a ~68% favorite,
a gap of 1 to a near coin flip that usually goes three sets, and so on --
the whole scale falls out of the rating instead of being hand-tuned.

FORM NOISE. A pure point-by-point model is far too sure of itself: it
would call a 6-game favorite a 94% lock. Real players have good and bad
days, and the ratings themselves carry estimation error. So the gap is
treated as a random variable

    d' ~ Normal(d, FORM_SD^2 + se_a^2 + se_b^2)      (clipped to +/-12)

and every probability is averaged over d' with a fixed 7-point
Gauss-Hermite rule (deterministic -- still no Monte Carlo). se comes from
the same formula the rating fit uses, 1/sqrt(ridge + matches).
FORM_SD (below) controls how upset-prone the tournament is.

Ratings are only comparable inside one pool (gender + singles/doubles +
flight). Every bracket here lives in exactly one pool, and the
cross-division "overall" ranking is fit on the whole pool, so gaps between
schools in different divisions are valid too.

The seed-committee prior is blended in at a small, backtested weight
(SEED_BLEND_WEIGHT): the rating gap already contains the results that
produced the seeds, so a heavy blend double counts them and hurts accuracy.
FORM_SD, the blend weight and the shape multipliers are all fit to a
walk-forward backtest -- see the constants block and calibrate_predictor.py.

OUTPUT
------
  - docs/csv/predictions/bracket_*.csv           (per-bracket seed odds)
  - docs/csv/predictions/team_predicted_*.csv    (per-division team points)
  - docs/csv/predictions/matches_*.csv           (every predicted match,
                                                  incl. expected margin)
  - docs/prediction_of_state.html                (standalone report)
"""

from __future__ import annotations

import csv
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

MAX_BRACKET = 32
VALID_FLIGHTS = {"1", "2", "3", "4"}

# ---- Power-rating model constants -----------------------------------------
# Must match mhsaa_seeding_v2.py (POWER_RATING_CAP / POWER_RATING_DEFAULT_RIDGE).
POWER_CAP = 12.0
POWER_RIDGE = 0.5

# ---- Constants FIT TO DATA (walk-forward backtest; see calibrate_predictor.py)
#
# Backtest: 2026 boys data, 13 daily cutoffs. At each cutoff the real seeding
# pipeline was run on everything before it, then the following day(s) of
# matches were predicted: 2,444 held-out matches, 693 of them between two
# players who both had 5+ matches (which is what a state bracket looks like).
#
# FORM_SD -- std-dev, in GAMES, of a player's day-to-day performance around
# their rating (on top of rating estimation error). Fit by maximum likelihood
# of the FULL observed scoreline, not just who won: the optimum is flat
# between 3 and 4, and 3.0 costs almost nothing in win-probability log-loss
# (0.3715 vs 0.3688 at the win-only optimum of 2.0). Mirror in build_site.py.
FORM_SD = 3.0

# Seed-committee prior: the higher seed wins ~95% of matches (19 years of
# MHSAA data). SEED_BLEND_WEIGHT is how much of that prior is mixed into the
# rating-based win probability, in logit space. Held-out log-loss (lower is
# better), matches with 5+ games played by both players:
#     w = 0.00 -> 0.3715    w = 0.05 -> 0.3706 (best)    w = 0.10 -> 0.3715
#     w = 0.15 -> 0.3741    w = 0.25 -> ~0.385 (the old default: WORSE)
#     w = 0.50 -> 0.4377
# The rating gap already contains the results that produced the seeds, so
# the prior is nearly redundant: a small weight helps a hair (the 95% CI on
# the gain includes zero), a big one clearly hurts. 0.05 keeps the blend ON
# at the accuracy-maximizing strength. Mirror in build_site.py.
SEED_PRIOR_ACCURACY = 0.950
SEED_BLEND_WEIGHT = 0.05

# The point-by-point model treats the two sets as independent given the
# match's form, so it over-predicts how often matches are competitive. On
# held-out matches it said 23.8% go to a 3rd set (actual 14.3%), 12.1%
# contain a 7-6 set (actual 5.6%) and 13.4% contain a 7-5 set (actual
# 11.0%). These multipliers apply those observed ratios to the REPORTED shape
# odds and to the straight-sets-vs-three-sets pick. Mass removed from
# three-setters moves to straight sets, so win probability is untouched.
THREE_SET_SCALE = 0.60

# The PRINTED scoreline's total game margin must EQUAL the seed-adjusted
# expected margin rounded to whole games (7-5 counts 1.5, 7-6 counts 1; see
# _set_margin / predict_match_details), so there is no tolerance constant.
TIEBREAK_SCALE = 0.46
SEVEN_FIVE_SCALE = 0.82


TABLE_STEP = 0.25   # rating-gap grid spacing for the precomputed match table

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
# 1.  Exact point -> game -> set -> match model
# ============================================================================

def _game_prob(p: float) -> float:
    """P(win a game) when each point is won with prob p (ad scoring)."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1.0 - p
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 5 * q ** 3 / (1 - 2 * p * q)


def _race_prob(p: float, n: int) -> float:
    """P(win) a first-to-n, win-by-2 points race (7-pt set tiebreak: n=7;
    10-pt match tiebreak: n=10) when each point is won with prob p."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    q = 1.0 - p
    s = sum(math.comb(n - 1 + k, k) * p ** n * q ** k for k in range(n - 1))
    s += math.comb(2 * (n - 1), n - 1) * (p * q) ** (n - 1) * p * p / (p * p + q * q)
    return s


def _set_dist(p: float) -> dict[tuple[int, int], float]:
    """Exact distribution of one set's score, from A's side: 6-0..6-4,
    7-5, and 7-6 (7-pt tiebreak at 6-6), plus the mirror images."""
    g = _game_prob(p)
    h = 1.0 - g
    t = _race_prob(p, 7)
    reach = {(0, 0): 1.0}
    out: dict[tuple[int, int], float] = defaultdict(float)
    for total in range(13):
        for ga in range(total + 1):
            gb = total - ga
            pr = reach.get((ga, gb), 0.0)
            if not pr:
                continue
            if ga == 6 and gb == 6:
                out[(7, 6)] += pr * t
                out[(6, 7)] += pr * (1 - t)
                continue
            for na, nb, w in ((ga + 1, gb, g), (ga, gb + 1, h)):
                if (na >= 6 and na - nb >= 2) or (nb >= 6 and nb - na >= 2) or na == 7 or nb == 7:
                    out[(na, nb)] += pr * w
                else:
                    reach[(na, nb)] = reach.get((na, nb), 0.0) + pr * w
    return dict(out)


def _super_tb_dist(p: float) -> dict[tuple[int, int], float]:
    """Exact 10-point match-tiebreak score from A's side. 10-0 .. 10-7 are
    exact; 10-8 also absorbs every longer deuce ending (11-9, 12-10, ...)
    so the display never needs a score the data doesn't use."""
    q = 1.0 - p
    out: dict[tuple[int, int], float] = {}
    a_exact = b_exact = 0.0
    for k in range(8):
        out[(10, k)] = math.comb(9 + k, k) * p ** 10 * q ** k
        out[(k, 10)] = math.comb(9 + k, k) * q ** 10 * p ** k
        a_exact += out[(10, k)]
        b_exact += out[(k, 10)]
    win = _race_prob(p, 10)
    out[(10, 8)] = max(0.0, win - a_exact)
    out[(8, 10)] = max(0.0, (1.0 - win) - b_exact)
    return out


def _expected_margin_for_point_prob(p: float) -> float:
    """E[signed game margin of A] for a best-of-3 match, using the SAME
    convention the power rating was fit on: a set's margin is its game
    difference (7-6 = +1) and the match tiebreak counts as +/-1 (an
    ordinary 7-6). Closed form via linearity -- both sets are always
    played; the 3rd only when they split."""
    sd = _set_dist(p)
    s_win = sum(v for (a, b), v in sd.items() if a > b)
    e_set = sum(v * (a - b) for (a, b), v in sd.items())
    t = _race_prob(p, 10)
    return 2.0 * e_set + 2.0 * s_win * (1.0 - s_win) * (2.0 * t - 1.0)


def _solve_point_prob(d: float) -> float:
    """Point-win probability p in [0.5, 1) whose best-of-3 expected game
    margin equals d (0 <= d <= POWER_CAP). Monotone -> bisection."""
    if d <= 0.0:
        return 0.5
    if d >= POWER_CAP:
        return 1.0
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _expected_margin_for_point_prob(mid) < d:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _set_margin(x: str) -> float:
    """Game margin a set score contributes, from the first number's side.
    7-5 counts as 1.5 and 7-6 (tiebreak) as 1; every other set is its plain
    game difference (6-3 = 3, 6-0 = 6, a lost 4-6 = -2, ...)."""
    a, b = (int(v) for v in x.split("-"))
    if (a, b) == (7, 5):
        return 1.5
    if (a, b) == (5, 7):
        return -1.5
    if (a, b) == (7, 6):
        return 1.0
    if (a, b) == (6, 7):
        return -1.0
    return float(a - b)


def _line_margin(sets) -> float:
    """Total game margin of a whole scoreline (winner's side)."""
    return sum(_set_margin(x) for x in sets)


def _fmt_set(a: int, b: int) -> str:
    return f"{a}-{b}"


def _canon2(s1, s2):
    """Straight-sets line from the WINNER's side, order-free. The two sets
    are independent draws, so (6-2, 6-3) and (6-3, 6-2) are the same
    outcome for ranking purposes; ranking ordered pairs split that mass and
    let the identical pair (6-2, 6-2) win every time. Merge the orderings
    and show the bigger win first."""
    a, b = sorted((s1, s2), key=lambda s: -(s[0] - s[1]))
    return (_fmt_set(*a), _fmt_set(*b))


def _canon3(s1, s2, t3):
    """Three-set line from the WINNER's side. The winner took two sets and
    lost one; the two sets they won are interchangeable draws, so merge
    their orderings (else the identical pair, e.g. 6-3 6-3, wins by
    default). Shown as: the set lost, the bigger win, then the tighter
    deciding set."""
    lost, won = (s1, s2) if s1[0] < s1[1] else (s2, s1)
    w_a, w_b = sorted((won, t3), key=lambda x: -(x[0] - x[1]))
    return (_fmt_set(*lost), _fmt_set(*w_a), _fmt_set(*w_b))


def _match_outcomes(p: float) -> dict:
    """Exact best-of-3 outcome summary for A vs B at point prob p >= 0.5
    (A is the favorite):

      stats  -> [P(A wins 2-0), P(A wins 2-1), P(B wins 2-0), P(B wins 2-1),
                 P(some set is 7-6), P(some set is 7-5)]
      a_wins -> {2: [(sets, prob)...], 3: [...]}  scorelines from the
                WINNER's side, top entries only
      b_wins -> same, for the underdog winning
    """
    sd = _set_dist(p)
    tb = sd   # the 3rd set is a real set, same distribution as sets 1-2
    pw2 = pw3 = pl2 = pl3 = p_tb = p_75 = 0.0
    a_lists: dict[int, dict] = {2: defaultdict(float), 3: defaultdict(float)}
    b_lists: dict[int, dict] = {2: defaultdict(float), 3: defaultdict(float)}

    for s1, p1 in sd.items():
        for s2, p2 in sd.items():
            joint = p1 * p2
            has_tb = {s1, s2} & {(7, 6), (6, 7)}
            has_75 = {s1, s2} & {(7, 5), (5, 7)}
            w1, w2 = s1[0] > s1[1], s2[0] > s2[1]
            if w1 == w2:
                if has_tb:
                    p_tb += joint
                if has_75:
                    p_75 += joint
                if w1:      # A wins 2-0
                    pw2 += joint
                    a_lists[2][_canon2(s1, s2)] += joint
                else:       # B wins 2-0 (shown from B's side)
                    pl2 += joint
                    b_lists[2][_canon2((s1[1], s1[0]), (s2[1], s2[0]))] += joint
            else:
                for t3, p3 in tb.items():
                    j3 = joint * p3
                    if has_tb or t3 in ((7, 6), (6, 7)):
                        p_tb += j3
                    if has_75 or t3 in ((7, 5), (5, 7)):
                        p_75 += j3
                    if t3[0] > t3[1]:   # A wins the 3rd set -> A wins 2-1
                        pw3 += j3
                        a_lists[3][_canon3(s1, s2, t3)] += j3
                    else:               # B wins 2-1 (B's side)
                        pl3 += j3
                        b_lists[3][_canon3((s1[1], s1[0]), (s2[1], s2[0]),
                                           (t3[1], t3[0]))] += j3

    def top(d):
        """Every candidate line, most likely first (there are only a few
        hundred), so the picker can also weigh game margin."""
        return sorted(((k, v) for k, v in d.items() if v > 0.0), key=lambda kv: -kv[1])

    return {
        "stats": [pw2, pw3, pl2, pl3, p_tb, p_75],
        "a_wins": {2: top(a_lists[2]), 3: top(a_lists[3])},
        "b_wins": {2: top(b_lists[2]), 3: top(b_lists[3])},
    }


_TABLE: list[dict] | None = None


def _table() -> list[dict]:
    """Lazily build the match table on a rating-gap grid 0..POWER_CAP."""
    global _TABLE
    if _TABLE is None:
        n = int(round(POWER_CAP / TABLE_STEP))
        _TABLE = [_match_outcomes(_solve_point_prob(i * TABLE_STEP)) for i in range(n + 1)]
    return _TABLE


def _stats_at(d: float) -> list[float]:
    """Interpolated [pw2, pw3, pl2, pl3, p_tb, p_75] for signed gap d
    (positive = A favored). Negative gaps use the mirror image."""
    tab = _table()
    x = min(abs(d), POWER_CAP) / TABLE_STEP
    i = min(int(x), len(tab) - 2)
    f = x - i
    lo, hi = tab[i]["stats"], tab[i + 1]["stats"]
    s = [lo[k] + f * (hi[k] - lo[k]) for k in range(6)]
    if d < 0:
        s = [s[2], s[3], s[0], s[1], s[4], s[5]]
    return s


# 7-point Gauss-Hermite rule for a standard normal (nodes, weights).
_GH_X = (0.0, 0.8162878828589647, 1.6735516287674714, 2.6519613568352334)
_GH_W = (0.8102646175568073, 0.4256072526101278, 0.05451558281912703, 0.0009717812450995)
_GH: list[tuple[float, float]] = []
for _x, _w in zip(_GH_X, _GH_W):
    _w_n = _w / math.sqrt(math.pi)
    if _x == 0.0:
        _GH.append((0.0, _w_n))
    else:
        _GH.append((math.sqrt(2.0) * _x, _w_n))
        _GH.append((-math.sqrt(2.0) * _x, _w_n))


def _mixture_stats(d: float, tau: float) -> list[float]:
    """Average _stats_at over d' ~ Normal(d, tau^2), clipped to +/-cap."""
    acc = [0.0] * 6
    for z, w in _GH:
        dd = max(-POWER_CAP, min(POWER_CAP, d + tau * z))
        s = _stats_at(dd)
        for k in range(6):
            acc[k] += w * s[k]
    return acc


# ============================================================================
# 2.  Reading a ranking row as a power rating
# ============================================================================

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


def _power_se(row: dict) -> float:
    """Standard error of the rating: the same 1/sqrt(ridge + matches) the
    rating fit reports, rebuilt from the row's record."""
    n = max(0, _to_int(row, "wins") + _to_int(row, "losses"))
    return 1.0 / math.sqrt(POWER_RIDGE + n)


def expected_margin(a: dict, b: dict) -> float:
    """Games A is expected to beat B by: rating gap, capped at +/-12."""
    return max(-POWER_CAP, min(POWER_CAP, _power(a) - _power(b)))


def _tau(a: dict, b: dict) -> float:
    return math.sqrt(FORM_SD ** 2 + _power_se(a) ** 2 + _power_se(b) ** 2)


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


_SEED_PRIOR_LOGIT = _logit(SEED_PRIOR_ACCURACY)


def _seed_number(row: dict) -> int | None:
    try:
        return int(row.get("rank"))
    except (TypeError, ValueError):
        return None


def _apply_seed_prior(p: float, a: dict, b: dict) -> float:
    """Optional logit-space blend with the seed prior (off by default)."""
    if SEED_BLEND_WEIGHT <= 0.0:
        return p
    sa, sb = _seed_number(a), _seed_number(b)
    if sa is None or sb is None or sa == sb:
        return p
    seed_logit = _SEED_PRIOR_LOGIT if sa < sb else -_SEED_PRIOR_LOGIT
    return _sigmoid((1.0 - SEED_BLEND_WEIGHT) * _logit(p) + SEED_BLEND_WEIGHT * seed_logit)


def match_win_prob(a, b) -> float:
    """P(a beats b): the exact best-of-3 model driven by the power-rating
    gap, averaged over form noise. A real player always beats a BYE."""
    if a is BYE and b is BYE:
        return 0.5
    if a is BYE:
        return 0.0
    if b is BYE:
        return 1.0
    s = _mixture_stats(expected_margin(a, b), _tau(a, b))
    return _apply_seed_prior(s[0] + s[1], a, b)


def effective_margin(a: dict, b: dict) -> float:
    """Signed expected game margin (positive = A favored) AFTER the seed
    boost. The seed prior nudges the win probability toward the higher seed,
    so the rating gap alone no longer describes the matchup. This inverts the
    blended win probability back into a rating-gap-equivalent: the gap d_eff
    whose plain (no-prior) win probability equals match_win_prob(a, b).
    Without a seed difference (or with the blend off) it is exactly the raw
    rating gap. The printed scoreline is built from this number, so the
    "Fav. By" column, the winner and the scoreline all agree."""
    d = expected_margin(a, b)
    if SEED_BLEND_WEIGHT <= 0.0:
        return d
    sa, sb = _seed_number(a), _seed_number(b)
    if sa is None or sb is None or sa == sb:
        return d
    tau = _tau(a, b)
    target = match_win_prob(a, b)

    def f(x: float) -> float:
        s = _mixture_stats(x, tau)
        return s[0] + s[1]

    lo, hi = -POWER_CAP, POWER_CAP
    if target >= f(hi):
        return hi
    if target <= f(lo):
        return lo
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
# 4.  Predicted scoreline (exact -- replaces the Monte Carlo engine)
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


def predict_match_details(a: dict, b: dict, winner_is_a: bool) -> dict:
    """
    Everything about one matchup, exact and deterministic:

      - "exp_margin": games the favorite is expected to win by, INCLUDING the
        seed boost (see effective_margin)
      - "score": the single representative scoreline, oriented so the
        FIRST number in each set is the predicted winner's games. Its total
        game margin EXACTLY equals exp_margin rounded to whole games (7-5
        counts as 1.5, 7-6 as 1, everything else is the plain game
        difference), e.g. a 4.6-game favorite prints a +5 line like 6-3 6-4
        or 6-4 6-3. The shape (straight sets vs three) is picked with the
        data-calibrated odds (THREE_SET_SCALE); then one line is DRAWN,
        weighted by likelihood, from the lines that hit the margin, with a
        seed built from the two players (see SCORE_SEED), so scores vary
        between matchups but are identical on every rebuild.
      - "prob_three_sets" / "prob_tiebreak" / "prob_75": chance the match
        goes to a 3rd set / contains a 7-6 set / contains a 7-5 set.

    Match-shape odds use the full form-noise mixture, scaled by the
    fitted shape multipliers; the exact scoreline text is read from the
    table entry nearest the seed-adjusted rating gap.
    """
    # Seed-adjusted gap: the win probability includes the small seed boost for
    # the higher seed, so the margin the scoreline must reflect does too.
    d = effective_margin(a, b)
    pw2, pw3, pl2, pl3, p_tb, p_75 = _mixture_stats(d, _tau(a, b))

    w2, w3 = (pw2, pw3) if winner_is_a else (pl2, pl3)

    # Shape weights: same straight-sets vs three-sets odds as before (the
    # calibrated THREE_SET_SCALE moves the over-predicted three-set mass back
    # onto straight sets).
    shape_w = {2: w2 + (1.0 - THREE_SET_SCALE) * w3, 3: THREE_SET_SCALE * w3}

    tab = _table()
    idx = min(int(round(min(abs(d), POWER_CAP) / TABLE_STEP)), len(tab) - 1)
    winner_is_favorite = (winner_is_a == (d >= 0))
    src = tab[idx]["a_wins" if winner_is_favorite else "b_wins"]

    # TARGET MARGIN: the printed line's total game margin must EQUAL the
    # favorite's expected margin rounded to a whole number of games (4.6 ->
    # 5, so 6-3 6-4 or 6-4 6-3). Margin counts 7-5 as 1.5 and 7-6 as 1;
    # every other set is its plain game difference (see _set_margin). A
    # winner is always at least 1 game ahead. (In the bracket the winner is
    # always the favorite; an underdog winner shows a 1-game win.)
    target = max(1, math.floor(abs(d) + 0.5)) if winner_is_favorite else 1
    seed = _matchup_seed(a, b, winner_is_a)

    # Every candidate line in each shape, weighted by how likely it is
    # (damped for the over-predicted 7-6 / 7-5 sets) and tagged with how far
    # its margin is from the target.
    by_shape: dict[int, list[tuple[list, float, float]]] = {}
    for shp in (2, 3):
        cands = []
        for sets, p in src[shp]:
            damp = 1.0
            for x in sets:
                if x in ("7-6", "6-7"):
                    damp *= TIEBREAK_SCALE
                elif x in ("7-5", "5-7"):
                    damp *= SEVEN_FIVE_SCALE
            cands.append((sets, p * damp, abs(_line_margin(sets) - target)))
        by_shape[shp] = cands

    # Step 1 -- pick the SHAPE (straight sets vs three sets) with the
    # data-calibrated odds, so the share of printed three-setters matches
    # real matches (see THREE_SET_SCALE). If the preferred shape has no line
    # that hits the target margin exactly (a straight-set win can't be +1; a
    # three-setter can't be +11), fall to the other shape.
    p3 = shape_w[3] / (shape_w[2] + shape_w[3]) if (shape_w[2] + shape_w[3]) > 0.0 else 0.0
    preferred = 3 if _uniform_from_seed((seed + 0x1B873593) & 0xFFFFFFFF) < p3 else 2
    order = (preferred, 5 - preferred)

    pool: list[tuple[list, float]] = []
    for shp in order:
        pool = [(sets, w) for sets, w, miss in by_shape[shp] if miss < 1e-9]
        if pool:
            break
    if not pool:
        # No line reaches the target at all (extreme tails): take the
        # closest margin available.
        best_miss = min((miss for shp in order for _, _, miss in by_shape[shp]), default=None)
        if best_miss is not None:
            for shp in order:
                pool = [(sets, w) for sets, w, miss in by_shape[shp] if miss <= best_miss + 1e-9]
                if pool:
                    break

    # Step 2 -- replicable weighted draw among the lines that hit the target
    # (e.g. 6-3 6-4 vs 6-2 6-3 for a 5-game margin): different matchups with
    # the same margin get different scorelines, the same matchup always gets
    # the same one.
    best = None
    total_w = sum(w for _, w in pool)
    if total_w > 0.0:
        r = _uniform_from_seed(seed) * total_w
        acc = 0.0
        for sets, w in pool:
            acc += w
            best = sets
            if r < acc:
                break
    score = list(best) if best else ["6-4", "6-4"]
    if len(score) == 3:
        # The line is stored as (set lost, won, won) with no order; deal it
        # out in a real sequence. The winner loses set 1 or set 2 (real data:
        # 51% / 49%) and ALWAYS wins the last set, and the two sets they won
        # can come in either order. Seeded, so it is repeatable.
        lost, w1, w2 = score
        lost_first = _uniform_from_seed((seed + 0x9E3779B9) & 0xFFFFFFFF) < 0.5
        swap = _uniform_from_seed((seed + 0x3C6EF372) & 0xFFFFFFFF) < 0.5
        if swap:
            w1, w2 = w2, w1
        score = [lost, w1, w2] if lost_first else [w1, lost, w2]
    elif len(score) == 2:
        # Straight sets are stored bigger-win-first; the order of the two
        # sets doesn't change the margin, so deal either order (6-3 6-4 or
        # 6-4 6-3), seeded so it is repeatable.
        if _uniform_from_seed((seed + 0x2545F491) & 0xFFFFFFFF) < 0.5:
            score = [score[1], score[0]]

    return {
        "score": score,
        "exp_margin": abs(d),
        "prob_three_sets": THREE_SET_SCALE * (pw3 + pl3),
        "prob_tiebreak": TIEBREAK_SCALE * p_tb,
        "prob_75": SEVEN_FIVE_SCALE * p_75,
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
    Every number below comes from one place: each player's <b>power
    rating</b>. The difference between two ratings is how many games the
    better player is expected to win by (12 = 6-0 6-0). That gap is turned
    into exact point-, game-, set- and match-level probabilities, then
    averaged over day-to-day form so upsets stay possible. Championship /
    final / semifinal odds are computed in closed form over the real seeded
    32-draw bracket (#1 and #2 can only meet in the final, etc.) -- no
    simulation or randomness anywhere, so re-running reproduces every
    number. The bracket path is the single most-likely outcome: the
    higher-rated side (after the small seed boost) always advances, and each
    printed scoreline's game margin equals the "Fav. By" number rounded to
    whole games (a 7-5 set counts as 1.5 games, a 7-6 set as 1). Form noise, the small
    seed-history blend and the match-shape odds were all fit to a
    walk-forward backtest of held-out matches.
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
