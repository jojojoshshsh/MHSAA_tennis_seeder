"""
predict_state.py
=================

Turns the per-(division, flight) rankings produced by mhsaa_seeding_v2.py
into a projected 32-draw state tournament: real single-elimination
brackets seeded 1-vs-32/16-vs-17/etc., exact win/finalist/semifinalist
probabilities computed from TrueSkill ratings (a closed recursive
calculation over the bracket tree -- no randomness involved in the
probabilities themselves), a single "most likely" run through every
bracket with realistic set-by-set scorelines, and a team point
projection (+1 team point per predicted match win) per gender/division.

This version replaces the old hash/ladder-based scoreline generator with
the same "hyper-realistic" Monte Carlo scoreline engine that
predict_matchup.py uses for one-off head-to-head predictions (dominance
proxy from wins/losses + sos + TGRS_scaled, simulated game-by-game with
Gaussian noise). See "SCORELINE ENGINE" below for details. It also no
longer touches docs/index.html -- instead it writes a brand-new,
self-contained HTML report.

Run this AFTER mhsaa_seeding_v2.py and build_site.py:

    python scripts/mhsaa_seeding_v2.py <matches.csv>
    python scripts/build_site.py          # optional, unrelated to this script now
    python scripts/predict_state.py

WHERE THE WIN PROBABILITIES COME FROM
---------------------------------------
Every ranking CSV row already carries ts_mu / ts_sigma -- the two
Gaussian parameters underlying ts_rating (= ts_mu - 3*ts_sigma).
Head-to-head win probability starts from the standard TrueSkill win
formula (identical to _match_probability() in trueskill_engine_v2.py,
and identical to win_probability() in predict_matchup.py):

    P(a beats b) = Phi( (mu_a - mu_b) / sqrt(2*BETA^2 + sigma_a^2 + sigma_b^2) )

That TrueSkill number is then blended with a SEED-HISTORY PRIOR: real
MHSAA seed-committee data (see the comment above SEED_PRIOR_ACCURACY,
section 1) showing the higher seed has won ~96.0% of matches across 19
years of tournament results. The blend happens in logit space via
_apply_seed_prior() so a lopsided TrueSkill gap can't be overwhelmed by a
prior calibrated mainly for close calls, and is fully tunable/disableable
via SEED_BLEND_WEIGHT. This means win probabilities are no longer *pure*
TrueSkill -- they're TrueSkill informed by real historical seed
reliability -- but the blend degrades gracefully back to pure TrueSkill
when SEED_BLEND_WEIGHT = 0, or when either player has no seed number.

This match_win_prob() -- TrueSkill blended with the seed prior -- is
used AS-IS, unmodified, for every bracket-advancement calculation in
this file (both the exact recursive probabilities in section 3 and the
single-path bracket walk in section 5). It is only reshaped for one
purpose: deriving the performance gap that flavors a match's *simulated
scoreline* -- see SCORELINE ENGINE below.

Given every pairwise probability, the probability of a seed reaching any
given round of a KNOWN bracket is NOT just "win N coin flips" -- it also
depends on who they'd face, who *that* opponent would have had to have
beaten to get there, and so on recursively. This module computes that
exactly with a standard bracket-probability recursion (split the bracket
in half, recursively get each half's "who emerges" distribution, then
convolve the two distributions through the head-to-head formula) rather
than approximating it by simulating many random brackets. No randomness
is used anywhere for these bracket-advancement probabilities -- every
number in the "Championship / Final / Semifinal Odds" tables is a
deterministic function of the input ratings (and, per above, the seed
prior).

SCORELINE ENGINE (now shared with predict_matchup.py)
--------------------------------------------------------
For the single "most likely path" bracket, the winner of each real match
is still simply whichever side has p >= 0.5 (the favorite always
advances -- no coin flips, so the bracket-path table stays consistent
with the exact probabilities above).

The *scoreline* attached to that match, however, now comes straight from
predict_matchup.py's engine instead of the old deterministic
hash-and-ladder generator:

  - A "dominance profile" is built per player from win_pct, sos, and
    TGRS_scaled -- an explicit PROXY (see _dominance_proxy() below,
    copied verbatim from predict_matchup.py), since true per-player
    dominance is never persisted per player (compute_match_margin() is a
    per-MATCH quantity, never aggregated and saved).
  - A performance gap is formed from an EFFECTIVE mu gap plus a weighted
    dominance-proxy gap (see _perf_diff_for_match() below). The
    effective mu gap is derived from match_win_prob(a, b) -- the SAME
    blended (TrueSkill + seed-prior) probability that decides who
    advances in the bracket -- rather than the raw, unblended
    mu_a - mu_b. This keeps the scoreline simulator's notion of "who's
    favored" identical to the bracket's notion of "who's favored"; see
    the docstring on _perf_diff_for_match() for why that matters (it
    fixes a real bug where close, seed-prior-flipped matchups could
    print a winner alongside a scoreline showing them losing 0-6, 0-6).
  - Before that probability is converted into a performance gap, it is
    first re-centered with a linear shift-and-scale toward 0.5 -- see
    WIN_PROB_SCORELINE_SCALE and _compress_win_prob() below. This is
    NOT a cap or clamp on the performance gap itself: it is a rescaling
    of the underlying probability distribution (which is naturally
    bounded to [0, 1], unlike the performance gap, which is unbounded).
    Shrinking the *probability* toward a coin flip before the
    unbounded-domain conversion (mu-gap via the inverse normal CDF)
    means the resulting performance gap is automatically bounded too,
    without ever touching or post-processing the gap value itself. See
    _compress_win_prob()'s docstring for the full rationale and the
    empirical bagel-rate problem this fixes.
  - Individual sets are simulated game-by-game via a logistic function
    of that (rescaled) performance gap plus fixed Gaussian noise
    (_simulate_set(), copied verbatim from predict_matchup.py), through
    a full best-of-3 match.
  - Because this uses real randomness (random.Random), and this script
    must be reproducible on re-run, each match's simulation is seeded
    deterministically from a hash of the two competitors' names and
    TrueSkill mu/sigma (same idea predict_matchup.py exposes via its
    `--seed` flag, applied per-match here). Several hundred trials are
    simulated per match; among the trials whose simulated outcome agrees
    with the TrueSkill favorite (so the printed scoreline never
    contradicts the printed winner), the single most common exact
    scoreline is what gets displayed -- the same "most common outcome
    across many simulated trials" logic predict_matchup.py's predict()
    uses for its own scoreline-distribution table.
  - Volatility is intentionally NOT used anywhere in this engine -- same
    as predict_matchup.py, it's neither persisted per-player in the
    ranking CSVs nor factored into the win-probability or
    scoreline-simulation math.

TEAM POINTS
-----------
Every real (non-bye) match in the single deterministic bracket run
awards +1 predicted point to the winner's school. Points are summed
across every flight and match_type (singles + doubles, flights 1-4)
within a (gender, division), matching how the real MHSAA team scoring
aggregates individual results into a team result.

OUTPUT
------
  - docs/csv/predictions/bracket_*.csv           (per-bracket seed odds)
  - docs/csv/predictions/team_predicted_*.csv    (per-division team points)
  - docs/prediction_of_state.html                (standalone report --
    this script no longer edits docs/index.html at all, but it does
    link BACK to docs/index.html so the page isn't a dead end)
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from trueskill_engine_v2 import BETA  # noqa: E402  (path insert above)
except ImportError:
    # Fallback so this script can still run standalone if trueskill_engine_v2
    # isn't importable for some reason -- matches its own default derivation,
    # and matches predict_matchup.py's own fallback.
    _MU, _SIGMA = 25.0, 25.0 / 3
    BETA = _SIGMA / 5

SRC_DIR = REPO_ROOT / "src" / "rankings_by_division_flight"
DOCS_DIR = REPO_ROOT / "docs"
PRED_CSV_DIR = DOCS_DIR / "csv" / "predictions"
PRED_HTML_PATH = DOCS_DIR / "prediction_of_state.html"

MAX_BRACKET = 32
VALID_FLIGHTS = {"1", "2", "3", "4"}
SCORELINE_TRIALS = 400  # Monte Carlo trials per bracket match, for the printed scoreline only

# Linear shift-and-scale applied to match_win_prob(a, b) -- NOT to the
# performance gap -- before that probability is converted into a
# performance gap for the scoreline simulator only. See
# _compress_win_prob() for the mechanism and rationale, and the
# SCORELINE ENGINE section of the module docstring for the empirical
# problem this fixes (double-bagel scorelines being wildly over-frequent
# for lopsided seed gaps). 1.0 = no rescaling (the probability is used
# as-is); 0.0 = every matchup's scoreline is simulated as a coin flip
# regardless of how lopsided the real matchup is. Empirically tuned (see
# the repo's calibration notes / git history for the check script used)
# so that a genuine blowout (99%+ win probability) tops out around a
# ~25% chance of a single 6-0 set and a ~6% chance of a full
# double-bagel match, rather than ~90%/~85%, while leaving close-to-
# moderate matchups (which were never the problem) close to untouched.
WIN_PROB_SCORELINE_SCALE = 0.5

FINISH_LABELS = {
    1: "Champion",
    2: "Runner-up",
    4: "Semifinalist",
    8: "Quarterfinalist",
    16: "Round of 16",
    32: "Round of 32",
}

BYE = object()  # sentinel for an empty bracket slot (small fields get first-round byes)


# ============================================================================
# 1.  TrueSkill head-to-head probability (identical math to predict_matchup.py)
# ============================================================================

def _Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def head_to_head_probability(mu_a: float, sigma_a: float, mu_b: float, sigma_b: float) -> float:
    """P(a beats b), the standard TrueSkill win-probability formula (eps=0).
    Same formula as win_probability() in predict_matchup.py."""
    c = math.sqrt(2.0 * BETA ** 2 + sigma_a ** 2 + sigma_b ** 2)
    if c <= 0:
        return 0.5
    return _Phi((mu_a - mu_b) / c)


def _player_mu_sigma(row: dict) -> tuple[float, float]:
    try:
        mu = float(row.get("ts_mu") or 25.0)
    except (TypeError, ValueError):
        mu = 25.0
    try:
        sigma = float(row.get("ts_sigma") or (25.0 / 3))
    except (TypeError, ValueError):
        sigma = 25.0 / 3
    return mu, sigma


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


def match_win_prob(a, b) -> float:
    """
    P(a beats b), with BYE handling (a real player always beats a bye),
    blended with the seed-history prior below (see
    SEED_PRIOR_ACCURACY / SEED_BLEND_WEIGHT). Set SEED_BLEND_WEIGHT = 0.0
    to fall back to pure TrueSkill, matching the original behavior.

    This is the number used for every bracket-advancement decision in
    this file (sections 3 and 5) and is returned/used completely
    unmodified there. It is only reshaped -- via _compress_win_prob(),
    downstream in _perf_diff_for_match() -- for the separate purpose of
    flavoring a simulated scoreline; it is never itself rescaled.
    """
    if a is BYE and b is BYE:
        return 0.5
    if a is BYE:
        return 0.0
    if b is BYE:
        return 1.0
    mu_a, sigma_a = _player_mu_sigma(a)
    mu_b, sigma_b = _player_mu_sigma(b)
    p_ts = head_to_head_probability(mu_a, sigma_a, mu_b, sigma_b)
    return _apply_seed_prior(p_ts, a, b)


# ----------------------------------------------------------------------------
# Seed-history prior
# ----------------------------------------------------------------------------
#
# Source: MHSAA Fall 2025 Teusink Seed Committee Report (compiled by Gary
# Ellis, Allegan). Its "Champions by Seed Number" table gives a per-year
# "Total Accuracy" figure -- the match-level rate, across every round of
# every flight/division, at which the better-seeded player actually won.
# Averaged over the 19 available years (2007 Spring/Fall through 2025,
# excluding the COVID-modified 2020 season):
#
#   2025 96.1  2024 95.3  2023 96.7  2022 97.7  2021 94.5  2019 97.6
#   2018 99.2  2017 98.4  2016 99.2  2015 97.6  2014 94.0  2013 96.1
#   2012 93.7  2011 95.3  2010 94.5  2009 94.5  2008 94.5
#   2007(Fall) 95.3  2007(Spring) 94.5
#   -> 19-year average = 96.0%
#
# That figure lines up closely with the Fall 2025 report's own semifinal
# breakdown (seeded players reaching the semis: 30/32, 31/32, 32/32, 30/32
# across the four divisions = 123/128 = 96.1%), which is a good sign it's
# a genuine match-level base rate rather than a champion-only statistic.
#
# CAVEAT: that 96% blends together every round of every bracket --
# including early rounds (e.g. #1 seed vs #32 seed) where TrueSkill alone
# would already call the outcome correctly almost every time. It does NOT
# tell us, on its own, how often seeds hold in a genuinely close matchup
# (e.g. a #1-vs-#2 final). Applying the full 96% prior uniformly, on top
# of what TrueSkill already captures, would double-count that signal for
# lopsided matchups. So this uses a BLEND WEIGHT (SEED_BLEND_WEIGHT) below
# 1.0: the seed prior pulls a close TrueSkill call meaningfully toward the
# higher seed, while a matchup TrueSkill already considers lopsided is
# barely moved (the prior can't push probability past what a full 96%
# read would imply, since it's blended, not substituted).
#
# HOW TO RE-TUNE: if the seed committee publishes gap-specific data later
# (e.g. accuracy broken out by how many seed positions apart the two
# players are), the flat SEED_PRIOR_ACCURACY below could be replaced with
# a lookup keyed on abs(seed_a - seed_b) for a more precise prior. Until
# then this flat rate + moderate blend weight is the most defensible
# reading of what's actually been reported.

SEED_PRIOR_ACCURACY = 0.960   # 19-year average "higher seed wins" rate, see note above
SEED_BLEND_WEIGHT = 0.35      # 0.0 = pure TrueSkill (old behavior); 1.0 = pure seed prior
_EPS = 1e-9
_STD_NORMAL = NormalDist()    # standard normal CDF/quantile, used by _implied_mu_gap() below


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


_SEED_PRIOR_LOGIT = _logit(SEED_PRIOR_ACCURACY)  # ~= 3.178 at 96.0%


def _seed_number(row: dict) -> int | None:
    try:
        return int(row.get("rank"))
    except (TypeError, ValueError):
        return None


def _apply_seed_prior(p_ts: float, a: dict, b: dict) -> float:
    """
    Blends the pure-TrueSkill probability `p_ts` (P(a beats b)) with the
    seed-history prior above, in logit space, so the result always stays
    a legal probability and a big TrueSkill gap isn't overwhelmed by a
    prior meant to matter most for close calls. Returns p_ts unchanged if
    either player is missing a seed number or the two are tied on seed
    (nothing to prefer).
    """
    if SEED_BLEND_WEIGHT <= 0.0:
        return p_ts
    seed_a, seed_b = _seed_number(a), _seed_number(b)
    if seed_a is None or seed_b is None or seed_a == seed_b:
        return p_ts

    seed_logit = _SEED_PRIOR_LOGIT if seed_a < seed_b else -_SEED_PRIOR_LOGIT
    blended_logit = (1.0 - SEED_BLEND_WEIGHT) * _logit(p_ts) + SEED_BLEND_WEIGHT * seed_logit
    return _sigmoid(blended_logit)


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
# 4.  Scoreline engine -- ported directly from predict_matchup.py
# ============================================================================
#
# Everything in this section mirrors predict_matchup.py's simulation code
# (_dominance_proxy, _simulate_set, _simulate_match) as closely as
# possible, operating on the same raw CSV row dicts predict_state.py
# already carries around. The additions here are (a) deterministic
# per-match seeding (predict_matchup.py exposes --seed for the whole CLI
# run; here each bracket match gets its own derived seed so re-running
# this script always reproduces the same bracket), (b) deriving the
# simulator's performance gap from the SAME blended win probability that
# decides bracket advancement, via _perf_diff_for_match() /
# _implied_mu_gap() below -- see those functions for why -- and (c)
# reshaping that probability first (_compress_win_prob()) so extreme
# seed mismatches don't turn every game of a set into a near-certainty
# for the favorite -- see the docstring on _compress_win_prob().

def _player_name(row: dict) -> str:
    return row.get("name") or row.get("pair_name") or "Unknown"


def _implied_mu_gap(p: float, c: float) -> float:
    """
    Inverts the TrueSkill win-probability formula
    P = Phi((mu_a - mu_b) / c) to recover an "effective mu gap" that is
    consistent with a given probability `p` -- which, via
    match_win_prob(), may already be blended with the seed-history prior
    (see _apply_seed_prior()), not just raw TrueSkill, and which the
    caller (_perf_diff_for_match()) may have additionally rescaled via
    _compress_win_prob() before this function ever sees it.

    Phi is the standard normal CDF (see _Phi() above), so its inverse is
    the standard normal quantile function, i.e. statistics.NormalDist's
    inv_cdf. When `p` is the raw, unblended, unrescaled p_ts for a pair,
    this returns mu_a - mu_b back out again (up to floating point), so
    behavior is unchanged whenever SEED_BLEND_WEIGHT = 0.0 and
    WIN_PROB_SCORELINE_SCALE = 1.0.
    """
    p = min(max(p, _EPS), 1.0 - _EPS)
    return c * _STD_NORMAL.inv_cdf(p)


def _compress_win_prob(p: float, scale: float = WIN_PROB_SCORELINE_SCALE) -> float:
    """
    Linearly shifts and scales the win probability `p` toward 0.5:

        p_scoreline = 0.5 + scale * (p - 0.5)

    This is deliberately a rescaling of the *probability distribution*
    (which lives on the bounded interval [0, 1]) rather than a cap or
    clamp applied to the performance gap the probability eventually
    becomes (which lives on an unbounded scale). Because probability is
    bounded, a plain linear rescale toward its center is automatically
    bounded too -- e.g. with scale=0.5, no matter how close the real
    match_win_prob() gets to 1.0, p_scoreline can never exceed 0.75 -- so
    nothing downstream needs an explicit cap or saturation function; the
    boundedness falls out naturally from doing the rescaling in
    probability space instead of gap space.

    WHY THIS IS NEEDED: match_win_prob() already blends TrueSkill with a
    seed-history prior that's been shown to be ~96% accurate (see
    SEED_PRIOR_ACCURACY), which routinely pushes very lopsided seed
    matchups (e.g. a #1 vs a #32) well past 99%. Converting a probability
    that close to 1 straight into a performance gap via
    _implied_mu_gap() (using the inverse normal CDF, which diverges
    toward +/-infinity as p approaches 0 or 1) produces enormous gaps.
    Fed into the logistic per-game formula in _simulate_set(), an
    enormous gap makes the favorite nearly unbeatable in every single
    game, so winning six straight games (a 6-0 set) stops being a rare
    event -- empirically, a genuine blowout matchup was printing a
    double-bagel (6-0, 6-0) something like a third of the time, which is
    far more lopsided than real high school tennis actually looks even
    in a clear mismatch (there's almost always at least one hard-fought
    game -- a service letdown, a nerves-driven game from the leader,
    etc.).

    Rescaling the probability first means the *scoreline simulator*
    never has to reason about a matchup as more lopsided than roughly a
    3-to-1 (75/25) proposition per set, no matter how one-sided the real
    seeding/rating gap is -- while match_win_prob() itself, and every
    bracket-advancement number derived from it in sections 3 and 5,
    stays completely untouched.

    scale=1.0 disables this entirely (p_scoreline == p); scale=0.0
    collapses every matchup to a coin flip for scoreline purposes only.
    """
    p = min(max(p, 0.0), 1.0)
    return 0.5 + scale * (p - 0.5)


def _dominance_proxy(row: dict) -> float:
    """
    NOT real per-match dominance data -- true per-player dominance is
    never persisted (see module docstring). This is the same explicit
    stand-in predict_matchup.py uses, built from three numbers that ARE
    persisted: win percentage, strength of schedule, and scaled TGRS.
    Used only to flavor the simulated scoreline, never the
    win-probability number, which comes from match_win_prob() (TrueSkill
    blended with the seed-history prior -- see _apply_seed_prior()), not
    from this dominance proxy.
    """
    wins = _to_int(row, "wins")
    losses = _to_int(row, "losses")
    matches = wins + losses
    win_pct = wins / matches if matches else 0.5
    sos = _to_float(row, "sos")
    sos_norm = min(max(sos / 50.0, 0.0), 1.0) if sos else 0.5
    tgrs_scaled = _to_float(row, "TGRS_scaled", default=50.0)
    tgrs_norm = min(max(tgrs_scaled / 100.0, 0.0), 1.0)
    return max(0.0, min(1.0, 0.5 * win_pct + 0.25 * sos_norm + 0.25 * tgrs_norm))


def _perf_diff_for_match(a: dict, b: dict) -> float:
    """
    The performance gap fed into the scoreline simulator (positive
    favors `a`): an effective mu-gap consistent with a RESCALED version
    of match_win_prob(a, b) -- see _compress_win_prob() -- plus the same
    dominance-proxy nudge predict_matchup.py's _simulate_match() uses.

    This used to be plain (mu_a - mu_b) + 8*(dom_a - dom_b), derived only
    from raw TrueSkill. That was fine for predict_matchup.py, which has
    no seed prior, but broke down once predict_state.py started blending
    in the seed-history prior for bracket-advancement decisions: for a
    close matchup (~50-57% win probability) where the prior flips or
    reinforces a near-tied TrueSkill call, the raw mu gap can still point
    the OTHER way. In that case nearly every one of the 400 simulated
    trials in predict_match_details() disagrees with the declared
    winner, so the "trials that agree with the favorite" filter is left
    hunting through a handful of rare, extreme upset trials -- and the
    most common scoreline among THOSE is a blowout in the *loser's*
    favor, printed next to the *winner's* name (e.g. a 51% favorite shown
    "winning" 0-6, 0-6).

    Deriving the effective mu gap from the same probability that decided
    the winner keeps the simulator's notion of "who's favored" identical
    to the bracket's, so trials naturally land on the declared winner's
    side most of the time. Before that conversion happens, though, the
    probability is passed through _compress_win_prob() -- a linear
    shift-and-scale toward 0.5 -- which is what keeps very lopsided seed
    gaps from producing an enormous, near-certain-every-game performance
    gap (see _compress_win_prob()'s docstring for the full story). This
    whole function reduces to the original, pre-fix behavior whenever
    SEED_BLEND_WEIGHT = 0 and WIN_PROB_SCORELINE_SCALE = 1.0.
    """
    sigma_a = _player_mu_sigma(a)[1]
    sigma_b = _player_mu_sigma(b)[1]
    mu_a = _player_mu_sigma(a)[0]
    mu_b = _player_mu_sigma(b)[0]
    p = match_win_prob(a, b)
    p_scoreline = _compress_win_prob(p)
    c = math.sqrt(2.0 * BETA ** 2 + sigma_a ** 2 + sigma_b ** 2)
    mu_gap_eff = _implied_mu_gap(p_scoreline, c) if c > 0 else (mu_a - mu_b)
    dom_a, dom_b = _dominance_proxy(a), _dominance_proxy(b)
    return mu_gap_eff + 8.0 * (dom_a - dom_b)


def _simulate_set(perf_diff: float, game_noise: float, rng: random.Random) -> tuple[int, int]:
    """
    One simulated set, identical logic to predict_matchup.py's
    _simulate_set(). perf_diff > 0 favors player A (expected to already
    be derived from a rescaled probability -- see
    _perf_diff_for_match()/_compress_win_prob()). Games are drawn one at
    a time from a logistic function of the per-game performance gap
    (plus fixed Gaussian noise) until someone reaches 6 with a 2-game
    lead, or a 7-6/7-5 breaker outcome.
    """
    games_a = games_b = 0
    while True:
        noise = rng.gauss(0.0, game_noise)
        p_a_game = 1.0 / (1.0 + math.exp(-(perf_diff + noise) / 6.0))
        if rng.random() < p_a_game:
            games_a += 1
        else:
            games_b += 1

        if games_a >= 6 and games_a - games_b >= 2:
            return games_a, games_b
        if games_b >= 6 and games_b - games_a >= 2:
            return games_a, games_b
        if games_a == 7 or games_b == 7:
            return games_a, games_b
        if games_a == 6 and games_b == 6:
            if rng.random() < 1.0 / (1.0 + math.exp(-perf_diff / 4.0)):
                return 7, 6
            return 6, 7


def _simulate_match_once(a: dict, b: dict, rng: random.Random,
                          perf_diff: float) -> tuple[bool, list[str]]:
    """
    One simulated best-of-3 match between two REAL players (never a
    BYE), identical mechanics to predict_matchup.py's _simulate_match(),
    except it also reports which side (a or b) won, and the set scores
    are always returned from `a`'s perspective ("a_games-b_games").

    `perf_diff` (positive favors a) is computed once per matchup by
    _perf_diff_for_match() -- see that function's docstring for why it's
    no longer derived from the raw mu_a - mu_b gap, and why the
    probability behind it is rescaled toward 0.5 first.
    """
    game_noise = 8.0  # fixed within-set randomness; no volatility input

    sets: list[str] = []
    a_sets = b_sets = 0
    while a_sets < 2 and b_sets < 2:
        ga, gb = _simulate_set(perf_diff, game_noise, rng)
        sets.append(f"{ga}-{gb}")
        if ga > gb:
            a_sets += 1
        else:
            b_sets += 1
    return a_sets > b_sets, sets


def flip_score(s: str) -> str:
    """Reverse a set score string to the other side's perspective, e.g.
    '6-4' -> '4-6'. 'BYE' passes through unchanged."""
    if s == "BYE":
        return s
    x, y = s.split("-")
    return f"{y}-{x}"


def _match_seed_int(a, b) -> int:
    """
    Deterministic seed for a single bracket match's Monte Carlo run,
    derived from both competitors' names + TrueSkill mu/sigma (order
    independent), so re-running this script always reproduces the same
    scoreline for the same matchup -- there is no time/OS-entropy
    randomness anywhere in this file.
    """
    def _sig(p):
        if p is BYE:
            return "BYE"
        mu, sigma = _player_mu_sigma(p)
        return f"{_player_name(p)}|{mu:.6f}|{sigma:.6f}"

    raw = "||".join(sorted([_sig(a), _sig(b)]))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def predict_match_details(a: dict, b: dict, winner_is_a: bool,
                           trials: int = SCORELINE_TRIALS) -> dict:
    """
    Runs predict_matchup.py's Monte Carlo engine `trials` times for this
    matchup (deterministically seeded, see _match_seed_int; performance
    gap held fixed per matchup, see _perf_diff_for_match) and returns:

      - "score": the single most common exact scoreline among the trials
        that agree with the TrueSkill/seed-prior favorite (`winner_is_a`)
        -- the same "most common simulated outcome" logic
        predict_matchup.py's predict() uses to pick its headline
        scoreline. Oriented so the FIRST number in each set is always
        the eventual bracket winner's game count (e.g. a winner who
        dropped set two reads ["6-4", "4-6", "7-5"]).
      - "sim_seed": the integer seed (from _match_seed_int) that drove
        this matchup's random.Random instance -- re-running this script
        for the same two players always reproduces this exact seed and
        therefore this exact scoreline/stat set.
      - "prob_three_sets" / "prob_tiebreak" / "prob_75": match-shape odds
        across ALL `trials` simulated matches for this matchup (not just
        the ones matching the favorite) -- identical definitions to
        predict_matchup.py's prob_three_sets / prob_any_tiebreak /
        prob_any_75 ("Goes to a 3rd set", "Contains a 7-6 tiebreak",
        "Contains a 7-5 set").
    """
    seed = _match_seed_int(a, b)
    rng = random.Random(seed)
    perf_diff = _perf_diff_for_match(a, b)

    matching: Counter[tuple[str, ...]] = Counter()
    last_sets: list[str] = ["6-4", "6-4"]  # harmless fallback, overwritten below
    three_setters = tiebreak_sets = seven_five_sets = 0

    for _ in range(trials):
        a_won, sets = _simulate_match_once(a, b, rng, perf_diff)
        last_sets = sets

        if len(sets) == 3:
            three_setters += 1
        for s in sets:
            gw, gl = (int(x) for x in s.split("-"))
            if {gw, gl} == {7, 6}:
                tiebreak_sets += 1
            if {gw, gl} == {7, 5}:
                seven_five_sets += 1

        if a_won == winner_is_a:
            matching[tuple(sets)] += 1

    if matching:
        chosen = list(matching.most_common(1)[0][0])
    else:
        # Vanishingly rare (would require a huge upset never once landing
        # on the favorite's side across `trials` runs) -- fall back to the
        # last simulated result rather than leaving the match unscored.
        chosen = last_sets

    # chosen is always in `a`'s perspective; re-orient to the winner's
    # perspective if the winner is actually b.
    score = chosen if winner_is_a else [flip_score(s) for s in chosen]

    return {
        "score": score,
        "sim_seed": seed,
        "prob_three_sets": three_setters / trials,
        "prob_tiebreak": tiebreak_sets / trials,
        "prob_75": seven_five_sets / trials,
    }


# ============================================================================
# 5.  Deterministic single-path bracket simulation
# ============================================================================

def simulate_bracket(players: list) -> list[list[dict]]:
    """
    One deterministic run through the whole bracket: in every match the
    favorite (p >= 0.5) always advances (no coin flips -- keeps this
    path consistent with the exact probabilities in section 3), and each
    match's scoreline comes from predict_match_scoreline() above. Bye
    matches are marked score=["BYE"]. `p` here is match_win_prob(a, b),
    used unmodified (see _compress_win_prob()'s docstring for why the
    scoreline generator alone sees a rescaled version of it instead).
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
                                 "score": ["BYE"], "p_fav": 1.0, "sim_seed": None,
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
                "sim_seed": details["sim_seed"],
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
                "sim_seed", "prob_three_sets_pct", "prob_7_6_tiebreak_pct",
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
                        m.get("sim_seed"),
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
            sim_seed = m.get("sim_seed")
            sim_seed_str = str(sim_seed) if sim_seed is not None else "--"
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
                f"<td>{sim_seed_str}</td>"
                f"<td>{p3_str}</td><td>{ptb_str}</td><td>{p75_str}</td></tr>"
            )
    return f"""
    <div class="pred-bracket">
      <h3>{_esc(label)}</h3>
      <table class="pred-table">
        <thead><tr>
          <th>Round</th><th>Matchup (seed #)</th><th>Predicted Winner</th>
          <th>Predicted Score</th><th>Win Prob.</th><th>Sim. Seed</th>
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
    Championship / final / semifinal odds below are computed once, in closed
    form, from each player's TrueSkill rating blended with a real
    seed-history prior (the higher seed has won ~96.0% of matches across 19
    years of MHSAA seed-committee data), and their position in a real seeded
    32-draw bracket (favorites can't meet early -- #1 and #2 can only meet
    in the final, etc.) -- no simulation or randomness is involved in those
    numbers, just a deterministic blend of the rating gap and that seed
    prior. The bracket path shown is the single most-likely outcome: the
    higher blended-probability side always advances, and each printed
    scoreline comes from the same Monte Carlo scoreline engine used by
    predict_matchup.py (flavored by a win-percentage/SOS/TGRS dominance
    proxy, simulated set-by-set, and seeded deterministically per matchup so
    re-running this report reproduces the same scorelines). Each matchup row
    in the bracket path below also lists both players' tournament seed
    numbers, the integer random seed that drove that matchup's simulation,
    and three match-shape odds computed across all simulated trials for that
    matchup: the chance it goes to a 3rd set, the chance it contains a 7-6
    tiebreak set, and the chance it contains a 7-5 set.
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
