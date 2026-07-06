# trueskill_engine_v2.py
#
# Margin-aware, volatility-aware TrueSkill for tennis
# (Herbrich et al., 2007 + margin extension + Glicko-2-inspired volatility).
# No external dependencies — uses only `math`, `csv`, `ast`, `collections`.
#
# WHY THIS EXISTS
# ----------------
# Plain win/loss TrueSkill treats every win as equally informative. But a
# 6-0 6-0 win and a 7-6 0-6 7-6 win are very different pieces of evidence
# about the true skill gap, even though the second one actually has a
# *negative* game differential (14 games to 18). Game differential is not
# a sufficient statistic in tennis because of the hierarchical
# points -> games -> sets -> match structure: once a set is won 6-0
# instead of 6-4, the extra games don't help you win the match, they just
# inflate a naive differential.
#
# So this module scores each match's *set-by-set* scoreline for
# "dominance" instead of raw game differential, and feeds that dominance
# into TrueSkill as a margin threshold (epsilon), reusing the same
# truncated-Gaussian factor-graph math that the original paper uses for
# draws. Setting eps=0 for every match exactly recovers plain TrueSkill,
# so this is a strict generalization, not a different algorithm.
#
# On top of that, this module tracks a per-entity VOLATILITY signal,
# inspired by Glicko-2. TrueSkill's sigma already shrinks with more
# games, but it shrinks the same way whether results have been steady or
# erratic relative to what the rating gap predicted. That's a blind spot
# for exactly the scenario that motivates this file: a team that only
# plays elite opposition and wins a handful of close, hard-fought matches
# against them (while losing others) should be recognized as carrying
# strong, high-value evidence — and should be able to move further and
# faster than a team racking up a long, low-variance streak against weak
# competition. Volatility is the lever that makes that possible; see the
# "volatility extension" section below for the full mechanism and why it
# is an adaptation of Glicko-2 rather than a literal port.
#
# Public API
# ----------
#   compute_trueskill(match_pairs)         -> {entity: Rating}   (unchanged, eps=0 always)
#   compute_trueskill_margin(match_triples) -> {entity: Rating}  (uses set scores)
#   compute_match_margin(set_score_str)     -> float in [0, 1]   (dominance score)
#   load_matches_from_csv(path)             -> list of (winner, loser, set_score, timestamp)
#   Rating.conservative                     -> mu - 3 * sigma (used for ranking)
#   Rating.volatility                       -> NEW: tracked erraticism, see below

import ast
import csv
import math
from collections import defaultdict
from dataclasses import dataclass

# ── Hyperparameters ────────────────────────────────────────────────────────────

MU = 25.0          # initial mean skill
SIGMA = MU / 3      # initial uncertainty  (~8.33)
BETA = SIGMA / 5     # performance noise /2   (~4.17)
TAU = SIGMA / 3     # dynamics factor /10     — keeps sigma from dying (~0.083)

# --- margin extension knobs -----------------------------------------------
# VOLATILITY_PENALTY: how much within-match set-to-set inconsistency
#   discounts the dominance score. A match like 7-6, 0-6, 7-6 has a
#   moderate mean set-margin but huge variance across sets — this should
#   pull it back down toward "just a regular win", not inflate it.
# MARGIN_SCALE: converts a [0,1] dominance score into an epsilon in the
#   same units as BETA (performance-noise units).
# MAX_EPS_FRACTION: caps epsilon as a fraction of BETA so a single very
#   lopsided scoreline can never push sigma negative or blow up mu.
VOLATILITY_PENALTY = 1.0
MARGIN_SCALE = 0.3
MAX_EPS_FRACTION = 0.3

# --- match-tiebreak (10-point breaker) handling -----------------------------
# Many leagues replace a third set with a single first-to-10-win-by-2 super
# tiebreak, recorded in the data as an ordinary-looking "set" token like
# "10-7". Its raw numbers are NOT games, so scoring it with the same
# (gw-gl)/(gw+gl) formula used for real sets distorts dominance: "10-7"
# evaluates to 0.176, nearly 2.3x more "dominant" than a real 7-6 set
# (0.077), purely because a breaker's point totals run higher than a set's
# game totals. A first-to-10 decider is, competitively, just a coin-flip
# tiebreak — the same role a 7-6 breaker plays inside a normal set — so it
# should always be scored as a bare, non-dominant win, independent of
# whether the actual breaker was tight (10-8) or lopsided (10-1).
MATCH_TIEBREAK_MIN_POINTS = 10   # a "set" token where either side reaches
                                  # this count is treated as a match tiebreak,
                                  # not a real set (real sets top out at 7-6).
MATCH_TIEBREAK_DOMINANCE = 1.0 / 13.0   # same dominance a bare 7-6 set gets

# --- volatility extension (Glicko-2-inspired) ------------------------------
# TrueSkill's sigma already shrinks as an entity accumulates games, but it
# shrinks the same way regardless of whether their results have tracked
# what the rating gap predicted or have been wildly erratic relative to
# it. Glicko-2 addresses an analogous blind spot with a per-player
# "volatility" parameter, solved via an iterative (Illinois-algorithm)
# root-find defined against Glicko-2's own logistic link function and its
# periodic (many-games-at-once) rating update.
#
# That machinery doesn't transplant literally into this module: TrueSkill
# is Gaussian (not logistic) and updates sequentially, one match at a
# time, not in periodic batches. So instead of porting Glicko-2's solver,
# we adapt its *idea*: track volatility as an exponentially-weighted
# moving average, in log-space (to keep it strictly positive), of how
# "surprising" each match's margin-threshold outcome was relative to the
# pre-match rating gap. That volatility then feeds back in as *extra*,
# temporary uncertainty at update time — analogous to Glicko-2's
# pre-period inflation phi* = sqrt(phi^2 + sigma^2) — so a volatile
# entity's sigma effectively widens right before the update is applied,
# letting that update move mu further than it otherwise would.
#
# Net effect on the motivating scenario: a team whose results closely
# track what their rating gap already predicts keeps low, decaying
# volatility and gets small, steady updates — even if they've played (and
# beaten) a lot of people. A team with erratic results relative to their
# rating gap — e.g. a handful of close, inconsistent wins earned only
# against elite opposition, mixed with losses to other elites — keeps
# high volatility, which widens their effective sigma and lets each
# subsequent result move their rating further. That's the lever that
# separates "under-rated because their sparse results are against brutal
# competition" from "accurately rated because they've beaten a lot of
# weak opponents in a row."
INITIAL_VOLATILITY = 0.06        # Glicko-2's own default starting volatility
MIN_VOLATILITY = 0.15
MAX_VOLATILITY = 0.50
VOLATILITY_LEARNING_RATE = 0.15  # how fast ln(volatility) reacts per match
VOLATILITY_SCALE = SIGMA         # converts dimensionless volatility into mu-units

# ── Normal-distribution helpers ───────────────────────────────────────────────

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _Phi(x: float) -> float:
    """Standard normal CDF via math.erfc for numerical stability at tails."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _v_margin(t: float, eps: float) -> float:
    """
    Generalized truncated-Gaussian mean factor for "won by more than eps".

    eps=0 reduces exactly to the plain win-case v(t) = phi(t) / Phi(t)
    from the original TrueSkill paper. eps>0 treats the observed win as
    evidence of a *larger* underlying performance gap than a bare win
    would imply, the same way the paper's draw-margin does for eps
    around zero, just shifted to one side.
    """
    denom = _Phi(t - eps)
    if denom < 1e-10:
        # deep in the tail: this margin was almost guaranteed given the
        # current rating gap, so the update carries little new information.
        return max(0.0, eps - t)
    return _phi(t - eps) / denom


def _w_margin(t: float, eps: float, v: float) -> float:
    """
    Generalized truncated-Gaussian variance factor for "won by more than eps".
    Clamps to [0, 1) so sigma never grows from a single update.
    """
    w = v * (v + t - eps)
    return min(max(w, 0.0), 1.0 - 1e-10)


# Backward-compatible aliases matching the original module's names/behavior.
def _v_win(t: float) -> float:
    return _v_margin(t, 0.0)


def _w_win(t: float, v: float) -> float:
    return _w_margin(t, 0.0, v)


def _match_probability(mu_a: float, mu_b: float, sigma_a: float, sigma_b: float, eps: float = 0.0) -> float:
    """
    Pre-match probability that entity a beats entity b by a margin of at
    least eps (mu-units), under the CURRENT (pre-update) ratings.

    This is exactly the denominator inside _v_margin, exposed standalone
    so it can be reused to score how surprising a realized match result
    was — the raw input to volatility tracking below.
    """
    c = math.sqrt(2.0 * BETA ** 2 + sigma_a ** 2 + sigma_b ** 2)
    t = (mu_a - mu_b) / c
    return _Phi(t - eps / c)


def _update_volatility(old_volatility: float, surprise: float) -> float:
    """
    EWMA update of volatility, done in log-space so it stays positive.

    `surprise` is 1 - P(observed outcome | pre-match ratings), in [0, 1]:
      * 0.0 -> the result was exactly what the rating gap already
        predicted (pure chalk): volatility decays toward normal.
      * 0.5 -> the neutral baseline of a genuine coin-flip match: no
        change (this is the "normal" amount of unpredictability, not
        evidence of erraticism).
      * 1.0 -> the result was the polar opposite of what the pre-match
        ratings predicted (a total upset): volatility grows.

    Clamped to [MIN_VOLATILITY, MAX_VOLATILITY] so a single freak result
    can't blow volatility up or collapse it to zero.
    """
    signal = surprise - 0.5
    new_ln_vol = math.log(old_volatility) + VOLATILITY_LEARNING_RATE * signal
    new_vol = math.exp(new_ln_vol)
    return min(max(new_vol, MIN_VOLATILITY), MAX_VOLATILITY)


# ── Rating dataclass ──────────────────────────────────────────────────────────

@dataclass
class Rating:
    mu: float = MU
    sigma: float = SIGMA
    volatility: float = INITIAL_VOLATILITY

    @property
    def conservative(self) -> float:
        """Lower-bound estimate used for ranking: mu − 3σ."""
        return self.mu - 3.0 * self.sigma

    def __repr__(self) -> str:
        return (
            f"Rating(mu={self.mu:.2f}, σ={self.sigma:.2f}, "
            f"ν={self.volatility:.3f}, cons={self.conservative:.2f})"
        )


# ── Core update (margin-aware + volatility-aware) ─────────────────────────────

def _update(r_win: Rating, r_lose: Rating, eps: float = 0.0) -> tuple[Rating, Rating]:
    """
    Apply one TrueSkill win/loss update, generalized with a margin
    threshold eps (performance-difference units) and a volatility signal
    inspired by Glicko-2.

    eps == 0.0  -> margin behavior identical to plain win/loss TrueSkill.
    eps  > 0.0  -> treats the result as "won by more than eps", which
                   produces a larger mu shift and larger sigma reduction
                   for the same rating gap, because a dominant scoreline
                   is more surprising (hence more informative) if the
                   players were actually close in skill.

    Step 0 — score how surprising this result was given the PRE-match
             ratings, and update each entity's volatility accordingly.
    Step 1 — add dynamics noise (TAU²) AND volatility-based inflation to
             both players' variance (Glicko-2's phi* step, adapted).
    Step 2 — compute the combined performance noise (c).
    Step 3 — compute v and w factors at the margin threshold eps.
    Step 4 — update mu and sigma for winner and loser.

    Returns new Rating objects (originals are not mutated).
    """
    # Step 0: volatility. Uses the pre-dynamics, pre-inflation sigmas —
    # i.e. exactly the uncertainty each entity carried in before this
    # match — mirroring Glicko-2's use of pre-period phi for its own
    # surprise term.
    e_win = _match_probability(r_win.mu, r_lose.mu, r_win.sigma, r_lose.sigma, eps)
    surprise = 1.0 - e_win
    vol_w_new = _update_volatility(r_win.volatility, surprise)
    vol_l_new = _update_volatility(r_lose.volatility, surprise)

    # Step 1: dynamics + volatility inflation. A player whose volatility
    # has climbed gets extra effective uncertainty stacked on top of
    # their tracked sigma for THIS update only — the boost isn't stored
    # directly, only however much of it survives the sigma-shrink in
    # Step 4 is carried forward, so volatility can't compound unboundedly.
    sw2 = r_win.sigma ** 2 + (VOLATILITY_SCALE * vol_w_new) ** 2 + TAU ** 2
    sl2 = r_lose.sigma ** 2 + (VOLATILITY_SCALE * vol_l_new) ** 2 + TAU ** 2

    # Step 2: combined noise
    c2 = 2.0 * BETA ** 2 + sw2 + sl2
    c = math.sqrt(c2)

    # Step 3: factors, evaluated at the margin threshold
    t = (r_win.mu - r_lose.mu) / c
    eps_std = eps / c  # eps is defined in mu-units; standardize like t
    v = _v_margin(t, eps_std)
    w = _w_margin(t, eps_std, v)

    # Step 4: updates
    mu_w_new = r_win.mu + (sw2 / c) * v
    mu_l_new = r_lose.mu - (sl2 / c) * v
    sigma_w_new = math.sqrt(sw2 * (1.0 - (sw2 / c2) * w))
    sigma_l_new = math.sqrt(sl2 * (1.0 - (sl2 / c2) * w))

    return (
        Rating(mu=mu_w_new, sigma=sigma_w_new, volatility=vol_w_new),
        Rating(mu=mu_l_new, sigma=sigma_l_new, volatility=vol_l_new),
    )


# ── Scoreline -> dominance/margin ─────────────────────────────────────────────

def _parse_set_score(set_score: str) -> list:
    """
    Parse a set-score string like "7-6 0-6 7-6" or "6-4(7) 7-6(3)" into
    a list of (winner_games, loser_games) tuples, one per set, from the
    match winner's perspective. Tiebreak point counts in parentheses are
    ignored (only games matter). Malformed tokens are skipped rather than
    raising, since real-world scraped data is messy (retirements, walkovers).
    """
    sets = []
    if not set_score:
        return sets
    for token in set_score.strip().split():
        token = token.split("(")[0]  # drop "(7)" tiebreak-point suffixes
        if "-" not in token:
            continue
        parts = token.split("-")
        if len(parts) != 2:
            continue
        try:
            gw, gl = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        sets.append((gw, gl))
    return sets


def compute_match_margin(set_score: str) -> float:
    """
    Turn a full scoreline into a single dominance score in [0, 1], used as
    the raw material for the margin epsilon.

    Design, matching the qualitative reasoning that game differential
    alone is misleading:
      * Each SET contributes one dominance sample (gw-gl)/(gw+gl),
        weighted equally regardless of how many games were in it. This
        stops a single bagel set (6-0) from being treated identically to
        a 12-game blowout, but also stops a 7-6 breaker from being
        treated as "close to 6-0" just because of raw game count.
      * The match-level score is the MEAN of the per-set dominance
        values, discounted by their standard deviation
        (VOLATILITY_PENALTY). This directly encodes "won 2 close sets
        and got bageled in between" as LOW-margin evidence (high
        variance cancels out a positive mean), rather than as
        "moderately dominant" — even though the winner clearly won.
      * The result is floored at 0. A win is never treated as *negative*
        evidence; in the worst case it's scored as a bare, non-dominant
        win (eps = 0), which is exactly the plain-TrueSkill behavior.

    Returns 0.0 for empty/unparseable scorelines (falls back to plain
    win/loss with no margin credit).
    """
    sets = _parse_set_score(set_score)
    doms = []
    for gw, gl in sets:
        total = gw + gl
        if total == 0:
            continue
        if max(gw, gl) >= MATCH_TIEBREAK_MIN_POINTS:
            # Match tiebreak (e.g. "10-7") — not a real set, so its raw
            # point margin isn't comparable to a set's game margin. Score
            # it exactly like a bare 7-6 set, no matter the actual points.
            doms.append(MATCH_TIEBREAK_DOMINANCE)
        else:
            doms.append((gw - gl) / total)

    if not doms:
        return 0.0

    mean_dom = sum(doms) / len(doms)
    if len(doms) > 1:
        variance = sum((d - mean_dom) ** 2 for d in doms) / len(doms)
        std_dom = math.sqrt(variance)
    else:
        std_dom = 0.0

    raw = mean_dom - VOLATILITY_PENALTY * std_dom
    return max(0.0, min(1.0, raw))


def _margin_to_eps(margin: float) -> float:
    """Map a [0,1] dominance score to an epsilon in mu-units, capped."""
    eps = margin * MARGIN_SCALE * BETA
    return min(eps, MAX_EPS_FRACTION * BETA)


# ── Public entry points ───────────────────────────────────────────────────────

def compute_trueskill(match_pairs: list) -> dict:
    """
    Plain win/loss TrueSkill (no margin), now also volatility-aware.
    Kept for backward compatibility and as a baseline to compare against.

    Parameters
    ----------
    match_pairs : list of (winner_entity, loser_entity), oldest first.

    Returns
    -------
    dict mapping entity -> Rating
    """
    ratings: dict = defaultdict(Rating)
    for winner, loser in match_pairs:
        ratings[winner], ratings[loser] = _update(ratings[winner], ratings[loser], eps=0.0)
    return dict(ratings)


def compute_trueskill_margin(match_triples: list) -> dict:
    """
    Margin-aware, volatility-aware TrueSkill. Same recursive replay as
    compute_trueskill, but each match also carries its set score, which
    is converted into a per-match epsilon via compute_match_margin(), and
    each entity's volatility is updated based on how surprising that
    match's margin-threshold outcome was given their pre-match ratings.

    Parameters
    ----------
    match_triples : list of (winner_entity, loser_entity, set_score_str),
        oldest first. set_score_str looks like "6-4 6-2" or "7-6 0-6 7-6".

    Returns
    -------
    dict mapping entity -> Rating
    """
    ratings: dict = defaultdict(Rating)
    for winner, loser, set_score in match_triples:
        eps = _margin_to_eps(compute_match_margin(set_score))
        ratings[winner], ratings[loser] = _update(ratings[winner], ratings[loser], eps=eps)
    return dict(ratings)


# ── CSV convenience loader ─────────────────────────────────────────────────────

def _parse_id_list(raw: str):
    """Parse a column like "['132109']" into ['132109']."""
    if not raw:
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return list(val)
        return [val]
    except (ValueError, SyntaxError):
        return [raw]


def load_matches_from_csv(path: str, sort_chronologically: bool = True) -> list:
    """
    Load a CSV with columns matching the tennis match export format:
    match_id, gender, match_type, flight, winner_names, loser_names,
    winner_school, loser_school, ..., winner_player_ids, loser_player_ids,
    set_score, match_updated_at, ...

    Singles matches use the single player id as the entity. Doubles
    matches use a sorted tuple of both player ids as a composite entity
    (a hashable "team" key, same trick suggested in the original module's
    docstring for pair keys).

    Returns a list of (winner_entity, loser_entity, set_score, timestamp)
    tuples, sorted oldest-first by match_updated_at if requested (needed
    since TrueSkill updates are order-dependent).
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            winner_ids = _parse_id_list(row.get("winner_player_ids", ""))
            loser_ids = _parse_id_list(row.get("loser_player_ids", ""))
            if not winner_ids or not loser_ids:
                continue

            if row.get("match_type", "").strip().lower() == "doubles":
                winner_entity = tuple(sorted(winner_ids))
                loser_entity = tuple(sorted(loser_ids))
            else:
                winner_entity = winner_ids[0]
                loser_entity = loser_ids[0]

            rows.append((
                winner_entity,
                loser_entity,
                row.get("set_score", ""),
                row.get("match_updated_at", ""),
            ))

    if sort_chronologically:
        rows.sort(key=lambda r: r[3])

    return rows


# ── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # A few illustrative scorelines, from most to least dominant.
    demo_scores = [
        ("6-0 6-0", "bagel-bagel"),
        ("6-2 6-3", "clearly dominant"),
        ("7-5 6-4", "moderately dominant"),
        ("7-6 7-6", "barely dominant"),
        ("7-6 0-6 7-6", "won, but wildly inconsistent"),
        ("2-6 6-3 10-7", "split sets, decided by match tiebreak"),
        ("6-1 4-6 10-2", "split sets, LOPSIDED match tiebreak"),
    ]
    print("scoreline        margin   eps      description")
    for score, desc in demo_scores:
        m = compute_match_margin(score)
        e = _margin_to_eps(m)
        print(f"{score:<16} {m:5.3f}   {e:5.3f}    {desc}")

    print("\nSame rating gap (mu diff held fixed at 0), effect of margin on update:")
    for score, desc in demo_scores:
        eps = _margin_to_eps(compute_match_margin(score))
        w, l = _update(Rating(), Rating(), eps=eps)
        print(f"{score:<16} winner -> {w}   loser -> {l}")

    # ── Volatility scenario: the motivating case from this module's redesign ──
    # Team A plays only elite opposition: a short, inconsistent run of
    # close wins and losses against much-higher-rated opponents.
    # Team B farms weak opposition: a long, near-undefeated streak against
    # much-lower-rated opponents, with one surprising loss thrown in.
    print("\nVolatility scenario: tough-schedule grinder vs. weak-schedule farmer")

    elite = Rating(mu=32.0, sigma=3.0)   # a stable, well-established elite
    weak = Rating(mu=18.0, sigma=3.0)    # a stable, well-established scrub

    team_a = Rating()  # starts at default MU/SIGMA, only plays `elite`
    team_a_matches = [
        ("7-6 6-4", True),    # narrow win over an elite
        ("4-6 3-6", False),   # loss to an elite
        ("6-7 7-6 7-6", True),  # another narrow win over an elite
        ("2-6 4-6", False),   # loss to an elite
        ("7-6 4-6 7-6", True),  # another narrow win over an elite
    ]
    for score, won in team_a_matches:
        eps = _margin_to_eps(compute_match_margin(score))
        if won:
            team_a, _ = _update(team_a, elite, eps=eps)
        else:
            _, team_a = _update(elite, team_a, eps=eps)

    team_b = Rating()  # starts at default MU/SIGMA, only plays `weak`
    team_b_matches = [
        ("6-1 6-2", True),
        ("6-0 6-1", True),
        ("6-2 6-3", True),
        ("4-6 3-6", False),   # one surprising loss to a weak opponent
        ("6-1 6-0", True),
        ("6-2 6-1", True),
        ("6-0 6-2", True),
    ]
    for score, won in team_b_matches:
        eps = _margin_to_eps(compute_match_margin(score))
        if won:
            team_b, _ = _update(team_b, weak, eps=eps)
        else:
            _, team_b = _update(weak, team_b, eps=eps)

    print(f"Team A (tough schedule, 3-2):   {team_a}")
    print(f"Team B (weak schedule, 6-1):    {team_b}")
