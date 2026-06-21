# trueskill_engine.py
#
# Lightweight TrueSkill implementation (Herbrich et al., 2007).
# No external dependencies — uses only `math` and `collections`.
#
# Public API
# ----------
#   compute_trueskill(match_pairs) -> {entity: Rating}
#   Rating.conservative           -> mu - 3 * sigma  (used for ranking)
#
# Design notes
# ------------
# * Uses the standard factor-graph / EP update rules for the win-loss
#   case (no draws; tennis never draws).
# * TAU (dynamics noise) prevents sigma from collapsing to zero so that
#   later matches always carry weight.
# * Each call to compute_trueskill() starts from a clean slate (MU, SIGMA)
#   and replays matches in chronological order.

import math
from collections import defaultdict
from dataclasses import dataclass

# ── Hyperparameters ────────────────────────────────────────────────────────────

MU = 25.0          # initial mean skill
SIGMA = MU / 3     # initial uncertainty  (~8.33)
BETA = SIGMA / 2   # performance noise    (~4.17)
TAU = SIGMA / 10   # dynamics factor      — keeps sigma from dying (~0.083)

# ── Normal-distribution helpers ───────────────────────────────────────────────

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _Phi(x: float) -> float:
    """Standard normal CDF via math.erfc for numerical stability at tails."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _v_win(t: float) -> float:
    """
    Truncated Gaussian mean factor (win case).
    v(t) = phi(t) / Phi(t)
    Clamps the denominator to avoid division by zero far in the tails.
    """
    denom = _Phi(t)
    if denom < 1e-10:
        # deep in the tail: winner is very likely to win, small update
        return max(0.0, -t)
    return _phi(t) / denom


def _w_win(t: float, v: float) -> float:
    """
    Truncated Gaussian variance factor (win case).
    w(t) = v(t) * (v(t) + t)
    Clamps to [0, 1) so sigma never grows from a single update.
    """
    return min(max(v * (v + t), 0.0), 1.0 - 1e-10)


# ── Rating dataclass ──────────────────────────────────────────────────────────

@dataclass
class Rating:
    mu: float = MU
    sigma: float = SIGMA

    @property
    def conservative(self) -> float:
        """Lower-bound estimate used for ranking: mu − 3σ."""
        return self.mu - 3.0 * self.sigma

    def __repr__(self) -> str:
        return f"Rating(mu={self.mu:.2f}, σ={self.sigma:.2f}, cons={self.conservative:.2f})"


# ── Core update ───────────────────────────────────────────────────────────────

def _update(r_win: Rating, r_lose: Rating) -> tuple[Rating, Rating]:
    """
    Apply one TrueSkill win/loss update.

    Step 1 — add dynamics noise (TAU²) to both players' variance.
    Step 2 — compute the combined performance noise (c).
    Step 3 — compute v and w factors.
    Step 4 — update mu and sigma for winner and loser.

    Returns new Rating objects (originals are not mutated).
    """
    # Step 1: dynamics
    sw2 = r_win.sigma ** 2 + TAU ** 2
    sl2 = r_lose.sigma ** 2 + TAU ** 2

    # Step 2: combined noise
    c2 = 2.0 * BETA ** 2 + sw2 + sl2
    c = math.sqrt(c2)

    # Step 3: factors
    t = (r_win.mu - r_lose.mu) / c
    v = _v_win(t)
    w = _w_win(t, v)

    # Step 4: updates
    mu_w_new = r_win.mu + (sw2 / c) * v
    mu_l_new = r_lose.mu - (sl2 / c) * v
    sigma_w_new = math.sqrt(sw2 * (1.0 - (sw2 / c2) * w))
    sigma_l_new = math.sqrt(sl2 * (1.0 - (sl2 / c2) * w))

    return (
        Rating(mu=mu_w_new, sigma=sigma_w_new),
        Rating(mu=mu_l_new, sigma=sigma_l_new),
    )


# ── Public entry point ────────────────────────────────────────────────────────

def compute_trueskill(
    match_pairs: list[tuple],
) -> dict:
    """
    Replay matches in chronological order and return final ratings.

    Parameters
    ----------
    match_pairs : list of (winner_entity, loser_entity)
        Entities can be any hashable type (str player IDs, tuple pair keys, …).
        Order matters — must be sorted oldest-first before calling.

    Returns
    -------
    dict mapping entity -> Rating
        Every entity that appeared in at least one match is present.
    """
    ratings: dict = defaultdict(Rating)

    for winner, loser in match_pairs:
        ratings[winner], ratings[loser] = _update(
            ratings[winner],
            ratings[loser],
        )

    return dict(ratings)
