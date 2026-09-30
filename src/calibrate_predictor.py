"""
calibrate_predictor.py
======================

Fits the data-driven match model used by predict_state.py and build_site.py
and writes it to  src/predictor_calibration.json  (both scripts read that one
file, so the site and the state predictions can never drift apart).

    python scripts/calibrate_predictor.py [path/to/all_matches.csv]

WHAT IS FIT
-----------
Honest, out-of-sample data only. Every pool (gender + singles/doubles +
flight) is split into 5 folds; power ratings are fit on 4 folds and used to
describe the 5th, so a rating never "saw" the match it is being scored on.
That gives ~19k held-out matches of (rating gap, what actually happened).

  1. WIN MODEL     P(favorite wins) = sigmoid(K * min(|gap|,12) / sqrt(1 + LAM*se))
                   K and LAM are fit by maximum likelihood. `se` is the
                   estimation error of the two ratings (fewer matches = less
                   sure), so thinly-sampled players get less extreme odds.

  2. SCORE MODEL   Everything about the scoreline is conditioned on ONE
                   number: z = logit(win probability of the player who won).
                   At a grid of z anchors, a Gaussian-kernel average over the
                   held-out matches gives
                       p3        chance the winner needed a 3rd set
                       p_tb/p_75 chance a set was 7-6 / 7-5
                       straight  distribution over straight-set lines
                       three     distribution over 3-set lines
                   Straight-set lines are ORDER-FREE (6-1 6-3 == 6-3 6-1).
                   Because a bigger rating gap means a higher win probability,
                   close matches get more 3-setters and lopsided scores, and
                   upset winners (z < 0) get their own, closer-looking scores,
                   all straight from the data with no hand-tuned multipliers.

Only matches with ordinary set scores feed the SCORE model (no pro-sets,
"2-0 2-0" placeholders, reversed/garbled scores).
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import mhsaa_seeding_v2 as seeding  # noqa: E402

OUT_PATH = REPO_ROOT / "src" / "predictor_calibration.json"
CAP = 12.0
RIDGE = 0.5
FOLDS = 5
Z_ANCHORS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
BANDWIDTH = 0.75          # kernel width in logit units
PRIOR_WEIGHT = 4.0        # pseudo-matches of the pooled average mixed into every anchor
MAX_STRAIGHT_LINES = 40
MAX_THREE_LINES = 60

# Selection knobs written to the JSON (predict_state / build_site read them).
PICK = {
    "shape_noise": 0.25,       # log-prob noise when comparing "3 sets" vs the best straight-set line
    "line_temp": 0.75,         # <1 favors the likeliest lines; 1 = sample exactly from the data
    "three_set_bias": 1.0,     # FIT below. >1 = more predicted 3-setters, <1 = fewer. 1.0 = the literal
                               # rule ("3 sets wins if its combined chance beats the best straight line")
}


# ---------------------------------------------------------------------------
# Score parsing (all from the MATCH WINNER's side: first number = winner's games)
# ---------------------------------------------------------------------------

def _toks(score: str):
    out = []
    for part in score.split():
        a, _, b = part.partition("-")
        try:
            out.append((int(a), int(b)))
        except ValueError:
            return None
    return out


def _won_set(t):     # winner took an ordinary set
    a, b = t
    return (a == 6 and 0 <= b <= 4) or (a == 7 and b in (5, 6))


def _lost_set(t):    # winner lost an ordinary set
    return _won_set((t[1], t[0]))


def _tb10(t):        # 10-point match tiebreak won by the winner
    a, b = t
    return (a == 10 and 0 <= b <= 8) or (a > 10 and a - b == 2)


def _fmt(t):
    return f"{t[0]}-{t[1]}"


def _margin(t):
    return t[0] - t[1]


def classify(score: str):
    """-> (kind, line_key, has_tb, has_75) or None if the score is unusable.
    kind is 2 (straight sets) or 3."""
    t = _toks(score)
    if not t:
        return None
    if len(t) == 2 and all(_won_set(s) for s in t):
        a, b = sorted(t, key=lambda s: -_margin(s))
        key = f"{_fmt(a)} {_fmt(b)}"
        sets = t
    elif len(t) == 3:
        first_two = t[:2]
        lost = [s for s in first_two if _lost_set(s)]
        won = [s for s in first_two if _won_set(s)]
        if len(lost) != 1 or len(won) != 1:
            return None
        if _tb10(t[2]):
            key = f"{_fmt(lost[0])} {_fmt(won[0])} {_fmt(t[2])}"
        elif _won_set(t[2]):
            a, b = sorted((won[0], t[2]), key=lambda s: -_margin(s))
            key = f"{_fmt(lost[0])} {_fmt(a)} {_fmt(b)}"
        else:
            return None
        sets = [s for s in t if not _tb10(s)]
    else:
        return None
    has_tb = any(s in ((7, 6), (6, 7)) for s in sets)
    has_75 = any(s in ((7, 5), (5, 7)) for s in sets)
    return len(t), key, has_tb, has_75


# ---------------------------------------------------------------------------
# 1. Out-of-sample dataset
# ---------------------------------------------------------------------------

def build_heldout(csv_path: str) -> list[dict]:
    meta = seeding.load_school_meta(csv_path)
    corr = seeding.load_correct_divisions(csv_path)
    matches = seeding.load_matches(csv_path, meta, corr)
    buckets = seeding.bucket_matches(matches)
    rows = []
    for key in sorted(buckets):
        elig = [m for m in buckets[key] if not m.get("is_ranking_excluded")]
        elig.sort(key=lambda x: (x["date"], x["winner"], x["loser"], x["score"]))
        rnd = random.Random(7)
        fold_of = [rnd.randrange(FOLDS) for _ in elig]
        for f in range(FOLDS):
            train = [m for m, ff in zip(elig, fold_of) if ff != f]
            test = [m for m, ff in zip(elig, fold_of) if ff == f]
            pr = seeding.compute_power_ratings(train)
            cnt: dict[str, int] = defaultdict(int)
            for m in train:
                cnt[m["winner"]] += 1
                cnt[m["loser"]] += 1
            for m in test:
                a, b = pr.get(m["winner"]), pr.get(m["loser"])
                if a is None or b is None:
                    continue
                rows.append({
                    "gap": a.rating - b.rating,          # winner minus loser
                    "se2": 1.0 / (RIDGE + cnt[m["winner"]]) + 1.0 / (RIDGE + cnt[m["loser"]]),
                    "score": m["score"],
                })
    return rows


# ---------------------------------------------------------------------------
# 2. Win model
# ---------------------------------------------------------------------------

def _fit_k(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """1-D logistic through the origin: P(y=1)=sigmoid(k*x). Newton."""
    k = 0.4
    for _ in range(60):
        p = 1 / (1 + np.exp(-k * x))
        g = np.sum(x * (y - p))
        h = np.sum(x * x * p * (1 - p)) + 1e-12
        k += g / h
    p = np.clip(1 / (1 + np.exp(-k * x)), 1e-9, 1 - 1e-9)
    return float(k), float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_win_model(rows):
    gap = np.array([r["gap"] for r in rows])
    se2 = np.array([r["se2"] for r in rows])
    fav_won = (gap >= 0).astype(float)
    ad = np.minimum(np.abs(gap), CAP)
    best = None
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        k, nll = _fit_k(ad / np.sqrt(1 + lam * se2), fav_won)
        if best is None or nll < best[2]:
            best = (k, lam, nll)
    return {"k": round(best[0], 4), "lam": best[1], "cap": CAP}, best[2]


def p_fav(win, ad, se2):
    return 1 / (1 + math.exp(-win["k"] * min(ad, win["cap"]) / math.sqrt(1 + win["lam"] * se2)))


# ---------------------------------------------------------------------------
# 3. Score model (kernel regression on z = logit P(winner wins))
# ---------------------------------------------------------------------------

def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_score_model(rows, win):
    recs = []
    for r in rows:
        c = classify(r["score"])
        if c is None:
            continue
        pf = p_fav(win, abs(r["gap"]), r["se2"])
        pw = 0.5 if r["gap"] == 0 else (pf if r["gap"] > 0 else 1 - pf)
        recs.append((_logit(pw), *c))
    n = len(recs)
    z = np.array([x[0] for x in recs])
    kind3 = np.array([x[1] == 3 for x in recs], float)
    tb = np.array([x[3] for x in recs], float)
    s75 = np.array([x[4] for x in recs], float)
    lines2 = [x[2] if x[1] == 2 else None for x in recs]
    lines3 = [x[2] if x[1] == 3 else None for x in recs]
    pooled = {
        "p3": kind3.mean(), "p_tb": tb.mean(), "p_75": s75.mean(),
        "two": Counter(l for l in lines2 if l), "three": Counter(l for l in lines3 if l),
    }
    tot2, tot3 = sum(pooled["two"].values()), sum(pooled["three"].values())

    anchors = []
    for a in Z_ANCHORS:
        w = np.exp(-0.5 * ((z - a) / BANDWIDTH) ** 2)
        W = w.sum() + PRIOR_WEIGHT
        mix = lambda v, prior: (float((w * v).sum()) + PRIOR_WEIGHT * prior) / W
        two, three = defaultdict(float), defaultdict(float)
        for wi, l2, l3 in zip(w, lines2, lines3):
            if l2:
                two[l2] += wi
            elif l3:
                three[l3] += wi
        for l, c in pooled["two"].items():
            two[l] += PRIOR_WEIGHT * c / tot2
        for l, c in pooled["three"].items():
            three[l] += PRIOR_WEIGHT * c / tot3

        def top(d, m):
            items = sorted(d.items(), key=lambda kv: -kv[1])[:m]
            s = sum(v for _, v in items)
            return [[k, round(v / s, 5)] for k, v in items]

        anchors.append({
            "z": a,
            "n_eff": round(float(w.sum()), 1),
            "p3": round(mix(kind3, pooled["p3"]), 5),
            "p_tb": round(mix(tb, pooled["p_tb"]), 5),
            "p_75": round(mix(s75, pooled["p_75"]), 5),
            "straight": top(two, MAX_STRAIGHT_LINES),
            "three": top(three, MAX_THREE_LINES),
        })
    return anchors, n


# ---------------------------------------------------------------------------
# 3b. Fit the 3-set comparison bias
# ---------------------------------------------------------------------------

def _interp_anchor(anchors, pw):
    z = min(max(_logit(pw), anchors[0]["z"]), anchors[-1]["z"])
    hi = next((i for i, a in enumerate(anchors) if a["z"] >= z), len(anchors) - 1)
    lo = max(hi - 1, 0)
    a0, a1 = anchors[lo], anchors[hi]
    f = 0.0 if a1["z"] == a0["z"] else (z - a0["z"]) / (a1["z"] - a0["z"])
    p3 = a0["p3"] + f * (a1["p3"] - a0["p3"])
    d0, d1 = dict(a0["straight"]), dict(a1["straight"])
    top = max(d0.get(k, 0.0) + f * (d1.get(k, 0.0) - d0.get(k, 0.0)) for k in set(d0) | set(d1))
    return p3, top


def predicted_three_set_rate(rows, win, anchors, bias):
    """Share of held-out matches where the rule picks a 3-setter:
    p3 * bias > (1 - p3) * (likeliest straight-set line's share), for the
    favorite (the side the bracket always advances)."""
    n3 = 0
    for r in rows:
        pf = p_fav(win, abs(r["gap"]), r["se2"])
        p3, top = _interp_anchor(anchors, pf)
        n3 += p3 * bias > (1.0 - p3) * top
    return n3 / len(rows)


def fit_three_set_bias(rows, win, anchors):
    """Bias at which the rule's predicted 3-set share equals the REAL share."""
    actual = sum(1 for r in rows if len(r["score"].split()) == 3) / len(rows)
    lo, hi = 0.05, 1.0
    if predicted_three_set_rate(rows, win, anchors, hi) <= actual:
        return 1.0, actual
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if predicted_three_set_rate(rows, win, anchors, mid) < actual:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 4), actual


# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------

def diagnostics(rows, win):
    print("\nWin-probability calibration (held-out):")
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.01]
    buckets = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        pf = p_fav(win, abs(r["gap"]), r["se2"])
        b = next(i for i in range(len(edges) - 1) if edges[i] <= pf < edges[i + 1])
        buckets[b][0] += 1
        buckets[b][1] += pf
        buckets[b][2] += 1.0 if r["gap"] >= 0 else 0.0
    for b in sorted(buckets):
        n, sp, sw = buckets[b]
        print(f"  predicted {edges[b]:.2f}-{edges[b+1]:.2f}   n={n:5d}   mean predicted {sp/n:.3f}   actual {sw/n:.3f}")


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "data" / "matches.csv")
    print(f"Calibrating from {csv_path}")
    rows = build_heldout(csv_path)
    print(f"  {len(rows):,} held-out matches")
    win, nll = fit_win_model(rows)
    print(f"  win model: K={win['k']}  LAM={win['lam']}  held-out log-loss {nll:.4f}")
    anchors, n_score = fit_score_model(rows, win)
    print(f"  score model fit on {n_score:,} matches with ordinary set scores")
    bias, actual3 = fit_three_set_bias(rows, win, anchors)
    PICK["three_set_bias"] = bias
    print(f"  3-set rule: literal (bias 1.0) predicts 3 sets in "
          f"{predicted_three_set_rate(rows, win, anchors, 1.0):.1%} of matches; "
          f"real rate is {actual3:.1%}")
    print(f"  fitted three_set_bias = {bias}  ->  predicts "
          f"{predicted_three_set_rate(rows, win, anchors, bias):.1%}")
    diagnostics(rows, win)
    out = {
        "version": 1,
        "n_heldout_matches": len(rows),
        "n_score_matches": n_score,
        "win": win,
        "bandwidth": BANDWIDTH,
        "pick": PICK,
        "anchors": anchors,
    }
    OUT_PATH.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"  wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
