"""
seed_accuracy.py
================
Scores the predicted seeds from mhsaa_seeding_v2.py against the OFFICIAL
state-tournament seeds in state_seeds.csv.

SCOPE: ONLY real seeds 1-6 are ever scored.  state_seeds.csv rows with
seed > 6 are ignored, because seeds below #6 don't reflect genuine seeding
decisions.  For every (gender, category, division, flight) combo, the six
real seeded players/pairs are looked up in the matching predicted CSV
(rankings_by_division_flight/{category}_{gender}_division_{division}.csv).

Metrics (reported overall and by gender/category, division, flight, real
seed position, and per division/flight combo):
    exact        predicted rank == real seed
    within 1     |predicted rank - real seed| <= 1
    within 5     |predicted rank - real seed| <= 5
    in top 6     real top-6 seed also has predicted rank <= 6
    mean diff    signed (predicted - real; + = engine ranked them lower)
    mean |diff|  average absolute error

Two denominators are shown because a real seed can be missing from the
predictions (e.g. didn't reach MIN_MATCHES, or a name mismatch):
    ALL      every real top-6 seed; unmatched ones count as misses
    MATCHED  only real top-6 seeds that were found in the predictions

Name matching mirrors mhsaa_seeding_v2.py: each half of a doubles pair is
run through the school-scoped alias table (normalize_player_name_with_school),
halves are alphabetised, and "A / B" vs "A/B" spacing is ignored.  If
mhsaa_seeding_v2 can be imported its functions are used directly, so the two
never drift apart; otherwise a mirrored copy of the alias table is used.

Usage
-----
    python seed_accuracy.py                      # default paths
    python seed_accuracy.py --year 2025
    python seed_accuracy.py --state-seeds ../historical/2025/state_seeds.csv \
                            --predicted-dir rankings_by_division_flight
    python seed_accuracy.py --min-top6-pct 60    # exit 1 if below 60 %
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

TOP_N = 6  # only real seeds 1..TOP_N are ever scored (fixed by design)

# ----------------------------------------------------------------------------
# Name handling — reuse the engine's own functions when possible
# ----------------------------------------------------------------------------
try:
    from mhsaa_seeding_v2 import (  # type: ignore
        normalize_player_name_with_school as _engine_normalize,
        _category_filename_stem as _engine_stem,
    )
    NAME_SOURCE = "mhsaa_seeding_v2 (imported)"
except Exception:  # missing deps, not on path, etc.
    _engine_normalize = None
    _engine_stem = None
    NAME_SOURCE = "built-in mirror of mhsaa_seeding_v2"

# Mirror of _PLAYER_ALIASES in mhsaa_seeding_v2.py (fallback only).
_FALLBACK_ALIASES: dict[tuple[str, str], str] = {
    ("ann arbor huron", "chengzhi ma"): "Charles Ma",
    ("ann arbor huron", "charles ma"): "Charles Ma",
}


def _fallback_normalize(name: str, school: str) -> str:
    parts = []
    for p in (name or "").split("/"):
        p = p.strip()
        parts.append(_FALLBACK_ALIASES.get((school.strip().lower(), p.lower()), p))
    if len(parts) > 1:
        parts = sorted(parts)
    return "/".join(parts)


def _clean_token(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().casefold()


def match_key(name: str, school: str = "") -> str:
    """
    Canonical identity string for a player or doubles pair:
      1. split on "/" and trim each half (handles "A / B" vs "A/B")
      2. apply the engine's school-scoped alias resolution + alphabetising
      3. casefold / strip accents / collapse whitespace
      4. re-sort the halves so order can never matter
    """
    raw = "/".join(p.strip() for p in (name or "").split("/"))
    if _engine_normalize is not None:
        try:
            raw = _engine_normalize(raw, school or "")
        except Exception:
            raw = _fallback_normalize(raw, school or "")
    else:
        raw = _fallback_normalize(raw, school or "")
    parts = sorted(_clean_token(p) for p in raw.split("/") if p.strip())
    return "/".join(parts)


def category_gender_stem(category: str, gender: str) -> tuple[str, str]:
    if _engine_stem is not None:
        try:
            return _engine_stem(category or "", gender or "")
        except Exception:
            pass
    cat = "singles" if (category or "").strip().lower().startswith("single") else "doubles"
    gen = "boys" if (gender or "").strip().lower().startswith("b") else "girls"
    return cat, gen


def norm_division(division: str) -> str:
    """'4_other' / 'Division 4' / 'D4' -> '4'; falls back to lowercase text."""
    d = (division or "").strip().lower()
    m = re.search(r"[1-4]", d)
    return m.group(0) if m else d


def norm_flight(flight: str) -> str:
    f = (flight or "").strip()
    m = re.search(r"\d+", f)
    return m.group(0) if m else f.lower()


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_real_top6(path: Path, year: str | None) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        for row in csv.DictReader(f, delimiter=delim):
            if year and (row.get("year") or "").strip() not in ("", str(year)):
                continue
            try:
                seed = int((row.get("seed") or "").strip())
            except ValueError:
                continue
            if 1 <= seed <= TOP_N:           # <-- top 6 only, always
                row["_seed"] = seed
                rows.append(row)
    return rows


def load_predicted(pred_dir: Path):
    """Returns (lookup, idx).
    lookup[(category, gender, division, flight, match_key)] = predicted rank.
    idx holds extra indexes used only to explain unmatched seeds:
      by_name   {(cat, gen, key)}        -> [(division, flight, rank)]
      by_player {(cat, gen, player_key)} -> [(division, flight, pair_name)]
      by_slot   {(cat, gen, div, flight)} -> [(key, raw_name)]
      files     {(cat, gen, div)} -> row count
    Only per-player files for divisions 1-4 are used (team_* and *_overall
    files are skipped)."""
    lookup: dict[tuple, int] = {}
    idx = {"by_name": defaultdict(list), "by_player": defaultdict(list),
           "by_slot": defaultdict(list), "files": {}}
    for path in sorted(pred_dir.glob("*_division_*.csv")):
        stem = path.stem
        if stem.startswith("team_") or "_division_" not in stem:
            continue
        cat_gender, division = stem.split("_division_", 1)
        if division.lower() not in {"1", "2", "3", "4"}:
            continue
        try:
            category, gender = cat_gender.split("_", 1)
        except ValueError:
            continue
        n_rows = 0
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw_name = row.get("pair_name") or row.get("name") or ""
                try:
                    rank = int((row.get("rank") or "").strip())
                except ValueError:
                    continue
                div, fl = norm_division(division), norm_flight(row.get("flight", ""))
                key = match_key(raw_name, row.get("school", ""))
                lk = (category, gender, div, fl, key)
                if lk not in lookup or rank < lookup[lk]:
                    lookup[lk] = rank
                idx["by_name"][(category, gender, key)].append((div, fl, rank))
                idx["by_slot"][(category, gender, div, fl)].append((key, raw_name))
                for part in key.split("/"):
                    idx["by_player"][(category, gender, part)].append((div, fl, raw_name))
                n_rows += 1
        idx["files"][(category, gender, norm_division(division))] = n_rows
    return lookup, idx


def diagnose(cat, gen, div, fl, key, idx) -> tuple[str, str]:
    """Explain why a real seed has no predicted row."""
    if (cat, gen, div) not in idx["files"]:
        return "no predicted file", f"{cat}_{gen}_division_{div}.csv not found/loaded"
    elsewhere = idx["by_name"].get((cat, gen, key))
    if elsewhere:
        d, f, r = elsewhere[0]
        return ("found in different division/flight",
                f"predicted in div {d} flight {f} at rank {r}")
    parts = key.split("/")
    if len(parts) > 1:
        hits = []
        for p in parts:
            for d, f, pair in idx["by_player"].get((cat, gen, p), []):
                hits.append(f"{p} -> '{pair}' (div {d} flt {f})")
        if hits:
            return "partner combo differs", "; ".join(hits[:2])
    slot = idx["by_slot"].get((cat, gen, div, fl), [])
    best, best_ratio = None, 0.0
    for k, raw in slot:
        r = difflib.SequenceMatcher(None, key, k).ratio()
        if r > best_ratio:
            best, best_ratio = raw, r
    if best and best_ratio >= 0.80:
        return "possible spelling mismatch", f"closest predicted name: '{best}' ({best_ratio:.0%})"
    if not slot:
        return "flight empty in predictions", f"no predicted rows for div {div} flight {fl}"
    return "not in any predicted file", "player/pair never ranked (no matches or below MIN_MATCHES)"


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def summarize(rows: list[dict]) -> dict:
    n_all = len(rows)
    matched = [r for r in rows if r["predicted_rank"] is not None]
    n_m = len(matched)

    def pct(k, n):
        return round(100.0 * k / n, 1) if n else None

    def cnt(pred, pool):
        return sum(1 for r in pool if pred(r))

    exact = lambda r: r["diff"] == 0
    w1 = lambda r: r["abs_diff"] <= 1
    w5 = lambda r: r["abs_diff"] <= 5
    t6 = lambda r: r["predicted_rank"] <= TOP_N

    diffs = [r["diff"] for r in matched]
    return {
        "real_seeds": n_all,
        "matched": n_m,
        "unmatched": n_all - n_m,
        # ALL denominators (unmatched = miss)
        "exact_pct_all": pct(cnt(exact, matched), n_all),
        "within1_pct_all": pct(cnt(w1, matched), n_all),
        "within5_pct_all": pct(cnt(w5, matched), n_all),
        "in_pred_top6_pct_all": pct(cnt(t6, matched), n_all),
        # MATCHED denominators
        "exact_pct_matched": pct(cnt(exact, matched), n_m),
        "within1_pct_matched": pct(cnt(w1, matched), n_m),
        "within5_pct_matched": pct(cnt(w5, matched), n_m),
        "in_pred_top6_pct_matched": pct(cnt(t6, matched), n_m),
        "exact_n": cnt(exact, matched),
        "within1_n": cnt(w1, matched),
        "within5_n": cnt(w5, matched),
        "in_pred_top6_n": cnt(t6, matched),
        "mean_diff": round(statistics.mean(diffs), 3) if diffs else None,
        "mean_abs_diff": round(statistics.mean(abs(d) for d in diffs), 3) if diffs else None,
        "median_abs_diff": round(statistics.median(abs(d) for d in diffs), 3) if diffs else None,
    }


def fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def build_breakdowns(rows: list[dict]) -> list[tuple[str, str, dict]]:
    """[(level, group label, stats)] — overall first, then each breakdown."""
    out = [("overall", "ALL", summarize(rows))]

    def add(level, keyfn, sortfn=None):
        groups = defaultdict(list)
        for r in rows:
            groups[keyfn(r)].append(r)
        for g in sorted(groups, key=sortfn) if sortfn else sorted(groups):
            out.append((level, g, summarize(groups[g])))

    add("gender/category", lambda r: f"{r['gender']}/{r['category']}")
    add("gender/category/division",
        lambda r: f"{r['gender']}/{r['category']}/div{r['division']}")
    add("gender/category/division/flight",
        lambda r: f"{r['gender']}/{r['category']}/div{r['division']}/flight{r['flight']}")
    add("flight", lambda r: f"flight{r['flight']}")
    add("real seed", lambda r: f"seed #{r['real_seed']}")
    return out


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def print_console(breakdowns, unmatched_rows):
    o = breakdowns[0][2]
    bar = "=" * 72
    print(f"\n{bar}\nOVERALL  (real seeds 1-{TOP_N} only)\n{bar}")
    print(f"  Real top-{TOP_N} seeds scored : {o['real_seeds']}")
    print(f"  Found in predictions       : {o['matched']}   (unmatched: {o['unmatched']})")
    print()
    print(f"  {'metric':<32}{'ALL (unmatched=miss)':>22}{'MATCHED only':>16}")
    for label, a, m, n in [
        ("Exactly correct",          "exact_pct_all",        "exact_pct_matched",        "exact_n"),
        ("Within 1 seed",            "within1_pct_all",      "within1_pct_matched",      "within1_n"),
        ("Within 5 seeds",           "within5_pct_all",      "within5_pct_matched",      "within5_n"),
        (f"Real top-{TOP_N} in predicted top-{TOP_N}",
                                     "in_pred_top6_pct_all", "in_pred_top6_pct_matched", "in_pred_top6_n"),
    ]:
        print(f"  {label:<32}{fmt(o[a], '%') + f' ({o[n]}/{o['real_seeds']})':>22}"
              f"{fmt(o[m], '%'):>16}")
    print(f"\n  Mean diff (signed, matched)  : {fmt(o['mean_diff'])}   (+ = engine ranked lower than reality)")
    print(f"  Mean absolute diff (matched) : {fmt(o['mean_abs_diff'])}")
    print(f"  Median absolute diff         : {fmt(o['median_abs_diff'])}")

    current = None
    for level, group, s in breakdowns[1:]:
        if level != current:
            current = level
            print(f"\n  By {level}:")
            print(f"    {'group':<44}{'n':>4}{'found':>6}{'exact':>8}{'±1':>8}{'±5':>8}{'top6':>8}")
        print(f"    {group:<44}{s['real_seeds']:>4}{s['matched']:>6}"
              f"{fmt(s['exact_pct_all']):>8}{fmt(s['within1_pct_all']):>8}"
              f"{fmt(s['within5_pct_all']):>8}{fmt(s['in_pred_top6_pct_all']):>8}")
    print("\n  (percentages above use the ALL denominator; unmatched real seeds count as misses)")

    if unmatched_rows:
        print(f"\n  {len(unmatched_rows)} real top-{TOP_N} seed(s) NOT found in predictions:")
        for r in unmatched_rows[:25]:
            print(f"    {r['gender']} {r['category']} div{r['division']} "
                  f"flight{r['flight']} seed {r['real_seed']}: {r['name']} ({r['school']})")
        if len(unmatched_rows) > 25:
            print(f"    ... and {len(unmatched_rows) - 25} more")


def write_markdown(path: Path, breakdowns, unmatched_rows, year, name_source):
    o = breakdowns[0][2]
    L = []
    L.append(f"# Seed Accuracy Report{f' — {year}' if year else ''}")
    L.append("")
    L.append(f"Scores the engine's predicted rank for each **real state seed 1–{TOP_N}** "
             f"(seeds below #{TOP_N} are ignored). Name matching: {name_source}.")
    L.append("")
    L.append(f"**{o['real_seeds']}** real top-{TOP_N} seeds scored · "
             f"**{o['matched']}** found in predictions · **{o['unmatched']}** not found")
    L.append("")
    L.append("| Metric | Count | % of all real seeds | % of matched |")
    L.append("|---|---:|---:|---:|")
    for label, n, a, m in [
        ("Exactly correct", "exact_n", "exact_pct_all", "exact_pct_matched"),
        ("Within 1 seed", "within1_n", "within1_pct_all", "within1_pct_matched"),
        ("Within 5 seeds", "within5_n", "within5_pct_all", "within5_pct_matched"),
        (f"Real top-{TOP_N} in predicted top-{TOP_N}", "in_pred_top6_n",
         "in_pred_top6_pct_all", "in_pred_top6_pct_matched"),
    ]:
        L.append(f"| {label} | {o[n]} | {fmt(o[a], '%')} | {fmt(o[m], '%')} |")
    L.append("")
    L.append(f"Mean signed diff: **{fmt(o['mean_diff'])}** (+ = engine ranked lower than reality) · "
             f"Mean |diff|: **{fmt(o['mean_abs_diff'])}** · Median |diff|: **{fmt(o['median_abs_diff'])}**")
    L.append("")
    L.append("_Breakdown percentages use the all-real-seeds denominator (unmatched = miss)._")

    current = None
    for level, group, s in breakdowns[1:]:
        if level != current:
            current = level
            L.append("")
            L.append(f"### By {level}")
            L.append("")
            L.append("| Group | Seeds | Found | Exact % | ±1 % | ±5 % | In top-6 % | Mean diff | Mean abs |")
            L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        L.append(f"| {group} | {s['real_seeds']} | {s['matched']} | {fmt(s['exact_pct_all'])} | "
                 f"{fmt(s['within1_pct_all'])} | {fmt(s['within5_pct_all'])} | "
                 f"{fmt(s['in_pred_top6_pct_all'])} | {fmt(s['mean_diff'])} | {fmt(s['mean_abs_diff'])} |")
    if unmatched_rows:
        L.append("")
        L.append(f"### Real top-{TOP_N} seeds not found in predictions ({len(unmatched_rows)})")
        L.append("")
        L.append("| Gender | Category | Div | Flight | Seed | Name | School | Reason |")
        L.append("|---|---|---|---|---:|---|---|---|")
        for r in unmatched_rows:
            L.append(f"| {r['gender']} | {r['category']} | {r['division']} | {r['flight']} | "
                     f"{r['real_seed']} | {r['name']} | {r['school']} | "
                     f"{r['unmatched_reason']}: {r['unmatched_detail']} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def resolve_default(candidates: list[str]) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", default=None,
                    help="Year; used to locate historical/{year}/state_seeds.csv and to "
                         "filter the 'year' column if present (default: config.YEAR or 2025)")
    ap.add_argument("--state-seeds", default=None, help="Path to state_seeds.csv")
    ap.add_argument("--predicted-dir", default="rankings_by_division_flight",
                    help="Directory of predicted *_division_*.csv files")
    ap.add_argument("--out", default="seed_accuracy_report.csv",
                    help="Row-by-row detail CSV (default: seed_accuracy_report.csv)")
    ap.add_argument("--summary-out", default="seed_accuracy_summary.csv",
                    help="Breakdown summary CSV (default: seed_accuracy_summary.csv)")
    ap.add_argument("--markdown-out", default="seed_accuracy_summary.md",
                    help="Markdown summary (default: seed_accuracy_summary.md)")
    ap.add_argument("--json-out", default="seed_accuracy_summary.json",
                    help="Overall stats as JSON (default: seed_accuracy_summary.json)")
    ap.add_argument("--min-top6-pct", type=float, default=None,
                    help="Exit with code 1 if overall %% of real top-6 seeds found in the "
                         "predicted top-6 is below this value")
    args = ap.parse_args()

    year = args.year
    if year is None:
        try:
            import config as _config  # type: ignore
            year = str(getattr(_config, "YEAR", "2025"))
        except Exception:
            year = "2025"

    state_arg = args.state_seeds or resolve_default([
        os.path.join("..", "historical", year, "state_seeds.csv"),
        os.path.join("historical", year, "state_seeds.csv"),
    ])
    state_path, pred_dir = Path(state_arg), Path(args.predicted_dir)
    if not state_path.exists():
        raise SystemExit(f"ERROR: state seeds file not found: {state_path}")
    if not pred_dir.exists():
        raise SystemExit(f"ERROR: predicted-dir not found: {pred_dir}")

    print(f"Name matching : {NAME_SOURCE}")
    real_rows = load_real_top6(state_path, year if args.year else None)
    print(f"Loaded {len(real_rows)} real top-{TOP_N} seed rows from {state_path}")
    lookup, idx = load_predicted(pred_dir)
    print(f"Loaded {len(lookup)} predicted rows from {pred_dir}")
    if not real_rows:
        raise SystemExit("ERROR: no real top-6 seed rows to score.")

    detail, unmatched = [], []
    for r in real_rows:
        cat, gen = category_gender_stem(r.get("category", ""), r.get("gender", ""))
        div, fl = norm_division(r.get("division", "")), norm_flight(r.get("flight", ""))
        name, school = r.get("name", ""), r.get("school", "")
        key = match_key(name, school)
        pred = lookup.get((cat, gen, div, fl, key))
        seed = r["_seed"]
        reason, why = ("", "") if pred is not None else diagnose(cat, gen, div, fl, key, idx)
        row = {
            "gender": gen, "category": cat, "division": div, "flight": fl,
            "name": name, "school": school, "real_seed": seed,
            "predicted_rank": pred,
            "diff": None if pred is None else pred - seed,
            "abs_diff": None if pred is None else abs(pred - seed),
            "in_predicted_top6": "" if pred is None else int(pred <= TOP_N),
            "unmatched_reason": reason, "unmatched_detail": why,
        }
        detail.append(row)
        if pred is None:
            unmatched.append(row)

    print("\nPredicted rows loaded per file:")
    for (c, g, d), n in sorted(idx["files"].items()):
        print(f"    {c}_{g}_division_{d}.csv : {n}")
    if unmatched:
        reasons = defaultdict(int)
        for u in unmatched:
            reasons[u["unmatched_reason"]] += 1
        print(f"\nWhy {len(unmatched)} real seeds were not found:")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>4}  {k}")
        print("  Examples:")
        shown = set()
        for u in unmatched:
            if u["unmatched_reason"] in shown:
                continue
            shown.add(u["unmatched_reason"])
            print(f"    [{u['unmatched_reason']}] {u['name']} ({u['school']}) "
                  f"{u['category']} d{u['division']} f{u['flight']} #{u['real_seed']}: {u['unmatched_detail']}")

    breakdowns = build_breakdowns(detail)
    print_console(breakdowns, unmatched)

    # ---- files ---------------------------------------------------------
    order = lambda d: (d["gender"], d["category"], d["division"], d["flight"], d["real_seed"])
    fields = ["gender", "category", "division", "flight", "name", "school", "real_seed",
              "predicted_rank", "diff", "abs_diff", "in_predicted_top6",
              "unmatched_reason", "unmatched_detail"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in sorted(detail, key=order):
            w.writerow({k: ("" if d[k] is None else d[k]) for k in fields})

    sfields = ["level", "group"] + list(breakdowns[0][2].keys())
    with open(args.summary_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for level, group, s in breakdowns:
            w.writerow({"level": level, "group": group, **{k: ("" if v is None else v) for k, v in s.items()}})

    write_markdown(Path(args.markdown_out), breakdowns, unmatched, year, NAME_SOURCE)
    Path(args.json_out).write_text(json.dumps(breakdowns[0][2], indent=2), encoding="utf-8")

    print(f"\nWrote: {args.out}, {args.summary_out}, {args.markdown_out}, {args.json_out}")

    # GitHub Actions job summary (no-op locally)
    gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            f.write(Path(args.markdown_out).read_text(encoding="utf-8"))

    if args.min_top6_pct is not None:
        got = breakdowns[0][2]["in_pred_top6_pct_all"] or 0.0
        if got < args.min_top6_pct:
            raise SystemExit(f"FAIL: top-6 overlap {got}% is below threshold {args.min_top6_pct}%")
        print(f"PASS: top-6 overlap {got}% >= {args.min_top6_pct}%")


if __name__ == "__main__":
    main()
