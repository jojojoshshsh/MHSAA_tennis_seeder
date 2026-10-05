import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# Make src/ importable so we can read config.YEAR for the site header
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from config import YEAR as SEASON_YEAR
except ImportError:
    SEASON_YEAR = None

src_dir = REPO_ROOT / "src" / "rankings_by_division_flight"
out_dir = REPO_ROOT / "docs"
out_dir.mkdir(exist_ok=True)

csv_dir = out_dir / "csv"
csv_dir.mkdir(exist_ok=True)

ALLOWED_FLIGHTS = {"1", "2", "3", "4"}

# ── Team rankings ─────────────────────────────────────────────────────────────
team_data = []
for csv_path in sorted(src_dir.glob("team_*.csv")):
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    stem = csv_path.stem
    gender = "Boys" if "_boys_" in stem else "Girls"
    division = stem.split("_division_")[-1].replace("_", " ")
    team_data.append({"gender": gender, "division": division, "df": df.head(10), "df_full": df})

DIVISION_ORDER_T = {"1": 0, "2": 1, "3": 2, "4": 3, "4 other": 4}
team_data.sort(key=lambda x: (
    DIVISION_ORDER_T.get(x["division"], 9),
    0 if x["gender"] == "Boys" else 1,
))

def _html_escape_py(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

def _norm_division(div_str):
    """Normalize a division label like '4_other' or '4 other' down to its
    leading numeral so the Division 1-4 filter checkboxes still match it."""
    tokens = str(div_str).replace("_", " ").split()
    return tokens[0] if tokens else str(div_str)

# ── Build team HTML ────────────────────────────────────────────────────────────
team_html = ""
for entry in team_data:
    is_overall_team = entry["division"] == "overall"
    if is_overall_team:
        label = f"Top 10 Teams · {entry['gender']} — All Divisions (General Ranking)"
        anchor = f"team_{entry['gender'].lower()}_overall"
    else:
        label = f"Top 10 Teams · {entry['gender']} Division {entry['division']}"
        anchor = f"team_{entry['gender'].lower()}_div{entry['division'].replace(' ','')}"
    df = entry["df"]
    df_full = entry["df_full"]
    div_attr = _norm_division(entry["division"])
    has_more = len(df_full) > len(df)

    COL_LABELS = {
        "rank": "Rank",
        "school": "School",
        "total_points": "Total Pts",
        "combined_score": "Combined",
        "depth_score": "Depth",
        "team_sos": "Team SOS",
        "team_local_sos": "Team Local SOS",
        "slots_counted": "Slots",
        "s1_pts": "S-F1",
        "s2_pts": "S-F2",
        "s3_pts": "S-F3",
        "s4_pts": "S-F4",
        "d1_pts": "D-F1",
        "d2_pts": "D-F2",
        "d3_pts": "D-F3",
        "d4_pts": "D-F4",
        "reason_below": "Why ranked below team above",
    }

    cols = list(df_full.columns)

    tbody_rows = ""
    for i, (_, row) in enumerate(df_full.iterrows()):
        cells = ""
        for col in cols:
            val = _html_escape_py(row[col])
            if col == "reason_below":
                cells += f'<td class="reason-cell">{val}</td>'
            elif col == "rank":
                cells += f'<td class="rank-cell">{val}</td>'
            elif col in ("total_points", "combined_score"):
                cells += f'<td class="pts-cell">{val}</td>'
            else:
                cells += f"<td>{val}</td>"
        row_class = ' class="extra-team-row" style="display:none;"' if i >= 10 else ""
        tbody_rows += f"<tr{row_class}>{cells}</tr>"

    show_all_btn = (
        f'<button type="button" class="dl-btn show-all-btn" onclick="toggleShowAllTeams(this, \'{anchor}\')">Show All Teams</button>'
        if has_more else ""
    )

    if is_overall_team:
        _tiers = [("1st", "12.5"), ("2nd", "11.25"), ("3rd – 4th", "10.0"),
                  ("5th – 8th", "7.5"), ("9th – 16th", "5.0"), ("17th – 32nd", "3.0"),
                  ("33rd – 64th", "1.25"), ("65th – 128th", "0.5"), ("129th+", "0")]
        _rows = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in _tiers)
        scoring_note_html = (
            '<details class="scoring-note scoring-dist"><summary>Score distribution</summary>'
            '<table><thead><tr><th>Place</th><th>Points</th></tr></thead>'
            f'<tbody>{_rows}</tbody></table></details>'
        )
    else:
        scoring_note_html = '<span class="scoring-note">Points: 1st=12.5 · 2nd=10 · 3rd–4th=7.5 · 5th–8th=5 · 9th–16th=2.5 · 17th–32nd=1</span>'

    div_attr_html = ' class="general-section"' if is_overall_team else f' data-division="{_html_escape_py(div_attr)}"'
    team_html += f"""
    <section id="{anchor}"{div_attr_html}>
      <div class="section-header">
        <h2>{_html_escape_py(label)}</h2>
        {scoring_note_html}
        {show_all_btn}
      </div>
      <div class="table-wrap"><table class="rankings-table team-table"><thead><tr>{"".join(
          f'<th onclick="sortTable(this)" title="{_html_escape_py(col)}">{_html_escape_py(COL_LABELS.get(col, col))}</th>'
          for col in cols
      )}</tr></thead><tbody>{tbody_rows}</tbody></table></div>
    </section>
    """

# ── Load all individual CSVs ──────────────────────────────────────────────────
# "overall" is a pseudo-division (see mhsaa_seeding_v2.py STEP 4b): a
# cross-division, flight-wide ranking published alongside (not instead of)
# the normal per-division files. Those go into general_data and get their
# own "General Rankings" section/nav further down, instead of being mixed
# into the Division 1-4 filterable tables in all_data.
all_data = []
general_data = []
all_rows_for_search = []

for csv_path in sorted(src_dir.glob("*.csv")):
    if csv_path.stem.startswith("team_"):
        continue

    df = pd.read_csv(csv_path)
    if "flight" in df.columns:
        df = df[df["flight"].astype(str).isin(ALLOWED_FLIGHTS)]
    if df.empty:
        continue

    dest = csv_dir / csv_path.name
    df.to_csv(dest, index=False)

    stem = csv_path.stem
    category = "singles" if stem.startswith("singles") else "doubles"
    gender = "boys" if "_boys_" in stem else "girls"

    if "division" in df.columns and "flight" in df.columns:
        for (division, flight), group in df.groupby(["division", "flight"]):
            is_overall = str(division) == "overall"
            entry = {
                "division": str(division),
                "flight": str(flight),
                "category": category,
                "gender": gender,
                "filename": csv_path.name,
                "df": group.copy(),
            }
            (general_data if is_overall else all_data).append(entry)

            for _, row in group.iterrows():
                school = str(row.get("school", ""))
                name = str(row.get("name", row.get("pair_name", "")))
                all_rows_for_search.append({
                    "school": school,
                    "name": name,
                    "division": str(division),
                    "flight": str(flight),
                    "category": category,
                    "gender": gender,
                    "filename": csv_path.name,
                })

DIVISION_ORDER = {"1": 0, "2": 1, "3": 2, "4": 3, "4_other": 4}
GENDER_ORDER = {"boys": 0, "girls": 1}
CAT_ORDER = {"singles": 0, "doubles": 1}

all_data.sort(key=lambda x: (
    DIVISION_ORDER.get(x["division"], 9),
    x["flight"],
    CAT_ORDER.get(x["category"], 9),
    GENDER_ORDER.get(x["gender"], 9),
))

general_data.sort(key=lambda x: (
    x["flight"],
    CAT_ORDER.get(x["category"], 9),
    GENDER_ORDER.get(x["gender"], 9),
))

all_schools = sorted(set(r["school"] for r in all_rows_for_search if r["school"]))

# nav_tree[division][ "Boys Singles" ] -> list of (flight, anchor)
# Powers the Division > Singles/Doubles > Flight dropdown nav.
nav_tree = defaultdict(lambda: defaultdict(list))

# general_nav[ "Boys Singles" ] -> list of (flight, anchor)
# Powers the General Rankings > Singles/Doubles > Flight dropdown nav
# (no division level, since these are the cross-division rankings).
general_nav = defaultdict(list)

# Column order for individual ranking tables — reason_below last so it
# doesn't crowd the important numeric columns on the left.
_PREVIEW_COL_ORDER = [
    "rank", "name", "pair_name", "school",
    "division", "flight", "wins", "losses",
    "ts_mu", "power_rating",
    "reachability",
    "sos", "local_sos", "quality_wins",
    "won_after_set1_loss", "vs_weaker_opp", "vs_mid_opp", "vs_top_opp",
    "last_match_date", "reason_below",
]

# Human-readable header labels for the individual ranking tables (team
# tables have their own COL_LABELS map above, built per-entry since it's
# scoped inside the team_html loop). Any column not listed here just uses
# its raw name, same as before.
INDIVIDUAL_COL_LABELS = {
    "power_rating": "Power Rating",
    "won_after_set1_loss": "Won After S1 Loss",
    "vs_weaker_opp": "vs Weaker Opp",
    "vs_mid_opp": "vs Mid Opp",
    "vs_top_opp": "vs Top Opp",
}


def _render_individual_table(entry, anchor_prefix, label_prefix, include_data_division):
    """Shared table-building logic for both the per-division tables
    (tables_html) and the cross-division General Rankings tables
    (general_tables_html). Returns (anchor, section_html)."""
    division = entry["division"]
    flight = entry["flight"]
    category = entry["category"].title()
    gender = entry["gender"].title()
    filename = entry["filename"]
    df = entry["df"]

    preview_cols = [c for c in _PREVIEW_COL_ORDER if c in df.columns]

    anchor = f"{anchor_prefix}flight{flight}_{gender.lower()}_{category.lower()}"
    if include_data_division:
        anchor = f"div{division}_flight{flight}_{gender.lower()}_{category.lower()}"
    label = f"{label_prefix}Flight {flight} · {gender} {category}"

    thead = "<thead><tr>" + "".join(
        f'<th onclick="sortTable(this)" title="{_html_escape_py(col)}">{_html_escape_py(INDIVIDUAL_COL_LABELS.get(col, col))}</th>'
        for col in preview_cols
    ) + "</tr></thead>"

    def _render_cell(col, val):
        escaped = _html_escape_py(val)
        if col == "reason_below":
            return f'<td class="reason-cell">{escaped}</td>'
        return f"<td>{escaped}</td>"

    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(_render_cell(col, row[col]) for col in preview_cols) + "</tr>"
        for _, row in df.head(32).iterrows()
    ) + "</tbody>"

    div_attrs = ""
    if include_data_division:
        div_attr = _norm_division(division)
        div_attrs = f' data-division="{_html_escape_py(div_attr)}"'
    else:
        div_attrs = ' class="general-section"'

    section_html = f"""
    <section id="{anchor}"{div_attrs} data-category="{_html_escape_py(entry['category'])}" data-flight="{_html_escape_py(flight)}">
      <div class="section-header">
        <h2>{_html_escape_py(label)}</h2>
        <a class="dl-btn" href="csv/{filename}">Download CSV</a>
      </div>
      <div class="table-wrap">
        <table class="rankings-table">{thead}{tbody}</table>
      </div>
    </section>
    """
    return anchor, section_html


tables_html = ""
for entry in all_data:
    division = entry["division"]
    gender = entry["gender"].title()
    category = entry["category"].title()
    anchor, section_html = _render_individual_table(
        entry, anchor_prefix="", label_prefix=f"Div {division} · ",
        include_data_division=True,
    )
    tables_html += section_html
    div_attr = _norm_division(division)
    nav_tree[div_attr][f"{gender} {category}"].append((entry["flight"], anchor))

# General Rankings (cross-division, per-flight) — same rendering, no
# division split and not part of the Division 1-4 filter checkboxes.
general_tables_html = ""
for entry in general_data:
    gender = entry["gender"].title()
    category = entry["category"].title()
    anchor, section_html = _render_individual_table(
        entry, anchor_prefix="general_", label_prefix="",
        include_data_division=False,
    )
    general_tables_html += section_html
    general_nav[f"{gender} {category}"].append((entry["flight"], anchor))

# ── Build full CSV data as JSON for JS search ─────────────────────────────────
csv_full_data = {}
for csv_path in sorted(src_dir.glob("*.csv")):
    if csv_path.stem.startswith("team_"):
        continue
    df = pd.read_csv(csv_path)
    if "flight" in df.columns:
        df = df[df["flight"].astype(str).isin(ALLOWED_FLIGHTS)]
    if df.empty:
        continue

    preview_cols = [c for c in _PREVIEW_COL_ORDER if c in df.columns]

    df = df[preview_cols].fillna("")
    csv_full_data[csv_path.stem] = {
        "cols": preview_cols,
        "rows": df.values.tolist(),
    }

schools_json = json.dumps(all_schools)
csv_data_json = json.dumps(csv_full_data)



# ── Build the Division > Singles/Doubles > Flight dropdown nav ───────────────
_DIVISION_SORT = {"1": 0, "2": 1, "3": 2, "4": 3}
_CATEGORY_NAV_ORDER = {"Boys Singles": 0, "Boys Doubles": 1, "Girls Singles": 2, "Girls Doubles": 3}

rankings_dropdown_html = ""
for div_key in sorted(nav_tree.keys(), key=lambda d: _DIVISION_SORT.get(d, 99)):
    cat_map = nav_tree[div_key]
    rankings_dropdown_html += f'<details><summary>Division {_html_escape_py(div_key)}</summary>'
    for cat_label in sorted(cat_map.keys(), key=lambda c: _CATEGORY_NAV_ORDER.get(c, 99)):
        flights_sorted = sorted(cat_map[cat_label], key=lambda x: x[0])
        rankings_dropdown_html += f'<details class="nav-subgroup"><summary>{_html_escape_py(cat_label)}</summary>'
        for flight_val, anchor in flights_sorted:
            rankings_dropdown_html += f'<a href="#{anchor}">Flight {_html_escape_py(flight_val)}</a>'
        rankings_dropdown_html += "</details>"
    rankings_dropdown_html += "</details>"

# ── Build the General Rankings (cross-division) > Singles/Doubles > Flight
#    dropdown nav — same shape as above, minus the Division level. ──────────
general_dropdown_html = ""
for cat_label in sorted(general_nav.keys(), key=lambda c: _CATEGORY_NAV_ORDER.get(c, 99)):
    flights_sorted = sorted(general_nav[cat_label], key=lambda x: x[0])
    general_dropdown_html += f'<details class="nav-subgroup"><summary>{_html_escape_py(cat_label)}</summary>'
    for flight_val, anchor in flights_sorted:
        general_dropdown_html += f'<a href="#{anchor}">Flight {_html_escape_py(flight_val)}</a>'
    general_dropdown_html += "</details>"

edt = timezone(timedelta(hours=-4))
updated = datetime.now(edt).strftime("%B %d, %Y at %I:%M %p EDT")
season_label = f"{SEASON_YEAR} season" if SEASON_YEAR else ""

# Only link to the state-tournament prediction page if predict_state.py
# has actually produced it — avoids a dead/404 nav link if that script
# hasn't run yet (e.g. first-time setup, or it failed/was skipped).
#
# NOTE: this check is now reliable because both GitHub Actions workflows
# run "Build state tournament predictions" (predict_state.py) BEFORE
# "Build HTML site" (this script) — see rank_publish.yml /
# fetch_rank_publish.yml. Previously build_site.py ran first every time,
# so this file never existed yet and the nav link was silently omitted
# on every single run.
_prediction_html_exists = (out_dir / "prediction_of_state.html").exists()
prediction_nav_link = (
    '<a class="nav-predict" href="prediction_of_state.html">&#127942; State Tournament Prediction</a>'
    if _prediction_html_exists else ""
)

# JS port of predict_state.py's power-rating engine (single braces on purpose:
# it is injected into the f-string below as a value, so it isn't re-parsed).
SIM_ENGINE_JS = r'''// ---- Simulate Matchup ------------------------------------------------
// JS port of predict_state.py's power-rating engine. A rating gap is the
// expected game margin (capped at 12 = 6-0 6-0). The gap is inverted into a
// per-POINT win probability, and the point -> game -> set -> best-of-3 match
// distribution is then computed EXACTLY (no Monte Carlo, no randomness).
// Averaging over day-to-day form (SIM_FORM_SD) keeps upsets possible.
// Constants mirror predict_state.py -- if you change one there, change it here.
const SIM_CAP = 12;
const SIM_RIDGE = 0.5;                 // POWER_RIDGE (rating fit)
const SIM_FORM_SD = 3.0;               // FORM_SD (games) -- fit to held-out data
const SIM_SEED_PRIOR_ACCURACY = 0.950;
const SIM_SEED_BLEND_WEIGHT = 0.05;    // backtested: 0.05 minimizes held-out log-loss
// Shape multipliers fit to held-out matches (the exact model over-predicts competitive matches)
const SIM_THREE_SET_SCALE = 0.60;
const SIM_TIEBREAK_SCALE = 0.46;
const SIM_SEVEN_FIVE_SCALE = 0.82;
const SIM_MARGIN_SD = 0.75;            // MARGIN_SD: how tightly the printed line's game margin tracks the rating gap
const SIM_SCORE_SEED = 2026;           // SCORE_SEED: change to reshuffle all predicted scorelines
const SIM_TABLE_STEP = 0.25;
const SIM_EPS = 1e-9;

function simLogit(p) { p = Math.min(Math.max(p, SIM_EPS), 1 - SIM_EPS); return Math.log(p / (1 - p)); }
function simSigmoid(z) { return 1 / (1 + Math.exp(-z)); }
const SIM_SEED_PRIOR_LOGIT = simLogit(SIM_SEED_PRIOR_ACCURACY);

function simComb(n, k) {
  let r = 1;
  for (let i = 1; i <= k; i++) r = r * (n - k + i) / i;
  return r;
}

// P(win a game) with point prob p (ad scoring)
function simGameProb(p) {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  const q = 1 - p;
  return Math.pow(p, 4) * (1 + 4*q + 10*q*q) + 20 * Math.pow(p, 5) * Math.pow(q, 3) / (1 - 2*p*q);
}

// P(win) a first-to-n, win-by-2 points race (7-pt set TB, 10-pt match TB)
function simRaceProb(p, n) {
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  const q = 1 - p;
  let s = 0;
  for (let k = 0; k < n - 1; k++) s += simComb(n - 1 + k, k) * Math.pow(p, n) * Math.pow(q, k);
  s += simComb(2*(n-1), n-1) * Math.pow(p*q, n-1) * p*p / (p*p + q*q);
  return s;
}

// One set's score distribution from A's side: {"6-4": prob, ...}
function simSetDist(p) {
  const g = simGameProb(p), h = 1 - g, t = simRaceProb(p, 7);
  const reach = {'0,0': 1};
  const out = {};
  const add = (k, v) => { out[k] = (out[k] || 0) + v; };
  for (let total = 0; total <= 12; total++) {
    for (let ga = 0; ga <= total; ga++) {
      const gb = total - ga;
      const pr = reach[ga + ',' + gb] || 0;
      if (!pr) continue;
      if (ga === 6 && gb === 6) { add('7-6', pr * t); add('6-7', pr * (1 - t)); continue; }
      const nexts = [[ga + 1, gb, g], [ga, gb + 1, h]];
      for (const [na, nb, w] of nexts) {
        if ((na >= 6 && na - nb >= 2) || (nb >= 6 && nb - na >= 2) || na === 7 || nb === 7) {
          add(na + '-' + nb, pr * w);
        } else {
          reach[na + ',' + nb] = (reach[na + ',' + nb] || 0) + pr * w;
        }
      }
    }
  }
  return out;
}

// 10-pt match tiebreak from A's side; deuce endings lumped into 10-8.
function simSuperTbDist(p) {
  const q = 1 - p;
  const out = {};
  let aExact = 0, bExact = 0;
  for (let k = 0; k < 8; k++) {
    out['10-' + k] = simComb(9 + k, k) * Math.pow(p, 10) * Math.pow(q, k);
    out[k + '-10'] = simComb(9 + k, k) * Math.pow(q, 10) * Math.pow(p, k);
    aExact += out['10-' + k]; bExact += out[k + '-10'];
  }
  const win = simRaceProb(p, 10);
  out['10-8'] = Math.max(0, win - aExact);
  out['8-10'] = Math.max(0, (1 - win) - bExact);
  return out;
}

const simSplit = s => s.split('-').map(Number);
const simFlip = s => { const x = simSplit(s); return x[1] + '-' + x[0]; };

// E[signed game margin of A]; a match tiebreak counts as +/-1 (same
// convention the power rating is fit on).
function simExpMarginForP(p) {
  const sd = simSetDist(p);
  let sWin = 0, eSet = 0;
  for (const k in sd) { const [a, b] = simSplit(k); if (a > b) sWin += sd[k]; eSet += sd[k] * (a - b); }
  const t = simRaceProb(p, 10);
  return 2 * eSet + 2 * sWin * (1 - sWin) * (2 * t - 1);
}

function simSolveP(d) {
  if (d <= 0) return 0.5;
  if (d >= SIM_CAP) return 1;
  let lo = 0.5, hi = 1;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (simExpMarginForP(mid) < d) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// Straight-sets line, winner's side, order-free (merge (6-2,6-3) with (6-3,6-2)
// so identical-set pairs no longer win by default); bigger win shown first.
function simCanon2(s1, s2) {
  const m = s => { const [a, b] = simSplit(s); return a - b; };
  return m(s1) >= m(s2) ? s1 + ' ' + s2 : s2 + ' ' + s1;
}
// Three-set line, winner's side: the set lost, then the bigger win, then the
// tighter deciding set. The two sets won are interchangeable draws, so their
// orderings are merged (else an identical pair like 6-3 6-3 wins by default).
function simCanon3(s1, s2, t3) {
  const [a1, b1] = simSplit(s1);
  const lost = a1 < b1 ? s1 : s2, won = a1 < b1 ? s2 : s1;
  const m = s => { const [a, b] = simSplit(s); return a - b; };
  const [wa, wb] = m(won) >= m(t3) ? [won, t3] : [t3, won];
  return lost + ' ' + wa + ' ' + wb;
}
// Exact best-of-3 summary for favorite A at point prob p >= 0.5.
function simMatchOutcomes(p) {
  const sd = simSetDist(p), tb = sd;  // 3rd set is a real set
  let pw2 = 0, pw3 = 0, pl2 = 0, pl3 = 0, pTb = 0, p75 = 0;
  const aL = {2: {}, 3: {}}, bL = {2: {}, 3: {}};
  const add = (o, k, v) => { o[k] = (o[k] || 0) + v; };
  const isTb = s => s === '7-6' || s === '6-7';
  const is75 = s => s === '7-5' || s === '5-7';
  for (const s1 in sd) {
    for (const s2 in sd) {
      const joint = sd[s1] * sd[s2];
      const [a1, b1] = simSplit(s1), [a2, b2] = simSplit(s2);
      const w1 = a1 > b1, w2 = a2 > b2;
      if (w1 === w2) { if (isTb(s1) || isTb(s2)) pTb += joint; if (is75(s1) || is75(s2)) p75 += joint; }
      if (w1 === w2) {
        if (w1) { pw2 += joint; add(aL[2], simCanon2(s1, s2), joint); }
        else    { pl2 += joint; add(bL[2], simCanon2(simFlip(s1), simFlip(s2)), joint); }
      } else {
        for (const t3 in tb) {
          const j3 = joint * tb[t3];
          if (isTb(s1) || isTb(s2) || isTb(t3)) pTb += j3;
          if (is75(s1) || is75(s2) || is75(t3)) p75 += j3;
          const [x, y] = simSplit(t3);
          if (x > y) { pw3 += j3; add(aL[3], simCanon3(s1, s2, t3), j3); }
          else       { pl3 += j3; add(bL[3], simCanon3(simFlip(s1), simFlip(s2), simFlip(t3)), j3); }
        }
      }
    }
  }
  const top = o => Object.entries(o).filter(e => e[1] > 0).sort((u, v) => v[1] - u[1]);
  return {
    stats: [pw2, pw3, pl2, pl3, pTb, p75],
    aWins: {2: top(aL[2]), 3: top(aL[3])},
    bWins: {2: top(bL[2]), 3: top(bL[3])},
  };
}

let SIM_TABLE = null;
function simTable() {
  if (!SIM_TABLE) {
    SIM_TABLE = [];
    const n = Math.round(SIM_CAP / SIM_TABLE_STEP);
    for (let i = 0; i <= n; i++) SIM_TABLE.push(simMatchOutcomes(simSolveP(i * SIM_TABLE_STEP)));
  }
  return SIM_TABLE;
}

// Interpolated [pw2, pw3, pl2, pl3, pTb, p75] for signed gap d (+ = A favored)
function simStatsAt(d) {
  const tab = simTable();
  const x = Math.min(Math.abs(d), SIM_CAP) / SIM_TABLE_STEP;
  const i = Math.min(Math.floor(x), tab.length - 2);
  const f = x - i;
  const lo = tab[i].stats, hi = tab[i + 1].stats;
  const s = lo.map((v, k) => v + f * (hi[k] - v));
  return d < 0 ? [s[2], s[3], s[0], s[1], s[4], s[5]] : s;
}

// 7-point Gauss-Hermite rule for a standard normal
const SIM_GH = (() => {
  const xs = [0.0, 0.8162878828589647, 1.6735516287674714, 2.6519613568352334];
  const ws = [0.8102646175568073, 0.4256072526101278, 0.05451558281912703, 0.0009717812450995];
  const out = [];
  xs.forEach((x, i) => {
    const w = ws[i] / Math.sqrt(Math.PI);
    if (x === 0) out.push([0, w]);
    else { out.push([Math.SQRT2 * x, w]); out.push([-Math.SQRT2 * x, w]); }
  });
  return out;
})();

function simMixtureStats(d, tau) {
  const acc = [0, 0, 0, 0, 0, 0];
  for (const [z, w] of SIM_GH) {
    const dd = Math.max(-SIM_CAP, Math.min(SIM_CAP, d + tau * z));
    const s = simStatsAt(dd);
    for (let k = 0; k < 6; k++) acc[k] += w * s[k];
  }
  return acc;
}

// Games A is expected to beat B by: rating gap capped at +/-12
function simExpectedMargin(a, b) {
  return Math.max(-SIM_CAP, Math.min(SIM_CAP, a.power - b.power));
}
function simSe(p) { return 1 / Math.sqrt(SIM_RIDGE + Math.max(0, p.wins + p.losses)); }
function simTau(a, b) { return Math.sqrt(SIM_FORM_SD * SIM_FORM_SD + simSe(a) ** 2 + simSe(b) ** 2); }

function simApplySeedPrior(p, a, b) {
  if (SIM_SEED_BLEND_WEIGHT <= 0 || !a.rank || !b.rank || a.rank === b.rank) return p;
  const seedLogit = a.rank < b.rank ? SIM_SEED_PRIOR_LOGIT : -SIM_SEED_PRIOR_LOGIT;
  return simSigmoid((1 - SIM_SEED_BLEND_WEIGHT) * simLogit(p) + SIM_SEED_BLEND_WEIGHT * seedLogit);
}

// P(a beats b) -- the "Win Prob." number shown in the results table.
function matchWinProb(a, b) {
  const s = simMixtureStats(simExpectedMargin(a, b), simTau(a, b));
  return simApplySeedPrior(s[0] + s[1], a, b);
}

// FNV-1a 32-bit hash of the UTF-8 text; mirrors _hash32() in predict_state.py.
function simHash32(text) {
  let h = 2166136261;
  for (const byte of new TextEncoder().encode(text)) h = Math.imul(h ^ byte, 16777619) >>> 0;
  return h >>> 0;
}
// First output of mulberry32(seed) in [0, 1); mirrors _uniform_from_seed().
function simUniform(seed) {
  let a = (seed + 0x6D2B79F5) >>> 0;
  let t = Math.imul(a ^ (a >>> 15), a | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function simMatchupSeed(a, b, winnerIsA) {
  const ka = a.name + '|' + (a.school || ''), kb = b.name + '|' + (b.school || '');
  return simHash32(SIM_SCORE_SEED + '~' + [ka, kb].sort().join('~') + '~' + (winnerIsA ? ka : kb));
}

// Everything about one matchup, exact. Mirrors predict_match_details() in
// predict_state.py: pick the more likely shape (straight sets vs three, using
// the data-calibrated odds), then the most likely exact score in that shape,
// oriented winner-first.
function predictMatchDetails(a, b, winnerIsA) {
  const d = simExpectedMargin(a, b);
  const s = simMixtureStats(d, simTau(a, b));
  const w2 = winnerIsA ? s[0] : s[2], w3 = winnerIsA ? s[1] : s[3];
  const shapeW = {2: w2 + (1 - SIM_THREE_SET_SCALE) * w3, 3: SIM_THREE_SET_SCALE * w3};
  const tab = simTable();
  const idx = Math.min(Math.round(Math.min(Math.abs(d), SIM_CAP) / SIM_TABLE_STEP), tab.length - 1);
  const winnerIsFavorite = (winnerIsA === (d >= 0));
  const src = winnerIsFavorite ? tab[idx].aWins : tab[idx].bWins;
  // Printed line's total game margin should track the expected margin (|d|),
  // floored at 1.5; an underdog winner shows a narrow win.
  const target = winnerIsFavorite ? Math.max(Math.abs(d), 1.5) : 1.5;
  // Step 1: pick the SHAPE (straight sets vs three) with the data-calibrated
  // odds so printed three-setters match real frequency; step 2: sample a line
  // inside that shape, matching the target margin as well as the shape allows.
  const seed = simMatchupSeed(a, b, winnerIsA);
  const p3 = (shapeW[2] + shapeW[3]) > 0 ? shapeW[3] / (shapeW[2] + shapeW[3]) : 0;
  let shapePick = simUniform((seed + 0x1B873593) >>> 0) < p3 ? 3 : 2;
  if (!src[shapePick].length) shapePick = 2;
  const lines = src[shapePick];
  // Straight-set wins are realistically never narrower than ~6-4 6-4 (+4).
  const tgt = shapePick === 2 ? Math.max(target, 4) : target;
  let tot = 0;
  for (const e of lines) tot += e[1];
  const cands = [];
  let totalW = 0;
  for (const [key, p] of lines) {
    const sets = key.split(' ');
    let margin = 0, damp = 1;
    for (const x of sets) {
      const [a1, b1] = simSplit(x);
      margin += a1 - b1;
      if (x === '7-6' || x === '6-7') damp *= SIM_TIEBREAK_SCALE;
      else if (x === '7-5' || x === '5-7') damp *= SIM_SEVEN_FIVE_SCALE;
    }
    const wgt = p / tot * damp * Math.exp(-((margin - tgt) ** 2) / (2 * SIM_MARGIN_SD ** 2));
    cands.push([sets, wgt]);
    totalW += wgt;
  }
  // Replicable draw: same matchup -> same line; different matchups vary.
  let best = null;
  if (totalW > 0) {
    const r = simUniform(seed) * totalW;
    let acc = 0;
    for (const [sets, w] of cands) { acc += w; best = sets; if (r < acc) break; }
  }
  let score = best || ['6-4', '6-4'];
  if (score.length === 3) {
    // Stored as (set lost, won, won) with no order; deal it out in a real
    // sequence: winner loses set 1 or set 2 (real data 51% / 49%), ALWAYS
    // wins the last set, and the two sets won can come in either order.
    let [lost, w1, w2] = score;
    const lostFirst = simUniform((seed + 0x9E3779B9) >>> 0) < 0.5;
    if (simUniform((seed + 0x3C6EF372) >>> 0) < 0.5) [w1, w2] = [w2, w1];
    score = lostFirst ? [lost, w1, w2] : [w1, lost, w2];
  }
  return {
    score: score,
    expMargin: Math.abs(d),
    prob3rd: SIM_THREE_SET_SCALE * (s[1] + s[3]),
  };
}

'''

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=1400">
<title>Michigan High School Tennis Rankings{' — ' + season_label if season_label else ''}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f7fa; color: #1a1a2e; line-height: 1.5; }}
  header {{ background: #1a3a5c; color: white; padding: 2rem 1.5rem 1.5rem; }}
  header h1 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: .4rem; }}
  header p {{ opacity: .8; font-size: .9rem; }}
  nav {{ background: #132d47; padding: .75rem 1.5rem; display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; }}
  .nav-group-label {{ color: #4a90c4; font-size: .7rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; padding: .2rem .5rem .2rem 0; margin-left: .5rem; }}
  .nav-group-label:first-child {{ margin-left: 0; }}
  nav a {{ color: #b8d8f0; text-decoration: none; font-size: .78rem; padding: .2rem .5rem; border-radius: 4px; border: 1px solid rgba(255,255,255,.1); }}
  nav a:hover {{ background: rgba(255,255,255,.12); }}
  .nav-about {{ color: #ffd580; font-size: .8rem; padding: .2rem .6rem; border-radius: 4px; border: 1px solid rgba(255,213,128,.3); text-decoration: none; margin-right: .4rem; }}
  .nav-about:hover {{ background: rgba(255,213,128,.1); }}
  .nav-predict {{ color: #ff9f8a; font-size: .8rem; padding: .2rem .6rem; border-radius: 4px; border: 1px solid rgba(255,159,138,.3); text-decoration: none; margin-right: .4rem; }}
  .nav-predict:hover {{ background: rgba(255,159,138,.1); }}
  .nav-tool {{ color: #a8e6c0; font-size: .8rem; padding: .2rem .6rem; border-radius: 4px; border: 1px solid rgba(168,230,192,.3); text-decoration: none; margin-right: .4rem; cursor: pointer; background: none; font-family: inherit; }}
  .nav-tool:hover {{ background: rgba(168,230,192,.1); }}
  main {{ max-width: 100%; width: 100%; margin: auto; padding: 1rem 1.25rem; }}
  section {{ background: white; border-radius: 10px; padding: 1rem; margin: 0 auto 1rem; max-width: 1600px; box-shadow: 0 1px 4px rgba(0,0,0,.07); }}
  .section-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: .75rem; flex-wrap: wrap; gap: .5rem; }}
  h2 {{ font-size: 1.05rem; font-weight: 600; color: #1a3a5c; }}
  .scoring-dist summary {{ cursor: pointer; }}
  .scoring-dist table {{ margin-top: .4rem; border-collapse: collapse; }}
  .scoring-dist th, .scoring-dist td {{ padding: .1rem .6rem; text-align: left; font-size: .72rem; }}
  .scoring-note {{ font-size: .72rem; color: #5a7a9a; background: #eef4fb; border: 1px solid #c0d4e8; border-radius: 5px; padding: .25rem .6rem; white-space: nowrap; }}
  .dl-btn {{ font-size: .8rem; color: #1a3a5c; text-decoration: none; border: 1px solid #c0d4e8; border-radius: 6px; padding: .3rem .7rem; background: #f8fafc; cursor: pointer; font-family: inherit; }}
  .dl-btn:hover {{ background: #e8f0f8; }}
  .table-wrap {{ overflow-x: auto; width: 100%; }}
  .rankings-table {{ width: 100%; table-layout: auto; border-collapse: collapse; font-size: .78rem; white-space: nowrap; }}
  .rankings-table th {{ background: #1a3a5c; color: white; padding: 5px 8px; text-align: left; font-weight: 500; cursor: pointer; user-select: none; }}
  .rankings-table th:hover {{ background: #245180; }}
  .rankings-table th.asc::after  {{ content: " ▲"; font-size: .65rem; }}
  .rankings-table th.desc::after {{ content: " ▼"; font-size: .65rem; }}
  .rankings-table td {{ padding: 4px 8px; border-bottom: 1px solid #eef0f3; }}
  .rankings-table tr:nth-child(even) td {{ background: #f8fafc; }}
  .rankings-table tr:hover td {{ background: #eef4fb; }}
  .rankings-table td:first-child {{ font-weight: 600; color: #1a3a5c; width: 36px; }}
  .highlight-row td {{ background: #fff3cd !important; font-weight: 600; }}

  /* reason_below column — shared by individual and team tables */
  .reason-cell {{
    font-size: .72rem;
    color: #7a5800;
    background: #fffbee;
    border-left: 3px solid #ffd580;
    padding-left: 8px !important;
    max-width: 280px;
    white-space: normal;
    line-height: 1.4;
  }}

  /* Team table specific */
  .team-table .rank-cell {{ font-weight: 700; color: #1a3a5c; font-size: .85rem; min-width: 48px; }}
  .team-table .pts-cell {{ font-weight: 700; color: #0a7c42; }}
  .team-table tr:first-child .reason-cell {{
    color: #888;
    background: transparent;
    border-left: none;
  }}

  footer {{ text-align: center; color: #888; font-size: .78rem; padding: 2rem; }}

  .tool-panel {{ display: none; }}
  .tool-panel.active {{ display: block; }}

  .search-box {{ position: relative; margin-bottom: 1rem; }}
  .search-box input {{ width: 100%; padding: .6rem 1rem; font-size: 1rem; border: 2px solid #c0d4e8; border-radius: 8px; outline: none; }}
  .search-box input:focus {{ border-color: #1a3a5c; }}
  .autocomplete-list {{ position: absolute; top: 100%; left: 0; right: 0; background: white; border: 1px solid #c0d4e8; border-top: none; border-radius: 0 0 8px 8px; max-height: 220px; overflow-y: auto; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,.1); }}
  .autocomplete-list div {{ padding: .5rem 1rem; cursor: pointer; font-size: .9rem; }}
  .autocomplete-list div:hover {{ background: #eef4fb; }}

  .compare-inputs {{ display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }}
  .compare-inputs .search-box {{ flex: 1; min-width: 200px; }}
  .compare-grid {{ display: grid; gap: 1rem; }}
  .compare-flight {{ background: #f8fafc; border-radius: 8px; padding: 1rem; border: 1px solid #e0e8f0; }}
  .compare-flight h3 {{ font-size: .95rem; color: #1a3a5c; margin-bottom: .75rem; }}
  .compare-cols {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .compare-col {{ flex: 1; min-width: 180px; }}
  .compare-col h4 {{ font-size: .82rem; font-weight: 600; color: #2c5f8a; margin-bottom: .4rem; border-bottom: 2px solid #c0d4e8; padding-bottom: .2rem; }}
  .compare-stat {{ display: flex; justify-content: space-between; font-size: .8rem; padding: .2rem 0; border-bottom: 1px solid #eef0f3; }}
  .compare-stat span:first-child {{ color: #555; }}
  .compare-stat span:last-child {{ font-weight: 600; color: #1a3a5c; }}
  .compare-winner {{ color: #0a7c42 !important; }}
  .compare-teams-list {{ display: flex; flex-wrap: wrap; gap: .4rem; margin: .75rem 0; }}
  .team-chip {{
    display: flex; align-items: center; gap: .35rem;
    background: #eef4fb; border: 1px solid #c0d4e8; border-radius: 16px;
    padding: .25rem .5rem .25rem .8rem; font-size: .8rem; color: #1a3a5c;
  }}
  .team-chip button {{
    background: none; border: none; color: #888; cursor: pointer;
    font-size: 1rem; line-height: 1; padding: 0 .15rem;
  }}
  .team-chip button:hover {{ color: #c0392b; }}
  .compare-flight-rank {{ font-weight: 700; color: #1a3a5c; }}
  .no-entry {{ color: #aaa; font-style: italic; font-size: .82rem; }}
  .sim-summary {{ background:#eef4fb;border:1px solid #c0d4e8;border-radius:8px;padding:.75rem 1rem;margin-bottom:1rem;font-size:.95rem;font-weight:600;color:#1a3a5c; }}
  .sim-flight-group {{ margin-bottom:1.25rem; }}
  .sim-flight-group h4 {{ font-size:.85rem;color:#2c5f8a;margin-bottom:.4rem; }}
  .sim-note {{ font-size:.72rem;color:#888;margin-top:.5rem; }}

  /* Dropdown nav (Individual Rankings + General Rankings + Filter) */
  .dropdown {{ position: relative; }}
  .dropdown-panel {{
    display: none;
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    background: white;
    border: 1px solid #c0d4e8;
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0,0,0,.18);
    padding: .6rem;
    min-width: 230px;
    max-height: 70vh;
    overflow-y: auto;
    z-index: 300;
  }}
  .dropdown-panel.open {{ display: block; }}
  .dropdown-panel a {{
    display: block;
    color: #1a3a5c;
    font-size: .8rem;
    text-decoration: none;
    padding: .25rem .5rem .25rem 1.1rem;
    border-radius: 4px;
  }}
  .dropdown-panel a:hover {{ background: #eef4fb; }}
  .dropdown-panel details {{ margin: .1rem 0; }}
  .dropdown-panel summary {{
    cursor: pointer;
    font-size: .85rem;
    font-weight: 600;
    color: #1a3a5c;
    padding: .3rem .4rem;
    border-radius: 4px;
    list-style: none;
  }}
  .dropdown-panel summary::-webkit-details-marker {{ display: none; }}
  .dropdown-panel summary::before {{ content: "▸ "; font-size: .68rem; color: #4a90c4; }}
  .dropdown-panel details[open] > summary::before {{ content: "▾ "; }}
  .dropdown-panel summary:hover {{ background: #eef4fb; }}
  .dropdown-panel .nav-subgroup {{ margin-left: .9rem; }}

  .filter-panel {{ min-width: 210px; }}
  .filter-group {{ margin-bottom: .6rem; }}
  .filter-group-title {{
    font-size: .7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #4a90c4;
    margin-bottom: .3rem;
  }}
  .filter-panel label {{
    display: flex;
    align-items: center;
    gap: .45rem;
    font-size: .82rem;
    padding: .18rem .2rem;
    color: #1a3a5c;
    cursor: pointer;
  }}
  .filter-actions {{
    display: flex;
    gap: .5rem;
    margin-top: .5rem;
    border-top: 1px solid #eef0f3;
    padding-top: .5rem;
  }}
  .filter-actions button {{
    flex: 1;
    font-size: .74rem;
    padding: .3rem .5rem;
    border: 1px solid #c0d4e8;
    border-radius: 6px;
    background: #f8fafc;
    color: #1a3a5c;
    cursor: pointer;
    font-family: inherit;
  }}
  .filter-actions button:hover {{ background: #eef4fb; }}

  @media (max-width: 600px) {{
    .scoring-note {{
      white-space: normal;
      font-size: .68rem;
      line-height: 1.4;
    }}
    .section-header {{
      flex-direction: column;
      align-items: flex-start;
    }}
  }}
</style>
</head>
<body>
<script async src="https://scripts.simpleanalyticscdn.com/latest.js"></script>
<header>
  <h1>Michigan High School Tennis Seed Prediction{' — ' + season_label if season_label else ''}</h1>
  <p>Updated automatically every day at 4am EDT. Last update: {updated}.</p>
</header>
<nav>
  <a class="nav-about" href="about.html">About &amp; Methodology</a>
  {prediction_nav_link}
  <button class="nav-tool" onclick="showTool('search')">&#128269; School Search</button>
  <button class="nav-tool" onclick="showTool('compare')">&#9878; Team Compare</button>
  <button class="nav-tool" onclick="showTool('simulate')">&#127922; Simulate Matchup</button>
  <div class="dropdown">
    <button class="nav-tool dropdown-toggle" type="button" onclick="toggleDropdown('rankings-dropdown', event)">&#127934; Individual Rankings &#9662;</button>
    <div class="dropdown-panel" id="rankings-dropdown">
      {rankings_dropdown_html}
    </div>
  </div>
  <div class="dropdown">
    <button class="nav-tool dropdown-toggle" type="button" onclick="toggleDropdown('general-dropdown', event)">&#127942; General Rankings &#9662;</button>
    <div class="dropdown-panel" id="general-dropdown">
      {general_dropdown_html}
    </div>
  </div>
  <div class="dropdown">
    <button class="nav-tool dropdown-toggle" type="button" onclick="toggleDropdown('filter-panel', event)">&#128269; Filter &#9662;</button>
    <div class="dropdown-panel filter-panel" id="filter-panel">
      <div class="filter-group">
        <div class="filter-group-title">Division</div>
        <label><input type="checkbox" class="filter-cb filter-division" value="1" checked onchange="applyFilters()"> Division 1</label>
        <label><input type="checkbox" class="filter-cb filter-division" value="2" checked onchange="applyFilters()"> Division 2</label>
        <label><input type="checkbox" class="filter-cb filter-division" value="3" checked onchange="applyFilters()"> Division 3</label>
        <label><input type="checkbox" class="filter-cb filter-division" value="4" checked onchange="applyFilters()"> Division 4</label>
      </div>
      <div class="filter-group">
        <div class="filter-group-title">Category</div>
        <label><input type="checkbox" class="filter-cb filter-category" value="singles" checked onchange="applyFilters()"> Singles</label>
        <label><input type="checkbox" class="filter-cb filter-category" value="doubles" checked onchange="applyFilters()"> Doubles</label>
      </div>
      <div class="filter-group">
        <div class="filter-group-title">Flight</div>
        <label><input type="checkbox" class="filter-cb filter-flight" value="1" checked onchange="applyFilters()"> Flight 1</label>
        <label><input type="checkbox" class="filter-cb filter-flight" value="2" checked onchange="applyFilters()"> Flight 2</label>
        <label><input type="checkbox" class="filter-cb filter-flight" value="3" checked onchange="applyFilters()"> Flight 3</label>
        <label><input type="checkbox" class="filter-cb filter-flight" value="4" checked onchange="applyFilters()"> Flight 4</label>
      </div>
      <div class="filter-group">
        <div class="filter-group-title">Rankings Scope</div>
        <label><input type="checkbox" class="filter-cb" id="filter-overall" checked onchange="applyFilters()"> Show Overall/General Rankings</label>
      </div>
      <div class="filter-actions">
        <button type="button" onclick="selectAllFilters()">Select all</button>
        <button type="button" onclick="clearAllFilters()">Clear all</button>
      </div>
      <p style="font-size:.68rem;color:#888;margin-top:.4rem;">Division/Category/Flight filters apply to division-specific tables only. Use the Overall/General Rankings toggle above to show or hide the cross-division rankings separately.</p>
    </div>
  </div>
</nav>
<main>

<section class="tool-panel" id="panel-search">
  <div class="section-header"><h2>School Search</h2></div>
  <div class="search-box">
    <input type="text" id="school-search-input" placeholder="Type a school name..." autocomplete="off">
    <div class="autocomplete-list" id="school-autocomplete"></div>
  </div>
  <div id="school-search-results"></div>
</section>

<section class="tool-panel" id="panel-compare">
  <div class="section-header"><h2>Team Compare</h2></div>
  <p style="font-size:.82rem;color:#555;margin-bottom:.75rem;">
    Add up to 16 teams, then Compare to see, for every division/flight where
    at least one of them has a ranked player/pair, how they rank against
    each other in that flight.
  </p>
  <div class="compare-inputs">
    <div class="search-box" style="flex:2;">
      <input type="text" id="cmp-input" placeholder="Type a school name..." autocomplete="off">
      <div class="autocomplete-list" id="cmp-auto"></div>
    </div>
    <button type="button" onclick="addCompareTeam(document.getElementById('cmp-input').value)" style="padding:.6rem 1.2rem;background:#f8fafc;color:#1a3a5c;border:1px solid #c0d4e8;border-radius:8px;cursor:pointer;font-size:.9rem;">Add Team</button>
    <button type="button" onclick="runCompare()" style="padding:.6rem 1.2rem;background:#1a3a5c;color:white;border:none;border-radius:8px;cursor:pointer;font-size:.9rem;">Compare</button>
    <button type="button" onclick="runCompare('desc')" title="1st = 8 pts ... 8th = 1 pt, per flight" style="padding:.6rem 1.2rem;background:#0a7c42;color:white;border:none;border-radius:8px;cursor:pointer;font-size:.9rem;">Points: Descending (8&rarr;1)</button>
    <button type="button" onclick="runCompare('match')" title="Points = bracket rounds advanced; bracket size = total teams added, same for every flight" style="padding:.6rem 1.2rem;background:#0a7c42;color:white;border:none;border-radius:8px;cursor:pointer;font-size:.9rem;">Points: Per-Match Bracket</button>
  </div>
  <div class="compare-teams-list" id="compare-teams-list"></div>
  <div id="compare-results"></div>
</section>

<section class="tool-panel" id="panel-simulate">
  <div class="section-header"><h2>Simulate Matchup</h2></div>
  <p style="font-size:.82rem;color:#555;margin-bottom:.75rem;">
    Picks each school's best-ranked player/pair in every flight where both
    schools have one, and predicts each flight using the same win-probability
    and scoreline model as the state tournament predictions.
  </p>
  <div class="compare-inputs">
    <div class="search-box">
      <input type="text" id="sim-input-a" placeholder="Team A..." autocomplete="off">
      <div class="autocomplete-list" id="sim-auto-a"></div>
    </div>
    <div class="search-box">
      <input type="text" id="sim-input-b" placeholder="Team B..." autocomplete="off">
      <div class="autocomplete-list" id="sim-auto-b"></div>
    </div>
    <button onclick="runSimulate()" style="padding:.6rem 1.2rem;background:#1a3a5c;color:white;border:none;border-radius:8px;cursor:pointer;font-size:.9rem;">Simulate</button>
  </div>
  <div id="simulate-results"></div>
</section>

{team_html}
{tables_html}
{general_tables_html}
</main>
<footer>Individual rankings computed using Power Rating + Graph Reachability; matchup and state predictions use the game-margin Power Rating. Team scores use MHSAA flight-finish point system. Data from TennisReporting.com.</footer>

<script>
const SCHOOLS = {schools_json};
const CSV_DATA = {csv_data_json};

function escapeHtml(value) {{
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

function showTool(name) {{
  document.querySelectorAll('.tool-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('panel-' + name);
  if (panel) {{
    panel.classList.add('active');
    panel.scrollIntoView({{behavior: 'smooth', block: 'start'}});
  }}
}}

function toggleDropdown(id, evt) {{
  if (evt) evt.stopPropagation();
  document.querySelectorAll('.dropdown-panel').forEach(p => {{
    if (p.id !== id) p.classList.remove('open');
  }});
  document.getElementById(id).classList.toggle('open');
}}

document.addEventListener('click', e => {{
  if (!e.target.closest('.dropdown')) {{
    document.querySelectorAll('.dropdown-panel').forEach(p => p.classList.remove('open'));
  }}
}});

function toggleShowAllTeams(btn, anchor) {{
  const section = document.getElementById(anchor);
  if (!section) return;
  const extraRows = section.querySelectorAll('.extra-team-row');
  const isHidden = extraRows.length > 0 && extraRows[0].style.display === 'none';
  extraRows.forEach(r => {{ r.style.display = isHidden ? '' : 'none'; }});
  btn.textContent = isHidden ? 'Show Top 10' : 'Show All Teams';
}}

// NOTE: only sections carrying a data-division attribute participate in
// the Division/Category/Flight filter. The cross-division General
// Rankings sections (team and individual) carry class="general-section"
// instead, and are shown/hidden independently via the "Show
// Overall/General Rankings" toggle.
function applyFilters() {{
  const divisions  = Array.from(document.querySelectorAll('.filter-division:checked')).map(cb => cb.value);
  const categories = Array.from(document.querySelectorAll('.filter-category:checked')).map(cb => cb.value);
  const flights     = Array.from(document.querySelectorAll('.filter-flight:checked')).map(cb => cb.value);
  const showOverall = document.getElementById('filter-overall').checked;

  document.querySelectorAll('main > section[data-division]').forEach(sec => {{
    let show = divisions.includes(sec.dataset.division);
    if (show && sec.dataset.category) show = categories.includes(sec.dataset.category);
    if (show && sec.dataset.flight)   show = flights.includes(sec.dataset.flight);
    sec.style.display = show ? '' : 'none';
  }});

  document.querySelectorAll('main > section.general-section').forEach(sec => {{
    sec.style.display = showOverall ? '' : 'none';
  }});
}}

function selectAllFilters() {{
  document.querySelectorAll('.filter-cb').forEach(cb => cb.checked = true);
  applyFilters();
}}

function clearAllFilters() {{
  document.querySelectorAll('.filter-cb').forEach(cb => cb.checked = false);
  applyFilters();
}}

function makeAutocomplete(inputId, listId, onSelect) {{
  const input = document.getElementById(inputId);
  const list  = document.getElementById(listId);

  document.addEventListener('click', e => {{
    if (!list.contains(e.target) && e.target !== input) {{
      list.innerHTML = '';
    }}
  }});

  function renderList(val) {{
    if (!val) {{ list.innerHTML = ''; return; }}
    const q = val.toLowerCase();
    const matches = SCHOOLS.filter(s => s.toLowerCase().includes(q)).slice(0, 12);
    list.innerHTML = '';
    matches.forEach(s => {{
      const div = document.createElement('div');
      div.textContent = s;
      div.addEventListener('mousedown', e => {{
        e.preventDefault();
        input.value = s;
        list.innerHTML = '';
        onSelect(s);
      }});
      list.appendChild(div);
    }});
  }}

  input.addEventListener('input',  () => renderList(input.value));
  input.addEventListener('focus',  () => {{ if (input.value) renderList(input.value); }});

  input.addEventListener('keydown', e => {{
    if (e.key === 'Enter') {{
      const first = list.querySelector('div');
      if (first) {{
        input.value = first.textContent;
        list.innerHTML = '';
        onSelect(input.value);
      }} else {{
        list.innerHTML = '';
        onSelect(input.value);
      }}
    }}
    if (e.key === 'Escape') list.innerHTML = '';
  }});
}}

makeAutocomplete('school-search-input', 'school-autocomplete', school => {{
  doSchoolSearch(school);
}});

document.getElementById('school-search-input').addEventListener('input', function() {{
  if (this.value.trim().length > 1) doSchoolSearch(this.value.trim());
}});

function renderCell(col, val) {{
  const escaped = escapeHtml(val);
  if (col === 'reason_below') return `<td class="reason-cell">${{escaped}}</td>`;
  return `<td>${{escaped}}</td>`;
}}

function doSchoolSearch(school) {{
  if (!school) return;
  const q = school.trim().toLowerCase();
  const results = document.getElementById('school-search-results');
  results.innerHTML = '';

  const byFlight = {{}};
  for (const [stem, data] of Object.entries(CSV_DATA)) {{
    if (stem.startsWith('team_')) continue;
    const cols = data.cols;
    const schoolIdx = cols.indexOf('school');
    if (schoolIdx === -1) continue;

    const category = stem.startsWith('singles') ? 'Singles' : 'Doubles';
    const gender   = stem.includes('_boys_') ? 'Boys' : 'Girls';
    const divIdx    = cols.indexOf('division');
    const flightIdx = cols.indexOf('flight');

    for (const row of data.rows) {{
      if (String(row[schoolIdx]).trim().toLowerCase() !== q) continue;
      const div    = divIdx    >= 0 ? row[divIdx]    : '?';
      const flight = flightIdx >= 0 ? row[flightIdx] : '?';
      const key = `${{gender}} ${{category}} · Div ${{div}} · Flight ${{flight}}`;
      if (!byFlight[key]) byFlight[key] = {{ cols, rows: [] }};
      byFlight[key].rows.push(row);
    }}
  }}

  if (Object.keys(byFlight).length === 0) {{
    results.innerHTML = '<p style="color:#888;margin-top:.5rem;">No results found.</p>';
    return;
  }}

  for (const [label, data] of Object.entries(byFlight).sort()) {{
    const rankIdx = data.cols.indexOf('rank');
    if (rankIdx >= 0) {{
      data.rows.sort((a, b) => Number(a[rankIdx]) - Number(b[rankIdx]));
    }}

    const thead = '<thead><tr>' +
      data.cols.map(c => `<th onclick="sortTable(this)">${{escapeHtml(INDIVIDUAL_COL_LABELS_JS[c] || c)}}</th>`).join('') +
      '</tr></thead>';
    const tbody = '<tbody>' +
      data.rows.map(r =>
        '<tr class="highlight-row">' +
          r.map((v, i) => renderCell(data.cols[i], v)).join('') +
        '</tr>'
      ).join('') +
    '</tbody>';

    results.innerHTML +=
      '<div style="margin-bottom:1.5rem;">' +
        `<h3 style="font-size:.95rem;color:#1a3a5c;margin-bottom:.5rem;">${{escapeHtml(label)}}</h3>` +
        '<div class="table-wrap"><table class="rankings-table">' + thead + tbody + '</table></div>' +
      '</div>';
  }}
}}

// Mirrors INDIVIDUAL_COL_LABELS on the Python side, for the School Search
// results tables which are built purely client-side from CSV_DATA.
const INDIVIDUAL_COL_LABELS_JS = {{
  won_after_set1_loss: 'Won After S1 Loss',
  vs_weaker_opp: 'vs Weaker Opp',
  vs_mid_opp: 'vs Mid Opp',
  vs_top_opp: 'vs Top Opp',
}};

let compareTeams = [];
const MAX_COMPARE_TEAMS = 16;

makeAutocomplete('cmp-input', 'cmp-auto', school => {{
  addCompareTeam(school);
}});

function addCompareTeam(school) {{
  school = (school || '').trim();
  if (!school) return;
  if (compareTeams.some(t => t.toLowerCase() === school.toLowerCase())) {{
    document.getElementById('cmp-input').value = '';
    return;
  }}
  if (compareTeams.length >= MAX_COMPARE_TEAMS) {{
    alert(`You can compare up to ${{MAX_COMPARE_TEAMS}} teams at a time.`);
    return;
  }}
  compareTeams.push(school);
  document.getElementById('cmp-input').value = '';
  document.getElementById('cmp-auto').innerHTML = '';
  renderCompareTeams();
}}

function removeCompareTeam(school) {{
  compareTeams = compareTeams.filter(t => t !== school);
  renderCompareTeams();
}}

function renderCompareTeams() {{
  const list = document.getElementById('compare-teams-list');
  list.innerHTML = compareTeams.map((t, i) =>
    `<span class="team-chip">${{escapeHtml(t)}} <button type="button" onclick="removeCompareTeam(compareTeams[${{i}}])">&times;</button></span>`
  ).join('');
}}

// A school's home division = first non-"overall" division it appears under.
function getSchoolDivision(school) {{
  const q = school.trim().toLowerCase();
  for (const data of Object.values(CSV_DATA)) {{
    const cols = data.cols;
    const s = cols.indexOf('school'), d = cols.indexOf('division');
    if (s === -1 || d === -1) continue;
    for (const row of data.rows) {{
      if (String(row[s]).trim().toLowerCase() !== q) continue;
      const div = String(row[d]);
      if (div !== 'overall') return div;
    }}
  }}
  return null;
}}

// All selected teams in one division -> that division's rankings.
// Mixed divisions -> the cross-division "overall" rankings.
function getCompareScope() {{
  const divs = new Set(compareTeams.map(getSchoolDivision).filter(d => d !== null));
  return divs.size === 1 ? Array.from(divs)[0] : 'overall';
}}

function getBestPerFlight(school, scope) {{
  const q = school.trim().toLowerCase();
  const result = {{}};

  for (const [stem, data] of Object.entries(CSV_DATA)) {{
    if (stem.startsWith('team_')) continue;
    const cols = data.cols;
    const schoolIdx = cols.indexOf('school');
    const rankIdx   = cols.indexOf('rank');
    const divIdx    = cols.indexOf('division');
    const flightIdx = cols.indexOf('flight');
    if (schoolIdx === -1) continue;

    const category = stem.startsWith('singles') ? 'Singles' : 'Doubles';
    const gender   = stem.includes('_boys_') ? 'Boys' : 'Girls';

    for (const row of data.rows) {{
      if (String(row[schoolIdx]).trim().toLowerCase() !== q) continue;
      const div    = divIdx    >= 0 ? String(row[divIdx])    : '?';
      if (div !== scope) continue;
      const flight = flightIdx >= 0 ? String(row[flightIdx]) : '?';
      const rank   = rankIdx   >= 0 ? Number(row[rankIdx])   : 9999;
      const key    = `${{gender}} ${{category}} · Flight ${{flight}}`;
      if (!result[key] || rank < result[key].rank) {{
        result[key] = {{ rank, cols, row }};
      }}
    }}
  }}
  return result;
}}

function statVal(cols, row, col) {{
  const i = cols.indexOf(col);
  return i >= 0 ? row[i] : null;
}}

function ceilLog2(x) {{
  let r = 0, v = 1;
  while (v < x) {{ v *= 2; r++; }}
  return r;
}}

// Option 1: 1st = 8 pts ... 8th = 1 pt, 9th+ = 0.
function pointsDescending(pos) {{
  return pos <= 8 ? 9 - pos : 0;
}}

// Option 2: points = bracket rounds advanced (byes count).
//   8 teams:  3,2,1,1,0,0,0,0
//   9 teams:  4,3,2,2,1,1,1,1,0
// formula: rounds - ceil(log2(place)), where rounds = ceil(log2(teams))
function pointsPerMatch(pos, n) {{
  if (n < 2) return 0;
  return Math.max(0, ceilLog2(n) - ceilLog2(pos));
}}

function runCompare(scoring) {{
  if (compareTeams.length === 0) {{ alert('Add at least one team to compare.'); return; }}

  const scope = getCompareScope();
  const scopeLabel = scope === 'overall'
    ? 'General Rankings (teams are from mixed divisions)'
    : `Division ${{scope}} rankings (all teams are in the same division)`;

  const byFlight = {{}};
  for (const team of compareTeams) {{
    const data = getBestPerFlight(team, scope);
    for (const [key, entry] of Object.entries(data)) {{
      if (!byFlight[key]) byFlight[key] = [];
      byFlight[key].push({{ team, rank: entry.rank, cols: entry.cols, row: entry.row }});
    }}
  }}

  const container = document.getElementById('compare-results');
  const keys = Object.keys(byFlight).sort();
  if (keys.length === 0) {{
    container.innerHTML = '<p style="color:#888">No ranked players/pairs found for the selected teams.</p>';
    return;
  }}

  const totals = {{}};
  compareTeams.forEach(t => {{ totals[t] = 0; }});
  const ptsHeader = scoring === 'desc' ? 'Pts (8→1)' : 'Pts (bracket)';

  let flightsHtml = '';
  for (const key of keys) {{
    const entries = byFlight[key].slice().sort((a, b) => a.rank - b.rank);
    // Bracket size is the same for every flight: total teams added.
    const n = compareTeams.length;
    const suffix = scoring === 'match' ? ` (${{n}}-team bracket)` : '';
    flightsHtml += `<div class="compare-flight"><h3>${{escapeHtml(key + suffix)}}</h3>`;
    flightsHtml += '<div class="table-wrap"><table class="rankings-table"><thead><tr>' +
      '<th>Place</th><th>Rank</th><th>School</th><th>Name</th><th>Record</th><th>SOS</th><th>Power Rating</th>' +
      (scoring ? `<th>${{ptsHeader}}</th>` : '') +
      '</tr></thead><tbody>';
    entries.forEach((e, idx) => {{
      const pos     = idx + 1;
      const name    = statVal(e.cols, e.row, 'pair_name') || statVal(e.cols, e.row, 'name') || '';
      const wins    = statVal(e.cols, e.row, 'wins');
      const losses  = statVal(e.cols, e.row, 'losses');
      const record  = (wins !== null && losses !== null && wins !== '' && losses !== '') ? `${{wins}}-${{losses}}` : '';
      const sos     = statVal(e.cols, e.row, 'sos') ?? '';
      const pr      = statVal(e.cols, e.row, 'power_rating') ?? '';
      let ptsCell = '';
      if (scoring) {{
        const pts = scoring === 'desc' ? pointsDescending(pos) : pointsPerMatch(pos, n);
        totals[e.team] += pts;
        ptsCell = `<td class="pts-cell">${{pts}}</td>`;
      }}
      flightsHtml += '<tr>' +
        `<td class="compare-flight-rank">${{pos}}</td>` +
        `<td>${{escapeHtml(e.rank)}}</td>` +
        `<td>${{escapeHtml(e.team)}}</td>` +
        `<td>${{escapeHtml(name)}}</td>` +
        `<td>${{escapeHtml(record)}}</td>` +
        `<td>${{escapeHtml(sos)}}</td>` +
        `<td>${{escapeHtml(pr)}}</td>` +
        ptsCell + '</tr>';
    }});
    flightsHtml += '</tbody></table></div></div>';
  }}

  let html = `<p class="sim-note" style="margin-bottom:.75rem;">Comparing using: <b>${{escapeHtml(scopeLabel)}}</b></p>`;

  if (scoring) {{
    const sorted = Object.entries(totals).sort((a, b) => b[1] - a[1]);
    const title = scoring === 'desc'
      ? 'Projected Points — Descending (1st = 8 … 8th = 1, per flight)'
      : 'Projected Points — Per-Match Bracket (points = rounds advanced, per flight)';
    html += `<div class="sim-summary">${{escapeHtml(title)}}</div>` +
      '<div class="table-wrap" style="margin-bottom:1rem;"><table class="rankings-table"><thead><tr>' +
      '<th>#</th><th>School</th><th>Projected Points</th></tr></thead><tbody>' +
      sorted.map(([team, pts], i) =>
        `<tr><td>${{i + 1}}</td><td>${{escapeHtml(team)}}</td><td class="pts-cell">${{pts}}</td></tr>`
      ).join('') +
      '</tbody></table></div>';
  }}

  container.innerHTML = html + flightsHtml;
}}

{SIM_ENGINE_JS}
function getSchoolFlights(school) {{
  const q = school.trim().toLowerCase();
  const result = {{}}; // baseKey -> {{ divisions: {{divValue: entry}}, overall: entry|null }}

  for (const [stem, data] of Object.entries(CSV_DATA)) {{
    if (stem.startsWith('team_')) continue;
    const cols = data.cols;
    const schoolIdx = cols.indexOf('school');
    const rankIdx   = cols.indexOf('rank');
    const divIdx    = cols.indexOf('division');
    const flightIdx = cols.indexOf('flight');
    if (schoolIdx === -1) continue;

    const category = stem.startsWith('singles') ? 'Singles' : 'Doubles';
    const gender   = stem.includes('_boys_') ? 'Boys' : 'Girls';

    for (const row of data.rows) {{
      if (String(row[schoolIdx]).trim().toLowerCase() !== q) continue;
      const div    = divIdx    >= 0 ? String(row[divIdx])    : '?';
      const flight = flightIdx >= 0 ? String(row[flightIdx]) : '?';
      const rank   = rankIdx   >= 0 ? Number(row[rankIdx])   : 9999;
      const baseKey = `${{gender}} ${{category}} · Flight ${{flight}}`;
      if (!result[baseKey]) result[baseKey] = {{ divisions: {{}}, overall: null }};
      const entry = {{ rank, cols, row }};
      if (div === 'overall') {{
        if (!result[baseKey].overall || rank < result[baseKey].overall.rank) {{
          result[baseKey].overall = entry;
        }}
      }} else {{
        if (!result[baseKey].divisions[div] || rank < result[baseKey].divisions[div].rank) {{
          result[baseKey].divisions[div] = entry;
        }}
      }}
    }}
  }}
  return result;
}}

// ---- UI wiring ---------------------------------------------------------
let simSelectedA = '';
let simSelectedB = '';
makeAutocomplete('sim-input-a', 'sim-auto-a', s => {{ simSelectedA = s; }});
makeAutocomplete('sim-input-b', 'sim-auto-b', s => {{ simSelectedB = s; }});
document.getElementById('sim-input-a').addEventListener('input', function() {{ simSelectedA = this.value; }});
document.getElementById('sim-input-b').addEventListener('input', function() {{ simSelectedB = this.value; }});

function simField(entry, col, fallback) {{
  const v = statVal(entry.cols, entry.row, col);
  return (v === null || v === '') ? fallback : v;
}}

function simBuildPlayer(entry, schoolLabel) {{
  const rawName = simField(entry, 'pair_name', null);
  return {{
    name: (rawName !== null && rawName !== '') ? rawName : simField(entry, 'name', 'Unknown'),
    school: schoolLabel,
    rank: Number(simField(entry, 'rank', 9999)),
    wins: Number(simField(entry, 'wins', 0)),
    losses: Number(simField(entry, 'losses', 0)),
    power: Number(simField(entry, 'power_rating', 0)),
  }};
}}

function runSimulate() {{
  const a = (simSelectedA || document.getElementById('sim-input-a').value).trim();
  const b = (simSelectedB || document.getElementById('sim-input-b').value).trim();
  if (!a || !b) {{ alert('Enter two school names to simulate.'); return; }}
  if (a.toLowerCase() === b.toLowerCase()) {{ alert('Pick two different schools.'); return; }}

  const flightsA = getSchoolFlights(a);
  const flightsB = getSchoolFlights(b);
  const baseKeys = Object.keys(flightsA).filter(k => flightsB[k]).sort();

  const container = document.getElementById('simulate-results');
  if (baseKeys.length === 0) {{
    container.innerHTML = '<p style="color:#888">No flights found where both schools have a ranked player/pair &mdash; check spelling (boys/girls and singles/doubles are matched separately).</p>';
    return;
  }}

  let winsA = 0, winsB = 0, expWinsA = 0, expWinsB = 0, groupsHtml = '';
  for (const baseKey of baseKeys) {{
    const fa = flightsA[baseKey];
    const fb = flightsB[baseKey];

    // Same division for both schools in this flight -> compare using
    // that division's own ranking. Different (or missing) divisions ->
    // fall back to each school's cross-division General Ranking entry,
    // so ranks are being compared on the same scale.
    let entryA = null, entryB = null, sourceLabel = '';
    const sharedDivisions = Object.keys(fa.divisions).filter(d => fb.divisions[d]);
    if (sharedDivisions.length > 0) {{
      const div = sharedDivisions.sort()[0];
      entryA = fa.divisions[div];
      entryB = fb.divisions[div];
      sourceLabel = `Div ${{div}}`;
    }} else if (fa.overall && fb.overall) {{
      entryA = fa.overall;
      entryB = fb.overall;
      sourceLabel = 'General Ranking (cross-division)';
    }}

    if (!entryA || !entryB) continue;

    const key = `${{baseKey}} · ${{sourceLabel}}`;
    const pa = simBuildPlayer(entryA, a);
    const pb = simBuildPlayer(entryB, b);
    const p = matchWinProb(pa, pb);
    const winnerIsA = p >= 0.5;
    const details = predictMatchDetails(pa, pb, winnerIsA);
    if (winnerIsA) winsA++; else winsB++;
    expWinsA += p;
    expWinsB += (1 - p);

    const winnerName = winnerIsA ? pa.name : pb.name;
    const pFav = Math.max(p, 1 - p);

    groupsHtml += '<div class="sim-flight-group">' +
      '<h4>' + escapeHtml(key) + '</h4>' +
      '<table class="rankings-table"><thead><tr>' +
      '<th>Matchup</th><th>Predicted Winner</th><th>Predicted Score</th><th>Win Prob.</th><th>Fav. By (games)</th><th>Goes to 3rd Set</th>' +
      '</tr></thead><tbody><tr>' +
      '<td>' + escapeHtml(pa.name) + ' (#' + pa.rank + ', ' + escapeHtml(a) + ') vs ' +
        escapeHtml(pb.name) + ' (#' + pb.rank + ', ' + escapeHtml(b) + ')</td>' +
      '<td><b>' + escapeHtml(winnerName) + '</b></td>' +
      '<td>' + escapeHtml(details.score.join(' ')) + '</td>' +
      '<td>' + (pFav*100).toFixed(0) + '%</td>' +
      '<td>' + details.expMargin.toFixed(1) + '</td>' +
      '<td>' + (details.prob3rd*100).toFixed(0) + '%</td>' +
      '</tr></tbody></table></div>';
  }}

  if (winsA === 0 && winsB === 0) {{
    container.innerHTML = '<p style="color:#888">No comparable flights found &mdash; neither school has an overlapping division or a General Ranking entry for the same flight.</p>';
    return;
  }}

  const summary = '<div class="sim-summary">Projected result: ' + escapeHtml(a) + ' ' + winsA +
    ' (' + expWinsA.toFixed(2) + ') &ndash; ' + winsB + ' (' + expWinsB.toFixed(2) + ') ' + escapeHtml(b) + '</div>' +
    '<p class="sim-note">Based on ' + (winsA + winsB) + ' flight' + ((winsA + winsB) === 1 ? '' : 's') +
    ' where both schools have a ranked player/pair. Same-division flights are compared using that ' +
    'division\\'s own ranking; flights where the schools are in different divisions use each school\\'s ' +
    'cross-division General Ranking entry instead. Everything comes from the power rating: the gap between two ' +
    'ratings is how many games the better player is expected to win by (12 = 6-0 6-0), converted into exact ' +
    'win, score and third-set odds (form noise and a small seed-history blend were fit to held-out matches). ' +
    'Numbers in parentheses are the expected (decimal) ' +
    'flight total, summing each flight\\'s win probability instead of just the predicted winner.</p>';

  container.innerHTML = summary + groupsHtml;
}}

function equalizeSectionWidths() {{
  const tables = document.querySelectorAll('.rankings-table');
  let maxContentWidth = 0;
  tables.forEach(t => {{
    const prevWidth = t.style.width;
    t.style.width = 'max-content';
    maxContentWidth = Math.max(maxContentWidth, t.scrollWidth);
    t.style.width = prevWidth;
  }});
  window._maxTableContentWidth = maxContentWidth;
  applySectionWidth();
}}

function applySectionWidth() {{
  const contentWidth = window._maxTableContentWidth || 0;
  if (!contentWidth) return;
  const sectionPadding = 34; // section's left+right padding (1rem each side) plus a hair of slack
  const desired = contentWidth + sectionPadding;
  const viewportCap = Math.max(320, window.innerWidth - 40); // leave a small page margin
  const finalWidth = Math.min(desired, viewportCap);
  document.querySelectorAll('main > section').forEach(sec => {{
    sec.style.maxWidth = finalWidth + 'px';
  }});
}}

window.addEventListener('load', equalizeSectionWidths);
window.addEventListener('resize', applySectionWidth);

function sortTable(th) {{
  const tbody = th.closest('table').querySelector('tbody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  const col   = Array.from(th.parentElement.children).indexOf(th);
  const asc   = !th.classList.contains('asc');
  th.closest('thead').querySelectorAll('th').forEach(h => h.classList.remove('asc', 'desc'));
  th.classList.add(asc ? 'asc' : 'desc');

  rows.sort((a, b) => {{
    const av = a.cells[col].textContent.trim();
    const bv = b.cells[col].textContent.trim();
    // Try to sort by a LEADING number in the cell (handles plain numbers
    // like "12.5" as well as our "3/5 (60%)" / "5-0 (100%)" style
    // formatted stats, which sort by the leading win-count number).
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</html>"""

# Output filename: only when a custom year was typed into the workflow
# (YEAR_OVERRIDE is set and non-blank) do we write {year}.html instead of
# index.html, so the main index page is left untouched by one-off year runs.
_year_override = os.environ.get("YEAR_OVERRIDE", "").strip()
output_name = f"{SEASON_YEAR}.html" if _year_override and SEASON_YEAR else "index.html"

(out_dir / output_name).write_text(html, encoding="utf-8")
print(f"Built docs/{output_name} with {len(all_data)} division section(s), "
      f"{len(general_data)} general-ranking section(s) (season: {SEASON_YEAR})")
