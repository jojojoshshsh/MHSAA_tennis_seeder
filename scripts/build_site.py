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
    team_data.append({"gender": gender, "division": division, "df": df.head(10)})

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
    label = f"Top 10 Teams · {entry['gender']} Division {entry['division']}"
    anchor = f"team_{entry['gender'].lower()}_div{entry['division'].replace(' ','')}"
    df = entry["df"]
    div_attr = _norm_division(entry["division"])

    COL_LABELS = {
        "rank": "Rank",
        "school": "School",
        "total_points": "Total Pts",
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

    cols = list(df.columns)

    tbody_rows = ""
    for _, row in df.iterrows():
        cells = ""
        for col in cols:
            val = _html_escape_py(row[col])
            if col == "reason_below":
                cells += f'<td class="reason-cell">{val}</td>'
            elif col == "rank":
                cells += f'<td class="rank-cell">{val}</td>'
            elif col == "total_points":
                cells += f'<td class="pts-cell">{val}</td>'
            else:
                cells += f"<td>{val}</td>"
        tbody_rows += f"<tr>{cells}</tr>"

    team_html += f"""
    <section id="{anchor}" data-division="{_html_escape_py(div_attr)}">
      <div class="section-header">
        <h2>{_html_escape_py(label)}</h2>
        <span class="scoring-note">Points: 1st=12.5 · 2nd=10 · 3rd–4th=7.5 · 5th–8th=5 · 9th–16th=2.5 · 17th–32nd=1</span>
      </div>
      <div class="table-wrap"><table class="rankings-table team-table"><thead><tr>{"".join(
          f'<th onclick="sortTable(this)" title="{_html_escape_py(col)}">{_html_escape_py(COL_LABELS.get(col, col))}</th>'
          for col in cols
      )}</tr></thead><tbody>{tbody_rows}</tbody></table></div>
    </section>
    """

# ── Load all individual CSVs ──────────────────────────────────────────────────
all_data = []
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
            entry = {
                "division": str(division),
                "flight": str(flight),
                "category": category,
                "gender": gender,
                "filename": csv_path.name,
                "df": group.copy(),
            }
            all_data.append(entry)

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
    GENDER_ORDER.get(x["gender"], 9),
    CAT_ORDER.get(x["category"], 9),
))

all_schools = sorted(set(r["school"] for r in all_rows_for_search if r["school"]))

# nav_tree[division][ "Boys Singles" ] -> list of (flight, anchor)
# Powers the Division > Singles/Doubles > Flight dropdown nav.
nav_tree = defaultdict(lambda: defaultdict(list))

# Column order for individual ranking tables — reason_below last so it
# doesn't crowd the important numeric columns on the left.
_PREVIEW_COL_ORDER = [
    "rank", "name", "pair_name", "school",
    "division", "flight", "wins", "losses",
    "TGRS", "TGRS_scaled", "ts_rating", "ts_mu", "ts_sigma",
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
    "won_after_set1_loss": "Won After S1 Loss",
    "vs_weaker_opp": "vs Weaker Opp",
    "vs_mid_opp": "vs Mid Opp",
    "vs_top_opp": "vs Top Opp",
}

tables_html = ""
for entry in all_data:
    division = entry["division"]
    flight = entry["flight"]
    category = entry["category"].title()
    gender = entry["gender"].title()
    filename = entry["filename"]
    df = entry["df"]

    preview_cols = [c for c in _PREVIEW_COL_ORDER if c in df.columns]

    anchor = f"div{division}_flight{flight}_{gender.lower()}_{category.lower()}"
    label = f"Div {division} · Flight {flight} · {gender} {category}"

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

    div_attr = _norm_division(division)

    tables_html += f"""
    <section id="{anchor}" data-division="{_html_escape_py(div_attr)}" data-category="{_html_escape_py(entry['category'])}" data-flight="{_html_escape_py(flight)}">
      <div class="section-header">
        <h2>{_html_escape_py(label)}</h2>
        <a class="dl-btn" href="csv/{filename}">Download CSV</a>
      </div>
      <div class="table-wrap">
        <table class="rankings-table">{thead}{tbody}</table>
      </div>
    </section>
    """
    nav_tree[div_attr][f"{gender} {category}"].append((flight, anchor))

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

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
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
  .scoring-note {{ font-size: .72rem; color: #5a7a9a; background: #eef4fb; border: 1px solid #c0d4e8; border-radius: 5px; padding: .25rem .6rem; white-space: nowrap; }}
  .dl-btn {{ font-size: .8rem; color: #1a3a5c; text-decoration: none; border: 1px solid #c0d4e8; border-radius: 6px; padding: .3rem .7rem; }}
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
  .no-entry {{ color: #aaa; font-style: italic; font-size: .82rem; }}

  /* Dropdown nav (Individual Rankings + Filter) */
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
  <div class="dropdown">
    <button class="nav-tool dropdown-toggle" type="button" onclick="toggleDropdown('rankings-dropdown', event)">&#127934; Individual Rankings &#9662;</button>
    <div class="dropdown-panel" id="rankings-dropdown">
      {rankings_dropdown_html}
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
      <div class="filter-actions">
        <button type="button" onclick="selectAllFilters()">Select all</button>
        <button type="button" onclick="clearAllFilters()">Clear all</button>
      </div>
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
  <div class="compare-inputs">
    <div class="search-box">
      <input type="text" id="cmp-input-a" placeholder="Team A..." autocomplete="off">
      <div class="autocomplete-list" id="cmp-auto-a"></div>
    </div>
    <div class="search-box">
      <input type="text" id="cmp-input-b" placeholder="Team B..." autocomplete="off">
      <div class="autocomplete-list" id="cmp-auto-b"></div>
    </div>
    <button onclick="runCompare()" style="padding:.6rem 1.2rem;background:#1a3a5c;color:white;border:none;border-radius:8px;cursor:pointer;font-size:.9rem;">Compare</button>
  </div>
  <div id="compare-results"></div>
</section>

{team_html}
{tables_html}
</main>
<footer>Individual rankings computed using TrueSkill + Graph Reachability (TGRS). Team scores use MHSAA flight-finish point system. Data from TennisReporting.com.</footer>

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

function applyFilters() {{
  const divisions  = Array.from(document.querySelectorAll('.filter-division:checked')).map(cb => cb.value);
  const categories = Array.from(document.querySelectorAll('.filter-category:checked')).map(cb => cb.value);
  const flights    = Array.from(document.querySelectorAll('.filter-flight:checked')).map(cb => cb.value);

  document.querySelectorAll('main > section[data-division]').forEach(sec => {{
    let show = divisions.includes(sec.dataset.division);
    if (show && sec.dataset.category) show = categories.includes(sec.dataset.category);
    if (show && sec.dataset.flight)   show = flights.includes(sec.dataset.flight);
    sec.style.display = show ? '' : 'none';
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
      if (!String(row[schoolIdx]).toLowerCase().includes(q)) continue;
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

let selectedA = '';
let selectedB = '';

makeAutocomplete('cmp-input-a', 'cmp-auto-a', s => {{ selectedA = s; }});
makeAutocomplete('cmp-input-b', 'cmp-auto-b', s => {{ selectedB = s; }});

document.getElementById('cmp-input-a').addEventListener('input', function() {{ selectedA = this.value; }});
document.getElementById('cmp-input-b').addEventListener('input', function() {{ selectedB = this.value; }});

function getBestPerFlight(school) {{
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
      if (!String(row[schoolIdx]).toLowerCase().includes(q)) continue;
      const div    = divIdx    >= 0 ? String(row[divIdx])    : '?';
      const flight = flightIdx >= 0 ? String(row[flightIdx]) : '?';
      const rank   = rankIdx   >= 0 ? Number(row[rankIdx])   : 9999;
      const key    = `${{gender}} ${{category}} · Div ${{div}} · Flight ${{flight}}`;
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

function runCompare() {{
  const a = (selectedA || document.getElementById('cmp-input-a').value).trim();
  const b = (selectedB || document.getElementById('cmp-input-b').value).trim();
  if (!a || !b) {{ alert('Enter two school names to compare.'); return; }}

  const dataA   = getBestPerFlight(a);
  const dataB   = getBestPerFlight(b);
  const allKeys = [...new Set([...Object.keys(dataA), ...Object.keys(dataB)])].sort();

  // reason_below is context-specific to each table and not meaningful
  // in a side-by-side compare view, so exclude it here. won_after_set1_loss
  // and the vs_*_opp records are formatted strings (not plain numbers), so
  // they're also excluded from this numeric-style compare view.
  const SHOW_COLS = [
    'rank', 'name', 'pair_name',
    'wins', 'losses',
    'TGRS', 'TGRS_scaled', 'ts_rating', 'ts_mu',
    'reachability',
    'sos', 'local_sos', 'quality_wins',
    'last_match_date'
  ];

  const LOWER_IS_BETTER = new Set(['rank', 'ts_sigma']);

  const container = document.getElementById('compare-results');
  if (allKeys.length === 0) {{
    container.innerHTML = '<p style="color:#888">No data found for either school.</p>';
    return;
  }}

  let html = '<div class="compare-grid">';
  for (const key of allKeys) {{
    const ea = dataA[key];
    const eb = dataB[key];
    html += `<div class="compare-flight"><h3>${{escapeHtml(key)}}</h3><div class="compare-cols">`;

    for (const [label, entry, other] of [[a, ea, eb], [b, eb, ea]]) {{
      html += `<div class="compare-col"><h4>${{escapeHtml(label)}}</h4>`;

      if (!entry) {{
        html += '<div class="no-entry">Not ranked in this slot</div>';
      }} else {{
        for (const col of SHOW_COLS) {{
          const v  = statVal(entry.cols, entry.row, col);
          const vo = other ? statVal(other.cols, other.row, col) : null;
          if (v === null || v === '') continue;

          const vn  = parseFloat(v);
          const von = parseFloat(vo);
          const better = !isNaN(vn) && !isNaN(von) && (
            LOWER_IS_BETTER.has(col) ? vn < von : vn > von
          );

          html +=
            '<div class="compare-stat">' +
              `<span>${{escapeHtml(col)}}</span>` +
              `<span class="${{better ? 'compare-winner' : ''}}">${{escapeHtml(v)}}</span>` +
            '</div>';
        }}
      }}
      html += '</div>';
    }}
    html += '</div></div>';
  }}
  html += '</div>';
  container.innerHTML = html;
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

(out_dir / "index.html").write_text(html, encoding="utf-8")
print(f"Built docs/index.html with {len(all_data)} sections (season: {SEASON_YEAR})")
