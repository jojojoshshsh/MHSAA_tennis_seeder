#!/usr/bin/env python3
"""
Update MHSAA boys-tennis (Lower Peninsula) divisions.

Scrapes the school -> division assignments for a school year from mhsaa.com,
matches the website's school names to the names in ``correct_divisions.csv``,
and writes ``correct_divisions_<year>.csv`` (e.g. ``correct_divisions_2026-27.csv``)
plus a human-readable report of everything that changed or could not be matched.

Usage (from the repo root or from inside ``data/``):

    python data/update_divisions.py                  # year is inferred from today's date
    python data/update_divisions.py --year 2026-27
    python data/update_divisions.py --html-file saved_page.html   # parse a saved page, no network

Dependencies:  pip install requests beautifulsoup4 lxml
Optional:      pip install playwright && playwright install chromium
               (only used if the division list turns out to be rendered by JavaScript)

How the data is found
---------------------
1. The Regional assignments page
   https://www.mhsaa.com/sports/boys-tennis/assignments/regional?uplpcode=lp&year=<year>
   only *links* to the "Division List (LP)" page on my.mhsaa.com, so the script
   follows that link (falling back to the standard URL pattern if it is missing).
2. The page is parsed with several layout strategies (table with a Division column,
   tables/lists grouped under "Division N" headings, one column per division, ...).
3. The division data is actually loaded by the page from an MHSAA API
   (my.mhsaa.com/DesktopModules/MHSAA-Endpoint/API/Tournament/EarlyRound?InstanceId=...), so if
   the static HTML has no usable data the script tries any InstanceIds found in the HTML, follows
   iframes, and finally renders the pages in headless Chromium (Playwright) while capturing the
   API responses. JSON and HTML responses are both understood.

Nothing is written unless at least ``--min-schools`` schools were scraped, so a layout
change on the website makes the run fail loudly instead of silently producing a bad CSV.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCRIPT_DIR = Path(__file__).resolve().parent

REGIONAL_URL = "https://www.mhsaa.com/sports/boys-tennis/assignments/regional"
DIVISION_LIST_URL = "https://my.mhsaa.com/Sports/Boys-Tennis/School-Division-List-{year}-LP"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
API_URL = "https://my.mhsaa.com/DesktopModules/MHSAA-Endpoint/API/Tournament/EarlyRound?InstanceId={uuid}"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
VALID_DIVISIONS = {1, 2, 3, 4}

# Short names that are just another name for a school already in the CSV.
# These rows are dropped so each school is matched only once.
ALIASES = {
    "Bridgman": "New Buffalo / Bridgman / Lake Michigan Catholic (LMC)",
    "Lake Fenton": "Lake Fenton/Linden",
}


# --------------------------------------------------------------------------- #
# Year handling
# --------------------------------------------------------------------------- #
def default_school_year(today: dt.date | None = None) -> str:
    """MHSAA announces next year's classifications in April, so from April on we
    target the school year that starts in the current calendar year."""
    today = today or dt.date.today()
    start = today.year if today.month >= 4 else today.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #
def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return session


def fetch(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_rendered(url: str, timeout_ms: int = 60_000) -> tuple[list[str], list[tuple[str, str]]]:
    """Render ``url`` in headless Chromium.

    Returns (html of the page and every frame, [(url, body)] of every API/JSON response the
    page triggered). The division data is loaded by the page from an MHSAA API endpoint, so
    capturing those responses is the most reliable way to get it.
    """
    from playwright.sync_api import sync_playwright  # imported lazily: optional dependency

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            responses = []
            page.on("response", lambda r: responses.append(r))
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_timeout(2000)
            htmls = [frame.content() for frame in page.frames]
            captured = []
            for r in responses:
                ctype = r.headers.get("content-type", "")
                if "/API/" in r.url or "json" in ctype:
                    try:
                        captured.append((r.url, r.text()))
                    except Exception:  # noqa: BLE001 - body may be unavailable (redirects etc.)
                        pass
            return htmls, captured
        finally:
            browser.close()


def save_debug(debug_dir: Path | None, name: str, html: str) -> None:
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / name).write_text(html, encoding="utf-8")


def find_division_list_link(html: str, base_url: str, year: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        if re.search(r"division\s*list", text, re.I) and re.search(r"\bLP\b", text):
            candidates.append((year in text, urljoin(base_url, a["href"])))
    candidates.sort(key=lambda c: not c[0])  # prefer the link that mentions our year
    return candidates[0][1] if candidates else None


# --------------------------------------------------------------------------- #
# Parsing: HTML -> [(school, division)]
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(?:lp\s+)?division\s*([1-4])\b", re.I)
_SKIP_CELLS = {"school", "schools", "team", "teams", "name", "member school", "enrollment"}


def _heading_division(text: str) -> int | None:
    """'Division 2', 'LP Division 2 (72 schools)' -> 2.  Long text is never a heading."""
    text = " ".join(text.split())
    if len(text) > 40:
        return None
    m = _HEADING_RE.match(text)
    return int(m.group(1)) if m else None


def _cell_division(text: str) -> int | None:
    m = re.search(r"\b([1-4])\b", text)
    return int(m.group(1)) if m else None


def _context_division(el) -> int | None:
    """Closest 'Division N' heading appearing before ``el`` in the document."""
    for prev in el.find_all_previous(True, limit=60):
        value = _heading_division(prev.get_text(" ", strip=True))
        if value:
            return value
    return None


def _rows(table):
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
        yield [c.get_text(" ", strip=True) for c in cells]


def _from_tables(soup) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for table in soup.find_all("table"):
        if table.find("table"):  # layout table wrapping other tables
            continue
        caption = table.find("caption")
        current = _heading_division(caption.get_text()) if caption else None
        current = current or _context_division(table)
        school_col = div_col = None
        col_div: dict[int, int] = {}  # layout: one column per division
        for cells in _rows(table):
            nonempty = [c for c in cells if c]
            if not nonempty:
                continue
            if school_col is None and not col_div:
                lowered = [c.lower().strip() for c in cells]
                s = [i for i, c in enumerate(lowered) if re.search(r"school|team|member", c)]
                d = [i for i, c in enumerate(lowered) if re.fullmatch(r"(?:lp\s*)?(?:div(?:ision)?\.?|class)", c)]
                if s and d:  # header: School | Division
                    school_col, div_col = s[0], d[0]
                    continue
                heads = {i: _heading_division(c) for i, c in enumerate(cells)}
                heads = {i: v for i, v in heads.items() if v}
                if len(heads) >= 2:  # header: Division 1 | Division 2 | ...
                    col_div = heads
                    continue
            if col_div:
                out += [(cells[i], v) for i, v in col_div.items() if i < len(cells) and cells[i]]
            elif school_col is not None:
                if max(school_col, div_col) < len(cells) and cells[school_col]:
                    value = _cell_division(cells[div_col])
                    if value:
                        out.append((cells[school_col], value))
            elif len(cells) >= 2 and re.fullmatch(r"[1-4]", cells[-1].strip()) and cells[0]:
                out.append((cells[0], int(cells[-1])))  # header-less: School | ... | 2
            elif len(nonempty) == 1 and _heading_division(nonempty[0]):
                current = _heading_division(nonempty[0])  # spanning "Division 2" row
            elif current and cells[0] and cells[0].lower() not in _SKIP_CELLS:
                out.append((cells[0], current))
    return out


def _from_lists(soup) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for lst in soup.find_all(["ul", "ol"]):
        if lst.find(["ul", "ol"]):
            continue
        division = _context_division(lst)
        if not division:
            continue
        for li in lst.find_all("li"):
            text = li.get_text(" ", strip=True)
            if text and len(text) <= 60 and not re.search(r"https?:|@", text):
                out.append((text, division))
    return out


def _from_text(soup) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    current = None
    for line in soup.get_text("\n").splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        if re.match(r"^(quick links|privacy policy|©)", line, re.I):
            current = None
        elif _heading_division(line):
            current = _heading_division(line)
        elif (current and 3 <= len(line) <= 60 and line.lower() not in _SKIP_CELLS
              and not re.search(r"https?:|@|:\s", line)):
            out.append((line, current))
    return out


def dedupe(pairs: list[tuple[str, int]]) -> tuple[dict[str, int], list[str]]:
    """First occurrence wins; returns (school -> division, conflict messages)."""
    result: dict[str, int] = {}
    conflicts: list[str] = []
    for school, division in pairs:
        school = " ".join(school.split())
        if division not in VALID_DIVISIONS or not school:
            continue
        if school in result and result[school] != division:
            conflicts.append(f"{school}: listed in Division {result[school]} and Division {division}")
        result.setdefault(school, division)
    return result, conflicts


def extract_pairs(html: str) -> tuple[dict[str, int], list[str]]:
    """Try every layout strategy and keep whichever finds the most schools.
    Returns ({school: division}, conflicts)."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    best: dict[str, int] = {}
    best_conflicts: list[str] = []
    for strategy in (_from_tables, _from_lists, _from_text):
        if strategy is _from_text and len(best) >= 20:
            break  # plain-text scraping is noisy: only a last resort
        found, conflicts = dedupe(strategy(soup))
        if len(found) > len(best):
            best, best_conflicts = found, conflicts
    return best, best_conflicts


_DIV_TEXT_RE = re.compile(r"division\s*([1-4])\b", re.I)
_SCHOOL_KEY_RE = re.compile(r"school(?:_?name)?|team(?:_?name)?", re.I)
_GENERIC_NAME_KEYS = ("SchoolName", "Name", "DisplayName", "Title")


def _json_division(node: dict) -> int | None:
    """Division declared by a JSON object: {'Division': 2}, {'Division': 'Division 2'},
    or any short string such as 'Division 2 Regional 5'."""
    for key, value in node.items():
        if isinstance(value, bool):
            continue
        if re.search(r"division|^div$", key, re.I):
            m = re.search(r"\b([1-4])\b", str(value))
            if m:
                return int(m.group(1))
        elif isinstance(value, str) and len(value) <= 80:
            m = _DIV_TEXT_RE.search(value)
            if m:
                return int(m.group(1))
    return None


def _name_from(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in _GENERIC_NAME_KEYS:
            if isinstance(value.get(key), str):
                return value[key]
    return None


def _walk_json(node, division: int | None, out: list[tuple[str, int]]) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_json(item, division, out)
        return
    if not isinstance(node, dict):
        return
    own = _json_division(node)
    division = own or division
    school = None
    for key, value in node.items():
        if _SCHOOL_KEY_RE.fullmatch(key) and _name_from(value):
            school = _name_from(value)
    if not school and own and any(re.search(r"division|^div$", k, re.I) for k in node):
        # {"Name": "Allen Park", "Division": 2}
        name = _name_from(node)
        if name and not re.match(r"(?i)\s*(regional|district|division|early)", name):
            school = name
    if school and division:
        out.append((school, division))
    for key, value in node.items():
        if isinstance(value, list) and _SCHOOL_KEY_RE.search(key) and value \
                and all(isinstance(x, str) for x in value):
            if division:
                out += [(x, division) for x in value]
        elif isinstance(value, (dict, list)):
            _walk_json(value, division, out)
        elif isinstance(value, str) and "<" in value and ("<table" in value or "<li" in value):
            html_found, _ = extract_pairs(value)  # HTML fragment embedded in JSON
            out += [(k, v) for k, v in html_found.items()]


def extract_any(text: str) -> tuple[dict[str, int], list[str]]:
    """Parse an API response / page body that may be JSON or HTML."""
    stripped = text.lstrip()
    if stripped[:1] in "{[":
        try:
            data = json.loads(stripped)
        except ValueError:
            data = None
        if data is not None:
            pairs: list[tuple[str, int]] = []
            _walk_json(data, None, pairs)
            return dedupe(pairs)
    return extract_pairs(text)


def iframe_urls(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = [urljoin(base_url, f["src"]) for f in soup.find_all("iframe", src=True)]
    return [u for u in urls if "googletagmanager" not in u]


# --------------------------------------------------------------------------- #
# Scraping orchestration
# --------------------------------------------------------------------------- #
def scrape(year: str, min_schools: int, use_browser: bool, debug_dir: Path | None):
    """Returns ({website school name: division}, source description, conflicts)."""
    session = make_session()
    best: dict[str, int] = {}
    best_conflicts: list[str] = []
    best_source = ""

    def consider(found, conflicts, source) -> bool:
        nonlocal best, best_conflicts, best_source
        if len(found) > len(best):
            best, best_conflicts, best_source = found, conflicts, source
        return len(best) >= min_schools

    regional_url = f"{REGIONAL_URL}?uplpcode=lp&year={year}"
    print(f"Fetching {regional_url}")
    regional_html = fetch(session, regional_url)
    save_debug(debug_dir, "regional.html", regional_html)
    if consider(*extract_pairs(regional_html), regional_url):
        return best, best_source, best_conflicts

    # The page loads its data from an API keyed by an InstanceId; try any ids embedded in the HTML.
    uuids = list(dict.fromkeys(u.lower() for u in UUID_RE.findall(regional_html)))
    for n, uuid in enumerate(uuids):
        api_url = API_URL.format(uuid=uuid)
        print(f"Trying API {api_url}")
        try:
            resp = session.get(api_url, timeout=30,
                               headers={"Origin": "https://www.mhsaa.com", "Referer": "https://www.mhsaa.com/",
                                        "Accept": "*/*"})
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  failed: {exc}")
            continue
        save_debug(debug_dir, f"api_{n}.txt", resp.text)
        if consider(*extract_any(resp.text), api_url):
            return best, best_source, best_conflicts

    list_url = find_division_list_link(regional_html, regional_url, year) or DIVISION_LIST_URL.format(year=year)
    print(f"Fetching {list_url}")
    list_html = fetch(session, list_url)
    save_debug(debug_dir, "division_list.html", list_html)
    if consider(*extract_pairs(list_html), list_url):
        return best, best_source, best_conflicts

    for i, frame_url in enumerate(iframe_urls(list_html, list_url)):
        print(f"  Trying iframe {frame_url}")
        try:
            frame_html = fetch(session, frame_url)
        except requests.RequestException as exc:
            print(f"    failed: {exc}")
            continue
        save_debug(debug_dir, f"division_list_iframe{i}.html", frame_html)
        if consider(*extract_pairs(frame_html), frame_url):
            return best, best_source, best_conflicts

    if use_browser:
        for label, url in (("regional", regional_url), ("division_list", list_url)):
            print(f"Rendering {url} in a headless browser and capturing API responses...")
            try:
                htmls, captured = fetch_rendered(url)
            except ImportError:
                print("  Playwright is not installed; skipping browser rendering.")
                break
            except Exception as exc:  # noqa: BLE001 - report and keep trying
                print(f"  Browser rendering failed: {exc}")
                continue
            merged: dict[str, int] = {}
            merged_conflicts: list[str] = []
            for i, (api_url, body) in enumerate(captured):
                save_debug(debug_dir, f"{label}_api{i}.txt", f"{api_url}\n\n{body}")
                found, conflicts = extract_any(body)
                print(f"  API response {api_url}: {len(found)} schools")
                for school, division in found.items():
                    merged.setdefault(school, division)
                merged_conflicts += conflicts
            if consider(merged, merged_conflicts, f"{url} (captured API responses)"):
                return best, best_source, best_conflicts
            for i, html in enumerate(htmls):
                save_debug(debug_dir, f"{label}_rendered{i}.html", html)
                if consider(*extract_pairs(html), f"{url} (browser-rendered)"):
                    return best, best_source, best_conflicts
    return best, best_source, best_conflicts


# --------------------------------------------------------------------------- #
# Name matching
# --------------------------------------------------------------------------- #
# Abbreviations that show up in MHSAA names; expanded token-by-token before comparing.
ALIASES = {
    "st": "saint", "ste": "sainte", "mt": "mount", "ft": "fort",
    "hts": "heights", "hgts": "heights", "twp": "township", "ctr": "center",
    "cent": "central", "ctrl": "central", "cath": "catholic", "chr": "christian",
    "univ": "university", "mich": "michigan",
    "gr": "grand rapids", "gpw": "grosse pointe woods", "gp": "grosse pointe",
    "aa": "ann arbor", "bh": "bloomfield hills", "tc": "traverse city", "fh": "forest hills",
}
DROP_TOKENS = {"high", "school", "hs", "the", "of", "and"}


def _tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " and ").replace("'", "").replace("\u2019", "")
    out: list[str] = []
    for tok in re.findall(r"[a-z0-9]+", s):
        out.extend(ALIASES.get(tok, tok).split())
    out = [t for t in out if t not in DROP_TOKENS]
    # "U-D" -> "u d" -> "ud"
    merged: list[str] = []
    run: list[str] = []
    for tok in out + [""]:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        if run:
            merged.append("".join(run))
            run = []
        if tok:
            merged.append(tok)
    return merged


def normalize(name: str) -> str:
    return " ".join(_tokens(name))


def _components(name: str) -> set[str]:
    """Co-op teams ('Lake Fenton/Linden') -> {'lake fenton', 'linden'}; empty if not a co-op."""
    if "/" not in name:
        return set()
    name = re.sub(r"\(.*?\)", " ", name)
    return {normalize(part) for part in name.split("/") if normalize(part)}


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def similarity(a: str, b: str) -> float:
    """0..1.  Every word of the shorter name must (nearly) appear in the longer one, which
    stops 'Walled Lake Central' from matching 'Walled Lake Western'."""
    ta, tb = _tokens(a), _tokens(b)
    if ta == tb:
        return 1.0
    ca, cb = _components(a), _components(b)
    if ca and cb and ca & cb:
        return 0.9
    if not ta or not tb:
        return 0.0
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    for tok in short:
        if max(_ratio(tok, other) for other in long_) < 0.8:
            return 0.0
    return max(_ratio(" ".join(ta), " ".join(tb)), _ratio(" ".join(sorted(ta)), " ".join(sorted(tb))))


def match_names(csv_names: list[str], web_names: list[str], threshold: float, margin: float):
    """One-to-one match of CSV names to website names.

    Returns (matches, ambiguous) where matches maps csv_name -> (web_name, score, kind)
    and ambiguous lists (csv_name, [(web_name, score), ...]) that were left unmatched
    because more than one website name was an equally good fit.
    """
    matches: dict[str, tuple[str, float, str]] = {}
    used_web: set[str] = set()

    by_key: dict[str, list[str]] = defaultdict(list)
    for w in web_names:
        by_key[normalize(w)].append(w)

    for c in csv_names:  # pass 1: identical after normalisation
        for w in by_key.get(normalize(c), []):
            if w not in used_web:
                kind = "exact" if w.casefold() == c.casefold() else "normalized"
                matches[c] = (w, 1.0, kind)
                used_web.add(w)
                break

    left_csv = [c for c in csv_names if c not in matches]
    left_web = [w for w in web_names if w not in used_web]
    scores = {(c, w): similarity(c, w) for c in left_csv for w in left_web}

    def runner_up(c: str, w: str) -> float:
        others = [scores[(c, w2)] for w2 in left_web if w2 != w]
        others += [scores[(c2, w)] for c2 in left_csv if c2 != c]
        return max(others, default=0.0)

    ambiguous: list[tuple[str, list[tuple[str, float]]]] = []
    candidates = sorted((p for p, s in scores.items() if s >= threshold), key=lambda p: -scores[p])
    for c, w in candidates:  # pass 2: fuzzy, best score first
        if c in matches or w in used_web:
            continue
        if scores[(c, w)] - runner_up(c, w) < margin:
            if any(a[0] == c for a in ambiguous):
                continue
            close = sorted(((w2, scores[(c, w2)]) for w2 in left_web if scores[(c, w2)] >= threshold),
                           key=lambda x: -x[1])
            ambiguous.append((c, close))
            continue
        matches[c] = (w, scores[(c, w)], "fuzzy")
        used_web.add(w)
    return matches, ambiguous


# --------------------------------------------------------------------------- #
# CSV in / out and report
# --------------------------------------------------------------------------- #
def read_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    for r in rows:
        r["school"] = r["school"].strip()
        r["division"] = r["division"].strip()
    return rows, newline


def write_csv(path: Path, rows: list[dict[str, str]], newline: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator=newline)
        writer.writerow(["school", "division"])
        for r in rows:
            writer.writerow([r["school"], r["division"]])


def build_report(year, source, n_web, n_csv, matches, changes, unmatched_csv, unmatched_web,
                 ambiguous, conflicts, added) -> str:
    lines = [
        f"# Boys tennis division update: {year}",
        "",
        f"Source: {source}",
        f"Schools on website: {n_web} | schools in CSV: {n_csv} | matched: {len(matches)}",
        f"Division changes: {len(changes)} | CSV schools not found on website: {len(unmatched_csv)} "
        f"| website schools not in CSV: {len(unmatched_web)}",
        "",
    ]

    def section(title, items):
        if items:
            lines.extend([f"## {title} ({len(items)})", "", *items, ""])

    section("Division changes", [f"- {s}: {old} -> {new}" for s, old, new in changes])
    section("Matched by normalization or fuzzy matching (worth a glance)",
            [f"- `{c}` <- `{w}` ({kind}, {score:.2f})"
             for c, (w, score, kind) in matches.items() if kind != "exact"])
    section("Ambiguous, left unmatched (CSV division kept)",
            [f"- `{c}`: " + ", ".join(f"`{w}` ({s:.2f})" for w, s in close) for c, close in ambiguous])
    section("In CSV but not found on website (old division kept)", [f"- {c}" for c in unmatched_csv])
    section("On website but not in CSV" + (" (added)" if added else " (not added; use --add-new)"),
            [f"- {w}: Division {d}" for w, d in unmatched_web])
    section("Website listed a school in more than one division (first kept)", [f"- {c}" for c in conflicts])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", help="School year like 2026-27 (default: inferred from today's date)")
    p.add_argument("--input", type=Path, default=SCRIPT_DIR / "correct_divisions.csv",
                   help="CSV with columns school,division (default: correct_divisions.csv next to this script)")
    p.add_argument("--output", type=Path, help="default: correct_divisions_<year>.csv next to this script")
    p.add_argument("--report", type=Path, help="default: division_update_report_<year>.md next to this script")
    p.add_argument("--html-file", type=Path, help="parse a saved HTML page instead of fetching (for testing)")
    p.add_argument("--min-schools", type=int, default=200,
                   help="abort without writing if fewer schools than this are scraped (default 200)")
    p.add_argument("--threshold", type=float, default=0.85, help="minimum fuzzy match score (default 0.85)")
    p.add_argument("--margin", type=float, default=0.05,
                   help="a fuzzy match must beat the runner-up by this much (default 0.05)")
    p.add_argument("--add-new", action="store_true", help="append website schools that are not in the CSV")
    p.add_argument("--no-browser", action="store_true", help="never fall back to headless-browser rendering")
    p.add_argument("--debug-dir", type=Path, help="save fetched HTML here for troubleshooting")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    year = args.year or default_school_year()
    output = args.output or SCRIPT_DIR / f"correct_divisions_{year}.csv"
    report_path = args.report or SCRIPT_DIR / f"division_update_report_{year}.md"

    if not args.input.exists():
        print(f"ERROR: input CSV not found: {args.input}", file=sys.stderr)
        return 1
    rows, newline = read_csv(args.input)
    rows = [r for r in rows if r["school"] not in ALIASES]

    if args.html_file:
        web, conflicts = extract_any(args.html_file.read_text(encoding="utf-8"))
        source = str(args.html_file)
    else:
        try:
            web, source, conflicts = scrape(year, args.min_schools, not args.no_browser, args.debug_dir)
        except requests.RequestException as exc:
            print(f"ERROR: could not fetch MHSAA pages: {exc}", file=sys.stderr)
            return 2
    print(f"Scraped {len(web)} schools from {source}")

    if len(web) < args.min_schools:
        print(
            f"ERROR: only {len(web)} schools found (expected at least {args.min_schools}). "
            "The page layout may have changed or the list is not published yet for "
            f"{year}. Nothing was written. Use --debug-dir to save the HTML for inspection.",
            file=sys.stderr,
        )
        return 2

    web_names = list(web)
    csv_names = [r["school"] for r in rows]
    matches, ambiguous = match_names(csv_names, web_names, args.threshold, args.margin)

    changes = []
    for r in rows:
        hit = matches.get(r["school"])
        if hit:
            new = str(web[hit[0]])
            if new != r["division"]:
                changes.append((r["school"], r["division"], new))
                r["division"] = new

    matched_web = {w for w, _, _ in matches.values()}
    unmatched_csv = [c for c in csv_names if c not in matches]
    unmatched_web = [(w, web[w]) for w in web_names if w not in matched_web]
    if args.add_new:
        rows += [{"school": w, "division": str(d)} for w, d in unmatched_web]

    write_csv(output, rows, newline)
    report = build_report(year, source, len(web), len(csv_names), matches, changes, unmatched_csv,
                          unmatched_web, ambiguous, conflicts, args.add_new)
    report_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\nWrote {output}")
    print(f"Wrote {report_path}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
