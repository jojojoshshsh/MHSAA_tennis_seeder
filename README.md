# Michigan HS Tennis Rankings

Automated ranking + static website for Michigan high school tennis,
built from match data crawled off tennisreporting.com.

## How it works

```
                 ┌─────────────────┐
   season picked │   src/config.py  │  ← change YEAR here
                 │     YEAR = 2025  │
                 └────────┬─────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                     │
 ┌──────▼───────┐                    ┌────────▼────────┐
 │ main_fetch.py │  crawls the API   │  run_ranking.py  │  re-ranks whatever
 │ (Workflow A)  │  for that season  │ (Workflow A & B) │  is in historical/
 └──────┬────────┘                   └────────┬─────────┘
        │ writes                              │ writes
        ▼                                      ▼
 historical/{YEAR}/                  src/rankings_by_division_flight/
   all_matches.csv                     singles_boys_division_1.csv
   all_matches_excluding_state.csv     doubles_girls_division_3.csv
   school_meta.json                    team_boys_division_1.csv
                                        ...
                                              │
                                              ▼
                                    scripts/build_site.py
                                              │
                                              ▼
                                          docs/index.html  ← published via
                                          docs/csv/*.csv      GitHub Pages
```

`src/config.py`'s `YEAR` is the one knob that controls the whole
pipeline — the fetch phase, the ranking phase, and the site's season
label all read it.

## Two workflows

| Workflow | File | What it does | When to use |
|---|---|---|---|
| **Fetch + Rank + Publish** | `.github/workflows/fetch_rank_publish.yml` | Crawls tennisreporting.com for `config.YEAR`, ranks it, builds the site. Slow (many HTTP calls). | New season data needs to be pulled in. Runs daily on a schedule by default. |
| **Rank + Publish** | `.github/workflows/rank_publish.yml` | Re-ranks whatever is already saved under `historical/{YEAR}/` and rebuilds the site. No network calls. Fast. | You fixed `correct_divisions.csv`, tweaked the ranking algorithm, or just want to rebuild the site without re-fetching. |

Both are manually triggerable from the **Actions** tab (`workflow_dispatch`).

## Setup

1. **Add GitHub Secrets** (Settings → Secrets and variables → Actions):
   - `TENNIS_EMAIL`
   - `TENNIS_PASSWORD`

   These are only used by `main_fetch.py` (Workflow A). Never commit
   credentials directly into any file — both workflows and all scripts
   read them from the environment only.

2. **Enable GitHub Pages**: Settings → Pages → Source: `docs/` folder
   on your default branch.

3. **Pick a season**: edit `src/config.py`'s `YEAR`, commit, then run
   "Fetch + Rank + Publish" once to populate `historical/{YEAR}/`.
   After that, "Rank + Publish" is enough for any re-ranking.

## Repo layout

```
src/
  config.py                  # YEAR + all tunable parameters
  api_fetcher.py              # HTTP/auth layer (env-based credentials)
  crawler.py                  # BFS crawl of schools/matches/brackets
  match_parser.py             # ⚠️ bring your own — not included, see below
  fetch_state_seeds.py        # ⚠️ bring your own — optional, only used by run_all_years.py
  main_fetch.py                # Workflow A entry point (fetch phase)
  run_ranking.py               # Workflow A & B entry point (rank phase)
  run_all_years.py             # manual local-only backfill of all seasons
  mhsaa_seeding_v2.py          # the ranking engine (cycle removal →
                                #   transitive closure → adjacent fix-up →
                                #   division split → CSV output)
  trueskill_engine.py          # lightweight TrueSkill implementation
  team_aggregation.py          # builds team_*.csv from player-level seeds
scripts/
  build_site.py                 # renders docs/index.html from the ranking CSVs
historical/{year}/              # raw crawled match data (committed by CI)
  all_matches.csv
  all_matches_excluding_state.csv
  school_meta.json
src/rankings_by_division_flight/  # ranking engine output (committed by CI)
  singles_boys_division_1.csv
  doubles_girls_division_2.csv
  team_boys_division_1.csv
  ...
docs/                            # published site (GitHub Pages root)
  index.html
  csv/*.csv
```

### ⚠️ Two modules are not included

`match_parser.py` (response parsing — used by `crawler.py`) and
`fetch_state_seeds.py` (state-tournament seed enrichment — only used by
the optional `run_all_years.py` backfill script) were referenced by the
original code but their source wasn't available when this repo was
assembled. Both workflows' fetch phase depends on `match_parser.py`
existing in `src/` — add your existing copy before running
"Fetch + Rank + Publish" for the first time.

## Ranking methodology

See the module docstring at the top of `src/mhsaa_seeding_v2.py` for
the full algorithm (cycle removal → transitive closure → adjacent
fix-up → division split). In short:

1. Build a win/loss graph across all divisions in a (gender, match
   type, flight) group.
2. Break every cycle by dropping the oldest contradicting match.
3. Rank by transitive win-reach (how many players you beat, directly or
   through a chain of wins), with recency/margin/TrueSkill as tiebreaks.
4. Walk the order top-to-bottom fixing up adjacent pairs using
   head-to-head, then common opponents, then dominance, then TrueSkill.
5. Split into divisions, preserving relative order.

Matches that end after one set, or with a literal "2-0 2-0" score, still
count in win/loss records but are excluded from every ranking
calculation.

### Team rankings (`team_*.csv`)

There was no existing team-ranking algorithm to port, so
`team_aggregation.py` implements a simple, documented default: each
school's seeded players are aggregated with a flight-depth weight
(flight 1 counts more than flight 4), producing a weighted-mean `TGRS`,
`ts_rating`, and `sos`, plus summed `wins`/`losses`/`quality_wins`. Tune
`_FLIGHT_WEIGHT` and `_compute_tgrs()` (in `mhsaa_seeding_v2.py`) if your
league's team-scoring rules differ.

## Local development

```bash
pip install -r requirements.txt

# Fetch (needs TENNIS_EMAIL / TENNIS_PASSWORD in your environment)
export TENNIS_EMAIL="you@example.com"
export TENNIS_PASSWORD="your-password"
cd src && python main_fetch.py

# Rank
python run_ranking.py

# Build the site
cd .. && python scripts/build_site.py
# → open docs/index.html
```
