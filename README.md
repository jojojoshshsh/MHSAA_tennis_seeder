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

