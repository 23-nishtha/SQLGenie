-- SQLGenie: Football (international results) schema (SQLite)
-- Day 8: the second built-in dataset, alongside Olist.
-- Source: results.csv from martj42/international_results — ONLY results.csv
-- is used; former_names.csv, goalscorers.csv and shootouts.csv are not part
-- of this dataset.

CREATE TABLE results (
  match_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- surrogate key: the source has no natural unique id
  date TEXT NOT NULL,                          -- 'YYYY-MM-DD'
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  home_score INTEGER NOT NULL,
  away_score INTEGER NOT NULL,
  tournament TEXT NOT NULL,
  city TEXT NOT NULL,
  country TEXT NOT NULL,
  neutral INTEGER NOT NULL,                    -- 0/1, converted from the source's TRUE/FALSE text
  winner TEXT NOT NULL,                        -- 'home' | 'away' | 'draw' — DERIVED, not in the source CSV
  goal_diff INTEGER NOT NULL                   -- abs(home_score - away_score) — DERIVED, not in the source CSV
);

CREATE INDEX idx_results_date ON results(date);
CREATE INDEX idx_results_home_team ON results(home_team);
CREATE INDEX idx_results_away_team ON results(away_team);
CREATE INDEX idx_results_tournament ON results(tournament);
CREATE INDEX idx_results_winner ON results(winner);
