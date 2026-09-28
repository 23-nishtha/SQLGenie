-- SQLGenie: Movies schema (SQLite)
-- Day 8: the third built-in dataset, alongside Olist and Football.
-- Source: imdb_top_1000.csv (IMDb Top 1000 Movies).

CREATE TABLE movies (
  movie_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- surrogate key: series_title alone isn't unique (e.g. remakes)
  series_title TEXT NOT NULL,
  released_year INTEGER NOT NULL,              -- see build_db.py: one row's source value was corrupted, and is fixed there
  certificate TEXT,                            -- NULL where missing in the source (101 of 1000 rows)
  runtime_minutes INTEGER NOT NULL,            -- source is text like '142 min'; the ' min' is stripped at load time
  genre TEXT NOT NULL,                         -- original multi-value text kept as-is, e.g. 'Action, Drama, Sport'
  primary_genre TEXT NOT NULL,                 -- DERIVED, not in the source CSV: the first genre listed in `genre`
  imdb_rating REAL NOT NULL,
  overview TEXT,
  meta_score REAL,                             -- NULL where missing in the source (157 of 1000 rows)
  director TEXT NOT NULL,
  star1 TEXT,
  star2 TEXT,
  star3 TEXT,
  star4 TEXT,
  no_of_votes INTEGER NOT NULL,
  gross INTEGER,                               -- NULL where missing in the source (169 of 1000 rows); commas stripped
  poster_link TEXT                             -- kept for a possible future frontend/visualization use
);

CREATE INDEX idx_movies_year ON movies(released_year);
CREATE INDEX idx_movies_rating ON movies(imdb_rating);
CREATE INDEX idx_movies_primary_genre ON movies(primary_genre);
CREATE INDEX idx_movies_director ON movies(director);
