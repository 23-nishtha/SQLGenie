"""Build database/movies/movies.db from database/movies/raw/imdb_top_1000.csv.

Day 8: the built-in Movies dataset (IMDb Top 1000).

Safe to re-run: deletes and recreates only movies.db. raw/imdb_top_1000.csv
is read-only and is never modified by this script.
Usage (from project root):  python database/movies/build_db.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_DIR = Path(__file__).resolve().parent
RAW_CSV = DB_DIR / "raw" / "imdb_top_1000.csv"
DB_PATH = DB_DIR / "movies.db"
SCHEMA_PATH = DB_DIR / "schema.sql"

# Source columns are CamelCase; renamed to snake_case to match the rest of
# SQLGenie's schema conventions. Runtime -> runtime_minutes since its VALUE
# also changes (text "142 min" -> integer 142), so the old name would lie.
RENAMES = {
    "Series_Title": "series_title",
    "Released_Year": "released_year",
    "Certificate": "certificate",
    "Runtime": "runtime_minutes",
    "Genre": "genre",
    "IMDB_Rating": "imdb_rating",
    "Overview": "overview",
    "Meta_score": "meta_score",
    "Director": "director",
    "Star1": "star1",
    "Star2": "star2",
    "Star3": "star3",
    "Star4": "star4",
    "No_of_Votes": "no_of_votes",
    "Gross": "gross",
    "Poster_Link": "poster_link",
}

# One known data-quality issue in the source CSV, found during inspection:
# this row's `Released_Year` cell literally contains the text "PG" instead
# of a year. Checking the RAW LINE (not just the parsed DataFrame) showed
# this is NOT a column shift — every other field in the row (Certificate,
# Runtime, Genre, Director, cast, votes, gross...) is already correctly
# positioned and correctly valued. It's a single bad cell. 1995 is Apollo
# 13's real, publicly documented release year; recorded here explicitly
# since nothing else in the row can supply a year. See database/movies's
# entry in the Day 8 chat log for the full investigation.
MANUAL_YEAR_FIXES = {
    "Apollo 13": 1995,
}


def _clean_runtime(value: str) -> int:
    # "142 min" -> 142
    return int(value.replace(" min", "").strip())


def _clean_gross(value):
    if pd.isna(value):
        return None
    return int(value.replace(",", ""))


def _clean_year(row) -> int:
    raw_year = row["released_year"]
    if raw_year.isdigit():
        return int(raw_year)
    fixed = MANUAL_YEAR_FIXES.get(row["series_title"])
    if fixed is None:
        raise ValueError(
            f"Row for {row['series_title']!r} has a non-numeric released_year "
            f"({raw_year!r}) and no manual fix is recorded for it in MANUAL_YEAR_FIXES. "
            "Add one rather than silently dropping the row."
        )
    return fixed


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()

    # keep_default_na=False + na_values=[""]: only a genuinely empty cell
    # becomes NULL (Certificate/Meta_score/Gross) — pandas' broader default
    # NA list (e.g. "NA", "None" as literal text) isn't relevant here and
    # could otherwise hide a real data value.
    df = pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False, na_values=[""])
    df = df.rename(columns=RENAMES)

    df["released_year"] = df.apply(_clean_year, axis=1)
    df["runtime_minutes"] = df["runtime_minutes"].apply(_clean_runtime)
    df["gross"] = df["gross"].apply(_clean_gross)
    df["meta_score"] = pd.to_numeric(df["meta_score"])   # NaN stays NaN -> SQL NULL
    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"])
    df["no_of_votes"] = pd.to_numeric(df["no_of_votes"]).astype(int)
    # certificate is left as-is: TEXT, NULL where the source left it empty.

    # DERIVED column, not in the source CSV: the first of possibly several
    # comma-separated genres (e.g. "Action, Drama, Sport" -> "Action"), so
    # a plain GROUP BY / WHERE on genre works cleanly. The original,
    # unsplit `genre` column is kept alongside it.
    df["primary_genre"] = df["genre"].apply(lambda g: g.split(",")[0].strip())

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        df.to_sql("movies", conn, if_exists="append", index=False)
        conn.commit()
        conn.execute("ANALYZE")
    finally:
        conn.close()
    print(f"Done: {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB, {len(df):,} rows)")


if __name__ == "__main__":
    main()
