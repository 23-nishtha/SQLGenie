"""Check that database/movies/movies.db was loaded correctly.
Usage (from project root):  python database/movies/verify_db.py
Exits with code 1 if any check fails.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "movies.db"
EXPECTED_ROWS = 1000

# Verified against raw/imdb_top_1000.csv at inspection time (Day 8).
EXPECTED_NULLS = {"certificate": 101, "meta_score": 157, "gross": 169}

failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures += 1


def scalar(conn, sql):
    return conn.execute(sql).fetchone()[0]


def main():
    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH} not found. Run: python database/movies/build_db.py")
    conn = sqlite3.connect(DB_PATH)

    actual_rows = scalar(conn, "SELECT COUNT(*) FROM movies")
    check("row count", actual_rows == EXPECTED_ROWS, f"(got {actual_rows:,}, expected {EXPECTED_ROWS:,})")

    check("integrity_check", scalar(conn, "PRAGMA integrity_check") == "ok")

    check("no NULLs in required (NOT NULL) columns",
          scalar(conn, "SELECT COUNT(*) FROM movies WHERE series_title IS NULL OR released_year IS NULL "
                       "OR runtime_minutes IS NULL OR genre IS NULL OR primary_genre IS NULL "
                       "OR imdb_rating IS NULL OR director IS NULL OR no_of_votes IS NULL") == 0)

    for col, expected in EXPECTED_NULLS.items():
        actual = scalar(conn, f"SELECT COUNT(*) FROM movies WHERE {col} IS NULL")
        check(f"NULLs in {col}", actual == expected, f"(got {actual}, expected {expected})")

    check("Apollo 13's released_year was repaired to 1995",
          scalar(conn, "SELECT released_year FROM movies WHERE series_title = 'Apollo 13'") == 1995)

    check("released_year is a plausible 4-digit year for every row",
          scalar(conn, "SELECT COUNT(*) FROM movies WHERE released_year < 1900 OR released_year > 2030") == 0)

    check("runtime_minutes has no leftover ' min' text (i.e. is a real integer column)",
          scalar(conn, "SELECT COUNT(*) FROM movies WHERE typeof(runtime_minutes) != 'integer'") == 0)

    check("gross has no leftover commas (i.e. is a real integer/NULL column)",
          scalar(conn, "SELECT COUNT(*) FROM movies WHERE gross IS NOT NULL "
                       "AND typeof(gross) != 'integer'") == 0)

    check("primary_genre is always the first word(s) of genre",
          scalar(conn, "SELECT COUNT(*) FROM movies "
                       "WHERE substr(genre, 1, length(primary_genre)) != primary_genre") == 0)

    print("\nSanity queries:")
    print("Rating range:", conn.execute("SELECT MIN(imdb_rating), MAX(imdb_rating) FROM movies").fetchone())
    print("Top 3 by rating:")
    for title, rating in conn.execute(
        "SELECT series_title, imdb_rating FROM movies ORDER BY imdb_rating DESC LIMIT 3"
    ):
        print(f"  {title:<30} {rating}")
    print("Top 3 primary genres by count:")
    for genre, n in conn.execute(
        "SELECT primary_genre, COUNT(*) FROM movies GROUP BY 1 ORDER BY 2 DESC LIMIT 3"
    ):
        print(f"  {genre:<15} {n}")

    conn.close()
    print(f"\n{'ALL CHECKS PASSED' if not failures else str(failures) + ' CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
