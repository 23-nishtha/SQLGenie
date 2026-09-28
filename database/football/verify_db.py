"""Check that database/football/football.db was loaded correctly.
Usage (from project root):  python database/football/verify_db.py
Exits with code 1 if any check fails.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "football.db"
EXPECTED_ROWS = 49547  # verified against raw/results.csv at inspection time (Day 8)

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
        sys.exit(f"{DB_PATH} not found. Run: python database/football/build_db.py")
    conn = sqlite3.connect(DB_PATH)

    actual_rows = scalar(conn, "SELECT COUNT(*) FROM results")
    check("row count", actual_rows == EXPECTED_ROWS, f"(got {actual_rows:,}, expected {EXPECTED_ROWS:,})")

    check("integrity_check", scalar(conn, "PRAGMA integrity_check") == "ok")

    check("no NULLs in any required column",
          scalar(conn, "SELECT COUNT(*) FROM results WHERE date IS NULL OR home_team IS NULL "
                       "OR away_team IS NULL OR home_score IS NULL OR away_score IS NULL "
                       "OR tournament IS NULL OR city IS NULL OR country IS NULL") == 0)

    check("neutral is only 0 or 1",
          scalar(conn, "SELECT COUNT(*) FROM results WHERE neutral NOT IN (0, 1)") == 0)

    check("winner is only home/away/draw",
          scalar(conn, "SELECT COUNT(*) FROM results WHERE winner NOT IN ('home','away','draw')") == 0)

    check("winner agrees with the scores",
          scalar(conn,
                 "SELECT COUNT(*) FROM results WHERE "
                 "(winner='home' AND home_score<=away_score) OR "
                 "(winner='away' AND away_score<=home_score) OR "
                 "(winner='draw' AND home_score!=away_score)") == 0)

    check("goal_diff = abs(home_score - away_score) for every row",
          scalar(conn, "SELECT COUNT(*) FROM results WHERE goal_diff != ABS(home_score - away_score)") == 0)

    print("\nSanity queries:")
    print("Date range:", conn.execute("SELECT MIN(date), MAX(date) FROM results").fetchone())
    print("Most wins (top 3):")
    for team, wins in conn.execute(
        "SELECT home_team, COUNT(*) FROM results WHERE winner='home' GROUP BY home_team "
        "ORDER BY 2 DESC LIMIT 3"
    ):
        print(f"  {team:<20} {wins:>6,} home wins")

    conn.close()
    print(f"\n{'ALL CHECKS PASSED' if not failures else str(failures) + ' CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
