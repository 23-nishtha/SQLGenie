"""Build database/football/football.db from database/football/raw/results.csv.

Day 8: the built-in Football dataset. Source: martj42/international_results
— ONLY results.csv is used (no former_names/goalscorers/shootouts files).

Safe to re-run: deletes and recreates only football.db. raw/results.csv is
read-only and is never modified by this script.
Usage (from project root):  python database/football/build_db.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_DIR = Path(__file__).resolve().parent
RAW_CSV = DB_DIR / "raw" / "results.csv"
DB_PATH = DB_DIR / "football.db"
SCHEMA_PATH = DB_DIR / "schema.sql"


def _winner(row) -> str:
    if row.home_score > row.away_score:
        return "home"
    if row.away_score > row.home_score:
        return "away"
    return "draw"


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()

    # dtype=str first, same approach as Day 1's Olist loader: keeps every
    # conversion explicit and under our control instead of pandas guessing.
    df = pd.read_csv(RAW_CSV, dtype=str)
    df["home_score"] = pd.to_numeric(df["home_score"]).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"]).astype(int)
    # The source stores this as the text "TRUE"/"FALSE"; SQLite has no
    # native boolean, so store the conventional 0/1 INTEGER instead.
    df["neutral"] = (df["neutral"] == "TRUE").astype(int)

    # Two columns Day 8 adds that are NOT in the source CSV — derived here
    # at load time, never written back to raw/results.csv.
    df["winner"] = df.apply(_winner, axis=1)
    df["goal_diff"] = (df["home_score"] - df["away_score"]).abs()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        df.to_sql("results", conn, if_exists="append", index=False)
        conn.commit()
        conn.execute("ANALYZE")
    finally:
        conn.close()
    print(f"Done: {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB, {len(df):,} rows)")


if __name__ == "__main__":
    main()
