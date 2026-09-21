"""Build database/olist.db from the Olist CSVs in database/raw/.

Safe to re-run: it deletes and recreates only olist.db. Raw CSVs are read-only.
Usage (from project root):  python database/build_db.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_DIR = Path(__file__).resolve().parent
RAW_DIR = DB_DIR / "raw"
DB_PATH = DB_DIR / "olist.db"
SCHEMA_PATH = DB_DIR / "schema.sql"

# Categories used by products but missing from the translation CSV.
EXTRA_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}

# (table, csv file, {csv column: db column} renames, integer cols, real cols)
# Load order matters: parents before children (foreign keys).
TABLES = [
    ("customers", "olist_customers_dataset.csv", {}, [], []),
    ("sellers", "olist_sellers_dataset.csv", {}, [], []),
    ("product_category_translation", "product_category_name_translation.csv", {}, [], []),
    (
        "products",
        "olist_products_dataset.csv",
        {
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        },
        [
            "product_name_length", "product_description_length", "product_photos_qty",
            "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
        ],
        [],
    ),
    ("orders", "olist_orders_dataset.csv", {}, [], []),
    (
        "order_items", "olist_order_items_dataset.csv", {},
        ["order_item_id"], ["price", "freight_value"],
    ),
    (
        "order_payments", "olist_order_payments_dataset.csv", {},
        ["payment_sequential", "payment_installments"], ["payment_value"],
    ),
    ("order_reviews", "olist_order_reviews_dataset.csv", {}, ["review_score"], []),
    (
        "geolocation", "olist_geolocation_dataset.csv", {},
        [], ["geolocation_lat", "geolocation_lng"],
    ),
]


def load_table(conn, table, csv_name, renames, int_cols, real_cols):
    # Read everything as text first (keeps zip codes like '01037'); utf-8-sig strips
    # the hidden BOM at the start of the translation CSV. Empty cells become NULL.
    df = pd.read_csv(RAW_DIR / csv_name, dtype=str, encoding="utf-8-sig")
    df = df.rename(columns=renames)
    for col in int_cols:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    for col in real_cols:
        df[col] = pd.to_numeric(df[col]).astype(float)
    # if_exists="append" keeps OUR schema from schema.sql instead of pandas guessing one
    df.to_sql(table, conn, if_exists="append", index=False, chunksize=50_000)
    print(f"  loaded {table:<30} {len(df):>10,} rows")


def build_zip_locations(conn):
    # One row per zip: average lat/lng, and the most frequent city/state spelling.
    conn.execute(
        """
        INSERT INTO zip_locations (zip_code_prefix, lat, lng, city, state)
        WITH avg_pos AS (
          SELECT geolocation_zip_code_prefix AS zip,
                 AVG(geolocation_lat) AS lat, AVG(geolocation_lng) AS lng
          FROM geolocation GROUP BY geolocation_zip_code_prefix
        ),
        ranked AS (
          SELECT geolocation_zip_code_prefix AS zip, geolocation_city AS city,
                 geolocation_state AS state,
                 ROW_NUMBER() OVER (
                   PARTITION BY geolocation_zip_code_prefix
                   ORDER BY COUNT(*) DESC, geolocation_city
                 ) AS rn
          FROM geolocation
          GROUP BY geolocation_zip_code_prefix, geolocation_city, geolocation_state
        )
        SELECT a.zip, a.lat, a.lng, r.city, r.state
        FROM avg_pos a JOIN ranked r ON r.zip = a.zip AND r.rn = 1
        """
    )
    n = conn.execute("SELECT COUNT(*) FROM zip_locations").fetchone()[0]
    print(f"  built  {'zip_locations':<30} {n:>10,} rows")


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        print("Loading CSVs...")
        for table, csv_name, renames, int_cols, real_cols in TABLES:
            load_table(conn, table, csv_name, renames, int_cols, real_cols)
            if table == "product_category_translation":
                conn.executemany(
                    "INSERT INTO product_category_translation VALUES (?, ?)",
                    EXTRA_TRANSLATIONS.items(),
                )
                print(f"  added  {len(EXTRA_TRANSLATIONS)} manual translations")
        build_zip_locations(conn)
        conn.commit()
        conn.execute("ANALYZE")
    finally:
        conn.close()
    print(f"Done: {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
