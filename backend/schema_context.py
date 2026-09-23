"""Builds a plain-text description of the database schema to hand to the LLM.

We read the schema straight out of SQLite (sqlite_master + PRAGMA table_info)
instead of re-parsing schema.sql. That way the description always matches
what's actually in olist.db, even if schema.sql ever drifts out of sync.
"""
import sqlite3

from backend.config import settings

# A few plain-English notes about things the raw column list can't convey.
# These come straight from the Day 1 design decisions in database/README.md.
SCHEMA_NOTES = """
Notes on this schema:
- "orders" is the central table. "order_items" links orders to products and sellers.
- customers.customer_id is unique per ORDER, not per person. Use
  customers.customer_unique_id to count or identify actual people.
- product_category_name is in Portuguese. Prefer the view
  v_order_items_detail, which already includes product_category_english.
- v_order_items_detail is a convenience view: one row per order line, with
  customer, product (English category), seller and order fields pre-joined.
  Use it for most "revenue / orders / customers by X" questions instead of
  writing the joins by hand.
- geolocation has many rows per zip code (raw samples). Use zip_locations
  instead, which has exactly one row per zip code prefix.
- Dates are stored as text in 'YYYY-MM-DD HH:MM:SS' format.
""".strip()


def _fetch_tables_and_views(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return [(name, type), ...] for every table/view, skipping SQLite internals."""
    rows = conn.execute(
        "SELECT name, type FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
        "ORDER BY type DESC, name"  # tables before views, alphabetical within each
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def build_schema_text(db_path=None) -> str:
    """Connect to the SQLite DB and build the schema description as one string.

    Called once at startup (see main.py) and cached, since the schema
    doesn't change while the API is running.
    """
    db_path = db_path or settings.DATABASE_PATH
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        lines = []
        for name, obj_type in _fetch_tables_and_views(conn):
            columns = conn.execute(f"PRAGMA table_info('{name}')").fetchall()
            # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
            col_descriptions = [f"{col[1]} {col[2]}".strip() for col in columns]
            label = "TABLE" if obj_type == "table" else "VIEW"
            lines.append(f"{label} {name}({', '.join(col_descriptions)})")
        return "\n".join(lines) + "\n\n" + SCHEMA_NOTES
    finally:
        conn.close()
