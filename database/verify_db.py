"""Check that database/olist.db was loaded correctly.
Usage (from project root):  python database/verify_db.py
Exits with code 1 if any check fails.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "olist.db"

EXPECTED_COUNTS = {
    "customers": 99441,
    "sellers": 3095,
    "product_category_translation": 71 + 2,  # 71 from CSV + 2 manual
    "products": 32951,
    "orders": 99441,
    "order_items": 112650,
    "order_payments": 103886,
    "order_reviews": 99224,
    "geolocation": 1000163,
}

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
        sys.exit(f"{DB_PATH} not found. Run: python database/build_db.py")
    conn = sqlite3.connect(DB_PATH)

    for table, expected in EXPECTED_COUNTS.items():
        actual = scalar(conn, f"SELECT COUNT(*) FROM {table}")
        check(f"row count {table}", actual == expected, f"(got {actual:,}, expected {expected:,})")

    zips = scalar(conn, "SELECT COUNT(DISTINCT geolocation_zip_code_prefix) FROM geolocation")
    check("zip_locations one row per zip",
          scalar(conn, "SELECT COUNT(*) FROM zip_locations") == zips, f"({zips:,} zips)")

    check("integrity_check", scalar(conn, "PRAGMA integrity_check") == "ok")
    bad = conn.execute("PRAGMA foreign_key_check").fetchall()
    check("foreign_key_check (no broken links)", not bad, f"({len(bad)} violations)")

    nulls = {
        "order_approved_at": 160,
        "order_delivered_carrier_date": 1783,
        "order_delivered_customer_date": 2965,
    }
    for col, expected in nulls.items():
        actual = scalar(conn, f"SELECT COUNT(*) FROM orders WHERE {col} IS NULL")
        check(f"NULLs in orders.{col}", actual == expected, f"(got {actual}, expected {expected})")

    check("zip leading zeros kept",
          scalar(conn, "SELECT COUNT(*) FROM customers WHERE customer_zip_code_prefix LIKE '0%'") > 0)
    check("unique people = 96096",
          scalar(conn, "SELECT COUNT(DISTINCT customer_unique_id) FROM customers") == 96096)
    check("delivered orders = 96478",
          scalar(conn, "SELECT COUNT(*) FROM orders WHERE order_status='delivered'") == 96478)
    check("view row count = order_items",
          scalar(conn, "SELECT COUNT(*) FROM v_order_items_detail") == EXPECTED_COUNTS["order_items"])
    check("no untranslated categories in products",
          scalar(conn, "SELECT COUNT(*) FROM products p LEFT JOIN product_category_translation t "
                       "ON t.product_category_name = p.product_category_name "
                       "WHERE p.product_category_name IS NOT NULL "
                       "AND t.product_category_name IS NULL") == 0)

    print("\nSanity queries:")
    print("Total revenue (price + freight): R$",
          f"{scalar(conn, 'SELECT SUM(price + freight_value) FROM order_items'):,.2f}")
    print("Orders per status:")
    for status, n in conn.execute(
        "SELECT order_status, COUNT(*) FROM orders GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"  {status:<12} {n:>7,}")
    print("Top 5 categories by revenue:")
    for cat, rev in conn.execute(
        "SELECT product_category_english, SUM(price) FROM v_order_items_detail "
        "GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
    ):
        print(f"  {str(cat):<25} R$ {rev:>12,.2f}")

    conn.close()
    print(f"\n{'ALL CHECKS PASSED' if not failures else str(failures) + ' CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
