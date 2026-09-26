"""Comprehensive tests for backend/sql_guard.py (Day 4).

Why this file exists
---------------------
sql_guard.py is SQLGenie's main security boundary: it's the code that
decides whether SQL an LLM invented is allowed anywhere near the real
database. Day 2 built the guard itself; this file is Day 4's job — proving,
case by case, that it actually does what it claims to. Every test below
calls the real `validate_and_prepare()`, no mocking, so a regression here
means the guard itself changed behavior, not a stand-in for it.

These tests need no OpenAI call and no mock LLM mode — sql_guard.py takes a
SQL string in and returns a SQL string out; it doesn't know or care where
the string came from.
"""
import pytest

from backend.config import settings
from backend.sql_guard import SqlValidationError, validate_and_prepare

# ---------------------------------------------------------------------------
# Queries that SHOULD be accepted.
# ---------------------------------------------------------------------------


def test_normal_select_is_accepted():
    sql = validate_and_prepare("SELECT * FROM orders")
    assert sql.strip().upper().startswith("SELECT")


def test_select_with_where_is_accepted():
    sql = validate_and_prepare("SELECT * FROM orders WHERE order_status = 'delivered'")
    assert "WHERE" in sql.upper()


def test_with_select_cte_is_accepted():
    sql = validate_and_prepare(
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent"
    )
    assert sql.strip().upper().startswith("SELECT")


def test_lowercase_select_is_accepted():
    """Keyword matching must be case-insensitive — real LLM output is
    inconsistent about casing."""
    sql = validate_and_prepare("select order_status, count(*) from orders group by 1")
    assert sql  # doesn't raise


# ---------------------------------------------------------------------------
# Queries that SHOULD be rejected: one per required-block keyword.
# ---------------------------------------------------------------------------

REJECTED_STATEMENTS = {
    "DELETE": "DELETE FROM orders",
    "UPDATE": "UPDATE orders SET order_status = 'delivered'",
    "INSERT": "INSERT INTO orders (order_id) VALUES ('x')",
    "DROP": "DROP TABLE orders",
    "ALTER": "ALTER TABLE orders ADD COLUMN hacked TEXT",
    "TRUNCATE": "TRUNCATE TABLE orders",
    "PRAGMA": "PRAGMA table_info(orders)",
    "ATTACH": "ATTACH DATABASE 'evil.db' AS evil",
    "CREATE": "CREATE TABLE hacked (id INTEGER)",
    "VACUUM": "VACUUM",
    "REPLACE": "REPLACE INTO orders (order_id) VALUES ('x')",
}


@pytest.mark.parametrize("keyword,sql", REJECTED_STATEMENTS.items(), ids=REJECTED_STATEMENTS.keys())
def test_write_and_admin_statements_are_rejected(keyword, sql):
    with pytest.raises(SqlValidationError):
        validate_and_prepare(sql)


def test_multiple_statements_are_rejected():
    """The classic SQL-injection shape: a valid query followed by a
    destructive one, separated by a semicolon."""
    with pytest.raises(SqlValidationError):
        validate_and_prepare("SELECT * FROM orders; DROP TABLE orders;")


def test_with_select_followed_by_delete_is_rejected():
    """SQLite allows a CTE to be followed by DELETE/UPDATE/INSERT instead of
    SELECT (e.g. `WITH t AS (...) DELETE FROM ...`). The statement still
    starts with the allowed keyword "WITH", so this only gets caught
    because the guard scans EVERY token for a forbidden keyword, not just
    the first one. See the comment in sql_guard.py for the full reasoning.
    """
    with pytest.raises(SqlValidationError):
        validate_and_prepare("WITH t AS (SELECT 1) DELETE FROM orders")


def test_union_select_injection_is_still_read_only():
    """A UNION-based injection attempt is still just a SELECT — the guard's
    job is "is this read-only", not "is this exactly the query we expected".
    It should be accepted (read-only), and it cannot modify data either way
    because db.py's connection is read-only regardless."""
    sql = validate_and_prepare(
        "SELECT customer_id FROM customers UNION SELECT seller_id FROM sellers"
    )
    assert sql.strip().upper().startswith("SELECT")


def test_classic_or_1_equals_1_is_still_just_a_select():
    """A classic injection payload embedded in a WHERE clause. Since the
    whole thing is still one read-only SELECT statement, the guard's job
    (read-only, single-statement) doesn't reject it — there's no data to
    leak beyond what the query already had access to, and it can't modify
    anything. Guards against WRITES; it isn't a general injection firewall
    for a read-only endpoint like this one."""
    sql = validate_and_prepare(
        "SELECT * FROM orders WHERE order_id = 'x' OR '1'='1'"
    )
    assert sql.strip().upper().startswith("SELECT")


def test_delete_inside_a_string_literal_is_not_rejected():
    """The whole reason sql_guard.py tokenizes instead of string-matching:
    "delete" appearing as DATA (inside quotes) must not trigger the
    forbidden-keyword check, which only looks at actual SQL keyword tokens.
    """
    sql = validate_and_prepare(
        "SELECT * FROM order_reviews WHERE review_comment_message LIKE '%delete%'"
    )
    assert "delete" in sql.lower()  # the word survives — it's just data
    assert sql.strip().upper().startswith("SELECT")


def test_update_word_inside_a_string_literal_is_not_rejected():
    """Same idea as above, for a different forbidden word, to make sure
    this isn't a coincidence specific to "delete"."""
    sql = validate_and_prepare(
        "SELECT * FROM order_reviews WHERE review_comment_title = 'please update your address'"
    )
    assert sql.strip().upper().startswith("SELECT")


# ---------------------------------------------------------------------------
# LIMIT enforcement.
# ---------------------------------------------------------------------------


def test_missing_limit_gets_auto_appended():
    sql = validate_and_prepare("SELECT * FROM order_items")
    assert "LIMIT" in sql.upper()
    assert str(settings.MAX_ROWS) in sql


def test_existing_limit_is_preserved_not_duplicated():
    sql = validate_and_prepare("SELECT * FROM orders LIMIT 5")
    # Should not be wrapped a second time — the original LIMIT 5 stands.
    assert sql.strip().upper().count("LIMIT") == 1
    assert "LIMIT 5" in sql.upper()


# ---------------------------------------------------------------------------
# Malformed / garbage input: the guard must fail SAFELY — either a clean
# SqlValidationError, or (for input that's at least SELECT-shaped) a normal
# return — and must NEVER crash with an unrelated, unhandled exception
# (IndexError, TypeError, a raw sqlparse error, etc.).
# ---------------------------------------------------------------------------

# Input with no recognizable SELECT/WITH shape at all: these must be
# rejected with our own clean SqlValidationError, not silently accepted.
UNPARSEABLE_INPUTS = [
    "",
    "   ",
    "asdkjaskjd",
    "-- just a comment, no query",
    ";;;",
]


@pytest.mark.parametrize("bad_sql", UNPARSEABLE_INPUTS)
def test_unparseable_input_is_rejected_safely_not_crashed(bad_sql):
    """Every one of these must raise SqlValidationError specifically —
    not IndexError, not a raw sqlparse exception, nothing that would leak
    as an unhandled 500 with a stack trace."""
    with pytest.raises(SqlValidationError):
        validate_and_prepare(bad_sql)


# Input that IS shaped like a SELECT but is syntactically broken in a way
# only a real SQL parser (not a keyword scanner) would catch. Checking full
# grammar isn't sql_guard.py's job — SQLite itself will reject these when
# db.py actually tries to run them, safely, as a QueryExecutionError. The
# only thing THIS test cares about is that the guard itself never crashes.
SYNTACTICALLY_BROKEN_BUT_SELECT_SHAPED = [
    "SELECT * FROM orders WHERE order_status = 'unterminated",
    "SELECT * FROM orders WHERE (",
    "SELECT FROM WHERE",
]


@pytest.mark.parametrize("odd_sql", SYNTACTICALLY_BROKEN_BUT_SELECT_SHAPED)
def test_broken_but_select_shaped_sql_does_not_crash_the_guard(odd_sql):
    try:
        result = validate_and_prepare(odd_sql)
        assert isinstance(result, str)
    except SqlValidationError:
        pass  # also fine — either outcome is "handled safely"


# ---------------------------------------------------------------------------
# Error messages should be clean and understandable, not raw exceptions.
# ---------------------------------------------------------------------------


def test_error_message_for_statement_level_write_is_understandable():
    """A bare `DELETE FROM orders` fails the FIRST check (must start with
    SELECT/WITH) before it ever reaches the keyword scan, so the message
    here is the allow-list's, not the deny-list's — still clean and
    actionable either way."""
    with pytest.raises(SqlValidationError, match="Only SELECT queries are allowed"):
        validate_and_prepare("DELETE FROM orders")


def test_error_message_names_the_rejected_keyword_when_found_mid_statement():
    """This is the case where the deny-list's specific message actually
    fires: a statement that starts with an allowed keyword (WITH) but
    contains a forbidden one later on."""
    with pytest.raises(SqlValidationError, match="DELETE"):
        validate_and_prepare("WITH t AS (SELECT 1) DELETE FROM orders")


def test_error_message_for_multiple_statements_is_understandable():
    with pytest.raises(SqlValidationError, match="one SQL statement"):
        validate_and_prepare("SELECT 1; SELECT 2;")


# ---------------------------------------------------------------------------
# Independent check: the DATABASE layer is read-only on its own, regardless
# of what sql_guard.py does. Defense-in-depth should be tested as
# independent, not just assumed — this test bypasses sql_guard entirely and
# tries to write directly through the same connection style db.py uses.
# ---------------------------------------------------------------------------


def test_database_connection_itself_rejects_writes():
    """This does NOT call sql_guard at all. It opens the database exactly
    the way backend/db.py does (mode=ro + PRAGMA query_only) and attempts a
    real write, proving the second safety layer works even if the guard
    were somehow bypassed entirely."""
    import sqlite3

    conn = sqlite3.connect(f"file:{settings.DATABASE_PATH}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = TRUE")
        with pytest.raises(sqlite3.Error):
            conn.execute("DELETE FROM orders WHERE order_id = 'this-should-never-work'")
    finally:
        conn.close()


def test_db_run_query_surfaces_a_write_attempt_as_a_clean_error():
    """End-to-end version of the above, through the real db.py function —
    confirms a write attempt (if it ever reached this far) comes back as a
    clean QueryExecutionError, not a raw sqlite3 exception or a crash."""
    from backend.db import QueryExecutionError, run_query

    with pytest.raises(QueryExecutionError):
        run_query("DELETE FROM orders WHERE order_id = 'this-should-never-work'")
