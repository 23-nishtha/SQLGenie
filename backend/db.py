"""Runs an already-validated SQL query against olist.db and returns the rows.

Second safety layer (the first is sql_guard.py): the connection itself is
opened read-only, and PRAGMA query_only is set, so even a query that slipped
past validation cannot modify the database.
"""
import sqlite3
import time

from backend.config import settings


class QueryExecutionError(RuntimeError):
    """Raised when the database rejects the query or it runs too long."""


def _make_progress_handler(deadline: float):
    """Returns a function sqlite3 calls periodically while a query runs.

    Returning a non-zero value tells SQLite to abort the query. This is our
    guard against a runaway query (e.g. an accidental cross join) hanging
    the request forever.
    """
    def handler():
        return time.monotonic() > deadline
    return handler


def run_query(sql: str) -> tuple[list[str], list[dict]]:
    """Execute `sql` (already validated as a single SELECT) read-only.

    Returns (column_names, rows) where rows is a list of {column: value} dicts.
    """
    if not settings.DATABASE_PATH.exists():
        raise QueryExecutionError(
            f"Database not found at {settings.DATABASE_PATH}. "
            "Run `python database/build_db.py` first."
        )

    # mode=ro opens the file read-only at the OS level; SQLite will refuse
    # any write even if our SQL validation somehow missed one.
    conn = sqlite3.connect(
        f"file:{settings.DATABASE_PATH}?mode=ro",
        uri=True,
        timeout=settings.QUERY_TIMEOUT_SECONDS,
    )
    try:
        # Belt-and-braces: also ask SQLite itself to refuse writes.
        conn.execute("PRAGMA query_only = TRUE")
        conn.row_factory = sqlite3.Row

        deadline = time.monotonic() + settings.QUERY_TIMEOUT_SECONDS
        # n=1000: check the deadline roughly every 1000 "steps" of work SQLite does.
        conn.set_progress_handler(_make_progress_handler(deadline), 1000)

        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description] if cursor.description else []
        return columns, [dict(row) for row in rows]
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise QueryExecutionError(
                f"Query took longer than {settings.QUERY_TIMEOUT_SECONDS}s and was stopped."
            ) from exc
        raise QueryExecutionError(f"Database error: {exc}") from exc
    except sqlite3.Error as exc:
        raise QueryExecutionError(f"Database error: {exc}") from exc
    finally:
        conn.close()
