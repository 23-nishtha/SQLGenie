"""Validates LLM-generated SQL before it's allowed anywhere near the database.

This is the first of two safety layers (the second is db.py opening the
connection read-only). Nothing here trusts the LLM's output:
  1. It must be exactly one statement (blocks "SELECT ...; DROP TABLE ...").
  2. It must be a read-only query (SELECT, optionally starting with WITH).
  3. It must not contain any data-changing or admin keyword, anywhere,
     including inside a subquery or CTE.
  4. If it has no LIMIT, we add one, so a broad question can't return the
     entire database.

We use the `sqlparse` library to tokenize the SQL rather than just
string-searching for forbidden words. That matters because a naive
`"DELETE" in sql.upper()` check would also block a harmless query like
`SELECT * FROM order_reviews WHERE review_comment_message LIKE '%delete%'`,
where "delete" is just data, not a SQL keyword.
"""
import sqlparse
from sqlparse.tokens import DDL, DML, Keyword

from backend.config import settings

# Keywords that would change data or database structure, or affect other
# connections/sessions. If any of these appear as an actual SQL keyword
# token anywhere in the statement, we reject it.
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE",
    "DROP", "ALTER", "CREATE", "TRUNCATE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
    "GRANT", "REVOKE",
    "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
}


class SqlValidationError(ValueError):
    """Raised when generated SQL fails validation. Message is safe to show the user."""


def _is_forbidden_keyword_token(token) -> bool:
    is_keyword_token = token.ttype in (DDL, DML, Keyword)
    return is_keyword_token and token.value.upper() in FORBIDDEN_KEYWORDS


def _starts_with_select_or_with(statement: sqlparse.sql.Statement) -> bool:
    first_keyword = statement.token_first(skip_cm=True)  # skip comments/whitespace
    if first_keyword is None:
        return False
    return first_keyword.value.upper() in ("SELECT", "WITH")


def _has_limit_clause(statement: sqlparse.sql.Statement) -> bool:
    return any(
        token.ttype is Keyword and token.value.upper() == "LIMIT"
        for token in statement.flatten()
    )


def validate_and_prepare(raw_sql: str) -> str:
    """Validate `raw_sql`; return the exact SQL string that is safe to execute.

    Raises SqlValidationError with a human-readable reason if the query is
    rejected.
    """
    sql = raw_sql.strip()
    if not sql:
        raise SqlValidationError("The model returned an empty SQL query.")

    # Exactly one statement. sqlparse.split() splits on top-level semicolons;
    # a trailing "select 1;" splits into ["select 1", ""], so drop empties.
    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) != 1:
        raise SqlValidationError(
            f"Expected exactly one SQL statement, got {len(statements)}."
        )

    parsed = sqlparse.parse(statements[0])[0]

    if not _starts_with_select_or_with(parsed):
        raise SqlValidationError("Only SELECT queries are allowed.")

    for token in parsed.flatten():
        if _is_forbidden_keyword_token(token):
            raise SqlValidationError(
                f"Query rejected: contains disallowed keyword '{token.value.upper()}'."
            )

    clean_sql = statements[0].strip().rstrip(";")

    if not _has_limit_clause(parsed):
        # Wrap rather than blindly append "LIMIT N", so this still works
        # even if the query already ends with ORDER BY, a comment, etc.
        clean_sql = f"SELECT * FROM ({clean_sql}) AS _sqlgenie_limited LIMIT {settings.MAX_ROWS}"

    return clean_sql
