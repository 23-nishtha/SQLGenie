"""Deterministic mock "LLM" for local development without an OpenAI budget.

Why this file exists
---------------------
SQLGenie's pipeline is:

    question -> [generate SQL] -> validate -> execute -> results

Only the "generate SQL" step needs a paid API call (backend/llm.py calling
OpenAI). Everything after it — backend/sql_guard.py and backend/db.py — is
ordinary Python running against our own free, local SQLite database. Mock
mode lets us build and test that entire second half of the pipeline (plus
the FastAPI endpoint wiring in main.py) without an OpenAI API key or quota.

It's controlled by one setting in .env:

    SQLGENIE_LLM_MODE=mock     # use this file
    SQLGENIE_LLM_MODE=openai   # use the real model in llm.py

backend/llm.py is the only file that reads that setting and decides which
one to call. Nothing else in the app — sql_guard.py, db.py, main.py — knows
or cares which mode produced the SQL, so switching back to OpenAI later
(once there's quota) is a one-line .env change, no code changes required.

How matching works
-------------------
This is a lookup table, not a real model: a fixed set of known questions,
each mapped to a hand-written SQL query. The incoming question is
"normalized" (lowercased, punctuation removed, extra spaces collapsed)
before the lookup, so small variations like a trailing "?" still match. An
unrecognized question raises a clear error listing what IS supported,
rather than guessing or failing silently.
"""
import re

# Every value here is a complete, valid SQLite SELECT statement — the exact
# same shape backend/llm.py's real OpenAI call would return. It still gets
# checked by sql_guard.py and executed by db.py like any other generated SQL.
_MOCK_ANSWERS: dict[str, str] = {
    "how many orders were delivered": (
        "SELECT COUNT(*) AS delivered_orders "
        "FROM orders WHERE order_status = 'delivered'"
    ),
    "how many orders are there": (
        "SELECT COUNT(*) AS total_orders FROM orders"
    ),
    "what are the top 5 product categories by revenue": (
        "SELECT product_category_english, SUM(price) AS revenue "
        "FROM v_order_items_detail "
        "GROUP BY product_category_english "
        "ORDER BY revenue DESC "
        "LIMIT 5"
    ),
    "what is the average order value": (
        "SELECT AVG(order_total) AS average_order_value FROM ("
        "  SELECT order_id, SUM(price + freight_value) AS order_total "
        "  FROM order_items GROUP BY order_id"
        ")"
    ),
    # A couple of extras, so mock mode is a slightly richer demo than the
    # minimum 4 questions.
    "how many customers are there": (
        "SELECT COUNT(DISTINCT customer_unique_id) AS unique_customers FROM customers"
    ),
    "what is the total revenue": (
        "SELECT SUM(price + freight_value) AS total_revenue FROM order_items"
    ),
}


class MockQuestionNotFound(ValueError):
    """Raised when the question isn't one mock mode knows how to answer."""


def _normalize(question: str) -> str:
    text = question.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)  # drop "?", ".", etc.
    text = re.sub(r"\s+", " ", text)     # collapse repeated whitespace
    return text


def generate_sql_mock(question: str) -> str:
    """Look up canned SQL for `question`. Raises MockQuestionNotFound if unknown."""
    key = _normalize(question)
    if key not in _MOCK_ANSWERS:
        supported = "\n  - ".join(_MOCK_ANSWERS)
        raise MockQuestionNotFound(
            "Mock LLM mode doesn't recognize this question. Try one of:\n"
            f"  - {supported}\n"
            "(Or set SQLGENIE_LLM_MODE=openai in .env to use the real model instead.)"
        )
    return _MOCK_ANSWERS[key]
