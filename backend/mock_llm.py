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

    # --- Day 8: Football dataset (table: results) ---
    # These only produce correct results when the "football" dataset is
    # active — same as the Olist answers above only make sense for
    # "olist". Mock mode doesn't know which dataset is active; it's just a
    # lookup table. Picking a question here while the WRONG dataset is
    # active fails at execution (no such table), exactly like it would for
    # a real LLM asked to write SQL against a schema it wasn't shown.
    "which team has the most wins": (
        "SELECT team, COUNT(*) AS wins FROM ("
        "  SELECT home_team AS team FROM results WHERE winner = 'home'"
        "  UNION ALL"
        "  SELECT away_team AS team FROM results WHERE winner = 'away'"
        ") GROUP BY team ORDER BY wins DESC LIMIT 1"
    ),
    "how many matches are there": "SELECT COUNT(*) AS total_matches FROM results",
    "what is the highest goal difference in a match": (
        "SELECT MAX(goal_diff) AS highest_goal_diff FROM results"
    ),
    "what is the average number of goals scored per match": (
        "SELECT AVG(home_score + away_score) AS average_goals FROM results"
    ),
    "how many matches has brazil played": (
        "SELECT COUNT(*) AS brazil_matches FROM results "
        "WHERE home_team = 'Brazil' OR away_team = 'Brazil'"
    ),

    # --- Day 8: Movies dataset (table: movies) ---
    "what are the top 5 highest rated movies": (
        "SELECT series_title, imdb_rating FROM movies ORDER BY imdb_rating DESC LIMIT 5"
    ),
    "what is the average imdb rating": "SELECT AVG(imdb_rating) AS average_rating FROM movies",
    "how many movies are there": "SELECT COUNT(*) AS total_movies FROM movies",
    "which director has the most movies": (
        "SELECT director, COUNT(*) AS movie_count FROM movies "
        "GROUP BY director ORDER BY movie_count DESC LIMIT 1"
    ),
    "what is the average rating by genre": (
        "SELECT primary_genre, AVG(imdb_rating) AS average_rating FROM movies "
        "GROUP BY primary_genre ORDER BY average_rating DESC"
    ),
}


# Day 5 self-correction scenarios: each entry scripts a small multi-step
# conversation, so backend/agent.py's error-driven retry loop can be tested
# end-to-end without any OpenAI call.
#
# "initial" is what generate_sql_mock() returns on the FIRST try — it's
# deliberately wrong SQL that passes sql_guard.py (it's a normal-looking
# SELECT) but fails for real against the real olist.db (a column/table
# that doesn't exist), producing a genuine sqlite3 error, not a fake one.
# "corrections" is the sequence of what correct_sql_mock() returns for the
# 1st, 2nd, ... correction attempt.
_MOCK_CORRECTION_SCENARIOS: dict[str, dict] = {
    "how many orders were placed in 2017": {
        "initial": "SELECT COUNT(*) FROM orders WHERE order_year = 2017",  # order_year doesn't exist
        "corrections": [
            # Fixed on the first correction attempt: no such column existed
            # above, so this uses the real column with strftime() instead.
            "SELECT COUNT(*) AS orders_in_2017 FROM orders "
            "WHERE strftime('%Y', order_purchase_timestamp) = '2017'",
        ],
    },
    "this question always fails even after correction": {
        "initial": "SELECT * FROM not_a_real_table",
        "corrections": [
            "SELECT * FROM still_not_a_real_table",  # 1st correction: still broken
            "SELECT * FROM also_not_a_real_table",   # 2nd correction: still broken -> retries exhausted
        ],
    },
    # Day 8: one self-correction scenario per new dataset, same shape as
    # Olist's above — a plausible-looking but wrong column name, fixed on
    # the first retry with the real column/expression.
    "how many matches were played in 2018": {
        "initial": "SELECT COUNT(*) FROM results WHERE year = 2018",  # `year` doesn't exist
        "corrections": [
            "SELECT COUNT(*) AS matches_in_2018 FROM results "
            "WHERE strftime('%Y', date) = '2018'",
        ],
    },
    "how many movies were released after 2010": {
        "initial": "SELECT COUNT(*) FROM movies WHERE release_year > 2010",  # typo: should be released_year
        "corrections": [
            "SELECT COUNT(*) AS movies_after_2010 FROM movies WHERE released_year > 2010",
        ],
    },
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
    if key in _MOCK_CORRECTION_SCENARIOS:
        return _MOCK_CORRECTION_SCENARIOS[key]["initial"]
    if key not in _MOCK_ANSWERS:
        supported = "\n  - ".join((*_MOCK_ANSWERS, *_MOCK_CORRECTION_SCENARIOS))
        raise MockQuestionNotFound(
            "Mock LLM mode doesn't recognize this question. Try one of:\n"
            f"  - {supported}\n"
            "(Or set SQLGENIE_LLM_MODE=openai in .env to use the real model instead.)"
        )
    return _MOCK_ANSWERS[key]


def correct_sql_mock(question: str, attempt_number: int) -> str:
    """Look up the scripted correction for `question`'s `attempt_number`
    (1 = first correction, 2 = second correction, ...).

    Real OpenAI correction reads the actual database error and reasons
    about it; mock mode can't do that, so it's a scripted stand-in — a
    fixed, numbered sequence of "next things to try" per question. That's
    enough to exercise the retry loop in backend/agent.py for real, without
    needing a real model to actually understand the error.
    """
    key = _normalize(question)
    scenario = _MOCK_CORRECTION_SCENARIOS.get(key)
    if scenario is None:
        raise MockQuestionNotFound(
            f"Mock LLM mode has no scripted self-correction for {question!r}. "
            "Self-correction scenarios are only defined for: "
            + ", ".join(_MOCK_CORRECTION_SCENARIOS)
        )
    corrections = scenario["corrections"]
    index = attempt_number - 1
    if index < 0 or index >= len(corrections):
        raise MockQuestionNotFound(
            f"Mock LLM mode ran out of scripted corrections for {question!r} "
            f"after {len(corrections)} attempt(s)."
        )
    return corrections[index]
