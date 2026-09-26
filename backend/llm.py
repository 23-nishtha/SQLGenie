"""Turns a question into SQL, using either the real OpenAI model or the
free local mock (see backend/mock_llm.py), depending on SQLGENIE_LLM_MODE.

Keeping the API call in one place means: (1) it's easy to test/mock later,
(2) swapping providers or models later only touches this file.

Day 5 adds a second entry point, correct_sql(), used by backend/agent.py's
self-correction loop when a first attempt's SQL fails against the real
database. It's a separate function (not a flag on generate_sql) because the
prompt is genuinely different: generate_sql only ever sees the question and
the schema, while correct_sql also needs to show the LLM what it tried
before and exactly how that failed.
"""
import re

from openai import OpenAI

from backend.config import settings
from backend.mock_llm import correct_sql_mock, generate_sql_mock

SYSTEM_PROMPT_TEMPLATE = """You are a SQL generator for a SQLite database.

You are given a curated, relevant SUBSET of the database schema below (not
necessarily every table that exists), selected for this specific question.
Given that schema and a natural-language question, output ONE valid SQLite
SELECT query that answers the question. Rules:
- Output ONLY the SQL query. No explanations, no markdown, no comments.
- Use only the tables/columns/views listed below. Do not invent columns.
- Only generate read-only queries: SELECT (optionally with WITH ... SELECT).
  Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or any other
  data-changing or admin statement.
- Prefer the v_order_items_detail view for questions spanning orders,
  products, customers or sellers, instead of writing the joins yourself.
- If the question is ambiguous, make a reasonable assumption rather than
  asking for clarification (you cannot ask follow-up questions).

Relevant schema for this question:
{schema}
"""

# Used only by correct_sql()'s OpenAI path. Sent as the user message, with
# the same SYSTEM_PROMPT_TEMPLATE (same schema, same rules) as the system
# message, so the model doesn't need the rules repeated here.
CORRECTION_USER_TEMPLATE = """The SQL query below was generated for this question, but failed when run against the real database. Fix it.

Original question: {question}

Previous SQL:
{failed_sql}

Exact database error when running it:
{db_error}

Return ONLY the corrected SQL query. Same rules as before: one read-only
SELECT (optionally WITH ... SELECT) statement, no explanations, no markdown.
"""

# Strips a ```sql ... ``` or ``` ... ``` fence if the model wraps its answer
# in one despite being told not to. Small models especially tend to do this.
_CODE_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _clean_sql_response(raw_text: str) -> str:
    text = raw_text.strip()
    text = _CODE_FENCE_RE.sub("", text).strip()
    return text


def generate_sql(question: str, schema_text: str) -> str:
    """Turn `question` into one SQL query. This is the only function the
    rest of the app calls — it decides mock vs. real OpenAI internally,
    based on settings.SQLGENIE_LLM_MODE, so callers never need to care.

    Returns the raw SQL text. It is NOT validated here — sql_guard.py does
    that before anything executes it.
    """
    if settings.SQLGENIE_LLM_MODE == "mock":
        # No network call, no API key needed — see mock_llm.py for why.
        return generate_sql_mock(question)

    if settings.SQLGENIE_LLM_MODE != "openai":
        raise RuntimeError(
            f"Unknown SQLGENIE_LLM_MODE '{settings.SQLGENIE_LLM_MODE}'. "
            "Expected 'mock' or 'openai' in .env."
        )

    return _generate_sql_openai(question, schema_text)


def correct_sql(
    question: str,
    schema_text: str,
    failed_sql: str,
    db_error: str,
    attempt_number: int = 1,
) -> str:
    """Ask for a corrected SQL query after `failed_sql` errored out against
    the real database with `db_error`. Same mock/OpenAI dispatch pattern as
    generate_sql() — backend/agent.py doesn't need to know which one runs.

    `db_error` must already be a clean, single-line message (see
    backend/db.py's QueryExecutionError) — never a raw Python traceback —
    since it gets shown to the LLM verbatim and, on total failure, may end
    up in the API's error response too.
    """
    if settings.SQLGENIE_LLM_MODE == "mock":
        return correct_sql_mock(question, attempt_number)

    if settings.SQLGENIE_LLM_MODE != "openai":
        raise RuntimeError(
            f"Unknown SQLGENIE_LLM_MODE '{settings.SQLGENIE_LLM_MODE}'. "
            "Expected 'mock' or 'openai' in .env."
        )

    return _correct_sql_openai(question, schema_text, failed_sql, db_error)


def _call_openai(system_content: str, user_content: str) -> str:
    """Shared by _generate_sql_openai and _correct_sql_openai: send one
    system + one user message, return the cleaned SQL text."""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=0,  # deterministic-ish output; we want SQL, not creativity
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )

    raw_text = response.choices[0].message.content or ""
    return _clean_sql_response(raw_text)


def _generate_sql_openai(question: str, schema_text: str) -> str:
    """The real OpenAI call. Only used when SQLGENIE_LLM_MODE=openai."""
    return _call_openai(SYSTEM_PROMPT_TEMPLATE.format(schema=schema_text), question)


def _correct_sql_openai(question: str, schema_text: str, failed_sql: str, db_error: str) -> str:
    """The real OpenAI correction call. Only used when SQLGENIE_LLM_MODE=openai."""
    user_content = CORRECTION_USER_TEMPLATE.format(
        question=question, failed_sql=failed_sql, db_error=db_error
    )
    return _call_openai(SYSTEM_PROMPT_TEMPLATE.format(schema=schema_text), user_content)
