"""Turns a question into SQL, using either the real OpenAI model or the
free local mock (see backend/mock_llm.py), depending on SQLGENIE_LLM_MODE.

Keeping the API call in one place means: (1) it's easy to test/mock later,
(2) swapping providers or models later only touches this file.
"""
import re

from openai import OpenAI

from backend.config import settings
from backend.mock_llm import generate_sql_mock

SYSTEM_PROMPT_TEMPLATE = """You are a SQL generator for a SQLite database.

Given a database schema and a natural-language question, output ONE valid
SQLite SELECT query that answers the question. Rules:
- Output ONLY the SQL query. No explanations, no markdown, no comments.
- Use only the tables/columns/views listed below. Do not invent columns.
- Only generate read-only queries: SELECT (optionally with WITH ... SELECT).
  Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or any other
  data-changing or admin statement.
- Prefer the v_order_items_detail view for questions spanning orders,
  products, customers or sellers, instead of writing the joins yourself.
- If the question is ambiguous, make a reasonable assumption rather than
  asking for clarification (you cannot ask follow-up questions).

Database schema:
{schema}
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


def _generate_sql_openai(question: str, schema_text: str) -> str:
    """The real OpenAI call. Only used when SQLGENIE_LLM_MODE=openai."""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=0,  # deterministic-ish output; we want SQL, not creativity
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(schema=schema_text)},
            {"role": "user", "content": question},
        ],
    )

    raw_text = response.choices[0].message.content or ""
    return _clean_sql_response(raw_text)
