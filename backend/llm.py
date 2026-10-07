"""Turns a question into SQL, using the free local mock (see
backend/mock_llm.py), the real OpenAI API, or a local Ollama model,
depending on SQLGENIE_LLM_MODE.

Keeping the API call in one place means: (1) it's easy to test/mock later,
(2) swapping providers or models later only touches this file.

Day 5 adds a second entry point, correct_sql(), used by backend/agent.py's
self-correction loop when a first attempt's SQL fails against the real
database. It's a separate function (not a flag on generate_sql) because the
prompt is genuinely different: generate_sql only ever sees the question and
the schema, while correct_sql also needs to show the LLM what it tried
before and exactly how that failed.

All three modes (mock/openai/ollama) share the SAME SYSTEM_PROMPT_TEMPLATE
and CORRECTION_USER_TEMPLATE below — only HOW the prompt gets to a model
(or a lookup table, for mock) differs. That's deliberate: switching modes
should never change what the model is asked to do, only which model answers.
"""
import json
import re
import urllib.error
import urllib.request

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
- If the schema below includes "General notes", follow them — they contain
  dataset-specific guidance (e.g. which columns to avoid, or a convenience
  view to prefer over writing joins by hand).
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


def _unknown_mode_error() -> RuntimeError:
    return RuntimeError(
        f"Unknown SQLGENIE_LLM_MODE '{settings.SQLGENIE_LLM_MODE}'. "
        "Expected 'mock', 'openai' or 'ollama' in .env."
    )


def generate_sql(question: str, schema_text: str) -> str:
    """Turn `question` into one SQL query. This is the only function the
    rest of the app calls — it decides mock vs. OpenAI vs. Ollama
    internally, based on settings.SQLGENIE_LLM_MODE, so callers never need
    to care which one actually runs.

    Returns the raw SQL text. It is NOT validated here — sql_guard.py does
    that before anything executes it.
    """
    mode = settings.SQLGENIE_LLM_MODE
    if mode == "mock":
        # No network call, no API key needed — see mock_llm.py for why.
        return generate_sql_mock(question)
    if mode == "openai":
        return _generate_sql_openai(question, schema_text)
    if mode == "ollama":
        return _generate_sql_ollama(question, schema_text)
    raise _unknown_mode_error()


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
    mode = settings.SQLGENIE_LLM_MODE
    if mode == "mock":
        return correct_sql_mock(question, attempt_number)
    if mode == "openai":
        return _correct_sql_openai(question, schema_text, failed_sql, db_error)
    if mode == "ollama":
        return _correct_sql_ollama(question, schema_text, failed_sql, db_error)
    raise _unknown_mode_error()


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


# ---------------------------------------------------------------------------
# Ollama (local model). Uses stdlib urllib only — no extra dependency just
# for one HTTP call, and it keeps this third mode as easy to read/test as
# the two above it.
# ---------------------------------------------------------------------------

_OLLAMA_CHAT_PATH = "/api/chat"
# Local models can be slow, especially on the very first call after the
# server starts (loading the model into memory) — a generous timeout
# avoids a spurious "connection" error that's actually just a cold start.
_OLLAMA_TIMEOUT_SECONDS = 120


def _call_ollama(system_content: str, user_content: str) -> str:
    """Shared by _generate_sql_ollama and _correct_sql_ollama: POST to
    Ollama's /api/chat with the same system+user shape _call_openai uses,
    return the cleaned SQL text.

    "stream": false — we want one complete response, not a token stream
    (this app isn't built to consume one).
    "think": false — Qwen3 (and other reasoning-capable Ollama models) can
    emit a <think>...</think> reasoning block before its answer; we only
    want the final SQL, so this is turned off at the request level rather
    than having _clean_sql_response() try to strip it out after the fact.
    """
    url = f"{settings.OLLAMA_BASE_URL}{_OLLAMA_CHAT_PATH}"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "think": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_OLLAMA_TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Ollama answers a missing/unpulled model with an HTTP error whose
        # body names it, e.g. {"error": "model \"qwen3:4b\" not found, try
        # pulling it first"}. Surface that, plus the model we asked for.
        error_text = exc.read().decode("utf-8", errors="replace")
        try:
            error_text = json.loads(error_text).get("error", error_text)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(
            f"Ollama returned an error for model '{settings.OLLAMA_MODEL}': {error_text}. "
            f"If it isn't pulled yet, run: ollama pull {settings.OLLAMA_MODEL}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Covers connection-refused (Ollama not running), DNS failure, and
        # a read/connect timeout alike — from the caller's point of view
        # all of these mean the same thing: couldn't talk to Ollama.
        raise RuntimeError(
            f"Could not connect to Ollama at {settings.OLLAMA_BASE_URL}. "
            "Make sure Ollama is running."
        ) from exc

    try:
        body = json.loads(raw_body)
        raw_text = body["message"]["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response from Ollama (model '{settings.OLLAMA_MODEL}'): "
            f"could not find a message in {raw_body[:200]!r}"
        ) from exc

    return _clean_sql_response(raw_text)


def _generate_sql_ollama(question: str, schema_text: str) -> str:
    """The local Ollama call. Only used when SQLGENIE_LLM_MODE=ollama."""
    return _call_ollama(SYSTEM_PROMPT_TEMPLATE.format(schema=schema_text), question)


# ---------------------------------------------------------------------------
# Embeddings, for backend/schema_retrieval.py's embedding-based retrieval.
#
# Unlike generate_sql()/correct_sql() above, this has no mock/openai/ollama
# dispatch: retrieval always embeds locally via Ollama's OLLAMA_EMBED_MODEL,
# regardless of SQLGENIE_LLM_MODE — even when that setting is "openai" (SQL
# generation uses the cloud, retrieval still doesn't) or "mock" (tests fake
# this call out; see tests/conftest.py).
# ---------------------------------------------------------------------------

_OLLAMA_EMBEDDINGS_PATH = "/api/embeddings"


def embed_text(text: str) -> list[float]:
    """Return OLLAMA_EMBED_MODEL's embedding vector for `text`, via Ollama's
    local /api/embeddings endpoint. Raises RuntimeError with a readable
    message on any failure (Ollama not running, model not pulled, a bad
    response) — same error-handling shape as _call_ollama above, just
    against a different endpoint and response shape (`{"embedding": [...]}`
    instead of `{"message": {"content": ...}}`).
    """
    url = f"{settings.OLLAMA_BASE_URL}{_OLLAMA_EMBEDDINGS_PATH}"
    payload = {"model": settings.OLLAMA_EMBED_MODEL, "prompt": text}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_OLLAMA_TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        try:
            error_text = json.loads(error_text).get("error", error_text)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(
            f"Ollama returned an error for embedding model '{settings.OLLAMA_EMBED_MODEL}': {error_text}. "
            f"If it isn't pulled yet, run: ollama pull {settings.OLLAMA_EMBED_MODEL}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {settings.OLLAMA_BASE_URL} for embeddings. "
            "Make sure Ollama is running."
        ) from exc

    try:
        body = json.loads(raw_body)
        embedding = body["embedding"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected response from Ollama embeddings (model '{settings.OLLAMA_EMBED_MODEL}'): "
            f"could not find an embedding in {raw_body[:200]!r}"
        ) from exc
    return embedding


def _correct_sql_ollama(question: str, schema_text: str, failed_sql: str, db_error: str) -> str:
    """The local Ollama correction call. Only used when SQLGENIE_LLM_MODE=ollama."""
    user_content = CORRECTION_USER_TEMPLATE.format(
        question=question, failed_sql=failed_sql, db_error=db_error
    )
    return _call_ollama(SYSTEM_PROMPT_TEMPLATE.format(schema=schema_text), user_content)
