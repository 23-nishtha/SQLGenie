"""Tests for the Ollama (local LLM) mode added to backend/llm.py.

No real Ollama server is ever contacted — urllib.request.urlopen is
monkeypatched to a fake, in-memory response for every test here. This also
means these tests are independent of tests/conftest.py's pinned
SQLGENIE_LLM_MODE=mock: each test that needs "ollama" sets it explicitly
via monkeypatch, which auto-reverts after the test.
"""
import io
import json
import urllib.error
import urllib.request

import pytest

from backend import llm
from backend.config import settings


class _FakeOllamaResponse:
    """Stands in for the object urllib.request.urlopen() normally returns —
    a context manager whose .read() gives raw bytes."""

    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _ollama_reply(content: str):
    """A fake urlopen() that returns `content` as the assistant message,
    and records the Request object it was called with."""
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        return _FakeOllamaResponse(
            {"model": settings.OLLAMA_MODEL, "message": {"role": "assistant", "content": content}}
        )

    return fake_urlopen, calls


@pytest.fixture
def ollama_mode(monkeypatch):
    """Switch to ollama mode for one test; auto-reverts afterward."""
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Dispatch: SQLGENIE_LLM_MODE=ollama routes to the Ollama functions.
# ---------------------------------------------------------------------------


def test_generate_sql_dispatches_to_ollama(ollama_mode, monkeypatch):
    fake_urlopen, calls = _ollama_reply("SELECT COUNT(*) FROM orders")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    sql = llm.generate_sql("How many orders?", "TABLE orders(order_id TEXT)")

    assert sql == "SELECT COUNT(*) FROM orders"
    assert len(calls) == 1


def test_correct_sql_dispatches_to_ollama(ollama_mode, monkeypatch):
    fake_urlopen, calls = _ollama_reply("SELECT COUNT(*) FROM orders WHERE status='x'")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    sql = llm.correct_sql(
        question="How many orders?",
        schema_text="TABLE orders(order_id TEXT)",
        failed_sql="SELECT COUNT(*) FROM orders WHERE bad_col=1",
        db_error="no such column: bad_col",
        attempt_number=1,
    )

    assert sql == "SELECT COUNT(*) FROM orders WHERE status='x'"
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Request shape: URL, payload, headers.
# ---------------------------------------------------------------------------


def test_generate_sql_request_shape_is_correct(ollama_mode, monkeypatch):
    fake_urlopen, calls = _ollama_reply("SELECT 1")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    llm.generate_sql("How many orders?", "TABLE orders(order_id TEXT)")

    request = calls[0]
    assert request.full_url == "http://localhost:11434/api/chat"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"

    payload = json.loads(request.data)
    assert payload["model"] == "qwen3:4b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["messages"] == [
        {"role": "system", "content": llm.SYSTEM_PROMPT_TEMPLATE.format(schema="TABLE orders(order_id TEXT)")},
        {"role": "user", "content": "How many orders?"},
    ]


def test_correct_sql_request_uses_the_correction_template(ollama_mode, monkeypatch):
    fake_urlopen, calls = _ollama_reply("SELECT 1")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    llm.correct_sql(
        question="How many orders?",
        schema_text="TABLE orders(order_id TEXT)",
        failed_sql="SELECT bad",
        db_error="no such column: bad",
        attempt_number=2,
    )

    payload = json.loads(calls[0].data)
    user_message = payload["messages"][1]["content"]
    assert user_message == llm.CORRECTION_USER_TEMPLATE.format(
        question="How many orders?", failed_sql="SELECT bad", db_error="no such column: bad",
    )
    assert "How many orders?" in user_message
    assert "SELECT bad" in user_message
    assert "no such column: bad" in user_message


# ---------------------------------------------------------------------------
# Markdown fence stripping (requirement 7 — reuses _clean_sql_response).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("  SELECT 1  ", "SELECT 1"),
])
def test_code_fences_are_stripped_from_ollama_responses(ollama_mode, monkeypatch, raw, expected):
    fake_urlopen, _ = _ollama_reply(raw)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert llm.generate_sql("q", "s") == expected


# ---------------------------------------------------------------------------
# Error handling (requirement 8).
# ---------------------------------------------------------------------------


def test_connection_failure_raises_clear_runtime_error(ollama_mode, monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Could not connect to Ollama at http://localhost:11434"):
        llm.generate_sql("q", "s")


def test_timeout_also_raises_the_connection_error_message(ollama_mode, monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        llm.generate_sql("q", "s")


def test_missing_model_error_names_the_configured_model(ollama_mode, monkeypatch):
    def fake_urlopen(request, timeout=None):
        body = b'{"error": "model \\"qwen3:4b\\" not found, try pulling it first"}'
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="qwen3:4b"):
        llm.generate_sql("q", "s")


def test_malformed_response_body_raises_a_clear_error(ollama_mode, monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeOllamaResponse({"unexpected": "shape"})  # no "message" key

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Unexpected response from Ollama"):
        llm.generate_sql("q", "s")


def test_correct_sql_connection_failure_also_raises_clearly(ollama_mode, monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        llm.correct_sql("q", "s", "SELECT bad", "no such column: bad")


# ---------------------------------------------------------------------------
# Regression: mock and openai modes are untouched.
# ---------------------------------------------------------------------------


def test_mock_mode_still_works(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "mock")
    sql = llm.generate_sql("How many orders are there?", "unused in mock mode")
    assert sql == "SELECT COUNT(*) AS total_orders FROM orders"


def test_openai_mode_still_dispatches_to_the_openai_function(monkeypatch):
    """Doesn't call the real OpenAI API — just confirms the dispatcher
    still routes to _generate_sql_openai (unchanged) rather than ollama."""
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "openai")
    monkeypatch.setattr(llm, "_generate_sql_openai", lambda q, s: "SELECT 'openai path used'")

    assert llm.generate_sql("q", "s") == "SELECT 'openai path used'"


def test_openai_correct_sql_still_dispatches_correctly(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "openai")
    monkeypatch.setattr(llm, "_correct_sql_openai", lambda q, s, f, e: "SELECT 'openai correction'")

    assert llm.correct_sql("q", "s", "bad sql", "an error") == "SELECT 'openai correction'"


def test_unknown_mode_error_lists_all_three_modes(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "something-else")
    with pytest.raises(RuntimeError, match="mock.*openai.*ollama"):
        llm.generate_sql("q", "s")
