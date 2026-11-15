"""Tests for the Gemini (Google cloud API) mode added to backend/llm.py.

No real Gemini API is ever contacted — google.genai.Client is monkeypatched
to an in-memory fake for every test here. Like tests/test_llm_ollama.py, each
test that needs "gemini" sets it explicitly via monkeypatch (auto-reverts
after the test), independent of tests/conftest.py's pinned
SQLGENIE_LLM_MODE=mock.
"""
import pytest
from google.genai import errors as genai_errors

from backend import llm
from backend.config import settings


# ---------------------------------------------------------------------------
# Fakes: stand in for google.genai.Client without any network call.
# ---------------------------------------------------------------------------


class _FakeGenerateContentResponse:
    """Stands in for types.GenerateContentResponse: `.text` is the only
    attribute _call_gemini reads off it."""

    def __init__(self, text):
        self.text = text


class _FakeContentEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedContentResponse:
    def __init__(self, values):
        self.embeddings = [_FakeContentEmbedding(values)] if values is not None else []


class _FakeModels:
    """Fake for client.models — records every call it receives and either
    returns a canned response or raises a canned exception."""

    def __init__(self, generate_result=None, embed_result=None):
        self._generate_result = generate_result
        self._embed_result = embed_result
        self.generate_calls: list[dict] = []
        self.embed_calls: list[dict] = []

    def generate_content(self, *, model, contents, config=None):
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self._generate_result, BaseException):
            raise self._generate_result
        return self._generate_result

    def embed_content(self, *, model, contents):
        self.embed_calls.append({"model": model, "contents": contents})
        if isinstance(self._embed_result, BaseException):
            raise self._embed_result
        return self._embed_result


def _install_fake_gemini_client(monkeypatch, generate_result=None, embed_result=None):
    """Monkeypatch genai.Client so every _gemini_client() call in llm.py
    returns one shared _FakeModels instance, whose recorded calls the test
    can inspect afterward."""
    fake_models = _FakeModels(generate_result=generate_result, embed_result=embed_result)

    class _FakeClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.models = fake_models

    monkeypatch.setattr(llm.genai, "Client", _FakeClient)
    return fake_models


@pytest.fixture
def gemini_mode(monkeypatch):
    """Switch to gemini mode for one test; auto-reverts afterward."""
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(settings, "GEMINI_EMBED_MODEL", "gemini-embedding-001")


def _client_error(code, message, status="RESOURCE_EXHAUSTED"):
    return genai_errors.ClientError(
        code, {"error": {"code": code, "message": message, "status": status}}
    )


# ---------------------------------------------------------------------------
# Dispatch: SQLGENIE_LLM_MODE=gemini routes to the Gemini functions.
# ---------------------------------------------------------------------------


def test_generate_sql_dispatches_to_gemini(gemini_mode, monkeypatch):
    fake_models = _install_fake_gemini_client(
        monkeypatch, generate_result=_FakeGenerateContentResponse("SELECT COUNT(*) FROM orders")
    )

    sql = llm.generate_sql("How many orders?", "TABLE orders(order_id TEXT)")

    assert sql == "SELECT COUNT(*) FROM orders"
    assert len(fake_models.generate_calls) == 1


def test_correct_sql_dispatches_to_gemini(gemini_mode, monkeypatch):
    fake_models = _install_fake_gemini_client(
        monkeypatch,
        generate_result=_FakeGenerateContentResponse("SELECT COUNT(*) FROM orders WHERE status='x'"),
    )

    sql = llm.correct_sql(
        question="How many orders?",
        schema_text="TABLE orders(order_id TEXT)",
        failed_sql="SELECT COUNT(*) FROM orders WHERE bad_col=1",
        db_error="no such column: bad_col",
        attempt_number=1,
    )

    assert sql == "SELECT COUNT(*) FROM orders WHERE status='x'"
    assert len(fake_models.generate_calls) == 1


# ---------------------------------------------------------------------------
# Request shape: model, system instruction, temperature, contents.
# ---------------------------------------------------------------------------


def test_generate_sql_request_shape_is_correct(gemini_mode, monkeypatch):
    fake_models = _install_fake_gemini_client(
        monkeypatch, generate_result=_FakeGenerateContentResponse("SELECT 1")
    )

    llm.generate_sql("How many orders?", "TABLE orders(order_id TEXT)")

    call = fake_models.generate_calls[0]
    assert call["model"] == "gemini-3.6-flash"
    assert call["contents"] == "How many orders?"
    assert call["config"].system_instruction == llm.SYSTEM_PROMPT_TEMPLATE.format(
        schema="TABLE orders(order_id TEXT)"
    )
    assert call["config"].temperature == 0


def test_correct_sql_request_uses_the_correction_template(gemini_mode, monkeypatch):
    fake_models = _install_fake_gemini_client(
        monkeypatch, generate_result=_FakeGenerateContentResponse("SELECT 1")
    )

    llm.correct_sql(
        question="How many orders?",
        schema_text="TABLE orders(order_id TEXT)",
        failed_sql="SELECT bad",
        db_error="no such column: bad",
        attempt_number=2,
    )

    user_content = fake_models.generate_calls[0]["contents"]
    assert user_content == llm.CORRECTION_USER_TEMPLATE.format(
        question="How many orders?", failed_sql="SELECT bad", db_error="no such column: bad",
    )


def test_api_key_is_passed_to_the_client(gemini_mode, monkeypatch):
    captured_kwargs = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.models = _FakeModels(generate_result=_FakeGenerateContentResponse("SELECT 1"))

    monkeypatch.setattr(llm.genai, "Client", _FakeClient)

    llm.generate_sql("q", "s")

    assert captured_kwargs["api_key"] == "test-gemini-key"


def test_429_is_not_in_the_retryable_status_codes(gemini_mode, monkeypatch):
    """Stage 2 requirement: don't burn extra quota-consuming attempts
    retrying a request that's already been told its quota is exhausted.
    5xx (transient, server-side) is still retried."""
    captured_kwargs = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.models = _FakeModels(generate_result=_FakeGenerateContentResponse("SELECT 1"))

    monkeypatch.setattr(llm.genai, "Client", _FakeClient)

    llm.generate_sql("q", "s")

    retry_codes = captured_kwargs["http_options"].retry_options.http_status_codes
    assert 429 not in retry_codes
    assert 503 in retry_codes


# ---------------------------------------------------------------------------
# Markdown fence stripping (reuses the shared _clean_sql_response).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("  SELECT 1  ", "SELECT 1"),
])
def test_code_fences_are_stripped_from_gemini_responses(gemini_mode, monkeypatch, raw, expected):
    _install_fake_gemini_client(monkeypatch, generate_result=_FakeGenerateContentResponse(raw))

    assert llm.generate_sql("q", "s") == expected


# ---------------------------------------------------------------------------
# Error handling: missing key, rate limit/quota (429), server errors,
# network failures, malformed responses.
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        llm.generate_sql("q", "s")


def test_rate_limit_error_is_reported_clearly(gemini_mode, monkeypatch):
    _install_fake_gemini_client(
        monkeypatch,
        generate_result=_client_error(429, "Quota exceeded for this model."),
    )

    with pytest.raises(RuntimeError, match="429"):
        llm.generate_sql("q", "s")


def test_rate_limit_error_message_names_the_model(gemini_mode, monkeypatch):
    _install_fake_gemini_client(
        monkeypatch,
        generate_result=_client_error(429, "Quota exceeded for this model."),
    )

    with pytest.raises(RuntimeError, match="gemini-3.6-flash"):
        llm.generate_sql("q", "s")


def test_server_error_is_reported_clearly(gemini_mode, monkeypatch):
    _install_fake_gemini_client(
        monkeypatch,
        generate_result=genai_errors.ServerError(
            503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}
        ),
    )

    with pytest.raises(RuntimeError, match="503"):
        llm.generate_sql("q", "s")


def test_network_failure_raises_a_clear_runtime_error(gemini_mode, monkeypatch):
    _install_fake_gemini_client(monkeypatch, generate_result=ConnectionError("connection reset"))

    with pytest.raises(RuntimeError, match="Could not reach the Gemini API"):
        llm.generate_sql("q", "s")


def test_empty_response_text_raises_a_clear_error(gemini_mode, monkeypatch):
    _install_fake_gemini_client(monkeypatch, generate_result=_FakeGenerateContentResponse(""))

    with pytest.raises(RuntimeError, match="Unexpected response from Gemini"):
        llm.generate_sql("q", "s")


def test_none_response_text_raises_a_clear_error(gemini_mode, monkeypatch):
    _install_fake_gemini_client(monkeypatch, generate_result=_FakeGenerateContentResponse(None))

    with pytest.raises(RuntimeError, match="Unexpected response from Gemini"):
        llm.generate_sql("q", "s")


def test_correct_sql_connection_failure_also_raises_clearly(gemini_mode, monkeypatch):
    _install_fake_gemini_client(monkeypatch, generate_result=ConnectionError("connection reset"))

    with pytest.raises(RuntimeError, match="Could not reach the Gemini API"):
        llm.correct_sql("q", "s", "SELECT bad", "no such column: bad")


# ---------------------------------------------------------------------------
# embed_text(): gemini mode uses the Gemini API; other modes still use
# Ollama exactly as before (the one new branch, everything else unchanged).
# ---------------------------------------------------------------------------


def test_embed_text_dispatches_to_gemini_in_gemini_mode(gemini_mode, monkeypatch):
    fake_models = _install_fake_gemini_client(
        monkeypatch, embed_result=_FakeEmbedContentResponse([0.1, 0.2, 0.3])
    )

    vector = llm.embed_text("how many orders are there?")

    assert vector == [0.1, 0.2, 0.3]
    assert fake_models.embed_calls == [
        {"model": "gemini-embedding-001", "contents": "how many orders are there?"}
    ]


def test_embed_text_rate_limit_error_is_reported_clearly(gemini_mode, monkeypatch):
    _install_fake_gemini_client(
        monkeypatch, embed_result=_client_error(429, "Quota exceeded for embeddings.")
    )

    with pytest.raises(RuntimeError, match="429"):
        llm.embed_text("q")


def test_embed_text_empty_values_raises_a_clear_error(gemini_mode, monkeypatch):
    _install_fake_gemini_client(monkeypatch, embed_result=_FakeEmbedContentResponse(None))

    with pytest.raises(RuntimeError, match="Unexpected response from Gemini embeddings"):
        llm.embed_text("q")


def test_embed_text_still_uses_ollama_outside_gemini_mode(monkeypatch):
    """Regression: mock/openai/ollama modes must keep embedding via Ollama,
    never Gemini — even if a GEMINI_API_KEY happens to be configured."""
    import json
    import urllib.request

    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "ollama")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-gemini-key")

    calls = []

    class _FakeOllamaResponse:
        def __init__(self, payload):
            self._data = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        return _FakeOllamaResponse({"embedding": [1.0, 2.0]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    vector = llm.embed_text("q")

    assert vector == [1.0, 2.0]
    assert len(calls) == 1
    assert calls[0].full_url.endswith("/api/embeddings")


# ---------------------------------------------------------------------------
# Regression: mock, openai and ollama generation/correction dispatch are
# untouched by adding the gemini branch.
# ---------------------------------------------------------------------------


def test_mock_mode_still_works(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "mock")
    sql = llm.generate_sql("How many orders are there?", "unused in mock mode")
    assert sql == "SELECT COUNT(*) AS total_orders FROM orders"


def test_openai_mode_still_dispatches_to_the_openai_function(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "openai")
    monkeypatch.setattr(llm, "_generate_sql_openai", lambda q, s: "SELECT 'openai path used'")

    assert llm.generate_sql("q", "s") == "SELECT 'openai path used'"


def test_ollama_mode_still_dispatches_to_the_ollama_function(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "ollama")
    monkeypatch.setattr(llm, "_generate_sql_ollama", lambda q, s: "SELECT 'ollama path used'")

    assert llm.generate_sql("q", "s") == "SELECT 'ollama path used'"


def test_unknown_mode_error_lists_all_four_modes(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "something-else")
    with pytest.raises(RuntimeError, match="mock.*openai.*ollama.*gemini"):
        llm.generate_sql("q", "s")
