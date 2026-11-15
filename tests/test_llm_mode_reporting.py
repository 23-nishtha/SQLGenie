"""Tests for backend/main.py's per-mode model reporting (_model_for_mode,
used by both /health and _llm_info()/the /ask response's `llm` field).

Regression coverage for a pre-existing bug: before this, every non-mock mode
other than "mock" fell through to OPENAI_MODEL in _llm_info() (the /ask
response), even when SQLGENIE_LLM_MODE was "ollama" — so an /ask response in
ollama mode incorrectly reported the configured OpenAI model name. /health
had already been fixed; this fixes _llm_info() the same way and extends both
to "gemini".
"""
import pytest

from backend.config import settings
from backend.main import _llm_info, _model_for_mode, health_check


@pytest.mark.parametrize("mode,attr,value", [
    ("openai", "OPENAI_MODEL", "gpt-test-model"),
    ("ollama", "OLLAMA_MODEL", "qwen3:4b"),
    ("gemini", "GEMINI_MODEL", "gemini-3.6-flash"),
])
def test_model_for_mode_reports_the_right_model(monkeypatch, mode, attr, value):
    monkeypatch.setattr(settings, attr, value)
    assert _model_for_mode(mode) == value


def test_model_for_mode_is_none_for_mock():
    assert _model_for_mode("mock") is None


def test_llm_info_reports_ollama_model_not_openai_model(monkeypatch):
    """The regression this file exists for: ollama mode must report
    OLLAMA_MODEL, never OPENAI_MODEL, in the /ask response's `llm` field."""
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "gpt-should-not-appear")

    info = _llm_info()

    assert info.mode == "ollama"
    assert info.model == "qwen3:4b"


def test_llm_info_reports_gemini_model(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "gemini")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.6-flash")

    info = _llm_info()

    assert info.mode == "gemini"
    assert info.model == "gemini-3.6-flash"


def test_llm_info_model_is_none_in_mock_mode(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "mock")

    info = _llm_info()

    assert info.mode == "mock"
    assert info.model is None


def test_health_endpoint_reports_gemini_model(monkeypatch):
    # Calls health_check() directly rather than going through a TestClient
    # (which would also run the app's startup hook — schema_retrieval
    # warm-up — and that would try to embed the active multi-table dataset
    # via the real Gemini API under this monkeypatched mode). health_check()
    # itself has no such side effect, so this is a plain unit test of it.
    monkeypatch.setattr(settings, "SQLGENIE_LLM_MODE", "gemini")
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.6-flash")

    body = health_check()

    assert body["llm_mode"] == "gemini"
    assert body["model"] == "gemini-3.6-flash"
