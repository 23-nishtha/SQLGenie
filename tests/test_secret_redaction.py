"""Tests for backend/trace.py's redact_secrets(): error text from an LLM
provider must never leak an API key back to a client. Covers the existing
OpenAI/"sk-..." behavior plus the Gemini/"AIza..." behavior added alongside
Gemini support.
"""
from backend.config import settings
from backend.trace import redact_secrets


def test_configured_openai_key_is_redacted(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-configuredkey12345")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)

    text = "Auth failed for key sk-configuredkey12345 on this request."
    assert "sk-configuredkey12345" not in redact_secrets(text)
    assert "[redacted]" in redact_secrets(text)


def test_configured_gemini_key_is_redacted(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "AIzaSyConfiguredGeminiKeyExample123")

    text = "Request rejected: key AIzaSyConfiguredGeminiKeyExample123 is invalid."
    assert "AIzaSyConfiguredGeminiKeyExample123" not in redact_secrets(text)
    assert "[redacted]" in redact_secrets(text)


def test_any_openai_shaped_key_is_redacted_even_if_not_the_configured_one(monkeypatch):
    """A provider's error text can echo back a DIFFERENT key than the one in
    settings (e.g. a stale key from a previous request) — the shape-based
    regex must still catch it."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)

    text = "Invalid key: sk-someOtherLeakedKeyValue"
    redacted = redact_secrets(text)
    assert "sk-someOtherLeakedKeyValue" not in redacted
    assert "[redacted]" in redacted


def test_any_google_shaped_key_is_redacted_even_if_not_the_configured_one(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)

    text = "Invalid key: AIzaSyUnconfiguredLeakedKeyExample456"
    redacted = redact_secrets(text)
    assert "AIzaSyUnconfiguredLeakedKeyExample456" not in redacted
    assert "[redacted]" in redacted


def test_text_with_no_secret_is_returned_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)

    text = "no such column: bad_col"
    assert redact_secrets(text) == text
