"""Tests for GET /debug/retrieve's Stage 2.1 gating: disabled by default in
every environment, 404 (not 403) when disabled, and — when explicitly
enabled for local development — still subject to the same question-length
limit /ask has. Rate limiting for this endpoint is covered in
tests/test_rate_limit.py.
"""
import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_debug_retrieve_is_disabled_by_default(client):
    response = client.get("/debug/retrieve", params={"question": "How many orders are there?"})
    assert response.status_code == 404


def test_disabled_response_reveals_nothing_about_the_feature(client):
    response = client.get("/debug/retrieve", params={"question": "How many orders are there?"})
    assert response.json() == {"detail": "Not Found"}


def test_debug_retrieve_works_when_explicitly_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_DEBUG_ENDPOINTS", True)

    response = client.get("/debug/retrieve", params={"question": "How many orders are there?"})

    assert response.status_code == 200
    body = response.json()
    assert "retrieved_tables" in body
    assert "schema_text_sent_to_llm" in body


def test_debug_retrieve_enforces_the_question_length_limit_when_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_DEBUG_ENDPOINTS", True)

    too_long = "a" * (settings.MAX_QUESTION_LENGTH + 1)
    response = client.get("/debug/retrieve", params={"question": too_long})

    assert response.status_code == 422


def test_debug_retrieve_still_disabled_even_with_a_short_question(client, monkeypatch):
    """ENABLE_DEBUG_ENDPOINTS is checked independently of the length limit
    -- a short, otherwise-valid question is still refused while disabled."""
    monkeypatch.setattr(settings, "ENABLE_DEBUG_ENDPOINTS", False)

    response = client.get("/debug/retrieve", params={"question": "short"})

    assert response.status_code == 404
