"""Tests for configurable CORS (Day 6).

Browsers refuse to let a page on one origin call an API on another unless the
API opts in. The future frontend runs separately from this backend, so the
backend must allow exactly that frontend's origin — and nobody else's.

tests/conftest.py pins CORS_ALLOWED_ORIGINS to the two local dev origins, so
these tests don't depend on a developer's personal .env.
"""
import pytest
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from backend import config
from backend.config import DEFAULT_CORS_ORIGINS, settings
from backend.main import app

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
DISALLOWED_ORIGIN = "http://evil.example.com"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def preflight(client, origin):
    """The OPTIONS request a browser sends before a cross-origin JSON POST."""
    return client.options(
        "/ask",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_default_origins_are_the_local_dev_origins_and_not_a_wildcard():
    assert DEFAULT_CORS_ORIGINS.split(",") == ALLOWED_ORIGINS
    assert "*" not in DEFAULT_CORS_ORIGINS


def test_settings_use_the_configured_origins():
    assert settings.CORS_ALLOWED_ORIGINS == ALLOWED_ORIGINS
    assert "*" not in settings.CORS_ALLOWED_ORIGINS


def test_origins_are_configurable_from_the_environment(monkeypatch):
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS", " https://app.example.com/ , http://localhost:3000,, "
    )
    assert config._get_list("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS) == [
        "https://app.example.com",   # whitespace and trailing slash stripped
        "http://localhost:3000",     # empty entries dropped
    ]


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_setting_falls_back_to_the_safe_default(monkeypatch, blank):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", blank)
    assert config._get_list("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS) == ALLOWED_ORIGINS


def test_unset_setting_falls_back_to_the_safe_default(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    assert config._get_list("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS) == ALLOWED_ORIGINS


def test_middleware_is_registered_with_the_configured_origins_and_no_credentials():
    cors = [m for m in app.user_middleware if m.cls is CORSMiddleware]
    assert len(cors) == 1
    options = getattr(cors[0], "kwargs", None) or getattr(cors[0], "options")
    assert options["allow_origins"] == settings.CORS_ALLOWED_ORIGINS
    assert "*" not in options["allow_origins"]
    assert options["allow_credentials"] is False


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_preflight_from_an_allowed_origin_succeeds(client, origin):
    response = preflight(client, origin)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in response.headers


def test_preflight_from_a_disallowed_origin_is_refused(client):
    response = preflight(client, DISALLOWED_ORIGIN)
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_real_requests_from_an_allowed_origin_carry_the_header(client, origin):
    dataset_response = client.get("/dataset", headers={"Origin": origin})
    ask_response = client.post(
        "/ask", json={"question": "How many orders are there?"}, headers={"Origin": origin}
    )
    assert dataset_response.headers["access-control-allow-origin"] == origin
    assert ask_response.status_code == 200
    assert ask_response.headers["access-control-allow-origin"] == origin


def test_failed_requests_also_carry_the_header(client):
    """A frontend must be able to read an error body too."""
    response = client.post(
        "/ask",
        json={"question": "what is the meaning of life"},
        headers={"Origin": ALLOWED_ORIGINS[0]},
    )
    assert response.status_code == 400
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGINS[0]


def test_requests_from_a_disallowed_origin_get_no_permission_header(client):
    response = client.get("/dataset", headers={"Origin": DISALLOWED_ORIGIN})
    assert "access-control-allow-origin" not in response.headers
