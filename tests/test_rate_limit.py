"""Tests for backend/rate_limit.py: client-IP resolution behind a (possibly
spoofed) reverse proxy, the sliding-window limiter itself, and the two
endpoints (/ask, /datasets/upload) actually enforcing it end to end.
"""
import time

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from backend import rate_limit
from backend.config import settings
from backend.main import app
from backend.rate_limit import SlidingWindowRateLimiter, get_client_ip


def _request_with_headers(headers: dict[str, str], client_host: str | None = "203.0.113.9") -> Request:
    """A minimal Starlette Request carrying only what get_client_ip() reads:
    headers and request.client.host."""
    encoded_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {
        "type": "http",
        "headers": encoded_headers,
        "client": (client_host, 12345) if client_host else None,
    }
    return Request(scope)


# ---------------------------------------------------------------------------
# get_client_ip(): the proxy-spoofing-safe part.
# ---------------------------------------------------------------------------


def test_x_forwarded_for_is_ignored_by_default(monkeypatch):
    """The safe default (settings.TRUST_PROXY_HEADERS=False): a
    client-supplied X-Forwarded-For has NO effect at all -- there is
    nothing here for a client to spoof unless this is explicitly turned
    on. This is the behavior a fresh/local deployment gets."""
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    request = _request_with_headers(
        {"x-forwarded-for": "9.9.9.9"}, client_host="203.0.113.9"
    )
    assert get_client_ip(request) == "203.0.113.9"  # the real TCP peer, not the header


def test_uses_the_last_x_forwarded_for_entry_not_the_first(monkeypatch):
    """With TRUST_PROXY_HEADERS=True: a single trusted proxy hop APPENDS
    to X-Forwarded-For rather than replacing it, so a client can prepend
    fake IPs. Only the LAST entry -- the one the actual trusted hop added
    -- is safe to trust."""
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    request = _request_with_headers({"x-forwarded-for": "1.2.3.4, 5.6.7.8, 9.9.9.9"})
    assert get_client_ip(request) == "9.9.9.9"


def test_single_hop_x_forwarded_for_is_used_directly(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    request = _request_with_headers({"x-forwarded-for": "198.51.100.7"})
    assert get_client_ip(request) == "198.51.100.7"


def test_falls_back_to_the_tcp_peer_when_no_proxy_header_is_present(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    request = _request_with_headers({}, client_host="127.0.0.1")
    assert get_client_ip(request) == "127.0.0.1"


def test_falls_back_to_unknown_when_neither_is_available(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    request = _request_with_headers({}, client_host=None)
    assert get_client_ip(request) == "unknown"


def test_whitespace_around_forwarded_entries_is_stripped(monkeypatch):
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)
    request = _request_with_headers({"x-forwarded-for": "1.2.3.4 ,  5.6.7.8  "})
    assert get_client_ip(request) == "5.6.7.8"


# ---------------------------------------------------------------------------
# SlidingWindowRateLimiter: the counting logic itself, in isolation.
# ---------------------------------------------------------------------------


def test_allows_up_to_the_configured_max_then_rejects():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60.0)
    assert [limiter.allow("ip-a") for _ in range(3)] == [True, True, True]
    assert limiter.allow("ip-a") is False


def test_different_keys_have_independent_budgets():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60.0)
    assert limiter.allow("ip-a") is True
    assert limiter.allow("ip-b") is True  # a different IP is never blocked by ip-a's usage
    assert limiter.allow("ip-a") is False
    assert limiter.allow("ip-b") is False


def test_old_hits_fall_out_of_the_window():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.05)
    assert limiter.allow("ip-a") is True
    assert limiter.allow("ip-a") is False
    time.sleep(0.07)
    assert limiter.allow("ip-a") is True


# ---------------------------------------------------------------------------
# End to end: /ask and /datasets/upload actually enforce it, with a 429.
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_ask_returns_429_after_exceeding_the_per_minute_limit(client, monkeypatch):
    monkeypatch.setattr(rate_limit.ask_limiter, "max_requests", 2)

    first = client.post("/ask", json={"question": "How many orders are there?"})
    second = client.post("/ask", json={"question": "How many orders are there?"})
    third = client.post("/ask", json={"question": "How many orders are there?"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert "rate limit" in third.json()["detail"].lower()


def test_ask_429_never_reaches_the_pipeline(client, monkeypatch):
    """The limit is checked BEFORE agent.run_pipeline() runs -- important so
    a flood of requests can't still rack up LLM calls before being refused."""
    import backend.main as main_module

    monkeypatch.setattr(rate_limit.ask_limiter, "max_requests", 1)

    first = client.post("/ask", json={"question": "How many orders are there?"})
    assert first.status_code == 200  # consumes the one allowed slot via the real (mock) pipeline

    calls = []
    monkeypatch.setattr(main_module.agent, "run_pipeline", lambda q: calls.append(q))
    response = client.post("/ask", json={"question": "How many orders are there?"})

    assert response.status_code == 429
    assert calls == []  # run_pipeline was never called for the blocked request


def test_upload_returns_429_after_exceeding_the_per_minute_limit(client, monkeypatch):
    monkeypatch.setattr(rate_limit.upload_limiter, "max_requests", 1)

    csv_bytes = b"a,b\n1,2\n"
    first = client.post("/datasets/upload", files={"file": ("t.csv", csv_bytes, "text/csv")})
    second = client.post("/datasets/upload", files={"file": ("t.csv", csv_bytes, "text/csv")})

    assert first.status_code == 200
    assert second.status_code == 429


def test_rate_limit_does_not_leak_a_stack_trace_or_internal_detail(client, monkeypatch):
    monkeypatch.setattr(rate_limit.ask_limiter, "max_requests", 1)

    client.post("/ask", json={"question": "How many orders are there?"})
    response = client.post("/ask", json={"question": "How many orders are there?"})

    body = response.json()
    assert set(body) == {"detail"}
    assert "Traceback" not in body["detail"]


def test_different_simulated_client_ips_get_independent_budgets(client, monkeypatch):
    monkeypatch.setattr(rate_limit.ask_limiter, "max_requests", 1)
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", True)

    first = client.post(
        "/ask",
        json={"question": "How many orders are there?"},
        headers={"X-Forwarded-For": "203.0.113.1"},
    )
    second = client.post(
        "/ask",
        json={"question": "How many orders are there?"},
        headers={"X-Forwarded-For": "203.0.113.2"},
    )

    assert first.status_code == 200
    assert second.status_code == 200  # different (claimed-last-hop) IP, so not blocked by the first


def test_spoofed_x_forwarded_for_cannot_bypass_the_limit_by_default(client, monkeypatch):
    """The end-to-end version of test_x_forwarded_for_is_ignored_by_default:
    without TRUST_PROXY_HEADERS, sending a different X-Forwarded-For on
    every request does NOT grant a fresh budget each time, because the key
    is always the real TestClient TCP peer, never the header."""
    monkeypatch.setattr(settings, "TRUST_PROXY_HEADERS", False)
    monkeypatch.setattr(rate_limit.ask_limiter, "max_requests", 1)

    first = client.post(
        "/ask",
        json={"question": "How many orders are there?"},
        headers={"X-Forwarded-For": "203.0.113.1"},
    )
    second = client.post(
        "/ask",
        json={"question": "How many orders are there?"},
        headers={"X-Forwarded-For": "203.0.113.2"},  # a different claimed IP each time
    )

    assert first.status_code == 200
    assert second.status_code == 429  # still blocked -- the header was never consulted


# ---------------------------------------------------------------------------
# Memory growth: expired entries must be removed from the dict, not just
# have their timestamp list emptied.
# ---------------------------------------------------------------------------


def test_expired_keys_are_fully_removed_not_just_emptied():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.05)
    limiter.allow("ip-a")
    assert "ip-a" in limiter._hits

    time.sleep(0.07)
    limiter.allow("ip-b")  # any later call triggers the prune pass

    assert "ip-a" not in limiter._hits  # gone from the dict entirely, not an empty list
    assert "ip-b" in limiter._hits


def test_an_untouched_keys_hits_are_pruned_by_someone_elses_call():
    """The dict-growth fix specifically: pruning on allow() must sweep
    EVERY key, not just the one passed in -- otherwise an IP that hits
    once and never returns would sit in the dict forever, since nothing
    else would ever revisit it."""
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=0.05)
    for key in ("ip-1", "ip-2", "ip-3"):
        limiter.allow(key)
    assert len(limiter._hits) == 3

    time.sleep(0.07)
    limiter.allow("ip-4")  # none of ip-1/2/3 are touched directly here

    assert set(limiter._hits) == {"ip-4"}


# ---------------------------------------------------------------------------
# /datasets/{id}/select and /debug/retrieve rate limiting.
# ---------------------------------------------------------------------------


def test_select_dataset_returns_429_after_exceeding_the_per_minute_limit(client, monkeypatch):
    monkeypatch.setattr(rate_limit.select_limiter, "max_requests", 1)

    first = client.post("/datasets/football/select")
    second = client.post("/datasets/movies/select")

    assert first.status_code == 200
    assert second.status_code == 429


def test_debug_retrieve_is_rate_limited_when_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_DEBUG_ENDPOINTS", True)
    monkeypatch.setattr(rate_limit.debug_limiter, "max_requests", 1)

    first = client.get("/debug/retrieve", params={"question": "How many orders are there?"})
    second = client.get("/debug/retrieve", params={"question": "How many orders are there?"})

    assert first.status_code == 200
    assert second.status_code == 429
