"""Tests for the request-size limits added alongside rate limiting:
- MAX_QUESTION_LENGTH (/ask)
- CSV_UPLOAD_MAX_BYTES enforced at the HTTP boundary, not just internally
- MAX_REQUEST_BODY_BYTES, the ASGI-level hard ceiling (RequestBodyLimitMiddleware)

Mock mode only, same as tests/test_validation_errors.py, whose envelope
helper this file reuses for the /ask case.
"""
import pytest
from fastapi import FastAPI, File, UploadFile
from fastapi.testclient import TestClient
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.main import app

ENVELOPE_KEYS = {"status", "detail", "error", "dataset", "llm", "trace"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# MAX_QUESTION_LENGTH: enforced by Pydantic's Field(max_length=...), so it
# goes through the SAME RequestValidationError handler as a missing/empty
# question -- no new error shape to maintain.
# ---------------------------------------------------------------------------


def test_question_over_the_length_limit_is_rejected_with_422(client):
    too_long = "a" * (settings.MAX_QUESTION_LENGTH + 1)
    response = client.post("/ask", json={"question": too_long})

    assert response.status_code == 422
    body = response.json()
    assert set(body) == ENVELOPE_KEYS
    assert body["error"]["type"] == "invalid_request"
    assert "question" in body["detail"]


def test_question_exactly_at_the_length_limit_is_allowed(client):
    exactly_at_limit = "How many orders are there?" + " " * (
        settings.MAX_QUESTION_LENGTH - len("How many orders are there?")
    )
    assert len(exactly_at_limit) == settings.MAX_QUESTION_LENGTH

    response = client.post("/ask", json={"question": exactly_at_limit})

    # Whatever the mock lookup decides about this (mangled) question, it
    # must not be rejected by the LENGTH check -- 400 (unsupported mock
    # question) is fine, 422 for a length violation is not.
    assert response.status_code in (200, 400)


def test_short_question_is_unaffected(client):
    response = client.post("/ask", json={"question": "How many orders are there?"})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# CSV_UPLOAD_MAX_BYTES, enforced at the HTTP boundary (csv_upload.py's limit
# already has unit-level coverage in test_csv_upload.py; this confirms it
# also applies when the setting is read through the real /datasets/upload
# endpoint, not just the internal function).
# ---------------------------------------------------------------------------


def test_oversized_csv_upload_returns_400(client, monkeypatch):
    monkeypatch.setattr(settings, "CSV_UPLOAD_MAX_BYTES", 10)  # 10 bytes

    csv_bytes = b"a,b,c\n1,2,3\n4,5,6\n"  # well over 10 bytes
    response = client.post(
        "/datasets/upload", files={"file": ("data.csv", csv_bytes, "text/csv")}
    )

    assert response.status_code == 400
    assert "limit" in response.json()["detail"].lower()


def test_non_csv_upload_is_rejected(client):
    response = client.post(
        "/datasets/upload", files={"file": ("data.txt", b"not a csv", "text/plain")}
    )
    assert response.status_code == 400
    assert "csv" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# MAX_REQUEST_BODY_BYTES / RequestBodyLimitMiddleware: the ASGI-level
# backstop. Testing this against the REAL app would mean sending a multi-MB
# payload (slow, and the real limit is derived from CSV_UPLOAD_MAX_BYTES at
# import time, so it can't be shrunk for a test without rebuilding the app).
# Instead: (1) confirm the real app is wired with it, at the configured
# size and in the right position relative to CORSMiddleware; (2) exercise
# the actual 413 behavior against a tiny throwaway app with a tiny limit,
# proving the wiring pattern itself (same middleware class, same kwarg)
# works as expected.
# ---------------------------------------------------------------------------


def test_production_app_has_the_body_limit_middleware_wired_correctly():
    middlewares = {m.cls: m for m in app.user_middleware}

    assert RequestBodyLimitMiddleware in middlewares
    assert middlewares[RequestBodyLimitMiddleware].kwargs["max_body_size"] == settings.MAX_REQUEST_BODY_BYTES

    # CORSMiddleware must be the OUTER one (added first = added earlier in
    # app.user_middleware) so its headers still land on a 413 that
    # RequestBodyLimitMiddleware produces -- see the comment in main.py.
    cors_index = app.user_middleware.index(middlewares[CORSMiddleware])
    body_limit_index = app.user_middleware.index(middlewares[RequestBodyLimitMiddleware])
    assert cors_index < body_limit_index


def test_body_limit_middleware_returns_413_over_the_limit():
    tiny_app = FastAPI()
    tiny_app.add_middleware(RequestBodyLimitMiddleware, max_body_size=16)

    @tiny_app.post("/echo")
    def echo(payload: dict):
        return payload

    with TestClient(tiny_app) as c:
        response = c.post("/echo", json={"x": "y" * 100})

    assert response.status_code == 413


def test_cors_header_still_lands_on_a_413_from_the_body_limit_middleware():
    """The actual point of the registration order in main.py (RBL added
    before CORSMiddleware, so CORS ends up OUTER): a cross-origin request
    that trips the body-size limit must still get a CORS header on its 413,
    or a browser would show the caller a generic network error instead of
    a readable 413. Reproduces the real app's middleware order in a tiny
    throwaway app with a tiny limit, rather than sending a multi-MB body at
    the real one."""
    tiny_app = FastAPI()
    tiny_app.add_middleware(RequestBodyLimitMiddleware, max_body_size=16)
    tiny_app.add_middleware(CORSMiddleware, allow_origins=["http://example.test"])

    @tiny_app.post("/echo")
    def echo(payload: dict):
        return payload

    with TestClient(tiny_app) as c:
        response = c.post(
            "/echo", json={"x": "y" * 100}, headers={"Origin": "http://example.test"}
        )

    assert response.status_code == 413
    assert response.headers.get("access-control-allow-origin") == "http://example.test"


def test_body_limit_middleware_returns_413_for_an_oversized_multipart_upload():
    """The real-world case this middleware exists for: a CSV upload, not a
    JSON echo. Reproduces the real app's registration order (body limit
    before CORS) in a tiny throwaway app with a tiny limit, so an actual
    multipart/form-data body -- not a generic payload -- is what trips it."""
    tiny_app = FastAPI()
    tiny_app.add_middleware(RequestBodyLimitMiddleware, max_body_size=64)
    tiny_app.add_middleware(CORSMiddleware, allow_origins=["http://example.test"])

    @tiny_app.post("/upload")
    def upload(file: UploadFile = File(...)):
        return {"filename": file.filename}

    oversized_csv = b"a,b,c\n" + (b"1,2,3\n" * 50)  # far more than 64 bytes once encoded

    with TestClient(tiny_app) as c:
        response = c.post(
            "/upload",
            files={"file": ("big.csv", oversized_csv, "text/csv")},
            headers={"Origin": "http://example.test"},
        )

    assert response.status_code == 413
    assert response.headers.get("access-control-allow-origin") == "http://example.test"


def test_body_limit_middleware_allows_a_normal_multipart_upload():
    """The other half of the same proof: a legitimately-sized multipart
    upload must sail through this middleware untouched."""
    tiny_app = FastAPI()
    tiny_app.add_middleware(RequestBodyLimitMiddleware, max_body_size=10_000)
    tiny_app.add_middleware(CORSMiddleware, allow_origins=["http://example.test"])

    @tiny_app.post("/upload")
    def upload(file: UploadFile = File(...)):
        return {"filename": file.filename, "size": len(file.file.read())}

    small_csv = b"a,b,c\n1,2,3\n"

    with TestClient(tiny_app) as c:
        response = c.post(
            "/upload",
            files={"file": ("small.csv", small_csv, "text/csv")},
            headers={"Origin": "http://example.test"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "small.csv"
    assert body["size"] == len(small_csv)
    assert response.headers.get("access-control-allow-origin") == "http://example.test"


def test_real_upload_endpoint_still_works_under_the_production_body_limit(client):
    """Confirms the production app itself (not a throwaway reproduction)
    still accepts an ordinary CSV upload with its REAL MAX_REQUEST_BODY_BYTES
    in place -- the ASGI-level backstop must not interfere with legitimate
    traffic."""
    csv_bytes = b"a,b,c\n1,2,3\n4,5,6\n"
    response = client.post(
        "/datasets/upload", files={"file": ("ok.csv", csv_bytes, "text/csv")}
    )
    assert response.status_code == 200


def test_body_limit_middleware_allows_requests_under_the_limit():
    tiny_app = FastAPI()
    tiny_app.add_middleware(RequestBodyLimitMiddleware, max_body_size=10_000)

    @tiny_app.post("/echo")
    def echo(payload: dict):
        return payload

    with TestClient(tiny_app) as c:
        response = c.post("/echo", json={"x": "y"})

    assert response.status_code == 200
