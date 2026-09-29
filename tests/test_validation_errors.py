"""Day 9: tests for the normalized request-validation error envelope.

Before this, a malformed /ask body (missing/empty "question") got FastAPI's
own default 422 shape — `{"detail": [ {...}, ... ]}`, an ARRAY — while every
other /ask failure returned `{"detail": "<string>", "error": {...}, ...}`.
Same status code, two incompatible shapes. backend/main.py now registers a
RequestValidationError handler that reshapes the former into the latter, so
a frontend only ever has to handle one 422 envelope.

Mock mode only — no OpenAI call needed anywhere in this file.
"""
import pytest
from fastapi.testclient import TestClient

from backend import dataset, llm
from backend.main import app

ENVELOPE_KEYS = {"status", "detail", "error", "dataset", "llm", "trace"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _assert_envelope_shape(body):
    """Every /ask error, regardless of cause, must have this exact shape."""
    assert set(body) == ENVELOPE_KEYS
    assert body["status"] == "error"
    assert isinstance(body["detail"], str)  # never an array — the whole point
    assert set(body["error"]) == {"type", "failed_step", "message"}
    assert isinstance(body["error"]["message"], str)
    assert set(body["dataset"]) == {"id", "name", "domain", "theme"}
    assert set(body["llm"]) == {"mode", "model"}
    assert set(body["trace"]) == {"summary", "steps", "dataset"}


# ---------------------------------------------------------------------------
# The two malformed-request cases the task calls out specifically.
# ---------------------------------------------------------------------------


def test_missing_question_returns_normalized_422(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422
    body = response.json()
    _assert_envelope_shape(body)
    assert body["error"]["type"] == "invalid_request"
    assert body["error"]["failed_step"] is None
    assert "question" in body["detail"]
    assert "required" in body["detail"].lower()


def test_empty_question_returns_normalized_422(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422
    body = response.json()
    _assert_envelope_shape(body)
    assert body["error"]["type"] == "invalid_request"
    assert "question" in body["detail"]
    assert "at least 1 character" in body["detail"]


def test_missing_and_empty_question_produce_different_readable_messages(client):
    """Sanity check that the message is actually informative, not a generic
    placeholder — the two cases should say different things."""
    missing = client.post("/ask", json={}).json()["detail"]
    empty = client.post("/ask", json={"question": ""}).json()["detail"]
    assert missing != empty


def test_wrong_type_for_question_also_normalized(client):
    """Any FastAPI validation failure, not just the two named cases —
    e.g. `question` present but not a string."""
    response = client.post("/ask", json={"question": 12345})
    assert response.status_code == 422
    body = response.json()
    _assert_envelope_shape(body)
    assert body["error"]["type"] == "invalid_request"


def test_empty_request_body_entirely_is_normalized(client):
    response = client.post("/ask", content=b"", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    _assert_envelope_shape(response.json())


# ---------------------------------------------------------------------------
# Existing application errors: same envelope shape, untouched values.
# ---------------------------------------------------------------------------


def test_sql_rejection_error_is_unchanged(client, monkeypatch):
    """The pre-existing sql_rejected path — must still work exactly as
    before, with the guard's own reason as the message, and a real trace
    (not the empty one the new handler produces)."""
    monkeypatch.setattr(llm, "generate_sql", lambda question, schema_text: "DELETE FROM orders")

    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 422
    body = response.json()
    _assert_envelope_shape(body)
    assert body["error"]["type"] == "sql_rejected"
    assert body["error"]["failed_step"] == "sql_guard"
    assert body["detail"] == "Only SELECT queries are allowed."
    # Unlike the invalid_request case, real pipeline steps DID run.
    assert len(body["trace"]["steps"]) > 0
    assert body["trace"]["summary"]["attempts"] == 1


def test_unsupported_question_error_is_unchanged(client):
    response = client.post("/ask", json={"question": "what is the meaning of life"})
    assert response.status_code == 400
    body = response.json()
    _assert_envelope_shape(body)
    assert body["error"]["type"] == "unsupported_question"


def test_exhausted_retries_error_is_unchanged(client):
    response = client.post(
        "/ask", json={"question": "This question always fails even after correction"}
    )
    assert response.status_code == 500
    body = response.json()
    _assert_envelope_shape(body)
    assert body["error"]["type"] == "execution_failed"


def test_llm_error_is_unchanged(client, monkeypatch):
    def boom(question, schema_text):
        raise RuntimeError("the model is unavailable")

    monkeypatch.setattr(llm, "generate_sql", boom)
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 502
    body = response.json()
    _assert_envelope_shape(body)
    assert body["error"]["type"] == "llm_error"


def test_httpexception_based_errors_are_not_wrapped_by_the_new_handler(client):
    """404 from /datasets/{id}/select is a plain HTTPException, a
    DIFFERENT exception type than RequestValidationError — must keep its
    own simple {"detail": "..."} shape, not gain the ErrorResponse fields."""
    response = client.post("/datasets/does-not-exist/select")
    assert response.status_code == 404
    assert set(response.json()) == {"detail"}
    assert isinstance(response.json()["detail"], str)


def test_upload_400_is_not_wrapped_by_the_new_handler(client):
    response = client.post(
        "/datasets/upload", files={"file": ("data.exe", b"not a csv", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert set(response.json()) == {"detail"}


# ---------------------------------------------------------------------------
# Requirements 5-8: nothing else changed.
# ---------------------------------------------------------------------------


def test_successful_ask_is_completely_unchanged(client):
    response = client.post("/ask", json={"question": "How many orders are there?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["rows"] == [{"total_orders": 99441}]
    assert body["corrected"] is False


def test_sql_guard_still_rejects_the_same_things(client, monkeypatch):
    """The guard's own behavior (what it rejects and why) must be
    unaffected — only the error ENVELOPE around a rejection changed.

    A bare "DROP TABLE orders" fails the guard's first check (must start
    with SELECT/WITH), not the mid-statement keyword scan — so its message
    is "Only SELECT queries are allowed.", not one naming DROP. That's the
    guard's own documented behavior (see sql_guard.py), unrelated to this
    change; asserted here to confirm this handler didn't alter it.
    """
    monkeypatch.setattr(llm, "generate_sql", lambda question, schema_text: "DROP TABLE orders")
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 422
    assert response.json()["detail"] == "Only SELECT queries are allowed."

    # This is the case where the guard's message DOES name the keyword —
    # a statement that starts with an allowed keyword (WITH) but contains
    # a forbidden one later on.
    monkeypatch.setattr(
        llm, "generate_sql", lambda question, schema_text: "WITH t AS (SELECT 1) DROP TABLE orders"
    )
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 422
    assert "DROP" in response.json()["detail"]


def test_dataset_switching_still_works(client):
    """Unrelated dataset behavior is untouched by this change."""
    response = client.post("/datasets/football/select")
    assert response.status_code == 200
    assert response.json()["id"] == "football"
    assert dataset.get_active_dataset().id == "football"
    client.post("/datasets/olist/select")  # restore for other tests in this module


def test_invalid_request_error_reflects_the_currently_active_dataset(client):
    """The new handler's dataset/llm fields aren't hardcoded — they reflect
    whatever is actually active, same as every other error path."""
    client.post("/datasets/football/select")
    try:
        response = client.post("/ask", json={})
        assert response.json()["dataset"]["id"] == "football"
    finally:
        client.post("/datasets/olist/select")
