"""Tests for Day 6's pipeline trace: the step-by-step record of what /ask did.

Everything runs in mock mode (tests/conftest.py forces it), so there is no
OpenAI call and no API key. The mock SQL in backend/mock_llm.py really does
fail or succeed against the real database/olist.db, so these traces record
genuine pipeline runs, not simulations.
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend import agent, llm
from backend.config import settings
from backend.main import app

ALL_STEP_TYPES = {
    "question", "schema_retrieval", "sql_generation", "sql_guard",
    "db_execution", "self_correction", "result",
}

CORRECTED_QUESTION = "How many orders were placed in 2017?"
ALWAYS_FAILS_QUESTION = "This question always fails even after correction"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def ask(client, question):
    return client.post("/ask", json={"question": question})


def step_types(body):
    return [step["type"] for step in body["trace"]["steps"]]


def steps_of(body, step_type):
    return [s for s in body["trace"]["steps"] if s["type"] == step_type]


# ---------------------------------------------------------------------------
# The two flows the frontend draws.
# ---------------------------------------------------------------------------


def test_successful_trace_has_the_expected_steps_in_order(client):
    response = ask(client, "How many orders are there?")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert step_types(body) == [
        "question", "schema_retrieval", "sql_generation",
        "sql_guard", "db_execution", "result",
    ]
    assert all(s["status"] == "success" for s in body["trace"]["steps"])
    assert all(s["attempt"] == 0 for s in body["trace"]["steps"])
    assert body["trace"]["summary"]["attempts"] == 1
    assert body["trace"]["summary"]["corrected"] is False
    assert body["trace"]["summary"]["max_correction_attempts"] == agent.MAX_CORRECTION_ATTEMPTS


def test_self_correction_trace_matches_the_documented_flow(client):
    """question -> retrieval -> generate -> guard -> execute (FAILS) ->
    self-correction -> generate -> guard -> execute (succeeds) -> result."""
    response = ask(client, CORRECTED_QUESTION)
    assert response.status_code == 200
    body = response.json()

    assert step_types(body) == [
        "question", "schema_retrieval",
        "sql_generation", "sql_guard", "db_execution",
        "self_correction",
        "sql_generation", "sql_guard", "db_execution",
        "result",
    ]
    steps = body["trace"]["steps"]
    assert [s["attempt"] for s in steps] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    first_execution, second_execution = steps_of(body, "db_execution")
    assert first_execution["status"] == "failed"
    assert "order_year" in first_execution["data"]["error"]
    assert second_execution["status"] == "success"

    first_generation, second_generation = steps_of(body, "sql_generation")
    assert first_generation["data"]["is_correction"] is False
    assert second_generation["data"]["is_correction"] is True
    assert first_generation["data"]["sql"] != second_generation["data"]["sql"]

    # The correction step carries the exact context the LLM was given.
    correction = steps_of(body, "self_correction")[0]["data"]
    assert correction["correction_number"] == 1
    assert correction["max_corrections"] == agent.MAX_CORRECTION_ATTEMPTS
    assert correction["failed_sql"] == first_generation["data"]["sql"]
    assert correction["db_error"] == first_execution["data"]["error"]

    assert body["trace"]["summary"]["attempts"] == 2
    assert body["trace"]["summary"]["corrected"] is True
    # ...and the older top-level fields agree with the trace.
    assert body["attempts"] == 2 and body["corrected"] is True


# ---------------------------------------------------------------------------
# Failures still return a trace, ending at the step that failed.
# ---------------------------------------------------------------------------


def test_exhausted_retries_return_a_structured_500_with_the_full_trace(client):
    response = ask(client, ALWAYS_FAILS_QUESTION)
    assert response.status_code == 500
    body = response.json()

    assert body["status"] == "error"
    assert body["error"]["type"] == "execution_failed"
    assert body["error"]["failed_step"] == "db_execution"
    assert isinstance(body["detail"], str)
    assert "Traceback" not in body["detail"]

    assert "result" not in step_types(body)
    executions = steps_of(body, "db_execution")
    assert len(executions) == agent.MAX_CORRECTION_ATTEMPTS + 1
    assert all(e["status"] == "failed" for e in executions)
    assert len(steps_of(body, "self_correction")) == agent.MAX_CORRECTION_ATTEMPTS
    assert body["trace"]["summary"]["attempts"] == agent.MAX_CORRECTION_ATTEMPTS + 1
    assert body["trace"]["summary"]["corrected"] is False


def test_guard_rejection_ends_at_the_guard_and_never_reaches_the_database(client, monkeypatch):
    monkeypatch.setattr(llm, "generate_sql", lambda question, schema_text: "DELETE FROM orders")

    response = ask(client, "anything, generate_sql is monkeypatched")
    assert response.status_code == 422
    body = response.json()

    assert body["status"] == "error"
    assert body["error"]["type"] == "sql_rejected"
    assert body["error"]["failed_step"] == "sql_guard"

    last = body["trace"]["steps"][-1]
    assert last["type"] == "sql_guard" and last["status"] == "failed"
    assert last["data"]["approved"] is False
    assert last["data"]["reason"]
    assert last["data"]["original_sql"] == "DELETE FROM orders"
    assert last["data"]["executed_sql"] is None
    # The whole point: rejected SQL never got as far as the database.
    assert steps_of(body, "db_execution") == []


def test_a_rejected_correction_is_visible_in_the_trace(client, monkeypatch):
    """The first SQL is safe-but-wrong and fails in SQLite; the 'correction'
    is a DROP TABLE. The trace should show the guard stopping it at attempt 1,
    with no database execution for that attempt."""
    monkeypatch.setattr(
        llm, "generate_sql", lambda question, schema_text: "SELECT * FROM not_a_real_table"
    )
    monkeypatch.setattr(llm, "correct_sql", lambda **kwargs: "DROP TABLE orders")

    response = ask(client, "anything, the llm functions are monkeypatched")
    assert response.status_code == 422
    body = response.json()

    assert body["error"]["type"] == "sql_rejected"
    assert step_types(body)[-4:] == ["db_execution", "self_correction", "sql_generation", "sql_guard"]
    last = body["trace"]["steps"][-1]
    assert last["attempt"] == 1 and last["status"] == "failed"
    assert [e["attempt"] for e in steps_of(body, "db_execution")] == [0]  # nothing ran for attempt 1


def test_unsupported_mock_question_ends_at_sql_generation(client):
    response = ask(client, "what is the meaning of life")
    assert response.status_code == 400
    body = response.json()

    assert body["error"]["type"] == "unsupported_question"
    assert body["error"]["failed_step"] == "sql_generation"
    last = body["trace"]["steps"][-1]
    assert last["type"] == "sql_generation" and last["status"] == "failed"
    assert last["data"]["sql"] is None and last["data"]["error"]
    assert steps_of(body, "sql_guard") == []


def test_llm_failure_is_a_502_with_the_llm_error_type(client, monkeypatch):
    def boom(question, schema_text):
        raise RuntimeError("the model is unavailable")

    monkeypatch.setattr(llm, "generate_sql", boom)

    response = ask(client, "anything")
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "llm_error"
    assert body["error"]["failed_step"] == "sql_generation"
    assert body["detail"].startswith("Could not generate SQL:")


@pytest.mark.parametrize(
    "question", [ALWAYS_FAILS_QUESTION, "what is the meaning of life"],
)
def test_run_pipeline_reports_expected_failures_instead_of_raising(question):
    run = agent.run_pipeline(question)
    assert run.result is None
    assert run.error is not None
    assert run.error_type is not None
    assert run.trace.steps  # there is always at least the question step


# ---------------------------------------------------------------------------
# What each step reports.
# ---------------------------------------------------------------------------


def test_schema_retrieval_step_reports_tables_and_reasons(client):
    question = "What is the average review score?"
    # Mock mode has no canned SQL for this question, so /ask would answer
    # 400 — but retrieval runs before generation, so the retrieval step is
    # still recorded. Check it through the pipeline directly, then
    # cross-check against the public debug endpoint.
    run = agent.run_pipeline(question)
    retrieval = next(s for s in run.trace.steps if s.type == "schema_retrieval")
    assert retrieval.status == "success"

    tables = {t.name: t for t in retrieval.data.tables}
    assert "order_reviews" in tables
    assert tables["order_reviews"].selected_by == "score"
    assert tables["order_reviews"].score > 0
    assert all(t.selected_by in {"score", "fk_expansion", "fallback"} for t in tables.values())
    assert "order_reviews.order_id -> orders.order_id" in retrieval.data.relationships
    assert 0 < retrieval.data.schema_chars < retrieval.data.full_schema_chars

    debug = client.get("/debug/retrieve", params={"question": question}).json()
    assert list(tables) == debug["retrieved_tables"]


def test_sql_guard_step_reports_whether_the_sql_was_modified(client):
    # No LIMIT in the generated SQL -> the guard adds one.
    no_limit = ask(client, "How many orders are there?").json()
    guard = steps_of(no_limit, "sql_guard")[0]["data"]
    assert guard["approved"] is True
    assert guard["sql_was_modified"] is True
    assert guard["executed_sql"] != guard["original_sql"]
    assert "LIMIT" in guard["executed_sql"].upper()

    # The generated SQL already has a LIMIT -> the guard leaves it alone.
    has_limit = ask(client, "What are the top 5 product categories by revenue?").json()
    guard = steps_of(has_limit, "sql_guard")[0]["data"]
    assert guard["sql_was_modified"] is False
    assert guard["executed_sql"] == guard["original_sql"]


@pytest.mark.parametrize("question", [
    "How many orders are there?", CORRECTED_QUESTION, ALWAYS_FAILS_QUESTION,
])
def test_trace_structural_invariants(client, question):
    body = ask(client, question).json()
    steps = body["trace"]["steps"]

    assert [s["index"] for s in steps] == list(range(len(steps)))
    assert all(s["type"] in ALL_STEP_TYPES for s in steps)
    assert all(s["status"] in {"success", "failed"} for s in steps)
    assert all(s["duration_ms"] >= 0 for s in steps)
    attempts = [s["attempt"] for s in steps]
    assert attempts == sorted(attempts)  # never goes backwards
    assert body["trace"]["summary"]["total_duration_ms"] >= 0
    assert steps[0]["type"] == "question" and steps[0]["data"]["text"] == question
    json.dumps(body)  # fully JSON-serializable


# ---------------------------------------------------------------------------
# Backward compatibility: the Day 2-5 response is still there, unchanged.
# ---------------------------------------------------------------------------


def test_original_response_fields_are_unchanged(client):
    body = ask(client, "How many orders are there?").json()

    assert body["question"] == "How many orders are there?"
    assert body["columns"] == ["total_orders"]
    assert body["rows"] == [{"total_orders": 99441}]
    assert body["row_count"] == 1
    assert body["attempts"] == 1
    assert body["corrected"] is False
    executed = steps_of(body, "db_execution")[0]["data"]["executed_sql"]
    assert body["sql"] == executed  # `sql` is still the SQL that actually ran


def test_health_endpoint_is_unchanged(client):
    body = client.get("/health").json()
    assert set(body) == {"status", "database_path", "database_found", "llm_mode", "model"}


def test_errors_keep_their_string_detail(client):
    """Day 2-5 callers read response.json()["detail"] as a string."""
    for question in ("what is the meaning of life", ALWAYS_FAILS_QUESTION):
        assert isinstance(ask(client, question).json()["detail"], str)


# ---------------------------------------------------------------------------
# Security: nothing sensitive reaches the browser.
# ---------------------------------------------------------------------------


def test_mock_mode_reports_no_model(client):
    assert ask(client, "How many orders are there?").json()["llm"] == {"mode": "mock", "model": None}


def test_responses_do_not_leak_keys_or_server_paths(client, monkeypatch):
    fake_key = "sk-test-secret-value-1234567890"

    # A normal success response, a failure that echoes the key back, and
    # the dataset endpoint: none may contain a secret or a server path.
    success = ask(client, "How many orders are there?")

    monkeypatch.setattr(settings, "OPENAI_API_KEY", fake_key)

    def leaky(question, schema_text):
        raise RuntimeError(f"Incorrect API key provided: {fake_key}")

    monkeypatch.setattr(llm, "generate_sql", leaky)
    failure = ask(client, "How many orders are there?")
    assert failure.status_code == 502

    db_path = str(settings.DATABASE_PATH)
    for response in (success, failure, client.get("/dataset")):
        text = response.text
        assert fake_key not in text
        assert db_path not in text
        assert json.dumps(db_path)[1:-1] not in text  # the JSON-escaped form (backslashes doubled)
        assert "olist.db" not in text
        assert "Traceback" not in text


def test_llm_error_text_is_redacted(client, monkeypatch):
    """Provider error messages can echo part of a credential. Both the
    configured key and anything key-shaped must be scrubbed."""
    configured = "configured-key-abcdef123456"
    monkeypatch.setattr(settings, "OPENAI_API_KEY", configured)

    def leaky(question, schema_text):
        raise RuntimeError(f"Bad key {configured} (also seen: sk-proj-AbCdEfGhIjKlMnOp)")

    monkeypatch.setattr(llm, "generate_sql", leaky)

    body = ask(client, "anything").json()
    assert configured not in body["detail"]
    assert "sk-proj-AbCdEfGhIjKlMnOp" not in body["detail"]
    assert "[redacted]" in body["detail"]
    # The same scrubbing applies inside the trace and the error block.
    assert configured not in json.dumps(body)
    assert "sk-proj-AbCdEfGhIjKlMnOp" not in json.dumps(body)


def test_answer_question_still_raises_the_original_exceptions(monkeypatch):
    """The Day 5 interface: same exception types as before."""
    from backend.mock_llm import MockQuestionNotFound
    from backend.sql_guard import SqlValidationError

    with pytest.raises(MockQuestionNotFound):
        agent.answer_question("what is the meaning of life")
    with pytest.raises(agent.PipelineError):
        agent.answer_question(ALWAYS_FAILS_QUESTION)

    monkeypatch.setattr(llm, "generate_sql", lambda question, schema_text: "DELETE FROM orders")
    with pytest.raises(SqlValidationError):
        agent.answer_question("anything")
