"""Tests for Day 5's error-driven self-correction loop (backend/agent.py).

All of these run in mock mode (forced by tests/conftest.py) — no OpenAI
call, no API key needed. The mock SQL in backend/mock_llm.py's
_MOCK_CORRECTION_SCENARIOS is genuinely wrong/right; it really executes
against the real database/olist.db and really fails or succeeds, so this
suite exercises the actual retry logic, not a simulation of it.
"""
import pytest

from backend import agent, db, llm, sql_guard
from backend.mock_llm import MockQuestionNotFound


def test_no_correction_needed_for_a_normal_question():
    """A question that works on the first try shouldn't look "corrected"."""
    result = agent.answer_question("How many orders are there?")
    assert result.corrected is False
    assert len(result.attempts) == 1
    assert result.attempts[0].error is None
    assert result.rows[0]["total_orders"] == 99441


def test_self_correction_succeeds_on_first_retry():
    """The scripted scenario: initial SQL references a column that doesn't
    exist (order_year), fails for real, gets corrected, and the corrected
    SQL succeeds."""
    result = agent.answer_question("How many orders were placed in 2017?")

    assert result.corrected is True
    assert len(result.attempts) == 2
    assert result.attempts[0].error is not None
    assert "order_year" in result.attempts[0].error.lower()
    assert result.attempts[1].error is None
    assert result.row_count == 1
    assert result.rows[0]["orders_in_2017"] > 0  # a real, sane count


def test_self_correction_exhausts_retries_and_raises_clean_error():
    """Both scripted corrections are also broken (nonexistent tables) — all
    3 attempts (1 initial + 2 corrections) should fail, and the resulting
    error must be clean: no Python traceback, just a plain-English summary.
    """
    with pytest.raises(agent.PipelineError) as exc_info:
        agent.answer_question("This question always fails even after correction")

    error = exc_info.value
    assert len(error.attempts) == 3
    assert all(a.error is not None for a in error.attempts)

    message = str(error)
    assert "Traceback" not in message
    assert ".py\", line" not in message  # no stack-trace-looking text
    assert "Attempt 1" in message and "Attempt 2" in message and "Attempt 3" in message


def test_never_exceeds_the_configured_max_correction_attempts():
    """Explicit numeric check on the retry cap, independent of the exact
    scenario data — ties the test to agent.MAX_CORRECTION_ATTEMPTS itself
    rather than a hardcoded "3", so it stays correct if that constant ever
    changes."""
    with pytest.raises(agent.PipelineError) as exc_info:
        agent.answer_question("This question always fails even after correction")
    assert len(exc_info.value.attempts) == agent.MAX_CORRECTION_ATTEMPTS + 1


def test_unknown_mock_question_is_not_treated_as_a_database_error():
    """generate_sql() failing outright (unrecognized mock question) should
    propagate immediately — this isn't a "SQL was wrong" case at all, so it
    must not be retried or wrapped in a PipelineError."""
    with pytest.raises(MockQuestionNotFound):
        agent.answer_question("what is the meaning of life")


def test_validation_failure_is_raised_immediately_not_retried(monkeypatch):
    """If the (mock or real) LLM ever returns SQL the guard would reject —
    e.g. it ignores the read-only rule — that must surface immediately as
    SqlValidationError, NOT be treated as a retryable execution failure.
    Retrying wouldn't help (there's no database error to explain), and
    quietly asking again would just be hoping for a different unsafe
    answer instead of failing loudly."""
    monkeypatch.setattr(llm, "generate_sql", lambda question, schema_text: "DELETE FROM orders")

    with pytest.raises(sql_guard.SqlValidationError):
        agent.answer_question("anything, generate_sql is monkeypatched above")


def test_corrected_sql_still_goes_through_the_guard_before_execution(monkeypatch):
    """THE core security test: if a correction attempt itself is unsafe
    (e.g. the LLM proposes a DROP TABLE while 'fixing' the query), it must
    be rejected by sql_guard BEFORE db.run_query is ever called — proving
    self-correction cannot be used to sneak a write past validation.
    """
    # First call to generate_sql returns SQL that's valid but wrong
    # (guaranteed to fail at execution); the "correction" then returns
    # something sql_guard must reject outright.
    monkeypatch.setattr(
        llm, "generate_sql", lambda question, schema_text: "SELECT * FROM not_a_real_table"
    )
    monkeypatch.setattr(
        llm,
        "correct_sql",
        lambda **kwargs: "DROP TABLE orders",
    )

    def _fake_run_query(sql):
        # A canary, not a real database: it simulates the genuine "table
        # doesn't exist" failure for the first (safe-but-wrong) attempt, so
        # the loop proceeds to ask for a correction — but if the DROP TABLE
        # "correction" ever reached this far, sql_guard failed to stop it.
        if "DROP" in sql.upper():
            raise AssertionError(f"db.run_query must never be called with unsafe SQL, got: {sql}")
        raise db.QueryExecutionError("no such table: not_a_real_table")

    monkeypatch.setattr(db, "run_query", _fake_run_query)

    with pytest.raises(sql_guard.SqlValidationError):
        agent.answer_question("anything, generate_sql/correct_sql are monkeypatched above")


def test_ask_endpoint_reports_correction_in_mock_mode():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "How many orders were placed in 2017?"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["corrected"] is True
    assert body["attempts"] == 2
    assert body["row_count"] == 1


def test_ask_endpoint_returns_clean_500_after_exhausting_corrections():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        response = client.post(
            "/ask", json={"question": "This question always fails even after correction"}
        )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "Traceback" not in detail
    assert ".py\", line" not in detail
    assert "attempt" in detail.lower()
