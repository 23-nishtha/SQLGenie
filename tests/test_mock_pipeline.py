"""Verifies the full mock-mode pipeline:

    question -> llm.generate_sql() [mock] -> sql_guard -> db.run_query()

These run against the real database/olist.db (read-only), never touch
OpenAI, and never need an API key — see tests/conftest.py, which forces
SQLGENIE_LLM_MODE=mock before backend.config is ever imported.
"""
import pytest

from backend import db, llm, sql_guard
from backend.mock_llm import MockQuestionNotFound

# The 4 questions Day 2 requires mock mode to support, plus the 2 extras
# defined in mock_llm.py.
KNOWN_QUESTIONS = [
    "How many orders were delivered?",
    "How many orders are there?",
    "What are the top 5 product categories by revenue?",
    "What is the average order value?",
    "How many customers are there?",
    "What is the total revenue?",
]


@pytest.mark.parametrize("question", KNOWN_QUESTIONS)
def test_mock_pipeline_end_to_end(question):
    """Each known question should flow all the way through to real rows."""
    raw_sql = llm.generate_sql(question, schema_text="not used in mock mode")
    safe_sql = sql_guard.validate_and_prepare(raw_sql)  # must pass the real guard
    columns, rows = db.run_query(safe_sql)               # must run on the real DB

    assert columns, f"expected at least one column for: {question!r}"
    assert len(rows) >= 1, f"expected at least one row for: {question!r}"


def test_mock_delivered_orders_matches_known_value():
    """Cross-check against the row count verified in Day 1 (verify_db.py)."""
    raw_sql = llm.generate_sql("How many orders were delivered?", schema_text="")
    safe_sql = sql_guard.validate_and_prepare(raw_sql)
    _, rows = db.run_query(safe_sql)
    assert rows[0]["delivered_orders"] == 96478


def test_mock_total_orders_matches_known_value():
    raw_sql = llm.generate_sql("How many orders are there?", schema_text="")
    safe_sql = sql_guard.validate_and_prepare(raw_sql)
    _, rows = db.run_query(safe_sql)
    assert rows[0]["total_orders"] == 99441


def test_mock_sql_passes_the_real_sql_guard():
    """Mock SQL isn't special-cased — it must be a genuinely safe SELECT."""
    raw_sql = llm.generate_sql("What are the top 5 product categories by revenue?", schema_text="")
    safe_sql = sql_guard.validate_and_prepare(raw_sql)
    assert safe_sql.strip().upper().startswith("SELECT")


def test_mock_unknown_question_raises_clear_error():
    with pytest.raises(MockQuestionNotFound):
        llm.generate_sql("What is the meaning of life?", schema_text="")


def test_mock_mode_never_requires_api_key(monkeypatch):
    """Mock mode must work even with no OpenAI key configured at all."""
    from backend.config import settings

    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    raw_sql = llm.generate_sql("How many orders are there?", schema_text="")
    assert "orders" in raw_sql.lower()


def test_ask_endpoint_works_in_mock_mode():
    """Full stack: HTTP request in, validated results out, no OpenAI call."""
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "How many orders are there?"})
    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 1
    assert body["rows"][0]["total_orders"] == 99441


def test_ask_endpoint_reports_llm_mode_in_health():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.json()["llm_mode"] == "mock"


def test_ask_endpoint_returns_400_for_unknown_mock_question():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What is the meaning of life?"})
    assert response.status_code == 400
