"""Day 7 regression suite: switching the active dataset must not corrupt
either dataset's behavior, and every safety property that holds for Olist
must hold identically for an uploaded dataset — nothing about the guard,
the read-only connection, or self-correction is Olist-specific.
"""
import io
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import agent, dataset, db, llm, schema_retrieval, sql_guard
from backend.main import app

FOOTBALL_CSV = "player,goals,team\nMessi,30,PSG\nRonaldo,25,Al Nassr\nMbappe,28,PSG\n"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _upload(client, csv_text=FOOTBALL_CSV, filename="football.csv"):
    response = client.post("/datasets/upload", files={"file": (filename, csv_text, "text/csv")})
    assert response.status_code == 200
    return response.json()


# ---------------------------------------------------------------------------
# Switching datasets doesn't corrupt either one.
# ---------------------------------------------------------------------------


def test_olist_still_works_after_uploading_and_switching_back(client):
    baseline = client.post("/ask", json={"question": "How many orders are there?"})
    assert baseline.status_code == 200
    assert baseline.json()["rows"] == [{"total_orders": 99441}]

    _upload(client)
    assert dataset.get_active_dataset().source == "uploaded"

    client.post("/datasets/olist/select")
    after = client.post("/ask", json={"question": "How many orders are there?"})
    assert after.status_code == 200
    # Compare everything except `trace`: its timing fields (duration_ms)
    # will legitimately differ between two separate calls even when the
    # rest of the answer is identical.
    after_body, baseline_body = after.json(), baseline.json()
    assert {k: v for k, v in after_body.items() if k != "trace"} == {
        k: v for k, v in baseline_body.items() if k != "trace"
    }
    assert [s["type"] for s in after_body["trace"]["steps"]] == [
        s["type"] for s in baseline_body["trace"]["steps"]
    ]


def test_switching_datasets_targets_the_correct_database(client, monkeypatch):
    """The clearest possible proof retrieval/execution follow the active
    dataset: ask the SAME literal question against two different datasets
    and confirm each answer comes from its own database."""
    uploaded = _upload(client, filename="players.csv")
    monkeypatch.setattr(
        llm, "generate_sql", lambda question, schema_text: 'SELECT COUNT(*) AS n FROM "players"'
    )
    football_answer = client.post("/ask", json={"question": "count"})
    assert football_answer.json()["rows"] == [{"n": 3}]
    assert football_answer.json()["dataset"]["id"] == uploaded["id"]

    client.post("/datasets/olist/select")
    monkeypatch.setattr(
        llm, "generate_sql", lambda question, schema_text: "SELECT COUNT(*) AS n FROM orders"
    )
    olist_answer = client.post("/ask", json={"question": "count"})
    assert olist_answer.json()["rows"] == [{"n": 99441}]
    assert olist_answer.json()["dataset"]["id"] == "olist"


def test_retrieval_does_not_leak_tables_between_datasets(client):
    """Olist's tables must never appear in a single-table upload's
    retrieval results, and vice versa."""
    upload = _upload(client, filename="players.csv")

    uploaded_tables = schema_retrieval.retrieve_relevant_tables("How many rows are there?")
    assert uploaded_tables == ["players"]
    assert "orders" not in uploaded_tables

    client.post("/datasets/olist/select")
    olist_tables = schema_retrieval.retrieve_relevant_tables("How many orders are there?")
    assert "players" not in olist_tables
    assert "orders" in olist_tables


# ---------------------------------------------------------------------------
# Every safety property holds identically for an uploaded dataset.
# ---------------------------------------------------------------------------


def test_sql_guard_rejects_writes_against_an_uploaded_dataset(client, monkeypatch):
    _upload(client, filename="players.csv")
    monkeypatch.setattr(llm, "generate_sql", lambda question, schema_text: 'DELETE FROM "players"')

    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "sql_rejected"


def test_uploaded_database_connection_is_independently_read_only(client):
    """Same proof Day 4 required for Olist, now for an upload: bypass the
    guard entirely and try to write straight through the connection style
    db.py uses. Must fail regardless of what the guard would have said."""
    _upload(client, filename="players.csv")
    db_path = dataset.get_active_database_path()

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = TRUE")
    with pytest.raises(sqlite3.Error):
        conn.execute('DELETE FROM "players"')
    conn.close()


def test_self_correction_works_against_an_uploaded_dataset(client, monkeypatch):
    """The Day 5 retry loop is dataset-agnostic: a bad column name should
    trigger a correction against an uploaded table exactly like it does
    for Olist."""
    _upload(client, filename="players.csv")

    attempts = {"count": 0}

    def flaky_generate(question, schema_text):
        return 'SELECT COUNT(*) FROM "players" WHERE nonexistent_column = 1'

    def flaky_correct(**kwargs):
        attempts["count"] += 1
        return 'SELECT COUNT(*) AS n FROM "players"'

    monkeypatch.setattr(llm, "generate_sql", flaky_generate)
    monkeypatch.setattr(llm, "correct_sql", flaky_correct)

    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 200
    assert response.json()["corrected"] is True
    assert response.json()["rows"] == [{"n": 3}]
    assert attempts["count"] == 1


def test_corrected_sql_against_an_uploaded_dataset_still_goes_through_the_guard(client, monkeypatch):
    """The Day 5 security property (a rejected correction never reaches
    db.run_query), proven again for an uploaded dataset specifically."""
    _upload(client, filename="players.csv")

    monkeypatch.setattr(
        llm, "generate_sql", lambda question, schema_text: 'SELECT * FROM "not_a_real_table"'
    )
    monkeypatch.setattr(llm, "correct_sql", lambda **kwargs: 'DROP TABLE "players"')

    def _run_query_guard(sql, *args, **kwargs):
        if "DROP" in sql.upper():
            raise AssertionError("an unsafe correction must never reach db.run_query")
        raise db.QueryExecutionError("no such table: not_a_real_table")

    monkeypatch.setattr(db, "run_query", _run_query_guard)

    with pytest.raises(sql_guard.SqlValidationError):
        agent.answer_question("anything")
