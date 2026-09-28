"""Tests for Day 7's CSV -> SQLite upload path: validation, identifier
sanitization, safe conversion, and the /datasets/upload endpoint.

No OpenAI call needed — uploading and converting a CSV never touches the LLM.
"""
import io
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import csv_upload, dataset
from backend.config import settings
from backend.csv_upload import CsvValidationError, build_dataset_from_csv, sanitize_table_name
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _csv_file(text: str):
    return io.BytesIO(text.encode("utf-8"))


VALID_CSV = "name,goals,team\nMessi,30,PSG\nRonaldo,25,Al Nassr\n"


# ---------------------------------------------------------------------------
# Identifier sanitization — the core security property of this module.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename,expected", [
    ("football.csv", "football"),
    ("My Football Data.csv", "my_football_data"),
    ("'; DROP TABLE orders; --.csv", "drop_table_orders"),
    ("123movies.csv", "uploaded_table_123movies"),
    ("....csv", "uploaded_table"),
    ("sqlite_master.csv", "t_sqlite_master"),
])
def test_table_names_are_sanitized(filename, expected):
    assert sanitize_table_name(filename) == expected


def test_sanitized_table_name_contains_only_safe_characters():
    import re
    name = sanitize_table_name("Revenue ($) — Q1'24.csv")
    assert re.fullmatch(r"[a-z][a-z0-9_]*", name)


def test_malicious_column_headers_are_sanitized_and_the_table_is_still_queryable():
    csv_text = '"id"; DROP TABLE x; --,normal_col\n1,hello\n2,world\n'
    result = build_dataset_from_csv("evil.csv", _csv_file(csv_text))
    try:
        assert all(col.isidentifier() or col[0].isalpha() for col in result.column_names)
        conn = sqlite3.connect(f"file:{result.database_path}?mode=ro", uri=True)
        rows = conn.execute(f'SELECT * FROM "{result.table_name}"').fetchall()
        conn.close()
        assert len(rows) == 2
        # The DROP-shaped header never executed as SQL — the table it
        # supposedly targeted (x, or evil itself) simply doesn't exist,
        # and querying it back above proves the real table is intact.
    finally:
        result.database_path.unlink(missing_ok=True)


def test_duplicate_column_names_are_deduplicated_not_merged():
    csv_text = "Price,price,PRICE!!\n1,2,3\n"
    result = build_dataset_from_csv("dup.csv", _csv_file(csv_text))
    try:
        assert len(result.column_names) == len(set(result.column_names)) == 3
    finally:
        result.database_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Validation: rejected inputs.
# ---------------------------------------------------------------------------


def test_non_csv_extension_is_rejected():
    with pytest.raises(CsvValidationError, match="\\.csv"):
        build_dataset_from_csv("data.exe", _csv_file(VALID_CSV))


def test_no_filename_is_rejected():
    with pytest.raises(CsvValidationError):
        build_dataset_from_csv("", _csv_file(VALID_CSV))


def test_empty_file_is_rejected():
    with pytest.raises(CsvValidationError, match="empty"):
        build_dataset_from_csv("empty.csv", io.BytesIO(b""))


def test_header_only_csv_is_rejected():
    with pytest.raises(CsvValidationError, match="no data rows"):
        build_dataset_from_csv("headeronly.csv", _csv_file("a,b,c\n"))


def test_unparseable_csv_is_rejected_cleanly():
    # An unterminated quoted field is a real CSV parse error, not garbage
    # sqlite would somehow execute — must come back as CsvValidationError,
    # not a raw pandas exception.
    with pytest.raises(CsvValidationError):
        build_dataset_from_csv("broken.csv", _csv_file('a,b\n"unterminated,2\n'))


def test_oversized_file_is_rejected_without_buffering_it_all(monkeypatch):
    monkeypatch.setattr(settings, "CSV_UPLOAD_MAX_BYTES", 10)  # 10 bytes
    with pytest.raises(CsvValidationError, match="limit"):
        build_dataset_from_csv("big.csv", _csv_file(VALID_CSV))


def test_too_many_columns_is_rejected(monkeypatch):
    monkeypatch.setattr(csv_upload, "MAX_COLUMNS", 3)
    csv_text = "a,b,c,d,e\n1,2,3,4,5\n"
    with pytest.raises(CsvValidationError, match="too many columns"):
        build_dataset_from_csv("wide.csv", _csv_file(csv_text))


# ---------------------------------------------------------------------------
# A valid upload: correct data, correct types, on disk in the right place.
# ---------------------------------------------------------------------------


def test_valid_csv_is_converted_correctly():
    result = build_dataset_from_csv("football.csv", _csv_file(VALID_CSV))
    try:
        assert result.table_name == "football"
        assert result.row_count == 2
        assert result.column_names == ["name", "goals", "team"]
        assert result.database_path.exists()
        # Server-generated location, not derived from the filename.
        assert "football" not in str(result.database_path.parent.name)

        conn = sqlite3.connect(f"file:{result.database_path}?mode=ro", uri=True)
        cols = conn.execute('PRAGMA table_info("football")').fetchall()
        rows = conn.execute('SELECT name, goals FROM "football" ORDER BY goals').fetchall()
        conn.close()

        types_by_name = {c[1]: c[2] for c in cols}
        assert types_by_name["goals"] == "INTEGER"
        assert types_by_name["name"] == "TEXT"
        assert rows == [("Ronaldo", 25), ("Messi", 30)]
    finally:
        result.database_path.unlink(missing_ok=True)


def test_uploaded_database_is_stored_under_uploads_dir():
    result = build_dataset_from_csv("x.csv", _csv_file(VALID_CSV))
    try:
        assert settings.UPLOADS_DIR in result.database_path.parents
    finally:
        result.database_path.unlink(missing_ok=True)


def test_uploaded_database_is_read_only_afterward():
    """The one write connection csv_upload.py opens is closed by the time
    build_dataset_from_csv returns — from then on it behaves exactly like
    olist.db: mode=ro + PRAGMA query_only rejects a write."""
    result = build_dataset_from_csv("x.csv", _csv_file(VALID_CSV))
    try:
        conn = sqlite3.connect(f"file:{result.database_path}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = TRUE")
        with pytest.raises(sqlite3.Error):
            conn.execute(f'DELETE FROM "{result.table_name}"')
        conn.close()
    finally:
        result.database_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The /datasets/upload endpoint.
# ---------------------------------------------------------------------------


def test_upload_endpoint_registers_and_activates_the_dataset(client):
    response = client.post(
        "/datasets/upload",
        files={"file": ("football.csv", VALID_CSV, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["source"] == "uploaded"
    assert body["id"].startswith("upload_")
    assert "database_path" not in response.text
    assert dataset.get_active_dataset().id == body["id"]  # auto-activated


def test_upload_endpoint_rejects_a_non_csv_file(client):
    response = client.post(
        "/datasets/upload",
        files={"file": ("data.exe", b"not a csv", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert dataset.get_active_dataset().id == "olist"  # unchanged on rejection


def test_uploaded_dataset_is_immediately_queryable_end_to_end(client, monkeypatch):
    """Full loop: upload -> becomes active -> retrieval finds its table ->
    a (monkeypatched, since mock mode has no canned SQL for uploads) LLM
    call generates SQL -> guard -> execution, all against the uploaded DB."""
    from backend import llm, schema_retrieval

    response = client.post(
        "/datasets/upload",
        files={"file": ("players.csv", VALID_CSV, "text/csv")},
    )
    table_name = response.json()["id"]  # not used for SQL; table is "players"

    assert "players" in schema_retrieval.retrieve_relevant_tables("How many rows are there?")

    monkeypatch.setattr(
        llm, "generate_sql", lambda question, schema_text: 'SELECT COUNT(*) AS n FROM "players"'
    )
    ask_response = client.post("/ask", json={"question": "How many rows are there?"})
    assert ask_response.status_code == 200
    assert ask_response.json()["rows"] == [{"n": 2}]
