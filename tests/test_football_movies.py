"""Day 8: tests for the built-in Football and Movies datasets — the built
databases themselves, dataset-aware retrieval against each, and /ask
(including self-correction) end-to-end in mock mode.

Requires database/football/football.db and database/movies/movies.db to
already exist (run `python database/football/build_db.py` and
`python database/movies/build_db.py` first) — same convention Olist's own
tests have always assumed for database/olist.db.
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import dataset, schema_retrieval
from backend.config import PROJECT_ROOT
from backend.main import app

FOOTBALL_DB = PROJECT_ROOT / "database/football/football.db"
MOVIES_DB = PROJECT_ROOT / "database/movies/movies.db"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# ---------------------------------------------------------------------------
# Football: the database itself.
# ---------------------------------------------------------------------------


def test_football_db_exists_and_row_count():
    assert FOOTBALL_DB.exists(), "run `python database/football/build_db.py` first"
    conn = _connect(FOOTBALL_DB)
    assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 49547
    conn.close()


def test_football_db_expected_columns():
    conn = _connect(FOOTBALL_DB)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    conn.close()
    assert columns == {
        "match_id", "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral", "winner", "goal_diff",
    }


def test_football_derived_columns_are_correct():
    """winner and goal_diff aren't in the source CSV at all — verify they
    were computed correctly, not just present."""
    conn = _connect(FOOTBALL_DB)
    # winner must always agree with the actual scores.
    mismatched = conn.execute(
        "SELECT COUNT(*) FROM results WHERE "
        "(winner='home' AND home_score<=away_score) OR "
        "(winner='away' AND away_score<=home_score) OR "
        "(winner='draw' AND home_score!=away_score)"
    ).fetchone()[0]
    assert mismatched == 0
    # goal_diff must always be the absolute difference.
    wrong_diff = conn.execute(
        "SELECT COUNT(*) FROM results WHERE goal_diff != ABS(home_score - away_score)"
    ).fetchone()[0]
    assert wrong_diff == 0
    # neutral was converted from the source's TEXT "TRUE"/"FALSE" to INTEGER 0/1.
    row = conn.execute("SELECT DISTINCT neutral FROM results").fetchall()
    assert set(v for (v,) in row) <= {0, 1}
    conn.close()


def test_football_db_is_read_only_via_the_normal_connection_style():
    conn = _connect(FOOTBALL_DB)
    conn.execute("PRAGMA query_only = TRUE")
    with pytest.raises(sqlite3.Error):
        conn.execute("DELETE FROM results WHERE match_id = 1")
    conn.close()


# ---------------------------------------------------------------------------
# Movies: the database itself.
# ---------------------------------------------------------------------------


def test_movies_db_exists_and_row_count():
    assert MOVIES_DB.exists(), "run `python database/movies/build_db.py` first"
    conn = _connect(MOVIES_DB)
    assert conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] == 1000
    conn.close()


def test_movies_db_expected_columns():
    conn = _connect(MOVIES_DB)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(movies)")}
    conn.close()
    assert columns == {
        "movie_id", "series_title", "released_year", "certificate", "runtime_minutes",
        "genre", "primary_genre", "imdb_rating", "overview", "meta_score", "director",
        "star1", "star2", "star3", "star4", "no_of_votes", "gross", "poster_link",
    }


def test_apollo_13_released_year_was_repaired():
    """The one known malformed row: released_year had literally been "PG"
    (the certificate value) instead of a year. Fixed, not dropped."""
    conn = _connect(MOVIES_DB)
    row = conn.execute(
        "SELECT released_year, certificate, runtime_minutes FROM movies WHERE series_title = 'Apollo 13'"
    ).fetchone()
    conn.close()
    assert row is not None, "Apollo 13 must still be present (fixed, not dropped)"
    assert row == (1995, "U", 140)


def test_runtime_and_gross_are_cleaned_numeric_columns():
    conn = _connect(MOVIES_DB)
    # runtime_minutes: was "142 min" text in the source; must be a plain integer now.
    non_integer_runtime = conn.execute(
        "SELECT COUNT(*) FROM movies WHERE typeof(runtime_minutes) != 'integer'"
    ).fetchone()[0]
    assert non_integer_runtime == 0
    # gross: was "28,341,469" text with commas; must be integer or NULL, never text.
    non_integer_gross = conn.execute(
        "SELECT COUNT(*) FROM movies WHERE gross IS NOT NULL AND typeof(gross) != 'integer'"
    ).fetchone()[0]
    assert non_integer_gross == 0
    conn.close()


def test_primary_genre_is_the_first_listed_genre():
    conn = _connect(MOVIES_DB)
    row = conn.execute(
        "SELECT genre, primary_genre FROM movies WHERE genre LIKE '%,%' LIMIT 1"
    ).fetchone()
    conn.close()
    genre, primary = row
    assert primary == genre.split(",")[0].strip()


def test_missing_values_preserved_as_null_not_dropped_or_zeroed():
    """101/157/169 rows are missing certificate/meta_score/gross in the
    source — must stay NULL (not "" or 0), and the rows themselves must
    still exist (1000 total, not 1000 minus the ones with gaps)."""
    conn = _connect(MOVIES_DB)
    assert conn.execute("SELECT COUNT(*) FROM movies WHERE certificate IS NULL").fetchone()[0] == 101
    assert conn.execute("SELECT COUNT(*) FROM movies WHERE meta_score IS NULL").fetchone()[0] == 157
    assert conn.execute("SELECT COUNT(*) FROM movies WHERE gross IS NULL").fetchone()[0] == 169
    conn.close()


def test_movies_db_is_read_only_via_the_normal_connection_style():
    conn = _connect(MOVIES_DB)
    conn.execute("PRAGMA query_only = TRUE")
    with pytest.raises(sqlite3.Error):
        conn.execute("DELETE FROM movies WHERE movie_id = 1")
    conn.close()


# ---------------------------------------------------------------------------
# Registration and metadata.
# ---------------------------------------------------------------------------


def test_all_three_datasets_are_registered_with_correct_metadata():
    profiles = {p.id: p for p in dataset.list_datasets()}
    assert set(profiles) >= {"olist", "football", "movies"}

    assert profiles["olist"].domain == "ecommerce" and profiles["olist"].theme == "commerce"
    assert profiles["football"].domain == "sports" and profiles["football"].theme == "football"
    assert profiles["movies"].domain == "entertainment" and profiles["movies"].theme == "entertainment"

    assert profiles["football"].source == "built_in"
    assert profiles["movies"].source == "built_in"


def test_datasets_endpoint_lists_all_three(client):
    ids = [d["id"] for d in client.get("/datasets").json()["datasets"]]
    assert {"olist", "football", "movies"} <= set(ids)


@pytest.mark.parametrize("dataset_id", ["football", "movies", "olist"])
def test_selecting_each_dataset_works(client, dataset_id):
    response = client.post(f"/datasets/{dataset_id}/select")
    assert response.status_code == 200
    assert response.json()["id"] == dataset_id
    assert client.get("/dataset").json()["id"] == dataset_id


# ---------------------------------------------------------------------------
# Dataset-specific schema retrieval.
# ---------------------------------------------------------------------------


def test_retrieval_finds_the_results_table_for_football():
    tables = schema_retrieval.retrieve_relevant_tables(
        "Which team has the most wins?", dataset_id="football", db_path=FOOTBALL_DB
    )
    assert tables == ["results"]


def test_retrieval_finds_the_movies_table_for_movies():
    tables = schema_retrieval.retrieve_relevant_tables(
        "What is the average rating by genre?", dataset_id="movies", db_path=MOVIES_DB
    )
    assert tables == ["movies"]


def test_football_schema_text_includes_curated_notes():
    text = schema_retrieval.build_relevant_schema_text(
        "How many matches are there?", dataset_id="football", db_path=FOOTBALL_DB
    )
    assert "goal_diff" in text
    assert "winner" in text


def test_movies_schema_text_includes_curated_notes():
    text = schema_retrieval.build_relevant_schema_text(
        "What are the top movies?", dataset_id="movies", db_path=MOVIES_DB
    )
    assert "primary_genre" in text


def test_retrieval_does_not_mix_up_football_and_movies_tables():
    football_tables = schema_retrieval.retrieve_relevant_tables(
        "How many matches are there?", dataset_id="football", db_path=FOOTBALL_DB
    )
    movies_tables = schema_retrieval.retrieve_relevant_tables(
        "How many movies are there?", dataset_id="movies", db_path=MOVIES_DB
    )
    assert "movies" not in football_tables
    assert "results" not in movies_tables


# ---------------------------------------------------------------------------
# /ask end-to-end, mock mode — including self-correction.
# ---------------------------------------------------------------------------


def test_ask_football_returns_correct_answer(client):
    client.post("/datasets/football/select")
    response = client.post("/ask", json={"question": "Which team has the most wins?"})
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == [{"team": "Brazil", "wins": 675}]
    assert body["dataset"]["id"] == "football"
    assert body["corrected"] is False


def test_ask_movies_returns_correct_answer(client):
    client.post("/datasets/movies/select")
    response = client.post("/ask", json={"question": "How many movies are there?"})
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == [{"total_movies": 1000}]
    assert body["dataset"]["id"] == "movies"


def test_self_correction_against_football(client):
    client.post("/datasets/football/select")
    response = client.post("/ask", json={"question": "How many matches were played in 2018?"})
    assert response.status_code == 200
    body = response.json()
    assert body["corrected"] is True
    assert body["rows"] == [{"matches_in_2018": 929}]
    trace_errors = [
        s["data"]["error"] for s in body["trace"]["steps"]
        if s["type"] == "db_execution" and s["status"] == "failed"
    ]
    assert any("year" in (e or "") for e in trace_errors)


def test_self_correction_against_movies(client):
    client.post("/datasets/movies/select")
    response = client.post("/ask", json={"question": "How many movies were released after 2010?"})
    assert response.status_code == 200
    body = response.json()
    assert body["corrected"] is True
    assert body["rows"] == [{"movies_after_2010": 225}]


@pytest.mark.parametrize("dataset_id", ["football", "movies"])
def test_every_example_question_works_in_mock_mode(client, dataset_id):
    """Same invariant Day 6 established for Olist's example_questions —
    a dataset's own listed starter questions must never be a dead end."""
    client.post(f"/datasets/{dataset_id}/select")
    for question in client.get("/dataset").json()["example_questions"]:
        response = client.post("/ask", json={"question": question})
        assert response.status_code == 200, f"{dataset_id}: {question!r} -> {response.status_code}"


def test_switching_datasets_does_not_leak_answers(client):
    """Ask the same literal question shape against two datasets in a row —
    each must answer from its own database, not the other's."""
    client.post("/datasets/football/select")
    football = client.post("/ask", json={"question": "How many matches are there?"})
    assert football.json()["rows"] == [{"total_matches": 49547}]

    client.post("/datasets/movies/select")
    movies = client.post("/ask", json={"question": "How many movies are there?"})
    assert movies.json()["rows"] == [{"total_movies": 1000}]

    client.post("/datasets/olist/select")
    olist = client.post("/ask", json={"question": "How many orders are there?"})
    assert olist.json()["rows"] == [{"total_orders": 99441}]
