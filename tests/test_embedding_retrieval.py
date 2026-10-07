"""Tests for the embedding-based retrieval swap.

Covers what tests/test_schema_retrieval.py doesn't: the embedding call
itself (backend/llm.embed_text — real wire format, error handling),
cosine similarity, caching (embeddings computed once per table, not once
per question), single-table datasets skipping embedding entirely, and
fallback/error behavior when the embedding call fails.

Most tests here rely on tests/conftest.py's session-scoped
`_fake_ollama_embeddings` fixture (zero network, deterministic). The wire
format tests below are the one place that deliberately overrides it with a
test-local `monkeypatch`, the same way tests/test_llm_ollama.py already
does for the chat endpoint — see that fixture's docstring for why this is
safe (the test-local monkeypatch simply overrides the session fake for the
duration of one test, then correctly reverts back to it).
"""
import json
import urllib.error
import urllib.request

import pytest

from backend import agent, dataset, llm, schema_retrieval
from backend.config import settings
from tests.conftest import fake_embedding_vector

# ---------------------------------------------------------------------------
# backend/llm.embed_text — real wire format (request shape, response
# parsing, error handling). Mirrors tests/test_llm_ollama.py's style for
# the chat endpoint, just against /api/embeddings.
# ---------------------------------------------------------------------------


class _FakeOllamaResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_embed_text_sends_the_configured_model_and_prompt(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        return _FakeOllamaResponse({"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    vector = llm.embed_text("TABLE orders(order_id TEXT)")

    assert vector == [0.1, 0.2, 0.3]
    assert len(calls) == 1
    request = calls[0]
    assert request.full_url == f"{settings.OLLAMA_BASE_URL}/api/embeddings"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data)
    assert payload["model"] == settings.OLLAMA_EMBED_MODEL
    assert payload["prompt"] == "TABLE orders(order_id TEXT)"


def test_embed_text_connection_failure_raises_a_clear_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
        llm.embed_text("some text")


def test_embed_text_missing_model_error_names_the_embed_model(monkeypatch):
    def fake_urlopen(request, timeout=None):
        body = (
            b'{"error": "model \\"nomic-embed-text:latest\\" not found, '
            b'try pulling it first"}'
        )
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=__import__("io").BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match=settings.OLLAMA_EMBED_MODEL.replace(":", r"\:")):
        llm.embed_text("some text")


def test_embed_text_malformed_response_raises_a_clear_error(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeOllamaResponse({"unexpected": "shape"})  # no "embedding" key

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Unexpected response from Ollama embeddings"):
        llm.embed_text("some text")


# ---------------------------------------------------------------------------
# Cosine similarity.
# ---------------------------------------------------------------------------


def test_cosine_similarity_of_identical_vectors_is_one():
    from backend.schema_retrieval import _cosine_similarity

    vector = [1.0, 2.0, 3.0]
    assert _cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    from backend.schema_retrieval import _cosine_similarity

    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    from backend.schema_retrieval import _cosine_similarity

    assert _cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


def test_cosine_similarity_handles_a_zero_vector_without_dividing_by_zero():
    from backend.schema_retrieval import _cosine_similarity

    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# Retrieval ranking: documents whose embedded text shares topical words
# with the question should rank above ones that don't — demonstrated with
# the same deterministic fake vectors the rest of the suite uses, so this
# is a genuine property of the ranking code, not a coincidence of one
# dataset's exact wording.
# ---------------------------------------------------------------------------


def test_ranking_prefers_a_document_with_more_topical_overlap():
    from backend.schema_retrieval import _cosine_similarity

    question = fake_embedding_vector("average review score rating")
    reviews_doc = fake_embedding_vector("Table: order_reviews\nKeywords: review, rating, score, feedback")
    sellers_doc = fake_embedding_vector("Table: sellers\nKeywords: seller, vendor, merchant, shop")

    assert _cosine_similarity(question, reviews_doc) > _cosine_similarity(question, sellers_doc)


def test_olist_review_question_ranks_order_reviews_by_embedding(client):
    """End-to-end through the real retrieval pipeline (Olist, 10 tables):
    the top-ranked table for a review question is order_reviews itself,
    not just "present somewhere in the expanded set"."""
    details = schema_retrieval.retrieve_with_details("What is the average review score?")
    ranked_by_score = sorted(details.tables, key=lambda t: t.score, reverse=True)
    assert ranked_by_score[0].name == "order_reviews"
    assert ranked_by_score[0].selected_by == "embedding"


# ---------------------------------------------------------------------------
# Caching: embeddings are computed once per table at warm-up, never once
# per question.
# ---------------------------------------------------------------------------


def test_table_embeddings_are_not_recomputed_per_question(monkeypatch):
    call_count = {"n": 0}
    real_embed_text = llm.embed_text

    def counting_embed_text(text):
        call_count["n"] += 1
        return real_embed_text(text)

    monkeypatch.setattr(llm, "embed_text", counting_embed_text)

    schema_retrieval.init_documents("olist")
    after_warmup = call_count["n"]
    assert after_warmup > 1  # one call per Olist table

    schema_retrieval.retrieve_with_details("How many orders are there?", dataset_id="olist")
    schema_retrieval.retrieve_with_details("What is the average review score?", dataset_id="olist")

    # Each question call embeds the QUESTION (one more call each) — the
    # per-table embeddings from warm-up must not be recomputed.
    assert call_count["n"] == after_warmup + 2


def test_init_documents_clears_and_rewarms_the_embedding_cache(monkeypatch):
    call_count = {"n": 0}
    real_embed_text = llm.embed_text

    def counting_embed_text(text):
        call_count["n"] += 1
        return real_embed_text(text)

    monkeypatch.setattr(llm, "embed_text", counting_embed_text)

    schema_retrieval.init_documents("olist")
    first_warmup_calls = call_count["n"]
    schema_retrieval.init_documents("olist")  # re-warm, e.g. after a dataset switch back
    assert call_count["n"] == first_warmup_calls * 2


# ---------------------------------------------------------------------------
# Single-table datasets (football, movies, and any CSV upload) must skip
# embedding entirely — there's nothing to rank with only one table.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dataset_id,db_path",
    [
        ("football", dataset.PROJECT_ROOT / "database/football/football.db"),
        ("movies", dataset.PROJECT_ROOT / "database/movies/movies.db"),
    ],
)
def test_single_table_builtin_datasets_make_no_embedding_call(dataset_id, db_path, monkeypatch):
    call_count = {"n": 0}
    real_embed_text = llm.embed_text

    def counting_embed_text(text):
        call_count["n"] += 1
        return real_embed_text(text)

    monkeypatch.setattr(llm, "embed_text", counting_embed_text)

    schema_retrieval.init_documents(dataset_id, db_path)
    details = schema_retrieval.retrieve_with_details(
        "anything at all", dataset_id=dataset_id, db_path=db_path
    )

    assert call_count["n"] == 0
    assert len(details.tables) == 1
    assert details.tables[0].selected_by == "embedding"
    assert details.tables[0].score == 1.0


def test_uploaded_single_table_csv_makes_no_embedding_call(monkeypatch, tmp_path):
    from backend import csv_upload

    call_count = {"n": 0}
    real_embed_text = llm.embed_text

    def counting_embed_text(text):
        call_count["n"] += 1
        return real_embed_text(text)

    monkeypatch.setattr(llm, "embed_text", counting_embed_text)

    csv_path = tmp_path / "titanic.csv"
    csv_path.write_text("passenger_id,survived,age\n1,1,22\n2,0,38\n3,1,26\n", encoding="utf-8")
    with open(csv_path, "rb") as f:
        result = csv_upload.build_dataset_from_csv("titanic.csv", f)

    from backend.dataset import DatasetProfile

    profile = DatasetProfile(
        id=result.dataset_id, name=result.display_name, domain=result.domain,
        theme=result.theme, source="uploaded",
        description="test upload", example_questions=[],
    )
    dataset.register_dataset(profile, result.database_path, schema_profile_id=result.dataset_id)

    schema_retrieval.init_documents(result.dataset_id, result.database_path)
    details = schema_retrieval.retrieve_with_details(
        "How many passengers survived?", dataset_id=result.dataset_id, db_path=result.database_path,
    )

    assert call_count["n"] == 0
    assert len(details.tables) == 1
    assert details.tables[0].name == result.table_name


# ---------------------------------------------------------------------------
# Fallback and error behavior.
# ---------------------------------------------------------------------------


def test_low_similarity_everywhere_triggers_the_fallback_table(monkeypatch):
    """If every document's cosine similarity to the question is below
    MIN_SIMILARITY, retrieval must fall back to the hub table rather than
    returning the (meaningless) top-ranked-but-unconfident result."""
    monkeypatch.setattr(llm, "embed_text", lambda text: [0.0] * 8)

    details = schema_retrieval.retrieve_with_details("irrelevant", dataset_id="olist")
    fallback_tables = [t for t in details.tables if t.selected_by == "fallback"]
    assert fallback_tables and fallback_tables[0].name == "orders"


def test_embedding_failure_surfaces_as_a_clean_pipeline_error_not_a_crash():
    """If Ollama's embedding call fails (down, model not pulled, ...), the
    whole /ask pipeline must still fail cleanly — agent.run_pipeline()
    never raises for an expected failure; it reports it in the trace, same
    as any other retrieval failure before this change."""

    def failing_embed_text(text):
        raise RuntimeError("Could not connect to Ollama at http://localhost:11434 for embeddings.")

    import backend.llm as llm_module
    original = llm_module.embed_text
    llm_module.embed_text = failing_embed_text
    try:
        run = agent.run_pipeline("How many orders are there?")
    finally:
        llm_module.embed_text = original

    assert run.result is None
    assert run.error is not None
    assert run.failed_step == "schema_retrieval"
    assert run.trace.steps[-1].type == "schema_retrieval"
    assert run.trace.steps[-1].status == "failed"


# ---------------------------------------------------------------------------
# Existing pipeline compatibility: guardrails, execution and self-correction
# are all still reached and still work, unchanged, downstream of the new
# retrieval.
# ---------------------------------------------------------------------------


def test_full_ask_pipeline_still_works_end_to_end_with_embedding_retrieval(client):
    response = client.post("/ask", json={"question": "How many orders are there?"})
    assert response.status_code == 200
    body = response.json()
    assert body["sql"].strip().upper().startswith("SELECT")
    assert body["row_count"] == 1

    retrieval_step = next(s for s in body["trace"]["steps"] if s["type"] == "schema_retrieval")
    assert retrieval_step["status"] == "success"
    assert all(
        t["selected_by"] in {"embedding", "fk_expansion", "fallback"}
        for t in retrieval_step["data"]["tables"]
    )


def test_self_correction_still_runs_after_embedding_based_retrieval(client):
    """Day 5's self-correction scenario (a wrong column name, fixed on
    retry) still exercises the same validate -> execute -> correct loop,
    unaffected by how schema_text was produced."""
    response = client.post("/ask", json={"question": "How many orders were placed in 2017?"})
    assert response.status_code == 200
    body = response.json()
    assert body["corrected"] is True
    assert body["attempts"] == 2


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from backend.main import app

    with TestClient(app) as c:
        yield c
