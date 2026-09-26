"""Tests for Day 6's dataset metadata (GET /dataset, the `dataset` block on
/ask) and for the OpenAPI description a frontend would read.

Mock mode only: no OpenAI call, no API key.
"""
import re
from typing import get_args

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import dataset
from backend.config import settings
from backend.dataset import DatasetProfile, ThemeKey, UnknownDatasetError
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /dataset
# ---------------------------------------------------------------------------


def test_dataset_endpoint_describes_olist(client):
    response = client.get("/dataset")
    assert response.status_code == 200
    body = response.json()

    assert body["id"] == "olist"
    assert body["name"] == "Olist Brazilian E-Commerce"
    assert body["domain"] == "ecommerce"
    assert body["theme"] == "commerce"
    assert body["description"]
    assert body["example_questions"]


def test_theme_is_one_of_the_allowed_keys(client):
    assert client.get("/dataset").json()["theme"] in get_args(ThemeKey)


def test_theme_is_a_key_not_css(client):
    """The backend names a theme; the frontend owns what it looks like.
    Nothing styling-related may leak into the API."""
    body = client.get("/dataset").json()

    assert re.fullmatch(r"[a-z_]+", body["theme"])
    forbidden_keys = ("css", "color", "colour", "style", "font", "background")
    assert not [k for k in body if any(word in k.lower() for word in forbidden_keys)]
    for field in ("id", "name", "domain", "theme"):
        value = body[field]
        assert "#" not in value and "rgb" not in value.lower() and "px" not in value.lower()


def test_invalid_theme_key_is_rejected():
    with pytest.raises(ValidationError):
        DatasetProfile(
            id="x", name="X", domain="x", theme="neon-pink",
            description="", example_questions=[],
        )


def test_every_example_question_works_in_mock_mode(client):
    """The frontend's empty state offers these; none may be a dead end."""
    for question in client.get("/dataset").json()["example_questions"]:
        response = client.post("/ask", json={"question": question})
        assert response.status_code == 200, f"{question!r} -> {response.status_code}"


# ---------------------------------------------------------------------------
# The `dataset` block on /ask
# ---------------------------------------------------------------------------


def test_ask_dataset_block_matches_the_dataset_endpoint(client):
    profile = client.get("/dataset").json()
    expected = {k: profile[k] for k in ("id", "name", "domain", "theme")}

    success = client.post("/ask", json={"question": "How many orders are there?"})
    failure = client.post("/ask", json={"question": "what is the meaning of life"})

    assert success.status_code == 200 and failure.status_code == 400
    assert success.json()["dataset"] == expected
    assert failure.json()["dataset"] == expected


# ---------------------------------------------------------------------------
# Choosing the dataset
# ---------------------------------------------------------------------------


def test_unknown_dataset_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_DATASET", "football")
    with pytest.raises(UnknownDatasetError, match="olist"):
        dataset.get_active_dataset()


def test_startup_fails_fast_on_an_unknown_dataset(monkeypatch):
    monkeypatch.setattr(settings, "SQLGENIE_DATASET", "football")
    with pytest.raises(UnknownDatasetError):
        with TestClient(app):
            pass


# ---------------------------------------------------------------------------
# OpenAPI: what a frontend (or a code generator) reads.
# ---------------------------------------------------------------------------

STEP_TYPES = {
    "question", "schema_retrieval", "sql_generation", "sql_guard",
    "db_execution", "self_correction", "result",
}


@pytest.fixture(scope="module")
def openapi(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def test_openapi_documents_the_new_endpoint_and_responses(openapi):
    assert "/dataset" in openapi["paths"]
    ask_responses = openapi["paths"]["/ask"]["post"]["responses"]
    assert {"200", "400", "422", "500", "502"} <= set(ask_responses)


def test_openapi_contains_the_response_and_trace_models(openapi):
    schemas = openapi["components"]["schemas"]
    expected = {
        "AskResponse", "ErrorResponse", "ErrorInfo", "Trace", "TraceSummary",
        "LlmInfo", "DatasetSummary", "DatasetProfile",
        "QuestionStep", "SchemaRetrievalStep", "SqlGenerationStep", "SqlGuardStep",
        "DbExecutionStep", "SelfCorrectionStep", "ResultStep",
        "QuestionData", "SchemaRetrievalData", "SqlGenerationData", "SqlGuardData",
        "DbExecutionData", "SelfCorrectionData", "ResultData", "RetrievedTable",
    }
    assert expected <= set(schemas)


def test_trace_steps_are_a_discriminated_union_on_type(openapi):
    items = openapi["components"]["schemas"]["Trace"]["properties"]["steps"]["items"]
    assert items["discriminator"]["propertyName"] == "type"
    assert set(items["discriminator"]["mapping"]) == STEP_TYPES


def test_ask_response_schema_keeps_the_original_fields_and_adds_the_new_ones(openapi):
    properties = set(openapi["components"]["schemas"]["AskResponse"]["properties"])
    original = {"question", "sql", "columns", "rows", "row_count", "attempts", "corrected"}
    added = {"status", "dataset", "llm", "trace"}
    assert original | added <= properties
