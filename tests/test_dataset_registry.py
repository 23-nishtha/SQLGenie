"""Tests for Day 7's dataset registry: identity, the active-dataset
pointer, and the split between the public DatasetProfile (no filesystem
path, ever) and the internal _DatasetRecord that carries one.

No OpenAI call, no upload — tests/test_csv_upload.py and
tests/test_multi_dataset_ask.py cover those. tests/conftest.py's
autouse fixture resets the registry/active pointer after every test here.
"""
import pytest
from fastapi.testclient import TestClient

from backend import dataset
from backend.dataset import DatasetProfile, UnknownDatasetError
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Pure registry behavior (no HTTP).
# ---------------------------------------------------------------------------


def test_olist_is_registered_and_active_by_default():
    active = dataset.get_active_dataset()
    assert active.id == "olist"
    assert active.source == "built_in"


def test_get_active_database_path_is_not_on_the_public_profile():
    """The internal path accessor works, but nothing about it is reachable
    from the public DatasetProfile model."""
    path = dataset.get_active_database_path()
    assert path.name == "olist.db"
    assert not hasattr(dataset.get_active_dataset(), "database_path")


def test_register_dataset_rejects_a_duplicate_id():
    profile = DatasetProfile(
        id="olist", name="dup", domain="x", theme="default", description="", source="uploaded",
    )
    with pytest.raises(ValueError, match="already registered"):
        dataset.register_dataset(profile, dataset.get_active_database_path(), "olist")


def test_set_active_dataset_switches_and_returns_the_profile():
    fake_path = dataset.get_active_database_path()  # reuse a real, valid file
    profile = DatasetProfile(
        id="football", name="Football", domain="sports", theme="football", description="", source="built_in",
    )
    dataset.register_dataset(profile, fake_path, schema_profile_id="football")

    result = dataset.set_active_dataset("football")
    assert result.id == "football"
    assert dataset.get_active_dataset().id == "football"
    assert dataset.get_active_database_path() == fake_path


def test_set_active_dataset_rejects_an_unknown_id():
    with pytest.raises(UnknownDatasetError, match="olist"):
        dataset.set_active_dataset("does-not-exist")
    # And the active dataset is unchanged after a rejected switch.
    assert dataset.get_active_dataset().id == "olist"


def test_list_datasets_includes_olist_first():
    profiles = dataset.list_datasets()
    assert profiles[0].id == "olist"
    assert all(isinstance(p, DatasetProfile) for p in profiles)


# ---------------------------------------------------------------------------
# HTTP endpoints.
# ---------------------------------------------------------------------------


def test_get_datasets_lists_olist_as_active(client):
    body = client.get("/datasets").json()
    assert body["active"] == "olist"
    ids = [d["id"] for d in body["datasets"]]
    assert "olist" in ids
    assert all("database_path" not in d for d in body["datasets"])


def test_select_endpoint_switches_the_active_dataset(client):
    fake_path = dataset.get_active_database_path()
    dataset.register_dataset(
        DatasetProfile(id="movies", name="Movies", domain="entertainment", theme="entertainment", description=""),
        fake_path,
        schema_profile_id="movies",
    )

    response = client.post("/datasets/movies/select")
    assert response.status_code == 200
    assert response.json() == {"id": "movies", "name": "Movies", "domain": "entertainment", "theme": "entertainment"}
    assert client.get("/dataset").json()["id"] == "movies"


def test_select_endpoint_404s_on_an_unknown_id(client):
    response = client.post("/datasets/does-not-exist/select")
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]
    # Still olist — a failed select must not change the active dataset.
    assert client.get("/dataset").json()["id"] == "olist"


def test_dataset_endpoints_never_mention_a_filesystem_path(client):
    dataset.register_dataset(
        DatasetProfile(id="football", name="Football", domain="sports", theme="football", description=""),
        dataset.get_active_database_path(),
        schema_profile_id="football",
    )
    responses = [
        client.get("/dataset"),
        client.get("/datasets"),
        client.post("/datasets/football/select"),
    ]
    db_path_str = str(dataset.get_active_database_path())
    for response in responses:
        text = response.text
        assert "database_path" not in text
        assert "olist.db" not in text
        assert db_path_str not in text
