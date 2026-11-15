"""Tests for scripts/fetch_olist_db.py: the Render build-time step that
downloads and SHA-256-verifies database/olist.db.

Uses small fake content in an isolated sandbox (pytest's tmp_path) --
never touches the real, ~183 MB database/olist.db, and never makes a real
network call (only local file:// URLs).
"""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "fetch_olist_db.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_olist_db_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fetch_module(tmp_path, monkeypatch):
    module = _load_module()
    (tmp_path / "database").mkdir()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "DB_PATH", tmp_path / "database" / "olist.db")
    monkeypatch.setattr(module, "CHECKSUM_PATH", tmp_path / "database" / "olist.db.sha256")
    monkeypatch.delenv("OLIST_DB_URL", raising=False)
    return module


def _write_checksum_for(module, content: bytes) -> None:
    digest = hashlib.sha256(content).hexdigest()
    module.CHECKSUM_PATH.write_text(f"{digest}  olist.db\n")


def test_committed_checksum_file_matches_the_real_database():
    """Regression lock: the checksum actually committed to the repo must
    match the actual database/olist.db file also committed... well, not
    committed (it's never committed -- see the README), but present
    locally. Catches the file silently drifting from the checksum."""
    real_db = Path(__file__).resolve().parent.parent / "database" / "olist.db"
    real_checksum_file = Path(__file__).resolve().parent.parent / "database" / "olist.db.sha256"
    if not real_db.exists():
        pytest.skip("database/olist.db not present locally")

    expected = real_checksum_file.read_text().strip().split()[0].lower()
    digest = hashlib.sha256()
    with real_db.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    assert digest.hexdigest() == expected


def test_fails_cleanly_when_file_missing_and_no_url_configured(fetch_module):
    _write_checksum_for(fetch_module, b"irrelevant for this case")

    assert fetch_module.main() == 1
    assert not fetch_module.DB_PATH.exists()


def test_fails_and_cleans_up_on_checksum_mismatch(fetch_module, tmp_path, monkeypatch):
    _write_checksum_for(fetch_module, b"the real expected content")
    wrong_source = tmp_path / "wrong.bin"
    wrong_source.write_bytes(b"not the expected content at all")
    monkeypatch.setenv("OLIST_DB_URL", wrong_source.as_uri())

    assert fetch_module.main() == 1
    assert not fetch_module.DB_PATH.exists()
    assert not fetch_module.DB_PATH.with_name("olist.db.partial").exists()


def test_downloads_and_installs_a_matching_file(fetch_module, tmp_path, monkeypatch):
    content = b"a small stand-in for the real olist database"
    _write_checksum_for(fetch_module, content)
    right_source = tmp_path / "right.bin"
    right_source.write_bytes(content)
    monkeypatch.setenv("OLIST_DB_URL", right_source.as_uri())

    assert fetch_module.main() == 0
    assert fetch_module.DB_PATH.read_bytes() == content


def test_skips_download_when_already_present_and_valid(fetch_module, monkeypatch):
    content = b"already here and already correct"
    _write_checksum_for(fetch_module, content)
    fetch_module.DB_PATH.write_bytes(content)

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("should not download when already valid")

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", _unexpected_call)

    assert fetch_module.main() == 0


def test_redownloads_when_present_but_checksum_is_stale(fetch_module, tmp_path, monkeypatch):
    """An on-disk file that doesn't match the committed checksum (e.g. an
    old/corrupted leftover from a previous build) is replaced, not trusted
    as-is."""
    fetch_module.DB_PATH.write_bytes(b"an old, no-longer-correct copy")

    fresh_content = b"the current correct content"
    _write_checksum_for(fetch_module, fresh_content)
    right_source = tmp_path / "right.bin"
    right_source.write_bytes(fresh_content)
    monkeypatch.setenv("OLIST_DB_URL", right_source.as_uri())

    assert fetch_module.main() == 0
    assert fetch_module.DB_PATH.read_bytes() == fresh_content
