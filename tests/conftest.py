"""Pytest configuration: forces mock LLM mode for the whole test session.

This must set the environment variable *before* any `backend.*` module is
imported anywhere in the suite, because backend/config.py reads
environment variables once, at import time, when the module is first
loaded. pytest always imports each directory's conftest.py before it
imports that directory's test modules, which is exactly the timing we need.

Using setdefault (not a plain assignment) means a developer who explicitly
exports SQLGENIE_LLM_MODE=openai in their shell before running pytest can
still opt into testing the real path; by default, tests never make a
network call or need an API key.

The same trick keeps the tests independent of whatever is in a developer's
personal .env: a real environment variable always wins over .env (see
backend/config.py), so pinning the dataset and CORS origins here means a
customized .env can't make these tests fail.
"""
import hashlib
import json
import os
import re
import urllib.request

import pytest

os.environ.setdefault("SQLGENIE_LLM_MODE", "mock")
os.environ.setdefault("SQLGENIE_DATASET", "olist")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")


@pytest.fixture(autouse=True)
def _reset_dataset_registry():
    """Day 7: the active dataset and the dataset registry are real, mutable,
    in-memory server state (backend/dataset.py) — a test that uploads a CSV
    or switches datasets would otherwise leak that into every test that
    runs after it. Snapshot both before each test and restore them after,
    the same way the env vars above are pinned rather than left to chance.
    """
    from backend import dataset as dataset_module
    from backend import schema_retrieval as schema_retrieval_module

    original_active = dataset_module._active_dataset_id
    original_registry = dict(dataset_module._registry)
    yield
    dataset_module._active_dataset_id = original_active
    dataset_module._registry.clear()
    dataset_module._registry.update(original_registry)
    # A test's uploaded/selected dataset may also have left retrieval
    # caches behind, keyed by a dataset id that no longer exists afterward.
    schema_retrieval_module._documents_cache.clear()
    schema_retrieval_module._full_schema_chars_cache.clear()
    schema_retrieval_module._embeddings_cache.clear()


_FAKE_EMBED_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Dropped before hashing: words too common/generic to carry any topical
# signal ("how", "many", "an", "one", ...). A literal bag-of-words fake
# can't tell "many" (the question word) from "many" (an incidental word in
# some table's free-text description) the way a real semantic embedding
# model would — filtering common connector words first is what keeps this
# fake's ranking behavior a reasonable proxy for the real one instead of
# drowning in incidental overlaps.
_FAKE_EMBED_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "in", "on", "for", "to", "by", "with", "from", "as", "at",
    "and", "or", "each", "it", "its", "this", "that", "these", "those",
    "there", "have", "has", "had", "can", "will", "would", "should",
    "what", "which", "who", "whom", "how", "many", "much", "do", "does",
    "did", "please", "show", "me", "list", "find", "get", "give",
    "one", "row", "rows", "per", "table", "columns", "column",
})


def fake_embedding_vector(prompt: str) -> list[float]:
    """A deterministic, content-aware stand-in for a real embedding: hashes
    each (non-stopword) token into one of _FAKE_EMBED_DIM buckets (feature
    hashing) and counts it there, so cosine similarity between two fake
    vectors tracks topical word overlap — enough to exercise ranking/
    fallback logic meaningfully (two questions/tables sharing no topical
    words get a similarity of exactly 0.0) — without any network call or
    real ML model. Deterministic across runs because it uses hashlib, not
    Python's randomized str hash(). Exported (no leading underscore) so
    tests/test_embedding_retrieval.py can build the same vectors a cached
    embedding would have, to assert on ranking directly.
    """
    vector = [0.0] * _FAKE_EMBED_DIM
    for token in _TOKEN_RE.findall(prompt.lower()):
        if token in _FAKE_EMBED_STOPWORDS:
            continue
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % _FAKE_EMBED_DIM
        vector[bucket] += 1.0
    return vector


class _FakeOllamaResponse:
    """Minimal stand-in for what urllib.request.urlopen() returns — a
    context manager whose .read() gives raw bytes. Same shape as the one
    tests/test_llm_ollama.py already uses for Ollama's /api/chat."""

    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture(scope="session", autouse=True)
def _fake_ollama_embeddings():
    """Default, zero-network stand-in for Ollama's /api/embeddings, applied
    for the whole test session — the same spirit as pinning
    SQLGENIE_LLM_MODE=mock above. backend/schema_retrieval.py now calls
    backend/llm.embed_text() for every multi-table dataset regardless of
    SQLGENIE_LLM_MODE, so without this the whole suite would need a live
    Ollama embedding server just to run.

    Session-scoped, and patched directly rather than through the
    function-scoped `monkeypatch` fixture, quite deliberately: several test
    files (test_football_movies.py, test_dataset_metadata.py,
    test_multi_dataset_ask.py, test_csv_upload.py) use a MODULE-scoped
    `client` fixture that builds the FastAPI TestClient — and so runs the
    app's startup hook, which now embeds every multi-table dataset's tables
    — once per module. Module-scoped fixtures are set up before any
    function-scoped fixture a given test also uses, so a function-scoped
    version of this fixture would still let that first startup hit a real
    network call before it ever got installed. Session scope guarantees
    this is in place before ANY fixture in ANY test file runs.

    Only requests to the embeddings path are faked; anything else is passed
    through to the real urlopen. A test that wants to exercise the REAL
    embeddings implementation's wire format (request/response shape, error
    handling) monkeypatches urlopen again inside its own body, with its own
    function-scoped `monkeypatch` — see tests/test_embedding_retrieval.py —
    and that later, test-local monkeypatch.setattr overrides this fixture's
    for the rest of that one test (and correctly reverts back to THIS fake,
    not the true original, once that test ends), exactly like
    tests/test_llm_ollama.py already does for the chat endpoint.
    """
    real_urlopen = urllib.request.urlopen

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/api/embeddings"):
            payload = json.loads(request.data)
            return _FakeOllamaResponse({"embedding": fake_embedding_vector(payload["prompt"])})
        return real_urlopen(request, timeout=timeout)

    urllib.request.urlopen = fake_urlopen
    yield
    urllib.request.urlopen = real_urlopen
