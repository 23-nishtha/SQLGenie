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
import os

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
