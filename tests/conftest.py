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

os.environ.setdefault("SQLGENIE_LLM_MODE", "mock")
os.environ.setdefault("SQLGENIE_DATASET", "olist")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
