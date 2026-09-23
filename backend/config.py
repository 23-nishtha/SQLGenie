"""Central place for settings, loaded from environment variables (.env).

Every other module imports `settings` from here instead of calling
os.getenv() itself, so all the configurable values live in one spot.
"""
from pathlib import Path

from dotenv import load_dotenv
import os

# Project root = one level up from this file's folder (backend/..).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load variables from a .env file at the project root, if one exists.
# Real environment variables (e.g. set by the OS or a deployment platform)
# still win over .env, which is what load_dotenv does by default.
load_dotenv(PROJECT_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to `default` if missing/invalid."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings:
    # --- LLM mode ---
    # "mock"   -> backend/mock_llm.py answers a small set of known questions
    #             with hand-written SQL. No network call, no API key needed.
    #             This is the default so the app works out of the box even
    #             without OpenAI credits.
    # "openai" -> the real call in backend/llm.py. Needs OPENAI_API_KEY.
    SQLGENIE_LLM_MODE: str = os.getenv("SQLGENIE_LLM_MODE", "mock").strip().lower()

    # --- OpenAI ---
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    # Configurable so we can swap models later without touching llm.py.
    # NOTE: "gpt-5.6-luna" (the current default) is not a model name we can
    # verify against OpenAI's public docs as of this writing. If API calls
    # fail with a "model not found" error, change OPENAI_MODEL in your .env
    # to a model your account has access to (e.g. "gpt-4o-mini").
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    # --- Database ---
    DATABASE_PATH: Path = PROJECT_ROOT / os.getenv("DATABASE_PATH", "database/olist.db")

    # --- Safety limits ---
    MAX_ROWS: int = _get_int("MAX_ROWS", 200)
    QUERY_TIMEOUT_SECONDS: int = _get_int("QUERY_TIMEOUT_SECONDS", 10)


settings = Settings()
