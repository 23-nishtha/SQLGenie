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


# Local dev servers for the frontend. 5173 is Vite's default port; 8080 is
# where the Day 9 Lovable frontend actually runs. Deliberately NOT "*": a
# wildcard would let ANY website's JavaScript call this API from a
# visitor's browser. When the real frontend is deployed, set
# CORS_ALLOWED_ORIGINS in .env to its exact origin(s) instead.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:8080,http://127.0.0.1:8080"
)


def _get_list(name: str, default: str) -> list[str]:
    """Read a comma-separated env var as a list. Unset or blank falls back
    to `default`. Whitespace and trailing slashes are stripped, since an
    origin is scheme + host + port with no path ("http://x:5173/" would
    never match what the browser sends)."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        raw = default
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


class Settings:
    # --- LLM mode ---
    # "mock"   -> backend/mock_llm.py answers a small set of known questions
    #             with hand-written SQL. No network call, no API key needed.
    #             This is the default so the app works out of the box even
    #             without OpenAI credits.
    # "openai" -> the real call in backend/llm.py. Needs OPENAI_API_KEY.
    # "ollama" -> a local model via Ollama's /api/chat (see OLLAMA_* below).
    #             No API key, no network call beyond localhost — but needs
    #             Ollama actually running (`ollama serve`) with the
    #             configured model pulled (`ollama pull <model>`).
    SQLGENIE_LLM_MODE: str = os.getenv("SQLGENIE_LLM_MODE", "mock").strip().lower()

    # --- OpenAI ---
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    # Configurable so we can swap models later without touching llm.py.
    # NOTE: "gpt-5.6-luna" (the current default) is not a model name we can
    # verify against OpenAI's public docs as of this writing. If API calls
    # fail with a "model not found" error, change OPENAI_MODEL in your .env
    # to a model your account has access to (e.g. "gpt-4o-mini").
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    # --- Ollama (local LLM) ---
    # Which locally-pulled Ollama model generates the SQL when
    # SQLGENIE_LLM_MODE=ollama. Only used in that mode.
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    # Ollama's own HTTP API, running locally by default. rstrip so
    # backend/llm.py can safely append "/api/chat" without a double slash.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    # Which locally-pulled Ollama model embeds schema/question text for
    # retrieval (backend/schema_retrieval.py). Always Ollama, regardless of
    # SQLGENIE_LLM_MODE/OLLAMA_MODEL above — retrieval never calls a cloud
    # API, even when SQL generation itself uses OpenAI.
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

    # --- Dataset (Day 6) ---
    # Which entry of backend/dataset.py's registry describes the database
    # being queried. Only "olist" exists today.
    SQLGENIE_DATASET: str = os.getenv("SQLGENIE_DATASET", "olist").strip().lower()

    # --- CORS (Day 6) ---
    # Origins whose browser JavaScript is allowed to call this API — needed
    # because the frontend runs on a different origin (port/domain) than
    # this backend. See backend/main.py, where this feeds CORSMiddleware.
    CORS_ALLOWED_ORIGINS: list[str] = _get_list("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ORIGINS)

    # --- Database ---
    DATABASE_PATH: Path = PROJECT_ROOT / os.getenv("DATABASE_PATH", "database/olist.db")

    # --- CSV upload (Day 7) ---
    # Where converted uploads live: one subfolder per upload, named by a
    # server-generated id — never the user's filename (see backend/csv_upload.py).
    UPLOADS_DIR: Path = PROJECT_ROOT / os.getenv("UPLOADS_DIR", "uploads")
    # 5 MB default: generous for a single CSV table, small enough that the
    # whole file is read into memory during conversion without a second thought.
    CSV_UPLOAD_MAX_BYTES: int = _get_int("CSV_UPLOAD_MAX_BYTES", 5 * 1024 * 1024)

    # --- Safety limits ---
    MAX_ROWS: int = _get_int("MAX_ROWS", 200)
    QUERY_TIMEOUT_SECONDS: int = _get_int("QUERY_TIMEOUT_SECONDS", 10)


settings = Settings()
