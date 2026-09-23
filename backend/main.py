"""FastAPI app: the /ask endpoint that ties the whole pipeline together.

Pipeline for POST /ask:
  question (from the request)
    -> llm.generate_sql()        turn it into SQL using OpenAI
    -> sql_guard.validate_and_prepare()   reject anything unsafe
    -> db.run_query()            execute against the read-only SQLite DB
    -> AskResponse               return question, sql, columns, rows

Run it with:  uvicorn backend.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException

from backend import db, llm, sql_guard
from backend.config import settings
from backend.mock_llm import MockQuestionNotFound
from backend.models import AskRequest, AskResponse
from backend.schema_context import build_schema_text

app = FastAPI(
    title="SQLGenie API",
    description="Natural language -> SQL -> SQLite results, for the Olist dataset.",
    version="0.1.0",
)

# The schema doesn't change while the server runs, so we read it once here
# instead of hitting the database on every request.
_schema_text_cache: str | None = None


@app.on_event("startup")
def load_schema_on_startup() -> None:
    global _schema_text_cache
    _schema_text_cache = build_schema_text()


@app.get("/health")
def health_check():
    """Quick check that the API is up and the database file is reachable."""
    return {
        "status": "ok",
        "database_path": str(settings.DATABASE_PATH),
        "database_found": settings.DATABASE_PATH.exists(),
        "llm_mode": settings.SQLGENIE_LLM_MODE,
        "model": settings.OPENAI_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    schema_text = _schema_text_cache or build_schema_text()

    # Step 1: natural language -> SQL (mock or OpenAI, see llm.py)
    try:
        raw_sql = llm.generate_sql(request.question, schema_text)
    except MockQuestionNotFound as exc:
        # The user's fault (an unsupported question in mock mode), not a
        # server/API failure, so this gets a 400 rather than a 502.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Covers missing API key, network errors, and OpenAI API errors alike.
        raise HTTPException(status_code=502, detail=f"Could not generate SQL: {exc}") from exc

    # Step 2: validate the SQL before it goes anywhere near the database
    try:
        safe_sql = sql_guard.validate_and_prepare(raw_sql)
    except sql_guard.SqlValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Step 3: execute against the read-only database
    try:
        columns, rows = db.run_query(safe_sql)
    except db.QueryExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(
        question=request.question,
        sql=safe_sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
    )
