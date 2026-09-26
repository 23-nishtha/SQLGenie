"""FastAPI app: the /ask endpoint that ties the whole pipeline together.

Pipeline for POST /ask (Day 3 adds the first step):
  question (from the request)
    -> schema_retrieval.build_relevant_schema_text()  retrieve only the
                                                        relevant tables/joins
    -> llm.generate_sql()        turn it into SQL (mock or OpenAI)
    -> sql_guard.validate_and_prepare()   reject anything unsafe
    -> db.run_query()            execute against the read-only SQLite DB
    -> AskResponse               return question, sql, columns, rows

Run it with:  uvicorn backend.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException

from backend import db, llm, schema_retrieval, sql_guard
from backend.config import settings
from backend.mock_llm import MockQuestionNotFound
from backend.models import AskRequest, AskResponse

app = FastAPI(
    title="SQLGenie API",
    description="Natural language -> SQL -> SQLite results, for the Olist dataset.",
    version="0.1.0",
)


@app.on_event("startup")
def load_schema_on_startup() -> None:
    # Reads the DB schema once and caches it inside schema_retrieval, so
    # each /ask request only does in-memory keyword scoring, not a fresh
    # SQLite read. See backend/schema_retrieval.py for the retrieval logic.
    schema_retrieval.init_documents()


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


@app.get("/debug/retrieve")
def debug_retrieve(question: str):
    """DEVELOPMENT/DEBUGGING ENDPOINT — not part of the normal user flow.

    Shows exactly what backend/schema_retrieval.py would send the LLM for a
    given question, WITHOUT calling OpenAI (no network call, no API key
    needed, works in either SQLGENIE_LLM_MODE). Handy for seeing/tuning
    which tables get retrieved for a question while building or debugging
    Day 3's retrieval logic.

    Try it at: /debug/retrieve?question=What is the average review score?
    """
    selected_tables = schema_retrieval.retrieve_relevant_tables(question)
    schema_text = schema_retrieval.build_relevant_schema_text(question)
    return {
        "question": question,
        "retrieved_tables": selected_tables,
        "schema_text_sent_to_llm": schema_text,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    # Step 1 (Day 3): retrieve only the schema relevant to this question,
    # instead of sending the LLM the entire database every time.
    schema_text = schema_retrieval.build_relevant_schema_text(request.question)

    # Step 2: natural language -> SQL (mock or OpenAI, see llm.py)
    try:
        raw_sql = llm.generate_sql(request.question, schema_text)
    except MockQuestionNotFound as exc:
        # The user's fault (an unsupported question in mock mode), not a
        # server/API failure, so this gets a 400 rather than a 502.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Covers missing API key, network errors, and OpenAI API errors alike.
        raise HTTPException(status_code=502, detail=f"Could not generate SQL: {exc}") from exc

    # Step 3: validate the SQL before it goes anywhere near the database
    try:
        safe_sql = sql_guard.validate_and_prepare(raw_sql)
    except sql_guard.SqlValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Step 4: execute against the read-only database
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
