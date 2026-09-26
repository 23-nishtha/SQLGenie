"""FastAPI app: the /ask endpoint that ties the whole pipeline together.

Pipeline for POST /ask (Day 5 wraps generation+validation+execution in an
error-driven self-correction loop — see backend/agent.py for the retry
logic itself; this file just calls it and maps the outcome to HTTP):
  question (from the request)
    -> agent.answer_question()
         -> schema_retrieval.build_relevant_schema_text()  retrieve schema
         -> llm.generate_sql()                             mock or OpenAI
         -> sql_guard.validate_and_prepare()                same guard, every attempt
         -> db.run_query()                                  read-only SQLite
         -> on failure: llm.correct_sql(), then retry the same validate+execute,
            up to agent.MAX_CORRECTION_ATTEMPTS times
    -> AskResponse       question, sql, columns, rows, attempts, corrected

Run it with:  uvicorn backend.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException

from backend import agent, schema_retrieval, sql_guard
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
    # backend/agent.py runs the whole pipeline: retrieval -> generate SQL ->
    # validate -> execute, and on a database error, asks the LLM to correct
    # the SQL and tries again (up to agent.MAX_CORRECTION_ATTEMPTS times).
    # Every attempt, including every correction, still goes through the
    # exact same sql_guard.validate_and_prepare() — see agent.py's docstring.
    try:
        result = agent.answer_question(request.question)
    except MockQuestionNotFound as exc:
        # The user's fault (an unsupported question in mock mode), not a
        # server/API failure, so this gets a 400 rather than a 502.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except sql_guard.SqlValidationError as exc:
        # A guard rejection is never retried (see agent.py) — surfaced
        # immediately, same behavior as before Day 5.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except agent.PipelineError as exc:
        # Every attempt (initial + all corrections) failed. str(exc) is
        # agent.py's own clean, multi-line summary — never a raw traceback.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        # Covers missing API key, network errors, and other OpenAI API
        # errors — from either the initial generation or a correction call.
        raise HTTPException(status_code=502, detail=f"Could not generate SQL: {exc}") from exc

    return AskResponse(
        question=result.question,
        sql=result.sql,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        attempts=len(result.attempts),
        corrected=result.corrected,
    )
