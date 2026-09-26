"""FastAPI app: the /ask endpoint that ties the whole pipeline together.

Pipeline for POST /ask (Day 5 wraps generation+validation+execution in an
error-driven self-correction loop; Day 6 records a step-by-step trace of
it — see backend/agent.py for the pipeline itself; this file just calls it
and maps the outcome to HTTP):
  question (from the request)
    -> agent.run_pipeline()
         -> schema_retrieval.retrieve_with_details()       retrieve schema
         -> llm.generate_sql()                             mock or OpenAI
         -> sql_guard.validate_and_prepare()                same guard, every attempt
         -> db.run_query()                                  read-only SQLite
         -> on failure: llm.correct_sql(), then retry the same validate+execute,
            up to agent.MAX_CORRECTION_ATTEMPTS times
    -> AskResponse (or ErrorResponse)   result + dataset + llm + trace

Run it with:  uvicorn backend.main:app --reload
Then open:    http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend import agent, dataset, schema_retrieval
from backend.config import settings
from backend.dataset import DatasetProfile
from backend.models import (
    AskRequest, AskResponse, ErrorInfo, ErrorResponse, ErrorType, LlmInfo,
)
from backend.trace import redact_secrets

app = FastAPI(
    title="SQLGenie API",
    description="Natural language -> SQL -> SQLite results, for the Olist dataset.",
    version="0.1.0",
)

# CORS: browsers block a web page from calling an API on a different origin
# (scheme + host + port) unless that API says it's allowed. The frontend runs
# separately from this backend, so without this its requests fail in the
# browser even though the server answered. The allowed origins come from
# .env (CORS_ALLOWED_ORIGINS) — never "*" by default.
#   allow_credentials=False: this API uses no cookies or login, so there is
#     nothing to protect with credentials, and leaving it off is the safe
#     choice (browsers refuse "*" combined with credentials anyway).
#   allow_methods: only what the API actually offers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
def load_schema_on_startup() -> None:
    # Reads the DB schema once and caches it inside schema_retrieval, so
    # each /ask request only does in-memory keyword scoring, not a fresh
    # SQLite read. See backend/schema_retrieval.py for the retrieval logic.
    schema_retrieval.init_documents()
    # Fail fast, with a readable message, if SQLGENIE_DATASET is misspelled.
    dataset.get_active_dataset()


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


@app.get("/dataset", response_model=DatasetProfile)
def get_dataset() -> DatasetProfile:
    """Which dataset is being queried: its name, domain, a `theme` key the
    frontend maps to its own predefined theme (never CSS), and example
    questions to start from."""
    return dataset.get_active_dataset()


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


# How each failure category maps to an HTTP status. These are the same codes
# /ask has used since Day 2 (400 unsupported mock question, 422 guard
# rejection, 500 exhausted retries, 502 LLM failure).
_ERROR_STATUS: dict[ErrorType, int] = {
    "unsupported_question": 400,
    "sql_rejected": 422,
    "execution_failed": 500,
    "llm_error": 502,
    "internal_error": 502,
}


def _llm_info() -> LlmInfo:
    mode = settings.SQLGENIE_LLM_MODE
    return LlmInfo(mode=mode, model=None if mode == "mock" else settings.OPENAI_MODEL)


def _error_message(run: agent.PipelineRun) -> str:
    message = str(run.error)
    if run.error_type in ("llm_error", "internal_error"):
        message = f"Could not generate SQL: {message}"
    # Text from an LLM provider can echo part of a credential back; never
    # forward that to a browser.
    return redact_secrets(message)


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported question (mock mode only)."},
        422: {
            "model": ErrorResponse,
            "description": (
                "The SQL guard rejected the generated SQL. (A malformed request "
                "body, such as an empty question, also returns 422, in FastAPI's "
                "standard validation shape.)"
            ),
        },
        500: {"model": ErrorResponse, "description": "Every attempt, including self-corrections, failed."},
        502: {"model": ErrorResponse, "description": "The LLM call failed (missing key, network, quota...)."},
    },
)
def ask(request: AskRequest):
    # backend/agent.py runs the whole pipeline: retrieval -> generate SQL ->
    # validate -> execute, and on a database error, asks the LLM to correct
    # the SQL and tries again (up to agent.MAX_CORRECTION_ATTEMPTS times).
    # Every attempt, including every correction, still goes through the
    # exact same sql_guard.validate_and_prepare() — see agent.py's docstring.
    run = agent.run_pipeline(request.question)
    dataset_summary = dataset.get_active_dataset().summary()

    if run.result is not None:
        result = run.result
        return AskResponse(
            question=result.question,
            sql=result.sql,
            columns=result.columns,
            rows=result.rows,
            row_count=result.row_count,
            attempts=len(result.attempts),
            corrected=result.corrected,
            dataset=dataset_summary,
            llm=_llm_info(),
            trace=run.trace,
        )

    # Failure: same HTTP status and same plain-English `detail` string as
    # before Day 6, plus a structured `error` and the trace so a UI can show
    # exactly which step failed. str(exc) is agent.py's own clean message —
    # never a raw traceback.
    message = _error_message(run)
    body = ErrorResponse(
        detail=message,
        error=ErrorInfo(type=run.error_type, failed_step=run.failed_step, message=message),
        dataset=dataset_summary,
        llm=_llm_info(),
        trace=run.trace,
    )
    return JSONResponse(
        status_code=_ERROR_STATUS[run.error_type],
        content=body.model_dump(mode="json"),
    )
