"""The pipeline that answers one question, with error-driven self-correction
(Day 5) and a step-by-step trace of everything it did (Day 6).

Why a separate module
----------------------
backend/main.py's job is turning HTTP requests into responses; it shouldn't
also contain "try up to 3 times, remembering what failed each time" retry
logic. Putting that here instead means:
  - It can be unit tested directly — call the function, inspect the
    result — without spinning up a FastAPI test client for every case.
  - main.py stays a thin translation layer: call this, map the result (or
    error) to an HTTP response.

Two entry points, one implementation
--------------------------------------
  run_pipeline(question)    -> PipelineRun. Never raises for an expected
                               failure; the outcome (result OR error) and a
                               full trace come back together. main.py uses
                               this so even a failed request can show the
                               frontend exactly where it failed.
  answer_question(question) -> PipelineResult, or raises the original
                               exception. The Day 5 API, kept as a thin
                               wrapper over run_pipeline so existing callers
                               and tests behave exactly as before.

What "self-correction" means here
-----------------------------------
The LLM (mock or real) sometimes generates SQL that's syntactically fine —
it passes sql_guard.py's checks — but is still WRONG in a way only SQLite
itself can catch: a column that doesn't exist, a table name it invented,
and so on. Rather than immediately failing the user's question, we give the
LLM a second chance: here's the question you were answering, here's the SQL
you wrote, here's the EXACT error SQLite gave — please fix it. This is
sometimes called an "agentic" workflow because the system reacts to the
outcome of its own action instead of just running once and reporting
success or failure.

Security note (the most important part of this file)
-------------------------------------------------------
Every single SQL string this loop ever produces — the first attempt AND
every correction — passes through the exact same sql_guard.validate_and_prepare()
before it ever touches the database, via _validate_and_execute() below.
Self-correction only changes how new SQL gets PROPOSED; it never changes
how SQL gets APPROVED or EXECUTED. If a correction is itself invalid (e.g.
the LLM tries a DELETE), it fails validation immediately and is NOT
retried — see the comment in the loop below for why that's a deliberate
choice, not an oversight.

The Day 6 trace only OBSERVES. Recording a step never decides anything, and
_validate_and_execute() is still the only place SQL goes from text to rows.
"""
from dataclasses import dataclass, field

from backend import db, llm, schema_retrieval, sql_guard
from backend.mock_llm import MockQuestionNotFound
from backend.models import (
    DbExecutionData, ErrorType, QuestionData, ResultData, RetrievedTable,
    SchemaRetrievalData, SelfCorrectionData, SqlGenerationData, SqlGuardData,
    StepType, Trace,
)
from backend.trace import TraceRecorder, elapsed_ms, redact_secrets, start_timer

# Total tries = 1 initial attempt + this many corrections. A plain constant
# (not another .env setting) is deliberate: this is a small, fixed safety
# limit, not something you'd want an end user or a bad .env value changing.
MAX_CORRECTION_ATTEMPTS = 2


@dataclass
class AttemptRecord:
    """One try at answering the question. Kept so the caller (a test, or
    the API's error response) can see exactly what was tried and why it
    failed — never a Python traceback, just the SQL and SQLite's own
    one-line error message."""
    sql: str
    error: str | None = None  # None means this attempt succeeded


@dataclass
class PipelineResult:
    question: str
    sql: str  # the SQL that actually succeeded
    columns: list[str]
    rows: list[dict]
    row_count: int
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def corrected(self) -> bool:
        """True if the first attempt failed and a later one succeeded."""
        return len(self.attempts) > 1


class PipelineError(RuntimeError):
    """Raised when every attempt (the initial one plus all corrections)
    failed. str(exc) is a single clean, multi-line summary safe to show a
    user directly — it lists each attempt's SQLite error, never a raw
    Python traceback. `attempts` carries the full history for logging/tests.
    """
    def __init__(self, attempts: list[AttemptRecord]):
        lines = [f"  Attempt {i}: {a.error}" for i, a in enumerate(attempts, start=1)]
        message = (
            f"Could not produce a working SQL query after {len(attempts)} attempt(s):\n"
            + "\n".join(lines)
            + "\nTry rephrasing your question."
        )
        super().__init__(message)
        self.attempts = attempts


@dataclass
class PipelineRun:
    """The outcome of run_pipeline(): a trace plus EITHER a result OR an error."""
    question: str
    trace: Trace
    result: PipelineResult | None = None
    error: Exception | None = None       # the original exception, kept so answer_question can re-raise it
    error_type: ErrorType | None = None
    failed_step: StepType | None = None


def _validate_and_execute(
    raw_sql: str, recorder: TraceRecorder | None = None, attempt_number: int = 0
) -> tuple[str, list[str], list[dict]]:
    """The one and only path from "SQL text" to "rows out of SQLite" — used
    for the very first attempt AND every correction, so a corrected query
    can never skip validation. This function IS the answer to "how do
    guardrails stay enforced during correction": there's only one door in.

    The recorder (optional, purely for the trace) notes what happened at
    each of the two stages. It records the outcome and then lets the
    original exception continue on unchanged — it never swallows one.
    """
    recorder = recorder or TraceRecorder()

    started = start_timer()
    try:
        safe_sql = sql_guard.validate_and_prepare(raw_sql)  # may raise SqlValidationError
    except sql_guard.SqlValidationError as exc:
        recorder.add(
            "sql_guard", "failed", attempt_number,
            SqlGuardData(approved=False, original_sql=raw_sql, reason=str(exc)),
            elapsed_ms(started),
        )
        raise
    recorder.add(
        "sql_guard", "success", attempt_number,
        SqlGuardData(
            approved=True,
            original_sql=raw_sql,
            executed_sql=safe_sql,
            # The guard may wrap the query to enforce a row limit; compare
            # against the original (minus a trailing semicolon) to tell.
            sql_was_modified=safe_sql != raw_sql.strip().rstrip(";").strip(),
        ),
        elapsed_ms(started),
    )

    started = start_timer()
    try:
        columns, rows = db.run_query(safe_sql)               # may raise QueryExecutionError
    except db.QueryExecutionError as exc:
        recorder.add(
            "db_execution", "failed", attempt_number,
            DbExecutionData(executed_sql=safe_sql, error=str(exc)),
            elapsed_ms(started),
        )
        raise
    recorder.add(
        "db_execution", "success", attempt_number,
        DbExecutionData(executed_sql=safe_sql, row_count=len(rows), columns=columns),
        elapsed_ms(started),
    )
    return safe_sql, columns, rows


def _classify_error(exc: Exception, failed_step: StepType | None) -> ErrorType:
    if isinstance(exc, MockQuestionNotFound):
        return "unsupported_question"
    if isinstance(exc, sql_guard.SqlValidationError):
        return "sql_rejected"
    if isinstance(exc, PipelineError):
        return "execution_failed"
    if failed_step == "sql_generation":
        return "llm_error"
    return "internal_error"


def run_pipeline(question: str) -> PipelineRun:
    """Run the full pipeline for one question: retrieve schema, generate
    SQL, validate, execute — and if execution fails against the real
    database, ask the LLM to correct it, up to MAX_CORRECTION_ATTEMPTS
    times, before giving up. Returns the outcome AND a trace of every step.

    Retries only happen on QueryExecutionError (the SQL passed validation
    but SQLite rejected it). A SqlValidationError is NOT retried: it means
    the SQL was unsafe or malformed in a way a "please fix the database
    error" prompt can't address, and quietly asking the LLM to try again
    would just be hoping for a different unsafe answer. It ends the run
    immediately, exactly as it did before Day 6.
    """
    recorder = TraceRecorder()
    recorder.add("question", "success", 0, QuestionData(text=question), 0.0)

    def fail(exc: Exception, failed_step: StepType | None, attempts_made: int) -> PipelineRun:
        return PipelineRun(
            question=question,
            trace=recorder.build(
                attempts=attempts_made,
                max_correction_attempts=MAX_CORRECTION_ATTEMPTS,
                corrected=False,
            ),
            error=exc,
            error_type=_classify_error(exc, failed_step),
            failed_step=failed_step,
        )

    # --- Step: schema retrieval -------------------------------------------
    started = start_timer()
    try:
        retrieval = schema_retrieval.retrieve_with_details(question)
    except Exception as exc:
        recorder.add(
            "schema_retrieval", "failed", 0,
            SchemaRetrievalData(error=redact_secrets(str(exc))),
            elapsed_ms(started),
        )
        return fail(exc, "schema_retrieval", 0)
    schema_text = retrieval.schema_text
    recorder.add(
        "schema_retrieval", "success", 0,
        SchemaRetrievalData(
            tables=[
                RetrievedTable(name=t.name, kind=t.kind, selected_by=t.selected_by, score=t.score)
                for t in retrieval.tables
            ],
            relationships=list(retrieval.relationships),
            schema_chars=len(schema_text),
            full_schema_chars=retrieval.full_schema_chars,
        ),
        elapsed_ms(started),
    )

    # --- Step: first SQL generation ---------------------------------------
    started = start_timer()
    try:
        raw_sql = llm.generate_sql(question, schema_text)
    except Exception as exc:
        recorder.add(
            "sql_generation", "failed", 0,
            SqlGenerationData(is_correction=False, error=redact_secrets(str(exc))),
            elapsed_ms(started),
        )
        return fail(exc, "sql_generation", 0)
    recorder.add(
        "sql_generation", "success", 0,
        SqlGenerationData(sql=raw_sql, is_correction=False),
        elapsed_ms(started),
    )

    attempts: list[AttemptRecord] = []

    # attempt_number: 0 = the initial try, 1..MAX = correction attempts.
    # A bounded `for` over a fixed `range` — not `while True`, not
    # recursion — so this loop provably cannot run more than
    # MAX_CORRECTION_ATTEMPTS + 1 times; there's no infinite-loop case to
    # reason about.
    for attempt_number in range(MAX_CORRECTION_ATTEMPTS + 1):
        try:
            safe_sql, columns, rows = _validate_and_execute(raw_sql, recorder, attempt_number)
        except db.QueryExecutionError as exc:
            attempts.append(AttemptRecord(sql=raw_sql, error=str(exc)))
            if attempt_number >= MAX_CORRECTION_ATTEMPTS:
                break  # out of retries — fall through to PipelineError below

            correction_number = attempt_number + 1
            recorder.add(
                "self_correction", "success", correction_number,
                SelfCorrectionData(
                    correction_number=correction_number,
                    max_corrections=MAX_CORRECTION_ATTEMPTS,
                    failed_sql=raw_sql,
                    db_error=str(exc),  # the exact database error, unmodified
                ),
                0.0,
            )
            started = start_timer()
            try:
                raw_sql = llm.correct_sql(
                    question=question,
                    schema_text=schema_text,
                    failed_sql=raw_sql,
                    db_error=str(exc),
                    attempt_number=correction_number,
                )
            except Exception as correction_exc:
                recorder.add(
                    "sql_generation", "failed", correction_number,
                    SqlGenerationData(is_correction=True, error=redact_secrets(str(correction_exc))),
                    elapsed_ms(started),
                )
                return fail(correction_exc, "sql_generation", len(attempts))
            recorder.add(
                "sql_generation", "success", correction_number,
                SqlGenerationData(sql=raw_sql, is_correction=True),
                elapsed_ms(started),
            )
            continue
        except sql_guard.SqlValidationError as exc:
            # Deliberately NOT retried (see the docstring). The trace already
            # ends at the failed sql_guard step, and no db_execution step
            # exists for this attempt — the SQL never reached the database.
            return fail(exc, "sql_guard", attempt_number + 1)
        except Exception as exc:
            # Something unexpected (a bug, not a bad query). Still handled,
            # so a request never dies with a raw traceback.
            return fail(exc, None, attempt_number + 1)

        attempts.append(AttemptRecord(sql=safe_sql, error=None))
        recorder.add("result", "success", attempt_number, ResultData(row_count=len(rows)), 0.0)
        result = PipelineResult(
            question=question,
            sql=safe_sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            attempts=attempts,
        )
        return PipelineRun(
            question=question,
            trace=recorder.build(
                attempts=len(attempts),
                max_correction_attempts=MAX_CORRECTION_ATTEMPTS,
                corrected=result.corrected,
            ),
            result=result,
        )

    return fail(PipelineError(attempts), "db_execution", len(attempts))


def answer_question(question: str) -> PipelineResult:
    """The Day 5 interface: return the result, or raise the original
    exception (MockQuestionNotFound, SqlValidationError, PipelineError, or
    whatever the LLM call raised). Kept unchanged in behavior; it just runs
    run_pipeline() and unpacks the outcome."""
    run = run_pipeline(question)
    if run.error is not None:
        raise run.error
    return run.result
