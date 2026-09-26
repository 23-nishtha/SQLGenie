"""Day 5: the self-correction loop that ties retrieval, generation,
validation and execution together with automatic error-driven retries.

Why a separate module
----------------------
backend/main.py's job is turning HTTP requests into responses; it shouldn't
also contain "try up to 3 times, remembering what failed each time" retry
logic. Putting that here instead means:
  - It can be unit tested directly — call answer_question(...), inspect the
    result — without spinning up a FastAPI test client for every case.
  - main.py stays a thin translation layer: call this, map the result (or
    exception) to an HTTP response.

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
"""
from dataclasses import dataclass, field

from backend import db, llm, schema_retrieval, sql_guard

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


def _validate_and_execute(raw_sql: str) -> tuple[str, list[str], list[dict]]:
    """The one and only path from "SQL text" to "rows out of SQLite" — used
    for the very first attempt AND every correction, so a corrected query
    can never skip validation. This function IS the answer to "how do
    guardrails stay enforced during correction": there's only one door in.
    """
    safe_sql = sql_guard.validate_and_prepare(raw_sql)  # may raise SqlValidationError
    columns, rows = db.run_query(safe_sql)               # may raise QueryExecutionError
    return safe_sql, columns, rows


def answer_question(question: str) -> PipelineResult:
    """Run the full pipeline for one question: retrieve schema, generate
    SQL, validate, execute — and if execution fails against the real
    database, ask the LLM to correct it, up to MAX_CORRECTION_ATTEMPTS
    times, before giving up.

    Retries only happen on QueryExecutionError (the SQL passed validation
    but SQLite rejected it). A SqlValidationError is NOT retried: it means
    the SQL was unsafe or malformed in a way a "please fix the database
    error" prompt can't address, and quietly asking the LLM to try again
    would just be hoping for a different unsafe answer. It's raised
    immediately instead, exactly as it was before Day 5.
    """
    schema_text = schema_retrieval.build_relevant_schema_text(question)
    attempts: list[AttemptRecord] = []

    raw_sql = llm.generate_sql(question, schema_text)

    # attempt_number: 0 = the initial try, 1..MAX = correction attempts.
    # A bounded `for` over a fixed `range` — not `while True`, not
    # recursion — so this loop provably cannot run more than
    # MAX_CORRECTION_ATTEMPTS + 1 times; there's no infinite-loop case to
    # reason about.
    for attempt_number in range(MAX_CORRECTION_ATTEMPTS + 1):
        try:
            safe_sql, columns, rows = _validate_and_execute(raw_sql)
        except db.QueryExecutionError as exc:
            attempts.append(AttemptRecord(sql=raw_sql, error=str(exc)))
            if attempt_number >= MAX_CORRECTION_ATTEMPTS:
                break  # out of retries — fall through to PipelineError below
            raw_sql = llm.correct_sql(
                question=question,
                schema_text=schema_text,
                failed_sql=raw_sql,
                db_error=str(exc),
                attempt_number=attempt_number + 1,
            )
            continue
        # Success: sql_guard.SqlValidationError is allowed to propagate
        # straight out of this function (see docstring above) — it is
        # deliberately NOT caught here.
        attempts.append(AttemptRecord(sql=safe_sql, error=None))
        return PipelineResult(
            question=question,
            sql=safe_sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            attempts=attempts,
        )

    raise PipelineError(attempts)
