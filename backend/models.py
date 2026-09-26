"""Pydantic models: the shapes of the /ask request and response JSON.

Day 6 adds a "trace": an ordered list of the steps the pipeline actually
ran, so a frontend can draw the journey of one question:

    question -> schema retrieval -> SQL generation -> SQL guard ->
    database execution -> result

and, when self-correction happens, the extra steps in between. Each step
type has its own small typed `data` model (rather than a free-form dict) so
that /openapi.json describes every shape exactly — a frontend, or a tool
generating frontend code, can read the types instead of guessing.
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from backend.dataset import DatasetSummary


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A natural-language question about the data.")


# ---------------------------------------------------------------------------
# Vocabulary shared by steps and errors.
# ---------------------------------------------------------------------------

StepType = Literal[
    "question", "schema_retrieval", "sql_generation", "sql_guard",
    "db_execution", "self_correction", "result",
]
StepStatus = Literal["success", "failed"]

# Machine-readable failure categories (the HTTP status is separate):
#   unsupported_question -> 400  mock mode doesn't know this question
#   sql_rejected         -> 422  the SQL guard refused the SQL
#   execution_failed     -> 500  every attempt (initial + corrections) failed in SQLite
#   llm_error            -> 502  the LLM call itself failed (missing key, network, quota...)
#   internal_error       -> 502  anything unexpected outside the LLM call
ErrorType = Literal[
    "unsupported_question", "sql_rejected", "execution_failed", "llm_error", "internal_error",
]


# ---------------------------------------------------------------------------
# Per-step `data` payloads: what each kind of step reports about itself.
# ---------------------------------------------------------------------------

class QuestionData(BaseModel):
    text: str


class RetrievedTable(BaseModel):
    name: str
    kind: Literal["table", "view"]
    # Why this table was included: it scored well against the question, it
    # is a direct foreign-key neighbor of one that did, or nothing matched
    # at all and the hub table `orders` was used as the fallback.
    selected_by: Literal["score", "fk_expansion", "fallback"]
    score: int


class SchemaRetrievalData(BaseModel):
    tables: list[RetrievedTable] = []
    relationships: list[str] = []
    schema_chars: int = 0        # size of the schema text actually sent to the LLM
    full_schema_chars: int = 0   # size the whole schema would have been (the Day 2 behavior)
    error: str | None = None


class SqlGenerationData(BaseModel):
    sql: str | None = None       # None only when generation itself failed
    is_correction: bool = False  # True when this SQL is a self-correction, not the first try
    error: str | None = None


class SqlGuardData(BaseModel):
    approved: bool
    original_sql: str            # exactly what the LLM produced
    executed_sql: str | None = None  # what will actually run (may add a LIMIT wrapper)
    sql_was_modified: bool = False
    reason: str | None = None    # why it was rejected, when approved is False


class DbExecutionData(BaseModel):
    executed_sql: str
    row_count: int | None = None
    columns: list[str] | None = None
    error: str | None = None


class SelfCorrectionData(BaseModel):
    correction_number: int       # 1 = first correction attempt
    max_corrections: int
    failed_sql: str
    db_error: str                # the exact database error that triggered the retry


class ResultData(BaseModel):
    row_count: int


# ---------------------------------------------------------------------------
# Steps: a common header (index/status/attempt/timing) plus typed `data`.
# `attempt` is 0 for the first try and 1..N for self-correction attempts.
# The literal `type` field is what lets the API tell step shapes apart
# (a "discriminated union"), and what a frontend switches on.
# ---------------------------------------------------------------------------

class _StepHeader(BaseModel):
    index: int
    status: StepStatus
    attempt: int
    duration_ms: float


class QuestionStep(_StepHeader):
    type: Literal["question"] = "question"
    data: QuestionData


class SchemaRetrievalStep(_StepHeader):
    type: Literal["schema_retrieval"] = "schema_retrieval"
    data: SchemaRetrievalData


class SqlGenerationStep(_StepHeader):
    type: Literal["sql_generation"] = "sql_generation"
    data: SqlGenerationData


class SqlGuardStep(_StepHeader):
    type: Literal["sql_guard"] = "sql_guard"
    data: SqlGuardData


class DbExecutionStep(_StepHeader):
    type: Literal["db_execution"] = "db_execution"
    data: DbExecutionData


class SelfCorrectionStep(_StepHeader):
    type: Literal["self_correction"] = "self_correction"
    data: SelfCorrectionData


class ResultStep(_StepHeader):
    type: Literal["result"] = "result"
    data: ResultData


PipelineStep = Annotated[
    Union[
        QuestionStep, SchemaRetrievalStep, SqlGenerationStep, SqlGuardStep,
        DbExecutionStep, SelfCorrectionStep, ResultStep,
    ],
    Field(discriminator="type"),
]


class TraceSummary(BaseModel):
    attempts: int                    # how many SQL candidates were submitted for validation/execution
    max_correction_attempts: int
    corrected: bool                  # True if an earlier attempt failed and a later one succeeded
    total_duration_ms: float


class Trace(BaseModel):
    """Only steps that actually ran appear here; on a failure the list ends
    at the failing step."""
    summary: TraceSummary
    steps: list[PipelineStep]


class LlmInfo(BaseModel):
    mode: str                        # "mock" or "openai"
    model: str | None = None         # None in mock mode (no model is called)


# ---------------------------------------------------------------------------
# Responses.
# ---------------------------------------------------------------------------

class AskResponse(BaseModel):
    status: Literal["success"] = "success"
    question: str
    sql: str                       # the exact SQL that was executed (after validation/LIMIT)
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    attempts: int = 1              # Day 5: how many tries it took (1 = no correction needed)
    corrected: bool = False        # Day 5: True if an earlier attempt failed and a later one fixed it
    dataset: DatasetSummary        # Day 6: which dataset answered (drives the frontend theme)
    llm: LlmInfo                   # Day 6: mock or real model
    trace: Trace                   # Day 6: the step-by-step pipeline journey


class ErrorInfo(BaseModel):
    type: ErrorType
    failed_step: StepType | None = None
    message: str


class ErrorResponse(BaseModel):
    """Body of every /ask failure. `detail` is the same plain-English string
    the API has always returned; the other fields add structure for a UI."""
    status: Literal["error"] = "error"
    detail: str
    error: ErrorInfo
    dataset: DatasetSummary
    llm: LlmInfo
    trace: Trace
