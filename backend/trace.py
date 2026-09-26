"""Day 6: records what the pipeline does, step by step, as a typed trace.

backend/agent.py calls TraceRecorder.add() every time a stage finishes
(retrieval, generation, guard, execution, ...), then build() turns the
recorded steps into the `Trace` that goes back to the frontend. Keeping the
bookkeeping here means agent.py reads as the pipeline itself, not a pile of
timing and formatting code.

Nothing here makes a decision or touches the database: it only observes.
Recording a step can never change whether SQL is validated or executed.
"""
import re
import time

from backend.config import settings
from backend.models import (
    DbExecutionStep, QuestionStep, ResultStep, SchemaRetrievalStep,
    SelfCorrectionStep, SqlGenerationStep, SqlGuardStep, StepStatus, StepType,
    Trace, TraceSummary,
)

_STEP_CLASSES = {
    "question": QuestionStep,
    "schema_retrieval": SchemaRetrievalStep,
    "sql_generation": SqlGenerationStep,
    "sql_guard": SqlGuardStep,
    "db_execution": DbExecutionStep,
    "self_correction": SelfCorrectionStep,
    "result": ResultStep,
}


def start_timer() -> float:
    return time.perf_counter()


def elapsed_ms(started: float) -> float:
    """Milliseconds since `started` (a value from start_timer())."""
    return round((time.perf_counter() - started) * 1000, 3)


# Error text from an LLM provider can echo back part of a credential, and
# these messages are shown to the frontend. So anything that came from
# outside our own code is scrubbed first.
_API_KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def redact_secrets(text: str) -> str:
    """Replace the configured API key, and anything shaped like an OpenAI
    key ("sk-..."), with a placeholder."""
    key = settings.OPENAI_API_KEY
    if key and len(key) >= 8:
        text = text.replace(key, "[redacted]")
    return _API_KEY_LIKE.sub("[redacted]", text)


class TraceRecorder:
    def __init__(self) -> None:
        self._steps: list = []
        self._started = start_timer()

    def add(
        self,
        step_type: StepType,
        status: StepStatus,
        attempt: int,
        data,
        duration_ms: float,
    ) -> None:
        step_class = _STEP_CLASSES[step_type]
        self._steps.append(
            step_class(
                index=len(self._steps),
                status=status,
                attempt=attempt,
                duration_ms=duration_ms,
                data=data,
            )
        )

    def build(self, *, attempts: int, max_correction_attempts: int, corrected: bool) -> Trace:
        return Trace(
            summary=TraceSummary(
                attempts=attempts,
                max_correction_attempts=max_correction_attempts,
                corrected=corrected,
                total_duration_ms=elapsed_ms(self._started),
            ),
            steps=list(self._steps),
        )
