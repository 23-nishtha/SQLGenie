"""Pydantic models: the shapes of the /ask request and response JSON."""
from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A natural-language question about the data.")


class AskResponse(BaseModel):
    question: str
    sql: str                       # the exact SQL that was executed (after validation/LIMIT)
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
