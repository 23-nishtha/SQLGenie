"""Day 6: which dataset is being queried, described for the frontend.

Why this exists
----------------
The frontend wants to know *what kind of data* it's showing — an e-commerce
database, a football database, a finance database — so it can switch to a
matching visual theme. This module is the one place that answers that
question.

The backend never sends CSS, colors, or any styling. `theme` below is just
an opaque KEY from a fixed list. The frontend owns a predefined
configuration for each key (colors, icons, fonts) and falls back to
"default" for a key it doesn't recognize. That keeps the boundary clean: the
backend says what the data IS, the frontend decides how it LOOKS.

`domain` and `theme` are separate on purpose. `domain` is a stable
description of the data ("ecommerce"); `theme` is a presentation hint that
could differ from it later (for example, two domains sharing one theme).

What this does NOT do (yet)
-----------------------------
This exposes a dataset's IDENTITY only. Other parts of the backend are still
written specifically for Olist: the retrieval metadata in
schema_retrieval.py, the notes in schema_context.py, the view hint in the
LLM prompt in llm.py, and the canned mock answers. Adding a football
database later means more than adding an entry here.
"""
from typing import Literal

from pydantic import BaseModel

from backend.config import settings

# The complete list of themes the backend may ever name. A typo in a profile
# fails validation at startup instead of silently shipping an unknown key to
# the frontend.
ThemeKey = Literal["commerce", "football", "finance", "entertainment", "property", "default"]


class DatasetSummary(BaseModel):
    """The small identity block attached to every /ask response."""
    id: str
    name: str
    domain: str
    theme: ThemeKey


class DatasetProfile(DatasetSummary):
    """The full description returned by GET /dataset."""
    description: str
    # Ready-made questions for the frontend's empty state. tests/ checks that
    # every one of these works in mock mode, so a demo never starts on a
    # question the mock can't answer.
    example_questions: list[str]

    def summary(self) -> DatasetSummary:
        return DatasetSummary(id=self.id, name=self.name, domain=self.domain, theme=self.theme)


OLIST_PROFILE = DatasetProfile(
    id="olist",
    name="Olist Brazilian E-Commerce",
    domain="ecommerce",
    theme="commerce",
    description=(
        "Public marketplace data from Olist, a Brazilian e-commerce platform: "
        "customers, orders, order items, payments, reviews, products and sellers."
    ),
    example_questions=[
        "How many orders were delivered?",
        "How many orders are there?",
        "What are the top 5 product categories by revenue?",
        "What is the average order value?",
        "How many customers are there?",
        "What is the total revenue?",
        "How many orders were placed in 2017?",
    ],
)

# Add a new dataset by adding an entry here (and pointing SQLGENIE_DATASET at it).
DATASET_PROFILES: dict[str, DatasetProfile] = {
    OLIST_PROFILE.id: OLIST_PROFILE,
}


class UnknownDatasetError(ValueError):
    """SQLGENIE_DATASET names a dataset that has no profile."""


def get_active_dataset() -> DatasetProfile:
    """Return the profile selected by SQLGENIE_DATASET, or raise a clear error.

    backend/main.py calls this at startup, so a misconfigured .env fails
    immediately with a readable message instead of on the first request.
    """
    profile = DATASET_PROFILES.get(settings.SQLGENIE_DATASET)
    if profile is None:
        available = ", ".join(sorted(DATASET_PROFILES))
        raise UnknownDatasetError(
            f"Unknown SQLGENIE_DATASET '{settings.SQLGENIE_DATASET}'. "
            f"Available datasets: {available}."
        )
    return profile
