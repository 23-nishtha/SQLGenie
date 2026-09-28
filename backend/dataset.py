"""Day 6/7: which dataset is being queried, described for the frontend.

Why this exists
----------------
The frontend wants to know *what kind of data* it's showing — an e-commerce
database, a football database, an uploaded CSV — so it can switch to a
matching visual theme, and so /ask knows which SQLite file to query. This
module is the one place that answers "what datasets exist, and which one is
active right now."

The backend never sends CSS, colors, or any styling. `theme` below is just
an opaque KEY from a fixed list. The frontend owns a predefined
configuration for each key (colors, icons, fonts) and falls back to
"default" for a key it doesn't recognize. That keeps the boundary clean: the
backend says what the data IS, the frontend decides how it LOOKS.

Split of responsibilities (Day 7)
------------------------------------
  dataset.py (this file)   identity: which datasets exist, which is active,
                            public metadata (name/domain/theme), and —
                            internally only — where each one's .db file is.
  schema_profiles.py       per-dataset schema-RETRIEVAL knowledge (keywords,
                            foreign keys, notes) that helps the LLM write
                            correct SQL. A dataset's schema_profile_id here
                            just names which profile to use.
  schema_retrieval.py      the generic retrieval algorithm, which knows
                            nothing about any specific dataset.

Public vs. internal: two different shapes on purpose
--------------------------------------------------------
`DatasetProfile` is what the API returns — it never contains a filesystem
path. Where a dataset's .db file actually lives is tracked separately, in
`_DatasetRecord`, a plain (non-Pydantic, never-serialized) dataclass. Only
this module ever touches `_DatasetRecord`; everything else — db.py,
schema_retrieval.py, main.py — asks for a `database_path` or a
`schema_profile_id` through the small functions at the bottom of this file,
and never sees the record itself. That's what makes "don't expose
filesystem paths to the frontend" structurally true rather than a rule
someone has to remember to follow in every response model.

The "active dataset" is real, mutable, in-memory server state
------------------------------------------------------------------
Through Day 6, the dataset was fixed for the whole process (one .env
setting, read once at startup). Day 7 adds the ability to switch it at
runtime — POST /datasets/{id}/select — and to add new datasets at runtime —
POST /datasets/upload. Both just update the module-level state below. This
is deliberately ONE global pointer, not a per-session/per-user one: nothing
else in this codebase has a session or user concept, and adding one here
would be scope creep for what is a local, single-user tool.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from backend.config import settings

# The complete list of themes the backend may ever name. A typo in a profile
# fails validation at startup instead of silently shipping an unknown key to
# the frontend.
ThemeKey = Literal["commerce", "football", "finance", "entertainment", "property", "default"]

DatasetSource = Literal["built_in", "uploaded"]


class DatasetSummary(BaseModel):
    """The small identity block attached to every /ask response."""
    id: str
    name: str
    domain: str
    theme: ThemeKey


class DatasetProfile(DatasetSummary):
    """The full, public description returned by GET /dataset and /datasets.
    Deliberately has no field for where the data actually lives on disk —
    see the module docstring."""
    description: str
    source: DatasetSource = "built_in"
    # Ready-made questions for the frontend's empty state. For built-in
    # datasets these are curated; for uploads they're auto-generated
    # (see csv_upload.py) since there's no hand-written knowledge yet.
    example_questions: list[str] = []

    def summary(self) -> DatasetSummary:
        return DatasetSummary(id=self.id, name=self.name, domain=self.domain, theme=self.theme)


class DatasetListResponse(BaseModel):
    active: str
    datasets: list[DatasetProfile]


class UnknownDatasetError(ValueError):
    """A dataset id doesn't exist in the registry — either the active one
    (a misconfigured SQLGENIE_DATASET) or one asked for by id (an unknown
    id passed to /datasets/{id}/select)."""


@dataclass
class _DatasetRecord:
    """Internal pairing of a public profile with where its data lives and
    which schema_profiles.py entry describes it. Never serialized — main.py
    only ever hands out `.profile`."""
    profile: DatasetProfile
    database_path: Path
    schema_profile_id: str


# ---------------------------------------------------------------------------
# The built-in Olist profile. Unchanged content from Day 6, now carrying
# source="built_in" explicitly and no example_questions default needed
# (kept explicit below for clarity).
# ---------------------------------------------------------------------------

OLIST_PROFILE = DatasetProfile(
    id="olist",
    name="Olist Brazilian E-Commerce",
    domain="ecommerce",
    theme="commerce",
    source="built_in",
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


# ---------------------------------------------------------------------------
# Registry + active pointer. Not exported directly (leading underscore) —
# everything outside this file goes through the functions below.
# ---------------------------------------------------------------------------

_registry: dict[str, _DatasetRecord] = {}
_active_dataset_id: str = settings.SQLGENIE_DATASET


def register_dataset(
    profile: DatasetProfile, database_path: Path, schema_profile_id: str
) -> None:
    """Add a dataset to the registry. Used once at import time for Olist,
    and by the /datasets/upload endpoint for each new upload.

    Raises ValueError on an id collision rather than silently overwriting —
    a collision means either a bug (re-registering Olist) or, for uploads,
    a UUID collision astronomically unlikely enough that treating it as a
    hard error is the right call rather than quietly replacing someone
    else's dataset.
    """
    if profile.id in _registry:
        raise ValueError(f"Dataset id '{profile.id}' is already registered.")
    _registry[profile.id] = _DatasetRecord(
        profile=profile, database_path=database_path, schema_profile_id=schema_profile_id
    )


register_dataset(OLIST_PROFILE, settings.DATABASE_PATH, schema_profile_id="olist")


def _get_active_record() -> _DatasetRecord:
    record = _registry.get(_active_dataset_id)
    if record is None:
        available = ", ".join(sorted(_registry)) or "(none registered)"
        raise UnknownDatasetError(
            f"Unknown active dataset '{_active_dataset_id}'. Available datasets: {available}."
        )
    return record


def get_active_dataset() -> DatasetProfile:
    """The public profile of whichever dataset is currently active.

    backend/main.py calls this at startup (so a misconfigured .env fails
    immediately with a readable message) and on every /dataset and /ask
    request.
    """
    return _get_active_record().profile


def get_active_database_path() -> Path:
    """Internal-use accessor: where the active dataset's SQLite file lives.
    Used by backend/db.py as its default when no explicit path is given —
    never returned from an API response."""
    return _get_active_record().database_path


def get_active_schema_profile_id() -> str:
    """Internal-use accessor: which schema_profiles.py entry describes the
    active dataset's tables. Used by backend/schema_retrieval.py."""
    return _get_active_record().schema_profile_id


def set_active_dataset(dataset_id: str) -> DatasetProfile:
    """Make `dataset_id` the active dataset. Raises UnknownDatasetError if
    it isn't registered (main.py maps that to 404 — "no such dataset")."""
    global _active_dataset_id
    if dataset_id not in _registry:
        available = ", ".join(sorted(_registry)) or "(none registered)"
        raise UnknownDatasetError(
            f"Unknown dataset id '{dataset_id}'. Available datasets: {available}."
        )
    _active_dataset_id = dataset_id
    return _registry[dataset_id].profile


def list_datasets() -> list[DatasetProfile]:
    """Every registered dataset's public profile, built-ins first (Olist is
    registered first, at import time) then uploads in the order they were
    added — a plain dict preserves insertion order."""
    return [record.profile for record in _registry.values()]
