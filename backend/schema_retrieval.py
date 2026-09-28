"""Day 3: lightweight, explainable schema-aware retrieval ("RAG" for SQLGenie).
Day 7: generalized to work against ANY dataset's schema, not just Olist's.

What "RAG" means in this project
---------------------------------
RAG = Retrieval-Augmented Generation: instead of handing a language model
everything it might ever need, you first RETRIEVE the small slice of
information relevant to the current request, then GENERATE using only that
slice. Usually the retrieved thing is documents from a knowledge base; here
it's schema information — which tables, columns and joins are relevant to
the question someone just asked.

Why this matters here
----------------------
Through Day 2, `backend/schema_context.build_schema_text()` sent the ENTIRE
database (every table, every column, every view) to the LLM on every single
question. That's harmless for a small schema, but it's still wasted tokens
(cost + latency) and it gives the model irrelevant columns it could get
confused by. This module instead:

  1. Represents each table/view as a "schema document" — a short
     description, some hand-picked concept keywords (e.g. "delivered",
     "shipped" for `orders`), and its real column list.
  2. Scores every document against the question using simple keyword
     overlap. No embeddings, no vector database, no extra services —
     just Python string matching, which is easy to read and debug.
  3. Picks the best-scoring documents, then walks the known foreign-key
     graph to pull in any directly-joined table that wasn't already picked
     (so the LLM always has what it needs to write a JOIN).
  4. Renders only that subset — plus a couple of universally-important
     notes — as the schema text handed to the LLM.

This is intentionally a lookup-and-score system, not a real search engine.
For a schema this small, hand-curated keywords are more explainable (and
easier to fix when retrieval picks the "wrong" table) than an
embeddings-based approach, and they need zero extra infrastructure.

Day 7: which dataset?
----------------------
This module has NO dataset-specific knowledge of its own. Descriptions,
keywords and foreign keys come from backend/schema_profiles.py, looked up by
`dataset_id`; column lists come live from whichever SQLite file `db_path`
points at. Both parameters default to whatever's currently active (see
backend/dataset.py) — so every function below still works exactly as it did
through Day 6 when called with no arguments at all, but now also works for
an uploaded CSV's single-table database, or a future football/movies
dataset, without any change to the algorithm itself.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

from backend import dataset
from backend.schema_context import build_schema_text, get_table_columns
from backend.schema_profiles import Relationship, SchemaProfile, get_schema_profile

# ---------------------------------------------------------------------------
# Schema documents: one entry per table/view, merging schema_profiles.py's
# hand-written knowledge (if any exists for this dataset) with the live
# column list read from SQLite.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaDocument:
    name: str                    # table/view name, e.g. "orders"
    kind: str                    # "table" or "view"
    description: str             # one or two plain-English sentences
    keywords: frozenset[str]     # extra concept words that should match this table
    columns: tuple[str, ...] = field(default_factory=tuple)  # filled in at load time

    def as_text(self) -> str:
        label = "TABLE" if self.kind == "table" else "VIEW"
        cols = ", ".join(self.columns)
        return f"{label} {self.name}({cols})\n  -- {self.description}"


# Words too common/generic to help scoring ("how", "many", "the", ...).
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "in", "on", "for", "to", "by", "with", "from", "as", "at",
    "and", "or", "each",
    "what", "which", "who", "whom", "how", "many", "much", "do", "does",
    "did", "please", "show", "me", "list", "find", "get", "give",
    "this", "that", "these", "those", "there", "have", "has", "had",
})

DEFAULT_TOP_K = 4          # how many documents scoring wins before FK expansion
MAX_DOCUMENTS = 6          # hard cap after expansion, so context stays small

# Both caches are keyed by (dataset_id, str(db_path)) — the exact pair a
# request resolved to — rather than a single global. That's what makes it
# safe for two different datasets' schemas to be cached at once (e.g. after
# switching the active dataset, or in a test that passes an explicit
# db_path): each combination gets its own entry, so switching datasets can
# never accidentally serve another dataset's cached documents.
_documents_cache: dict[tuple[str, str], dict[str, SchemaDocument]] = {}
_full_schema_chars_cache: dict[tuple[str, str], int] = {}


def _resolve_dataset_id(dataset_id: str | None) -> str:
    return dataset_id if dataset_id is not None else dataset.get_active_dataset().id


def _resolve_db_path(db_path) -> Path:
    return Path(db_path) if db_path is not None else dataset.get_active_database_path()


def _load_documents(dataset_id: str | None = None, db_path=None) -> dict[str, SchemaDocument]:
    """Build {name: SchemaDocument}, merging schema_profiles.py's knowledge
    (if this dataset has a curated profile) with the live column list from
    SQLite (via schema_context.get_table_columns)."""
    resolved_id = _resolve_dataset_id(dataset_id)
    resolved_path = _resolve_db_path(db_path)
    cache_key = (resolved_id, str(resolved_path))
    if cache_key in _documents_cache:
        return _documents_cache[cache_key]

    profile: SchemaProfile = get_schema_profile(resolved_id)
    tables = get_table_columns(resolved_path)
    documents: dict[str, SchemaDocument] = {}
    for name, info in tables.items():
        description, keywords = profile.metadata.get(
            name, ("(no description available)", frozenset())
        )
        documents[name] = SchemaDocument(
            name=name,
            kind=info["kind"],
            description=description,
            keywords=keywords,
            columns=tuple(info["columns"]),
        )

    _documents_cache[cache_key] = documents
    return documents


def init_documents(dataset_id: str | None = None, db_path=None) -> None:
    """Warm the cache for one dataset (the active one by default) — called
    at startup and after selecting/uploading a dataset (see backend/main.py)
    — so a request only pays for scoring (pure Python), not a fresh SQLite
    read. Clears everything cached first: cheap, and correct if a dataset's
    underlying .db file was ever rebuilt out from under a running server.
    """
    _documents_cache.clear()
    _full_schema_chars_cache.clear()
    _load_documents(dataset_id, db_path)
    _full_schema_chars(dataset_id, db_path)


def _full_schema_chars(dataset_id: str | None = None, db_path=None) -> int:
    """Length of the FULL schema text (every table, unfiltered) for one
    dataset — used only so the frontend can show how much smaller the
    retrieved schema is."""
    resolved_id = _resolve_dataset_id(dataset_id)
    resolved_path = _resolve_db_path(db_path)
    cache_key = (resolved_id, str(resolved_path))
    if cache_key in _full_schema_chars_cache:
        return _full_schema_chars_cache[cache_key]
    size = len(build_schema_text(resolved_path))
    _full_schema_chars_cache[cache_key] = size
    return size


def _tokenize(question: str) -> list[str]:
    text = question.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # drop punctuation
    return [word for word in text.split() if word and word not in _STOPWORDS]


def _score_document(tokens: list[str], doc: SchemaDocument) -> int:
    """Simple weighted keyword overlap. Higher weight = stronger signal:
    the table's own name matching is the strongest hint of all, then a
    curated keyword, then a column name.

    Deliberately NOT scored against: the free-text `description`. An
    earlier version did, and a purely incidental word there (e.g.
    "average latitude/longitude" in a description matching a question about
    "average review score") was enough to pull in that table — and then its
    foreign-key neighbors too — for a question that had nothing to do with
    it. Descriptions are for humans reading the rendered schema text, not a
    retrieval signal: only the deliberately curated `keywords` set should
    trigger a match beyond the exact table/column names.

    This also means a dataset with NO curated profile (every upload, until
    someone writes it one) still gets usable retrieval: table-name and
    column-name matches don't depend on schema_profiles.py at all.
    """
    name_words = set(doc.name.split("_"))
    column_words = {
        word
        for column in doc.columns
        for word in re.split(r"[_\s]+", column.lower())
    }

    score = 0
    for token in tokens:
        if token == doc.name or token in name_words:
            score += 5
        if token in doc.keywords:
            score += 3
        if token in column_words:
            score += 2
    return score


def _expand_with_join_neighbors(
    selected: list[str], relationships: tuple[Relationship, ...], cap: int
) -> list[str]:
    """Add any table directly connected (by FK) to an ORIGINALLY selected
    table, so the LLM can always see how to join what it was given.

    Strictly ONE hop: `original` is frozen before the loop starts, and every
    membership check that decides whether to add a new table is against
    `original`, never against `result`. That matters — an earlier version
    checked against the growing `result` list instead, so a table added
    *during* expansion could itself trigger adding a further table in the
    same pass. Freezing `original` means only tables directly joined to a
    table the SCORER picked can be added; a table that got in purely
    through expansion is never itself used as a new jumping-off point.

    A dataset with no relationships defined (the common case for an
    uploaded single-table CSV) just does nothing here — `result` comes back
    unchanged, which is correct: there's nothing to join.
    """
    original = frozenset(selected)
    result = list(selected)
    for rel in relationships:
        if len(result) >= cap:
            break
        if rel.from_table in original and rel.to_table not in result:
            result.append(rel.to_table)
        elif rel.to_table in original and rel.from_table not in result:
            result.append(rel.from_table)
    return result[:cap]


@dataclass(frozen=True)
class RetrievedTableInfo:
    """One table/view that retrieval chose, and why."""
    name: str
    kind: str          # "table" or "view"
    selected_by: str   # "score" | "fk_expansion" | "fallback"
    score: int         # 0 for tables that were only pulled in as join partners


@dataclass(frozen=True)
class RetrievalDetails:
    """Everything retrieval decided for one question."""
    tables: tuple[RetrievedTableInfo, ...]
    relationships: tuple[str, ...]
    schema_text: str          # exactly what is handed to the LLM
    full_schema_chars: int    # size of the whole schema, for comparison


def retrieve_with_details(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    db_path=None,
    dataset_id: str | None = None,
) -> RetrievalDetails:
    """The whole retrieval step for one question, with the reasoning kept.

    Scoring, selection, the hub-table fallback and the one-hop join
    expansion are exactly what they have been since Day 3 — this only
    remembers each table's score and whether it was picked by scoring,
    pulled in as a foreign-key neighbor, or chosen as the fallback.

    `dataset_id` and `db_path` both default to whatever's currently active
    (backend/dataset.py) — pass them explicitly to inspect a different
    dataset (or a test fixture) without touching the active pointer.
    """
    resolved_id = _resolve_dataset_id(dataset_id)
    resolved_path = _resolve_db_path(db_path)
    documents = _load_documents(resolved_id, resolved_path)
    profile = get_schema_profile(resolved_id)

    tokens = _tokenize(question)
    scores = {doc.name: _score_document(tokens, doc) for doc in documents.values()}

    scored = sorted(documents.values(), key=lambda doc: scores[doc.name], reverse=True)
    selected = [doc.name for doc in scored if scores[doc.name] > 0][:top_k]

    used_fallback = not selected
    if used_fallback:
        # Nothing matched at all (e.g. a very generic or off-topic
        # question) — fall back to a hub table rather than sending nothing.
        # "orders" is the historical Olist fallback; for a dataset with no
        # table named "orders" (any upload, most other datasets), fall back
        # to whichever table happens to be first, so there's always
        # something to send rather than an empty schema.
        fallback_name = "orders" if "orders" in documents else next(iter(documents), None)
        selected = [fallback_name] if fallback_name else []

    names = _expand_with_join_neighbors(selected, profile.relationships, cap=MAX_DOCUMENTS)

    initial_reason = "fallback" if used_fallback else "score"
    tables = tuple(
        RetrievedTableInfo(
            name=name,
            kind=documents[name].kind,
            selected_by=initial_reason if name in selected else "fk_expansion",
            score=scores.get(name, 0),
        )
        for name in names
        if name in documents
    )
    relationships = tuple(
        rel.as_text()
        for rel in profile.relationships
        if rel.from_table in names and rel.to_table in names
    )

    return RetrievalDetails(
        tables=tables,
        relationships=relationships,
        schema_text=_render_schema_text(documents, names, relationships, profile.general_notes),
        full_schema_chars=_full_schema_chars(resolved_id, resolved_path),
    )


def _render_schema_text(
    documents: dict[str, SchemaDocument],
    names: list[str],
    relationship_lines: tuple[str, ...],
    general_notes: str,
) -> str:
    doc_lines = [documents[name].as_text() for name in names if name in documents]

    parts = ["Relevant tables for this question (not the full database):", ""]
    parts += doc_lines
    if relationship_lines:
        parts += ["", "How these tables join together:"] + list(relationship_lines)
    if general_notes:
        parts += ["", general_notes]
    return "\n".join(parts)


def retrieve_relevant_tables(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    db_path=None,
    dataset_id: str | None = None,
) -> list[str]:
    """Return the names of the tables/views most relevant to `question`,
    already expanded with their join partners. This is the "retrieve" step
    of RAG, kept separate from text rendering so it's easy to unit test.
    """
    details = retrieve_with_details(question, top_k=top_k, db_path=db_path, dataset_id=dataset_id)
    return [table.name for table in details.tables]


def build_relevant_schema_text(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    db_path=None,
    dataset_id: str | None = None,
) -> str:
    """The full retrieval step for one question: pick relevant tables, then
    render them (plus the relationships between them) as the schema text
    handed to the LLM in place of the full database schema.
    """
    return retrieve_with_details(
        question, top_k=top_k, db_path=db_path, dataset_id=dataset_id
    ).schema_text
