"""Schema-aware retrieval ("RAG" for SQLGenie) — now embedding-based.

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
Sending the ENTIRE database schema (every table, every column, every view)
to the LLM on every question is wasted tokens (cost + latency) and gives
the model irrelevant columns it could get confused by. This module instead:

  1. Represents each table/view as a "schema document" — a short
     description, some hand-picked concept keywords (e.g. "delivered",
     "shipped" for `orders`), and its real column list (see
     SchemaDocument/_embedding_text below).
  2. Embeds each document once, at cache warm-up — never once per question
     — using a local Ollama embedding model (backend/llm.embed_text(),
     OLLAMA_EMBED_MODEL). Embeds the incoming question the same way, then
     ranks documents by cosine similarity against it.
  3. Picks the best-ranked documents, then walks the known foreign-key
     graph to pull in any directly-joined table that wasn't already picked
     (so the LLM always has what it needs to write a JOIN).
  4. Renders only that subset — plus a couple of universally-important
     notes — as the schema text handed to the LLM.

This used to be pure keyword-overlap scoring (no embeddings, no network).
Embedding-based ranking replaces step 2 only — steps 1, 3 and 4, the public
function signatures, and the shape of what callers get back are unchanged,
so backend/agent.py and backend/llm.py don't need to know retrieval changed
at all; they still just get a `schema_text` string.

Dataset-generic, same as before
---------------------------------
This module still has NO dataset-specific knowledge of its own. Descriptions,
keywords and foreign keys come from backend/schema_profiles.py, looked up by
`dataset_id`; column lists come live from whichever SQLite file `db_path`
points at. Both parameters default to whatever's currently active (see
backend/dataset.py).

Single-table datasets (every CSV upload, and the built-in football/movies
datasets) skip embedding entirely — there's nothing to rank when there's
only one table, so `_load_embeddings` short-circuits and the one table is
used directly (see `selected_by="embedding"`, `score=1.0` below). This
keeps uploads and single-table datasets exactly as network-call-free as
multi-table ones were before Day 3, and avoids paying for an embedding call
that could never change the outcome.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

from backend import dataset, llm
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


DEFAULT_TOP_K = 4          # how many documents rank highest before FK expansion
MAX_DOCUMENTS = 6          # hard cap after expansion, so context stays small

# Below this cosine similarity, the best-ranked document isn't a confident
# match — treat it like "nothing matched" and use the fallback table
# instead. Deliberately low: real embedding models rarely produce a
# literal 0.0 for unrelated text the way keyword overlap did, so this only
# needs to catch the clearly-nothing-in-common case (also what the
# deterministic fake embedding used in tests produces for a question that
# shares no tokens with any table) — not finely calibrated against
# nomic-embed-text's exact distribution, which would need live tuning.
MIN_SIMILARITY = 0.05

# All three caches are keyed by (dataset_id, str(db_path)) — the exact pair
# a request resolved to — rather than a single global. That's what makes it
# safe for two different datasets' schemas to be cached at once (e.g. after
# switching the active dataset, or in a test that passes an explicit
# db_path): each combination gets its own entry, so switching datasets can
# never accidentally serve another dataset's cached documents/embeddings.
_documents_cache: dict[tuple[str, str], dict[str, SchemaDocument]] = {}
_full_schema_chars_cache: dict[tuple[str, str], int] = {}
# None means "skipped — this dataset has <= 1 table, nothing to rank".
_embeddings_cache: dict[tuple[str, str], dict[str, list[float]] | None] = {}


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
    — so a request only pays for one embedding call (the question's), not a
    fresh SQLite read AND an embedding call per table. Clears everything
    cached first: cheap, and correct if a dataset's underlying .db file was
    ever rebuilt out from under a running server.
    """
    _documents_cache.clear()
    _full_schema_chars_cache.clear()
    _embeddings_cache.clear()
    _load_documents(dataset_id, db_path)
    _full_schema_chars(dataset_id, db_path)
    _load_embeddings(dataset_id, db_path)


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


def _embedding_text(doc: SchemaDocument, relationships: tuple[Relationship, ...]) -> str:
    """The text embedded for ONE schema document (once, at cache warm-up —
    never once per question): its name, its real columns, its curated
    description/keywords if this dataset has a profile for it, and the
    relationships that directly involve it. This is the embedding
    counterpart of the old scorer's signals (table name, columns, curated
    keywords) plus relationship context, which the old keyword scorer
    didn't use directly but is cheap and useful context for an embedding
    model deciding "is this table relevant to this question."
    """
    lines = [f"Table: {doc.name}", f"Columns: {', '.join(doc.columns)}"]
    if doc.description:
        lines.append(f"Description: {doc.description}")
    if doc.keywords:
        lines.append(f"Keywords: {', '.join(sorted(doc.keywords))}")
    related = [
        rel.as_text() for rel in relationships
        if rel.from_table == doc.name or rel.to_table == doc.name
    ]
    if related:
        lines.append("Relationships: " + "; ".join(related))
    return "\n".join(lines)


def _load_embeddings(
    dataset_id: str | None = None, db_path=None
) -> dict[str, list[float]] | None:
    """{table_name: embedding vector} for every document in this dataset,
    computed once and cached — or None if this dataset has 0 or 1 tables,
    meaning there's nothing to rank and no embedding call is made at all
    (see the module docstring's "single-table datasets" section).
    """
    resolved_id = _resolve_dataset_id(dataset_id)
    resolved_path = _resolve_db_path(db_path)
    cache_key = (resolved_id, str(resolved_path))
    if cache_key in _embeddings_cache:
        return _embeddings_cache[cache_key]

    documents = _load_documents(resolved_id, resolved_path)
    if len(documents) <= 1:
        _embeddings_cache[cache_key] = None
        return None

    profile = get_schema_profile(resolved_id)
    embeddings = {
        name: llm.embed_text(_embedding_text(doc, profile.relationships))
        for name, doc in documents.items()
    }
    _embeddings_cache[cache_key] = embeddings
    return embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two embedding vectors, in [-1, 1] (practically
    [0, 1] for same-model text embeddings). Pure Python — these vectors are
    small and there are at most a handful of tables per dataset, so this
    needs no numeric library."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


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
    selected_by: str   # "embedding" | "fk_expansion" | "fallback"
    score: float       # cosine similarity; 0.0 for fk_expansion/fallback-only tables


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

    Ranking is now embedding cosine similarity (see _load_embeddings/
    _cosine_similarity above) instead of keyword overlap; selection, the
    hub-table fallback and the one-hop join expansion are otherwise exactly
    what they were before — this still remembers each table's score and
    whether it was picked by ranking, pulled in as a foreign-key neighbor,
    or chosen as the fallback.

    `dataset_id` and `db_path` both default to whatever's currently active
    (backend/dataset.py) — pass them explicitly to inspect a different
    dataset (or a test fixture) without touching the active pointer.
    """
    resolved_id = _resolve_dataset_id(dataset_id)
    resolved_path = _resolve_db_path(db_path)
    documents = _load_documents(resolved_id, resolved_path)
    profile = get_schema_profile(resolved_id)

    embeddings = _load_embeddings(resolved_id, resolved_path)

    if embeddings is None:
        # 0 or 1 tables: nothing to rank, and no embedding call made at all
        # (see _load_embeddings) — just use the one table there is, if any.
        names = list(documents)[:1]
        tables = tuple(
            RetrievedTableInfo(name=name, kind=documents[name].kind, selected_by="embedding", score=1.0)
            for name in names
        )
        relationships: tuple[str, ...] = ()
    else:
        question_vector = llm.embed_text(question)
        scores = {
            name: _cosine_similarity(question_vector, vector)
            for name, vector in embeddings.items()
        }
        ranked = sorted(embeddings, key=lambda name: scores[name], reverse=True)

        used_fallback = not ranked or scores[ranked[0]] < MIN_SIMILARITY
        if used_fallback:
            # Nothing matched confidently (e.g. a very generic or
            # off-topic question) — fall back to a hub table rather than
            # sending nothing. "orders" is the historical Olist fallback;
            # for a dataset with no table named "orders" (most other
            # datasets), fall back to whichever table happens to be first,
            # so there's always something to send rather than an empty
            # schema.
            fallback_name = "orders" if "orders" in documents else next(iter(documents), None)
            selected = [fallback_name] if fallback_name else []
        else:
            selected = ranked[:top_k]

        names = _expand_with_join_neighbors(selected, profile.relationships, cap=MAX_DOCUMENTS)

        initial_reason = "fallback" if used_fallback else "embedding"
        tables = tuple(
            RetrievedTableInfo(
                name=name,
                kind=documents[name].kind,
                selected_by=initial_reason if name in selected else "fk_expansion",
                score=scores.get(name, 0.0),
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
