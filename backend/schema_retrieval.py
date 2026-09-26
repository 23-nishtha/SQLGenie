"""Day 3: lightweight, explainable schema-aware retrieval ("RAG" for SQLGenie).

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
question. That's harmless for an 11-object schema, but it's still wasted
tokens (cost + latency) and it gives the model irrelevant columns it could
get confused by. This module instead:

  1. Represents each table/view as a "schema document" — a short
     description, some hand-picked concept keywords (e.g. "delivered",
     "shipped" for `orders`), and its real column list.
  2. Scores every document against the question using simple keyword
     overlap. No embeddings, no vector database, no extra services —
     just Python string matching, which is easy to read and debug.
  3. Picks the best-scoring documents, then walks the known foreign-key
     graph to pull in any directly-joined table that wasn't already picked
     (so the LLM always has what it needs to write a JOIN, per Day 3's
     requirement to preserve relationships).
  4. Renders only that subset — plus a couple of universally-important
     notes — as the schema text handed to the LLM.

This is intentionally a lookup-and-score system, not a real search engine.
For a schema this small, hand-curated keywords are more explainable (and
easier to fix when retrieval picks the "wrong" table) than an
embeddings-based approach, and they need zero extra infrastructure.
"""
import re
from dataclasses import dataclass, field

from backend.schema_context import get_table_columns

# ---------------------------------------------------------------------------
# Schema documents: one entry per table/view, with the business knowledge
# that plain column names don't convey. This is the "corpus" retrieval
# scores against. Column lists are NOT hand-written here — they're filled
# in from the live database by _load_documents(), so they can't drift out
# of sync with olist.db.
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


@dataclass(frozen=True)
class Relationship:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    enforced: bool = True  # False = a "soft" link (e.g. zip codes), not a real FK

    def as_text(self) -> str:
        arrow = "->" if self.enforced else "~>"  # ~> hints this isn't a strict FK
        return f"{self.from_table}.{self.from_column} {arrow} {self.to_table}.{self.to_column}"


# name -> (description, keywords). Keywords are extra words a question
# might use that don't already appear in the table/column names themselves.
_SCHEMA_METADATA: dict[str, tuple[str, frozenset[str]]] = {
    "orders": (
        "One row per customer order. The central/hub table: order_items, "
        "order_payments and order_reviews all link back to it.",
        frozenset({
            "order", "orders", "purchase", "purchased", "bought", "status",
            "delivered", "delivery", "shipped", "shipping", "canceled",
            "cancelled", "approved", "estimated", "placed",
        }),
    ),
    "customers": (
        "One row per customer PER ORDER (customer_id is unique per order). "
        "Use customer_unique_id to count or identify actual distinct people.",
        frozenset({
            "customer", "customers", "buyer", "buyers", "client", "clients",
            "zip", "city", "state", "location", "people", "person",
        }),
    ),
    "sellers": (
        "One row per seller/merchant who lists products on the marketplace.",
        frozenset({
            "seller", "sellers", "vendor", "vendors", "merchant", "merchants",
            "shop", "shops", "zip", "city", "state",
        }),
    ),
    "products": (
        "One row per product: its category (in Portuguese) and physical "
        "dimensions/weight.",
        frozenset({
            "product", "products", "item", "items", "category", "categories",
            "weight", "dimensions", "size", "photo", "photos",
        }),
    ),
    "product_category_translation": (
        "Maps a Portuguese product_category_name to its English translation.",
        frozenset({
            "category", "categories", "translation", "english", "portuguese",
            "name",
        }),
    ),
    "order_items": (
        "One row per product line within an order (the bridge between "
        "orders, products and sellers). Holds price and freight per line.",
        frozenset({
            "item", "items", "line", "price", "prices", "freight", "shipping",
            "revenue", "sales", "sold", "cost",
        }),
    ),
    "order_payments": (
        "One row per payment made on an order (an order can have more than one).",
        frozenset({
            "payment", "payments", "paid", "pay", "installment",
            "installments", "credit", "card", "boleto", "voucher", "debit",
        }),
    ),
    "order_reviews": (
        "One row per customer review of an order, with a 1-5 score.",
        frozenset({
            "review", "reviews", "rating", "ratings", "score", "scores",
            "comment", "comments", "feedback", "satisfaction", "opinion",
        }),
    ),
    "geolocation": (
        "Raw lat/lng samples for zip codes — many rows per zip. Prefer "
        "zip_locations (one row per zip) for most questions.",
        frozenset({
            "geolocation", "coordinates", "latitude", "longitude", "lat",
            "lng", "zip", "map",
        }),
    ),
    "zip_locations": (
        "One row per zip code prefix: average latitude/longitude and the "
        "most common city/state for that zip. Derived from geolocation.",
        frozenset({
            "zip", "location", "locations", "city", "state", "latitude",
            "longitude", "map", "geography", "region", "where",
        }),
    ),
    "v_order_items_detail": (
        "Convenience VIEW: one row per order line, with order, customer, "
        "product (English category) and seller fields already joined. Use "
        "this for most revenue / top-N / group-by questions instead of "
        "writing the joins by hand.",
        frozenset({
            "revenue", "sales", "total", "top", "best", "ranking", "rank",
            "category", "categories", "seller", "sellers", "customer",
            "customers", "order", "orders", "product", "products", "state",
            "city", "average", "sum",
        }),
    ),
}

# Real, enforced foreign keys (from database/schema.sql) plus a couple of
# "soft" links that aren't declared FKs but are still how a human would
# join these tables (e.g. category name, zip code prefix).
_RELATIONSHIPS: tuple[Relationship, ...] = (
    Relationship("orders", "customer_id", "customers", "customer_id"),
    Relationship("order_items", "order_id", "orders", "order_id"),
    Relationship("order_items", "product_id", "products", "product_id"),
    Relationship("order_items", "seller_id", "sellers", "seller_id"),
    Relationship("order_payments", "order_id", "orders", "order_id"),
    Relationship("order_reviews", "order_id", "orders", "order_id"),
    Relationship(
        "products", "product_category_name",
        "product_category_translation", "product_category_name",
        enforced=False,  # ~610 products have no category; not a real FK
    ),
    Relationship(
        "customers", "customer_zip_code_prefix",
        "zip_locations", "zip_code_prefix", enforced=False,
    ),
    Relationship(
        "sellers", "seller_zip_code_prefix",
        "zip_locations", "zip_code_prefix", enforced=False,
    ),
)

# Small, universally-useful notes kept even when the table they describe
# isn't in the retrieved set — cheap (a few dozen tokens) and they head off
# the two most common mistakes an LLM makes on this schema.
_GENERAL_NOTES = (
    "General notes:\n"
    "- customers.customer_id is unique per ORDER, not per person. Use "
    "customers.customer_unique_id to count or identify real people.\n"
    "- Dates are stored as text in 'YYYY-MM-DD HH:MM:SS' format."
)

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

# Populated once by _load_documents() and reused; a document's description/
# keywords never change at runtime, only its live column list could (and
# that only changes if someone rebuilds olist.db, which requires a restart
# anyway — see backend/main.py's startup hook).
_documents_cache: dict[str, "SchemaDocument"] | None = None


def _load_documents(db_path=None) -> dict[str, SchemaDocument]:
    """Build {name: SchemaDocument}, merging hand-written metadata with the
    live column list from SQLite (via schema_context.get_table_columns)."""
    global _documents_cache
    if _documents_cache is not None and db_path is None:
        return _documents_cache

    tables = get_table_columns(db_path)
    documents: dict[str, SchemaDocument] = {}
    for name, info in tables.items():
        description, keywords = _SCHEMA_METADATA.get(
            name, ("(no description available)", frozenset())
        )
        documents[name] = SchemaDocument(
            name=name,
            kind=info["kind"],
            description=description,
            keywords=keywords,
            columns=tuple(info["columns"]),
        )

    if db_path is None:
        _documents_cache = documents
    return documents


def init_documents(db_path=None) -> None:
    """Warm the document cache once at startup (see backend/main.py), so
    each request only pays for scoring (pure Python), not a fresh DB read."""
    global _documents_cache
    _documents_cache = None
    _load_documents(db_path)


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
    "average latitude/longitude" in zip_locations' description matching a
    question about "average review score") was enough to pull in that
    table — and then its foreign-key neighbors too — for a question that
    had nothing to do with it. Descriptions are for humans reading the
    rendered schema text, not a retrieval signal: only the deliberately
    curated `keywords` set should trigger a match beyond the exact
    table/column names.
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


def _expand_with_join_neighbors(selected: list[str], cap: int) -> list[str]:
    """Add any table directly connected (by FK) to an ORIGINALLY selected
    table, so the LLM can always see how to join what it was given.

    Strictly ONE hop: `original` is frozen before the loop starts, and every
    membership check that decides whether to add a new table is against
    `original`, never against `result`. That matters — an earlier version
    checked against the growing `result` list instead, so a table added
    *during* expansion could itself trigger adding a further table in the
    same pass (e.g. selecting "orders" pulled in "customers" via scoring,
    which then pulled in "order_items" as one hop from "orders", which then
    pulled in "products" and "sellers" as one hop from "order_items" — three
    hops away from anything the question actually matched). Freezing
    `original` means only tables directly joined to a table the SCORER
    picked can be added; a table that got in purely through expansion is
    never itself used as a new jumping-off point.
    """
    original = frozenset(selected)
    result = list(selected)
    for rel in _RELATIONSHIPS:
        if len(result) >= cap:
            break
        if rel.from_table in original and rel.to_table not in result:
            result.append(rel.to_table)
        elif rel.to_table in original and rel.from_table not in result:
            result.append(rel.from_table)
    return result[:cap]


def retrieve_relevant_tables(
    question: str, top_k: int = DEFAULT_TOP_K, db_path=None
) -> list[str]:
    """Return the names of the tables/views most relevant to `question`,
    already expanded with their join partners. This is the "retrieve" step
    of RAG, kept separate from text rendering so it's easy to unit test.
    """
    documents = _load_documents(db_path)
    tokens = _tokenize(question)

    scored = sorted(
        documents.values(),
        key=lambda doc: _score_document(tokens, doc),
        reverse=True,
    )
    selected = [
        doc.name for doc in scored if _score_document(tokens, doc) > 0
    ][:top_k]

    if not selected:
        # Nothing matched at all (e.g. a very generic or off-topic
        # question) — fall back to the hub table rather than sending
        # nothing, since almost every real question touches orders.
        selected = ["orders"]

    return _expand_with_join_neighbors(selected, cap=MAX_DOCUMENTS)


def build_relevant_schema_text(
    question: str, top_k: int = DEFAULT_TOP_K, db_path=None
) -> str:
    """The full retrieval step for one question: pick relevant tables, then
    render them (plus the relationships between them) as the schema text
    handed to the LLM in place of the full database schema.
    """
    documents = _load_documents(db_path)
    selected_names = retrieve_relevant_tables(question, top_k=top_k, db_path=db_path)

    doc_lines = [
        documents[name].as_text() for name in selected_names if name in documents
    ]
    relationship_lines = [
        rel.as_text()
        for rel in _RELATIONSHIPS
        if rel.from_table in selected_names and rel.to_table in selected_names
    ]

    parts = ["Relevant tables for this question (not the full database):", ""]
    parts += doc_lines
    if relationship_lines:
        parts += ["", "How these tables join together:"] + relationship_lines
    parts += ["", _GENERAL_NOTES]
    return "\n".join(parts)
