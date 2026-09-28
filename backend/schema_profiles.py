"""Day 7: dataset-specific knowledge for schema-aware retrieval.

Split of responsibilities (see backend/dataset.py and backend/schema_retrieval.py):
  - dataset.py            identity: which datasets exist, which is active,
                           its name/domain/theme, where its .db file lives.
  - schema_profiles.py     (this file) per-dataset RETRIEVAL knowledge: the
                           hand-curated table descriptions/keywords, known
                           foreign keys, and general notes that help the LLM
                           write correct SQL for THAT dataset's schema.
  - schema_retrieval.py    the generic algorithm (tokenize, score, expand by
                           foreign key, render as text) that works off
                           whichever SchemaProfile it's given. It has no
                           Olist-specific knowledge at all.

Through Day 6, schema_retrieval.py had Olist's table metadata and foreign
keys hard-coded as module constants — fine when there was only one dataset,
wrong now that a football database or an uploaded CSV needs its own (or no)
curated knowledge. Moving them here, unchanged in content, means adding a
new BUILT-IN dataset later is "write one more SchemaProfile", and an
UPLOADED dataset (which has no hand-curated knowledge at all) can use
DEFAULT_SCHEMA_PROFILE and still work — see its docstring for why.
"""
from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class SchemaProfile:
    """Everything schema_retrieval.py needs to know about ONE dataset's
    schema, beyond what it can read live from SQLite (table/column names).

    metadata: {table_name: (description, keywords)} — human-written context
        a column list alone can't convey. Keywords are extra words a
        question might use that don't already appear in the table/column
        names themselves (e.g. "delivered" for an `orders` table).
    relationships: known foreign keys (or "soft" links like zip codes) used
        to pull in a table's join partners once it's selected.
    general_notes: short, always-included notes — kept small since they're
        added to every question's schema text regardless of which tables
        were retrieved.
    """
    metadata: dict[str, tuple[str, frozenset[str]]] = field(default_factory=dict)
    relationships: tuple[Relationship, ...] = field(default_factory=tuple)
    general_notes: str = ""


# ---------------------------------------------------------------------------
# Olist (moved from schema_retrieval.py, content unchanged).
# ---------------------------------------------------------------------------

_OLIST_METADATA: dict[str, tuple[str, frozenset[str]]] = {
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
_OLIST_RELATIONSHIPS: tuple[Relationship, ...] = (
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
# the most common mistakes an LLM makes on this schema. The
# v_order_items_detail hint used to live in llm.py's fixed system prompt;
# it moved here in Day 7 so llm.py has no Olist-specific knowledge at all.
_OLIST_NOTES = (
    "General notes:\n"
    "- customers.customer_id is unique per ORDER, not per person. Use "
    "customers.customer_unique_id to count or identify real people.\n"
    "- Dates are stored as text in 'YYYY-MM-DD HH:MM:SS' format.\n"
    "- Prefer the v_order_items_detail view for questions spanning orders, "
    "products, customers or sellers, instead of writing the joins yourself."
)

OLIST_SCHEMA_PROFILE = SchemaProfile(
    metadata=_OLIST_METADATA,
    relationships=_OLIST_RELATIONSHIPS,
    general_notes=_OLIST_NOTES,
)


# ---------------------------------------------------------------------------
# Football (Day 8). One table, so `relationships` is empty — there's
# nothing to join. Keywords/description still help retrieval and give the
# LLM's prompt useful hints even with only one table to pick from.
# ---------------------------------------------------------------------------

_FOOTBALL_METADATA: dict[str, tuple[str, frozenset[str]]] = {
    "results": (
        "One row per international football (soccer) match, 1872-present: "
        "the two teams, final score, competition, and where it was played. "
        "winner and goal_diff are derived columns (not in the original data).",
        frozenset({
            "match", "matches", "game", "games", "result", "results",
            "score", "scores", "scored", "win", "wins", "won", "winner",
            "loss", "loses", "lost", "draw", "draws", "drew", "tie",
            "team", "teams", "goal", "goals", "difference", "diff",
            "tournament", "tournaments", "competition", "cup", "world cup",
            "qualifier", "qualification", "friendly", "friendlies",
            "season", "year", "home", "away", "neutral", "venue", "city",
            "country", "opponent", "versus", "vs",
        }),
    ),
}

_FOOTBALL_NOTES = (
    "General notes:\n"
    "- goal_diff is always a non-negative number (it's an absolute "
    "difference); use home_score - away_score directly if you need the "
    "SIGNED difference (positive = home team ahead).\n"
    "- winner is 'home', 'away' or 'draw' — a team can appear in either "
    "home_team or away_team, so compare against both when asking about one "
    "team's overall record (e.g. wins as home_team OR away_team).\n"
    "- neutral is 1 when the match was played at a neutral venue (INTEGER "
    "0/1, not TEXT).\n"
    "- date is text in 'YYYY-MM-DD' format; extract the year with "
    "strftime('%Y', date) or substr(date, 1, 4) for season/year questions."
)

FOOTBALL_SCHEMA_PROFILE = SchemaProfile(
    metadata=_FOOTBALL_METADATA,
    relationships=(),
    general_notes=_FOOTBALL_NOTES,
)


# ---------------------------------------------------------------------------
# Movies (Day 8). Also one table.
# ---------------------------------------------------------------------------

_MOVIES_METADATA: dict[str, tuple[str, frozenset[str]]] = {
    "movies": (
        "One row per movie (IMDb Top 1000): title, year, rating, genre, "
        "director, cast, votes and box-office gross. primary_genre is a "
        "derived column (the first of possibly several genres listed).",
        frozenset({
            "movie", "movies", "film", "films", "title", "titles",
            "rating", "ratings", "rated", "score", "imdb", "vote", "votes",
            "genre", "genres", "category", "categories", "director",
            "directors", "directed", "cast", "actor", "actors", "actress",
            "star", "stars", "starring", "year", "years", "released",
            "release", "decade", "runtime", "duration", "minutes", "long",
            "certificate", "rated", "gross", "revenue", "earned",
            "box office", "highest", "best", "top", "worst",
        }),
    ),
}

_MOVIES_NOTES = (
    "General notes:\n"
    "- Use primary_genre for grouping/filtering by a single genre — genre "
    "holds the original comma-separated list (e.g. 'Action, Drama, Sport') "
    "and grouping by it directly treats each combination as one category.\n"
    "- gross and meta_score and certificate can be NULL (missing in the "
    "source data) — use IS NOT NULL if a question implies only movies with "
    "that data available (e.g. 'highest grossing').\n"
    "- runtime_minutes is a plain INTEGER (already stripped of ' min').\n"
    "- released_year is a plain INTEGER."
)

MOVIES_SCHEMA_PROFILE = SchemaProfile(
    metadata=_MOVIES_METADATA,
    relationships=(),
    general_notes=_MOVIES_NOTES,
)


# The profile used for any dataset with no curated entry below — every
# upload, and any future built-in dataset before someone writes it a real
# profile. Empty metadata/relationships isn't a broken state: retrieval
# still works off live table/column names alone (see schema_retrieval.py's
# _score_document — table-name and column-name matches don't need curated
# keywords), and a single-table upload barely needs retrieval at all, since
# there's only one table to pick.
DEFAULT_SCHEMA_PROFILE = SchemaProfile(
    metadata={},
    relationships=(),
    general_notes="General notes:\n- No dataset-specific notes are available for this dataset yet.",
)

# dataset id -> its SchemaProfile. Datasets not listed here get
# DEFAULT_SCHEMA_PROFILE (see get_schema_profile below).
_SCHEMA_PROFILES: dict[str, SchemaProfile] = {
    "olist": OLIST_SCHEMA_PROFILE,
    "football": FOOTBALL_SCHEMA_PROFILE,
    "movies": MOVIES_SCHEMA_PROFILE,
}


def get_schema_profile(dataset_id: str) -> SchemaProfile:
    """The SchemaProfile for `dataset_id`, or the safe generic default if
    none has been curated. Never raises — an unrecognized or uploaded
    dataset id is an expected case, not an error."""
    return _SCHEMA_PROFILES.get(dataset_id, DEFAULT_SCHEMA_PROFILE)
