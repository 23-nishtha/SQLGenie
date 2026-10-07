"""Tests for schema-aware retrieval (backend/schema_retrieval.py).

These exercise ranking/selection logic and a read-only SQLite connection
for column names. Retrieval now embeds each table and the question via
Ollama (backend/llm.embed_text()) — no real network call happens here,
though: tests/conftest.py's session-scoped `_fake_ollama_embeddings`
fixture intercepts every /api/embeddings call with a deterministic,
content-aware fake vector, so these tests need no live Ollama server, no
API key, and stay fast and reproducible. See tests/test_embedding_retrieval.py
for tests of the real embedding call's wire format and of caching.
tests/conftest.py also forces SQLGENIE_LLM_MODE=mock for the whole test
session, but that setting doesn't matter for these tests either way:
retrieval happens before the SQL-generation LLM is ever called.
"""
import pytest

from backend.schema_retrieval import (
    build_relevant_schema_text,
    retrieve_relevant_tables,
)

# question -> table names we expect to see somewhere in the retrieved set.
# (Retrieval may also pull in extra join-partner tables — that's fine and
# expected; we only assert the *expected* ones are present, not that
# nothing else is.)
EXPECTED_TABLES_FOR_QUESTION = {
    "How many orders were delivered?": {"orders"},
    "What are the top 5 product categories by revenue?": {
        "v_order_items_detail", "product_category_translation", "products",
    },
    "Which sellers have the highest sales?": {"sellers"},
    "What is the average review score?": {"order_reviews"},
    "How many customers are from each state?": {"customers"},
}


@pytest.mark.parametrize("question,expected_subset", EXPECTED_TABLES_FOR_QUESTION.items())
def test_retrieves_expected_tables(question, expected_subset):
    retrieved = set(retrieve_relevant_tables(question))
    missing = expected_subset - retrieved
    assert not missing, (
        f"question={question!r} expected {expected_subset} to be a subset of "
        f"retrieved {retrieved}, but was missing {missing}"
    )


def test_fk_expansion_does_not_recurse_past_one_hop():
    """FK expansion must only add a table DIRECTLY joined to one that was
    actually ranked/selected — never a table reachable by going one hop
    further, through a table that itself only got in via expansion (that
    would be two hops, not one). This is the one-hop-only contract
    backend/schema_retrieval._expand_with_join_neighbors() guarantees
    regardless of what ranked the original selection (embedding similarity
    now, keyword overlap before): `original` is frozen before expansion
    starts, so only tables directly joined to an ORIGINALLY selected table
    can ever be added.

    Rather than pin this to one question's exact top-K (which depends on
    the embedding model's ranking, not on the mechanism being tested), this
    directly selects a table and checks what one-hop expansion adds from
    it — exercising the real relationship graph (backend/schema_profiles.
    OLIST_SCHEMA_PROFILE) a question never could alone.
    """
    from backend.schema_profiles import OLIST_SCHEMA_PROFILE
    from backend.schema_retrieval import _expand_with_join_neighbors

    # orders.customer_id -> customers, and three tables link TO orders:
    # order_items, order_payments, order_reviews. All four are one hop from
    # orders; products and sellers are only reachable via a SECOND hop,
    # through order_items — which, starting from ["orders"] alone, is never
    # itself an originally-selected table.
    expanded = set(_expand_with_join_neighbors(
        ["orders"], OLIST_SCHEMA_PROFILE.relationships, cap=10
    ))

    assert {"orders", "customers", "order_items", "order_payments", "order_reviews"}.issubset(expanded)
    assert "products" not in expanded, "products is 2 hops away (via order_items) — should not be pulled in"
    assert "sellers" not in expanded, "sellers is 2 hops away (via order_items) — should not be pulled in"


def test_a_real_question_about_orders_also_gets_orders_and_a_join_partner():
    """End-to-end sanity check (through the public retrieval function, not
    just the expansion helper above): a question clearly about orders
    retrieves `orders` itself plus at least one of its real join partners,
    regardless of which other tables the embedding model also ranks highly."""
    retrieved = set(retrieve_relevant_tables("How many orders were delivered?"))
    assert "orders" in retrieved
    assert retrieved & {"customers", "order_items", "order_payments", "order_reviews"}


def test_retrieval_stays_small():
    """Retrieval should never balloon back into "the whole schema" —
    that would defeat the point of doing retrieval at all."""
    for question in EXPECTED_TABLES_FOR_QUESTION:
        retrieved = retrieve_relevant_tables(question)
        assert 1 <= len(retrieved) <= 6


def test_orders_and_customers_join_is_preserved():
    """A question that clearly needs orders+customers should retrieve both,
    thanks to foreign-key expansion, even if scoring alone favored one."""
    retrieved = set(retrieve_relevant_tables("How many customers placed orders?"))
    assert {"orders", "customers"}.issubset(retrieved)


def test_order_items_pulls_in_its_join_partners():
    """order_items joins to orders, products and sellers — retrieval should
    surface at least one of those partners when order_items is relevant."""
    retrieved = set(retrieve_relevant_tables("What is the total freight value on order items?"))
    assert "order_items" in retrieved
    assert retrieved & {"orders", "products", "sellers"}


def test_unmatched_question_falls_back_to_orders():
    """A question with no recognizable keywords still gets a sensible,
    non-empty default (orders, the hub table) rather than an empty schema.
    FK expansion then pulls in Orders' direct join partners too, same as
    it would for any other selection — that's expected, not a bug."""
    retrieved = retrieve_relevant_tables("asdf qwerty zzz")
    assert retrieved[0] == "orders"
    assert len(retrieved) <= 6


def test_build_relevant_schema_text_is_smaller_than_full_schema():
    from backend.schema_context import build_schema_text

    full_text = build_schema_text()
    partial_text = build_relevant_schema_text("What is the average review score?")

    assert len(partial_text) < len(full_text)
    assert "order_reviews" in partial_text
    # Should NOT be pulling in the obviously unrelated raw geolocation table
    # (its ~1M-row cousin zip_locations mentioning it in a description
    # doesn't count as "pulling it in" — check for the TABLE line itself).
    assert "TABLE geolocation(" not in partial_text


def test_build_relevant_schema_text_includes_join_relationship():
    """When products and its category translation are both retrieved, the
    relationship line connecting them should be present, not just their
    column lists in isolation."""
    text = build_relevant_schema_text("What are the top 5 product categories by revenue?")
    assert "product_category_translation" in text
    # The soft relationship uses "~>" (not a real FK); either arrow is fine
    # as long as the join path is documented somewhere in the text.
    assert "product_category_name" in text


def test_general_notes_always_included():
    """The customer_unique_id gotcha is important enough to always include,
    regardless of which tables were retrieved."""
    text = build_relevant_schema_text("How many orders were delivered?")
    assert "customer_unique_id" in text
