"""Tests for Day 3's schema-aware retrieval (backend/schema_retrieval.py).

These only exercise scoring/selection logic and a read-only SQLite
connection for column names — no OpenAI call, no network, no API key.
tests/conftest.py forces SQLGENIE_LLM_MODE=mock for the whole test session,
but that setting doesn't even matter for these tests: retrieval happens
before the LLM is ever called.
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
    """"How many orders were delivered?" only scores on `orders` and
    `v_order_items_detail` (no other table's name/keywords/columns match).
    `orders` is directly joined to customers, order_items, order_payments
    and order_reviews (one hop) — those should all appear. `products` and
    `sellers` are only reachable by going one hop further, THROUGH
    order_items (a table that only got in via expansion, not scoring) — a
    correct one-hop-only implementation must NOT add them, even though an
    earlier (buggy) version did, because it re-checked membership against
    the growing result list instead of the frozen original selection."""
    retrieved = set(retrieve_relevant_tables("How many orders were delivered?"))

    assert {"orders", "v_order_items_detail"}.issubset(retrieved)
    assert {"customers", "order_items", "order_payments", "order_reviews"}.issubset(retrieved)
    assert "products" not in retrieved, "products is 2 hops away (via order_items) — should not be pulled in"
    assert "sellers" not in retrieved, "sellers is 2 hops away (via order_items) — should not be pulled in"


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
