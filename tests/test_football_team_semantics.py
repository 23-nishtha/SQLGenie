"""Regression tests for a semantic SQL bug in the Football dataset.

The bug: `results.winner` holds only the literal text 'home'/'away'/'draw'
— WHICH SIDE won — never an actual team name. Asked "which team has the
most wins?", SQLGenie answered with `GROUP BY winner`, producing
`winner | win_count` rows like `home | 24276` — a real, computable number,
just the answer to a different question ("how many home wins were there")
than the one asked. The fix has two parts:

  1. backend/schema_profiles.py's FOOTBALL_SCHEMA_PROFILE now explicitly
     tells the LLM winner is not a team name, and gives it the exact
     UNION ALL pattern to derive one — this is what the REAL model (Ollama
     or OpenAI) reads.
  2. backend/mock_llm.py gained canned, correct answers for all 8 questions
     from the bug report, so this is testable without a real LLM call.

Every expected number here was independently computed directly against
database/football/football.db before being hard-coded, not derived from
the code under test.
"""
import pytest
from fastapi.testclient import TestClient

from backend import schema_profiles
from backend.main import app


@pytest.fixture
def client():
    """Function-scoped (not module-scoped) and self-restoring on purpose:
    conftest.py's autouse dataset-registry reset snapshots the active
    dataset at the START of each test function. A module-scoped fixture
    that selects "football" runs BEFORE that per-test snapshot, so the
    snapshot would capture "football" as if it were the session's original
    baseline — leaking it into every test in every file that runs after
    this one. Selecting it here, per-test, keeps the snapshot/restore in
    conftest.py meaningful regardless of fixture setup order.
    """
    with TestClient(app) as c:
        c.post("/datasets/football/select")
        yield c


# ---------------------------------------------------------------------------
# Questions 1-6: must use actual team names, never "home"/"away"/"draw".
# ---------------------------------------------------------------------------


def _assert_team_name_answer(body, team_key):
    """The whole point of the fix: the answer names a real team, not a side."""
    assert body["row_count"] == 1
    team_value = body["rows"][0][team_key]
    assert team_value not in ("home", "away", "draw"), (
        f"Got {team_value!r} — this is a SIDE, not a team name. "
        "The winner-column bug has regressed."
    )
    # A real team name is a non-trivial string, not a number or empty.
    assert isinstance(team_value, str) and len(team_value) > 2


def test_most_wins_names_an_actual_team_not_home_or_away(client):
    response = client.post("/ask", json={"question": "Which team has the most wins?"})
    assert response.status_code == 200
    body = response.json()
    _assert_team_name_answer(body, "team")
    assert body["rows"] == [{"team": "Brazil", "wins": 675}]
    # The specific shape of the original bug report, confirmed absent:
    assert "winner" not in body["columns"]


def test_most_losses_names_an_actual_team(client):
    response = client.post("/ask", json={"question": "Which team has the most losses?"})
    assert response.status_code == 200
    body = response.json()
    _assert_team_name_answer(body, "team")
    assert body["rows"] == [{"team": "Finland", "losses": 436}]


def test_brazil_wins_counts_both_home_and_away_wins(client):
    response = client.post("/ask", json={"question": "How many matches has Brazil won?"})
    assert response.status_code == 200
    assert response.json()["rows"] == [{"brazil_wins": 675}]


def test_brazil_losses_counts_both_home_and_away_losses(client):
    response = client.post("/ask", json={"question": "How many matches has Brazil lost?"})
    assert response.status_code == 200
    assert response.json()["rows"] == [{"brazil_losses": 172}]


def test_most_goals_scored_names_an_actual_team(client):
    response = client.post("/ask", json={"question": "Which team has scored the most goals?"})
    assert response.status_code == 200
    body = response.json()
    _assert_team_name_answer(body, "team")
    assert body["rows"] == [{"team": "England", "goals_scored": 2401}]


def test_most_goals_conceded_names_an_actual_team(client):
    response = client.post("/ask", json={"question": "Which team has conceded the most goals?"})
    assert response.status_code == 200
    body = response.json()
    _assert_team_name_answer(body, "team")
    assert body["rows"] == [{"team": "Finland", "goals_conceded": 1675}]


# ---------------------------------------------------------------------------
# Questions 7-8: grouping by the raw `winner` column IS correct here —
# these ask about the SIDE, not a team.
# ---------------------------------------------------------------------------


def test_home_wins_count_uses_the_winner_column_directly(client):
    response = client.post("/ask", json={"question": "How many home wins were there?"})
    assert response.status_code == 200
    assert response.json()["rows"] == [{"home_wins": 24276}]


def test_away_wins_count_uses_the_winner_column_directly(client):
    response = client.post("/ask", json={"question": "How many away wins were there?"})
    assert response.status_code == 200
    assert response.json()["rows"] == [{"away_wins": 14010}]


# ---------------------------------------------------------------------------
# Internal consistency: the 8 answers must agree with each other and with
# the underlying data, independent of which question produced them.
# ---------------------------------------------------------------------------


def test_home_plus_away_plus_draw_equals_total_matches(client):
    home = client.post("/ask", json={"question": "How many home wins were there?"}).json()
    away = client.post("/ask", json={"question": "How many away wins were there?"}).json()
    total = client.post("/ask", json={"question": "How many matches are there?"}).json()

    draws = total["rows"][0]["total_matches"] - home["rows"][0]["home_wins"] - away["rows"][0]["away_wins"]
    assert draws == 11261  # independently verified against the real database


def test_brazil_wins_plus_losses_is_at_most_its_total_matches_played(client):
    wins = client.post("/ask", json={"question": "How many matches has Brazil won?"}).json()
    losses = client.post("/ask", json={"question": "How many matches has Brazil lost?"}).json()
    played = client.post("/ask", json={"question": "How many matches has Brazil played?"}).json()

    total_decided = wins["rows"][0]["brazil_wins"] + losses["rows"][0]["brazil_losses"]
    assert total_decided <= played["rows"][0]["brazil_matches"]  # the rest are draws


def test_the_most_wins_question_does_not_reproduce_the_original_bug_report(client):
    """The literal bug report: asking "which team has the most wins"
    should never come back as {"winner": "home", "win_count": 24276} or
    anything shaped like it."""
    response = client.post("/ask", json={"question": "Which team has the most wins?"})
    body = response.json()
    assert body["rows"] != [{"winner": "home", "win_count": 24276}]
    assert "home" != body["rows"][0].get("team")


# ---------------------------------------------------------------------------
# The schema guidance itself: this is what a REAL model (Ollama/OpenAI)
# reads, so it's tested directly too, not just through mock mode's canned
# answers above.
# ---------------------------------------------------------------------------


def test_football_schema_notes_warn_that_winner_is_not_a_team_name():
    notes = schema_profiles.FOOTBALL_SCHEMA_PROFILE.general_notes
    assert "winner is NOT a team name" in notes
    assert "GROUP BY winner" in notes  # names the exact mistake to avoid


def test_football_schema_notes_include_the_union_all_pattern():
    notes = schema_profiles.FOOTBALL_SCHEMA_PROFILE.general_notes
    assert "UNION ALL" in notes
    assert "home_team AS team" in notes
    assert "away_team AS team" in notes


def test_football_schema_notes_still_permit_winner_for_side_only_questions():
    """The fix must not throw out the legitimate use of `winner` for
    home/away-side questions — only forbid it for team-name questions."""
    notes = schema_profiles.FOOTBALL_SCHEMA_PROFILE.general_notes
    assert "home wins were there" in notes
