"""V1.1B1.1: SUBJECT intent must mean confirmed primary route subject only."""
from __future__ import annotations

import pytest

from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retrieval_intents import (
    _subject_phrase,
    decompose_movement_query,
    primary_route_subject,
)

POSSESSIVE = (
    "Trace Lucullus's campaign movements against Mithridates.",
    "Lucullus",
)

NON_POSSESSIVE = [
    "Trace Lucullus during the war against Mithridates.",
    "Trace the route of Lucullus during the war against Mithridates.",
    "Follow Lucullus from Pontus to Armenia.",
    "Show Pompey fleeing after Pharsalus.",
]

POSSESSIVE_REPLAY = [
    ("Trace Pompey's route after the Battle of Pharsalus to Egypt.", "Pompey"),
    (
        "Trace Lucullus's campaign movements against Mithridates from Pontus to Armenia.",
        "Lucullus",
    ),
    ("Trace Alexander's march against Porus toward the Hydaspes.", "Alexander"),
    ("Trace Sulla's campaign against Mithridates in Greece.", "Sulla"),
]


def _kinds(query: str) -> set[str]:
    return {intent.kind for intent in decompose_movement_query(query)}


def _subject_queries(query: str) -> list[str]:
    return [intent.query for intent in decompose_movement_query(query) if intent.kind == "SUBJECT"]


def test_possessive_query_emits_confirmed_subject_intent() -> None:
    query, expected = POSSESSIVE
    assert primary_route_subject(query) == expected
    assert _subject_queries(query) == [expected]


@pytest.mark.parametrize("query", NON_POSSESSIVE)
def test_non_possessive_queries_emit_no_subject_intent(query: str) -> None:
    assert primary_route_subject(query) is None
    assert _subject_queries(query) == []


@pytest.mark.parametrize("query", NON_POSSESSIVE)
def test_keyword_bag_not_published_as_subject(query: str) -> None:
    roles = analyze_query(query)
    bag = _subject_phrase(query, roles)
    if not bag:
        return
    for subject_query in _subject_queries(query):
        assert subject_query != bag


def test_lucullus_during_war_episode_has_no_contaminated_mover_prefix() -> None:
    query = "Trace Lucullus during the war against Mithridates."
    roles = analyze_query(query)
    assert primary_route_subject(query) is None
    assert _subject_phrase(query, roles) == "Lucullus Mithridates"
    intents = {intent.kind: intent.query for intent in decompose_movement_query(query)}
    assert "SUBJECT" not in intents
    episode = intents.get("EPISODE", "")
    assert "Lucullus Mithridates" not in episode
    assert "Mithridates" not in episode.split()


@pytest.mark.parametrize("query,expected", POSSESSIVE_REPLAY)
def test_possessive_replay_retains_clean_subject(query: str, expected: str) -> None:
    assert _subject_queries(query) == [expected]


@pytest.mark.parametrize("query", NON_POSSESSIVE)
def test_non_possessive_replay_has_no_subject(query: str) -> None:
    assert "SUBJECT" not in _kinds(query)
