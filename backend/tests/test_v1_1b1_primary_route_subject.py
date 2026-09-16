"""V1.1B1: primary route subject intent contract for possessive movement queries."""
from __future__ import annotations

import pytest

from backend.app.rag.query_roles import analyze_query
from backend.app.rag.retrieval_intents import (
    _subject_phrase,
    decompose_movement_query,
    primary_route_subject,
)

MATRIX = [
    (
        "Trace Pompey's route after the Battle of Pharsalus to Egypt.",
        "Pompey",
        frozenset({"pompey"}),
        None,
        frozenset(),
    ),
    (
        "Trace Lucullus's campaign movements against Mithridates from Pontus to Armenia.",
        "Lucullus",
        frozenset({"lucullus", "mithridates"}),
        "Lucullus Mithridates",
        frozenset({"mithridates"}),
    ),
    (
        "Trace Caesar's route across the Adriatic into Epirus.",
        "Caesar",
        frozenset({"caesar"}),
        None,
        frozenset(),
    ),
    (
        "Trace Alexander's march against Porus toward the Hydaspes.",
        "Alexander",
        frozenset({"alexander", "porus"}),
        "Alexander Porus",
        frozenset({"porus"}),
    ),
    (
        "Trace Sulla's campaign against Mithridates in Greece.",
        "Sulla",
        frozenset({"sulla", "mithridates"}),
        None,
        frozenset({"mithridates"}),
    ),
    (
        "Trace Commander Alpha's route toward Delta Bay.",
        "Commander Alpha",
        frozenset({"alpha", "bay", "commander"}),
        "Commander Alpha Bay",
        frozenset(),
    ),
    (
        "Trace Commander Alpha's campaign against Commander Beta from Gamma Harbor.",
        "Commander Alpha",
        frozenset({"alpha", "beta", "commander", "harbor"}),
        "Commander Alpha Beta Harbor",
        frozenset({"beta"}),
    ),
]


@pytest.mark.parametrize(
    ("query", "expected_primary", "expected_person_terms", "polluted_phrase", "opponents"),
    MATRIX,
)
def test_primary_route_subject_matrix(
    query: str,
    expected_primary: str,
    expected_person_terms: frozenset[str],
    polluted_phrase: str | None,
    opponents: frozenset[str],
) -> None:
    roles = analyze_query(query)
    assert primary_route_subject(query, roles) == expected_primary
    assert expected_person_terms <= roles.person_terms
    if polluted_phrase is not None:
        assert _subject_phrase(query, roles) == polluted_phrase


def test_non_possessive_query_has_no_primary_route_subject() -> None:
    query = "Trace Lucullus during the war against Mithridates."
    assert primary_route_subject(query) is None


def _intent_queries(query: str) -> dict[str, list[str]]:
    intents = decompose_movement_query(query)
    grouped: dict[str, list[str]] = {}
    for intent in intents:
        grouped.setdefault(intent.kind, []).append(intent.query)
    return grouped


def _subject_prefixed_movement(grouped: dict[str, list[str]], expected_primary: str) -> str:
    matches = [
        query
        for query in grouped.get("MOVEMENT", [])
        if query.startswith(expected_primary)
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("query,expected_primary,_,__,opponents", MATRIX)
def test_retrieval_intent_prefixes_use_primary_subject_only(
    query: str,
    expected_primary: str,
    _: frozenset[str],
    __: str | None,
    opponents: frozenset[str],
) -> None:
    grouped = _intent_queries(query)
    assert grouped["SUBJECT"] == [expected_primary]

    for episode in grouped.get("EPISODE", []):
        assert episode.startswith(expected_primary)
        for opponent in opponents:
            assert opponent.casefold() not in {
                token.casefold() for token in episode.split()
            }

    subject_movement = _subject_prefixed_movement(grouped, expected_primary)
    for opponent in opponents:
        assert opponent.casefold() not in {
            token.casefold() for token in subject_movement.split()
        }

    for endpoint in grouped.get("ENDPOINT", []):
        assert endpoint.startswith(expected_primary)
        for opponent in opponents:
            assert opponent.casefold() not in {
                token.casefold() for token in endpoint.split()
            }
