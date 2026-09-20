"""V1.1G9G-R2R4.2: structured query role parsing without cross-role contamination."""
from __future__ import annotations

import pytest

from backend.app.routes.query_route_admission import parse_query_route_scope

EQUIVALENT_SCOPE = {
    "subject": "Commander Alpha",
    "origin": "Port A",
    "destination": "City B",
    "episode": "Campaign Red",
    "has_episode_constraint": True,
}


def _assert_equivalent_scope(query: str) -> None:
    scope = parse_query_route_scope((query,))
    assert scope.subject == EQUIVALENT_SCOPE["subject"]
    assert scope.origin == EQUIVALENT_SCOPE["origin"]
    assert scope.destination == EQUIVALENT_SCOPE["destination"]
    assert scope.episode == EQUIVALENT_SCOPE["episode"]
    assert scope.has_episode_constraint is EQUIVALENT_SCOPE["has_episode_constraint"]


@pytest.mark.parametrize(
    "query",
    [
        "Trace Commander Alpha from Port A to City B during Campaign Red.",
        "During Campaign Red, trace Commander Alpha from Port A to City B.",
        "From Port A to City B, trace Commander Alpha during Campaign Red.",
        "Commander Alpha's route from Port A to City B in Campaign Red.",
    ],
)
def test_alternate_word_orders_parse_equivalent_scope(query: str):
    _assert_equivalent_scope(query)


def test_episode_first_query_roles_do_not_swallow_trailing_clauses():
    scope = parse_query_route_scope(
        ("During Campaign Red, trace Commander Alpha from Port A to City B.",),
    )
    assert scope.episode == "Campaign Red"
    assert "trace" not in (scope.episode or "").casefold()
    assert "port a" not in (scope.episode or "").casefold()


def test_endpoints_first_destination_stops_before_trace_clause():
    scope = parse_query_route_scope(
        ("From Port A to City B, trace Commander Alpha during Campaign Red.",),
    )
    assert scope.destination == "City B"
    assert scope.origin == "Port A"
    assert scope.subject == "Commander Alpha"


def test_possessive_route_subject_without_leading_command():
    scope = parse_query_route_scope(
        ("Commander Alpha's route from Port A to City B in Campaign Red.",),
    )
    assert scope.subject == "Commander Alpha"
    assert scope.episode == "Campaign Red"


def test_across_feature_does_not_contaminate_origin_or_destination():
    scope = parse_query_route_scope(
        ("Trace Commander Alpha from Port A across River X to City B during Campaign Red.",),
    )
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
    assert "river x" not in (scope.origin or "").casefold()
    assert "river x" not in (scope.destination or "").casefold()


@pytest.mark.parametrize(
    "query",
    [
        "Show routes during Campaign Red.",
        "Trace movement from Port A to City B.",
        "Show routes into Region C.",
    ],
)
def test_subjectless_controls_remain_subjectless(query: str):
    assert parse_query_route_scope((query,)).subject is None


def test_person_like_place_names_are_not_promoted_to_subject():
    scope = parse_query_route_scope(
        ("Trace movement from Alexander to Victoria during Campaign Red.",),
    )
    assert scope.subject is None
    assert scope.origin == "Alexander"
    assert scope.destination == "Victoria"
    assert scope.episode == "Campaign Red"


def test_place_like_person_name_in_explicit_actor_phrase():
    scope = parse_query_route_scope(
        ("Trace Commander River from Port A to City B during Campaign Red.",),
    )
    assert scope.subject == "Commander River"
    assert scope.origin == "Port A"
    assert scope.destination == "City B"
