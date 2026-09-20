"""V1.1G9G-R2R4.1: subjectless route queries retain episode and endpoint scope."""
from __future__ import annotations

import pytest

from backend.app.routes.query_route_admission import parse_query_route_scope


@pytest.mark.parametrize(
    "query",
    [
        "Show historical military routes.",
        "Show routes during Campaign Red.",
        "Trace movement from Port A to City B.",
        "Show routes into Region C.",
        "Show movements around City B.",
    ],
)
def test_subjectless_route_scope_never_fabricates_actor(query: str):
    assert parse_query_route_scope((query,)).subject is None


@pytest.mark.parametrize(
    ("query", "subject"),
    [
        ("Show Commander Alpha's historical route.", "Commander Alpha"),
        ("Trace Commander Alpha's movements.", "Commander Alpha"),
    ],
)
def test_explicit_actor_route_scope_retains_subject(query: str, subject: str):
    assert parse_query_route_scope((query,)).subject == subject


def test_subjectless_campaign_scope_retains_episode():
    scope = parse_query_route_scope(("Show routes during Campaign Red.",))
    assert scope.subject is None
    assert scope.episode == "Campaign Red"
    assert scope.has_episode_constraint is True
