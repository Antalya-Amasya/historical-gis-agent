"""R4-C5: extend depart movement family with departure at query time."""
from __future__ import annotations

from backend.app.rag.query_roles import _movement_family_expansions, analyze_query, movement_scoring_terms


def test_departure_query_expands_depart_movement_family():
    roles = analyze_query("What departure is attested before the provincial council met?")
    assert "departure" in roles.action_terms
    assert roles.movement_inflection_terms >= {"depart", "departed", "departing"}
    assert movement_scoring_terms(roles) >= {"departure", "departed"}


def test_unrelated_query_does_not_gain_depart_family_terms():
    roles = analyze_query("What policy changes did the senate approve in Rome?")
    depart_family = frozenset({"depart", "departed", "departing", "departure"})
    scored = roles.action_terms | roles.expanded_action_terms | roles.movement_inflection_terms
    assert depart_family.isdisjoint(scored)


def test_movement_family_helper_includes_departure():
    expanded = _movement_family_expansions(frozenset({"departure"}))
    assert expanded >= {"depart", "departed", "departing"}
