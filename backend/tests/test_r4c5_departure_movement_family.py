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


def test_voyage_query_expands_sail_movement_family():
    roles = analyze_query("What voyage did Paulus report when accounting for his campaign?")
    assert "voyage" in roles.action_terms
    assert roles.movement_inflection_terms >= {"sail", "sailed", "sailing"}
    assert "sailed" in movement_scoring_terms(roles)


def test_existing_sail_query_still_expands_sail_family():
    roles = analyze_query("Pompey sailed to Egypt")
    scored = roles.action_terms | roles.expanded_action_terms | roles.movement_inflection_terms
    assert scored >= {"sail", "sailed", "sailing"}
    assert "voyage" not in roles.action_terms


def test_voyage_query_is_not_coverage_route_query():
    from backend.app.rag.evidence_ranking import is_route_or_movement_query

    query = "What voyage did Paulus report when accounting for his campaign?"
    assert not is_route_or_movement_query(query)


def test_movement_family_helper_includes_departure():
    expanded = _movement_family_expansions(frozenset({"departure"}))
    assert expanded >= {"depart", "departed", "departing"}
