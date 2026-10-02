"""Movement claims preserve bound semantic edges without supplying route authority."""
import pytest

from backend.app.models import EventActorStatus
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.tests.test_g4a_movement_semantics import evidence
from backend.tests.test_v2auth1_strict_endpoint_admission import build

LOCAL = HistoricalPlaceMentionExtractor(tuple(
    HistoricalPlaceAlias(name, (name.lower(),))
    for name in ("A", "B", "C", "D", "Neralon", "Darsena", "Galveth")
))
DOCUMENTARY = (
    "Sulla left them and sailed for Greece, and thence passed on to Italy "
    "with the greater part of his army."
)


def claims(text, extractor=LOCAL, **metadata):
    return extractor.movement_claims([evidence("source-1", text, **metadata)], event_id="control")


def pairs(values):
    return [(c.source_place, c.destination_place) for c in values]


@pytest.mark.parametrize("text,expected", [
    ("Marcus marched from A to B.", ("A", "B")),
    ("Nikanor Demetrios marched from Neralon to Darsena.", ("Neralon", "Darsena")),
    ("Ariston did not hesitate and marched from A to B.", ("A", "B")),
    ("The army marched from A to B.", ("A", "B")),
])
def test_bound_direct_edge_survives_recheck(text, expected):
    semantic = analyze_sentence(text, LOCAL.aliases_in(text))
    assert len(semantic.edges) == 1
    result = claims(text)
    assert pairs(result) == [expected]
    assert result[0].supporting_evidence_ids == ["source-1"]
    assert result[0].source_documents == ["Source"]
    assert result[0].textual_basis == text


def test_documentary_edge_retains_identity_and_provenance_once():
    extractor = HistoricalPlaceMentionExtractor()
    semantic = analyze_sentence(DOCUMENTARY, extractor.aliases_in(DOCUMENTARY))
    assert [(e.origin.place_name, e.destination.place_name) for e in semantic.edges] == [("Greece", "Italia")]
    result = claims(DOCUMENTARY, extractor)
    assert pairs(result) == [("Greece", "Italia")]
    assert result[0].movement_relation == "thence_passed_on_to"
    assert result[0].supporting_evidence_ids == ["source-1"]
    assert result[0].textual_basis == DOCUMENTARY


@pytest.mark.parametrize("text", [
    "Marcus advanced from A toward B but stopped before reaching it.",
    "Marcus was prevented from marching from A to B.",
    "Marcus planned to march from A to B.",
    "Marcus attempted to march from A to B.",
    "Marcus did not march from A to B.",
    "Marcus could march from A to B.",
    "Marcus sailed for A, and planned to pass on to B.",
])
def test_non_completion_does_not_become_completed_claim(text):
    assert claims(text) == []


def test_contrast_preserves_only_completed_edge():
    result = claims("Marcus did not march from A to B, but marched from C to D.")
    assert pairs(result) == [("C", "D")]


@pytest.mark.parametrize("text", [
    "Marcus reached A, while Pompey moved to B.",
    "Marcus sailed for A, and Pompey thence passed on to B.",
    "Marcus sailed for A, and in a different campaign thence passed on to B.",
    "Marcus sailed for A. Pompey passed on to B.",
])
def test_independent_context_cannot_supply_missing_edge(text):
    assert claims(text) == []


def test_separate_actors_keep_their_direct_edges_without_bridge():
    result = claims("Marcus marched from A to B. Pompey later moved from C to D.")
    assert pairs(result) == [("A", "B"), ("C", "D")]
    assert len({c.id for c in result}) == 2


def test_supported_same_actor_two_legs_have_distinct_claim_ids():
    result = claims("Marcus sailed from A to B, and then marched from B to C.")
    assert pairs(result) == [("A", "B"), ("B", "C")]
    assert len({c.id for c in result}) == 2
    assert all(c.supporting_evidence_ids == ["source-1"] for c in result)


def test_collective_actor_is_not_upgraded_by_claim_retention():
    text = "The army marched from A to B."
    assert pairs(claims(text)) == [("A", "B")]
    events, _ = EvidenceGroundedHistoricalEventExtractor(LOCAL).extract([evidence("source-1", text)])
    assert all(e.actor.actor_status is EventActorStatus.UNKNOWN for e in events)


@pytest.mark.parametrize("text", [
    "Marcus marched from A to B.",
    "Marcus marched from nowhere to nowhereelse.",
    "Marcus remained at A and discussed B.",
])
def test_unbound_or_non_movement_strings_do_not_create_claims(text):
    assert claims(text, HistoricalPlaceMentionExtractor()) == []


def test_parent_metadata_does_not_supply_missing_endpoint():
    assert claims("Marcus sailed for A.", parent_text="Marcus marched from A to B.", query="A to B") == []


@pytest.mark.parametrize("origin,destination", [
    ("Rome", "Capua"), ("Roma", "Capua"),
    ("Oricum", "Brundisium"), ("Orikon", "Brundisium"),
    ("Oricum", "Brundusium"),
])
def test_production_aliases_survive(origin, destination):
    x = HistoricalPlaceMentionExtractor()
    text = f"Libo sailed from {origin} to {destination}."
    semantic = analyze_sentence(text, x.aliases_in(text))
    assert len(semantic.edges) == 1
    edge = semantic.edges[0]
    assert pairs(claims(text, x)) == [(edge.origin.place_name, edge.destination.place_name)]


def test_retaining_real_claim_does_not_admit_wrong_subject():
    text = "Pompey marched from Rome to Capua."
    assert pairs(claims(text, HistoricalPlaceMentionExtractor())) == [("Roma", "Capua")]
    result, _, admissions = build([text], query="Show Caesar route from Rome to Capua.")
    assert result.route is None
    assert admissions and all(not a.admitted for a in admissions)


def test_retaining_real_claim_does_not_admit_wrong_strict_phase():
    text = "Marcus marched from Capua to Brundisium."
    assert pairs(claims(text, HistoricalPlaceMentionExtractor())) == [("Capua", "Brundisium")]
    result, _, admissions = build([text])
    assert result.route is None
    assert admissions and all(not a.admitted for a in admissions)


def test_fresh_cross_clause_relation_is_preserved():
    text = "Nikanor Demetrios sailed for Neralon, and thence passed on to Darsena."
    assert len(analyze_sentence(text, LOCAL.aliases_in(text)).edges) == 1
    assert pairs(claims(text)) == [("Neralon", "Darsena")]


def test_established_edge_still_requires_occurrence_continuity():
    text = "Nikanor sailed for Neralon, and thence passed on to Darsena in a different campaign."
    assert len(analyze_sentence(text, LOCAL.aliases_in(text)).edges) == 1
    assert claims(text) == []


@pytest.mark.parametrize("text", [
    "Nikanor left Neralon and arrived at Darsena.",
    "Nikanor left Neralon, crossed Galveth, and arrived at Darsena.",
])
def test_established_departure_arrival_preserves_bounded_continuation(text):
    assert len(analyze_sentence(text, LOCAL.aliases_in(text)).edges) == 1
    assert pairs(claims(text)) == [("Neralon", "Darsena")]


@pytest.mark.parametrize("text", [
    "Nikanor left Neralon, Orestes arrived at Darsena.",
    "Nikanor left Neralon, crossed Galveth, and arrived at Darsena in another campaign.",
    "Nikanor left Neralon, planned to cross Galveth, and arrived at Darsena.",
])
def test_departure_arrival_cannot_bypass_actor_occurrence_or_completion(text):
    assert claims(text) == []
