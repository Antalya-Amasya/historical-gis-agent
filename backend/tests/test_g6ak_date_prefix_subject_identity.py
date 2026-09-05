"""G6AK: temporal era markers must not become narrative subject identity."""

from __future__ import annotations

from backend.app.models import (
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
    TemporalPrecision,
)
from backend.app.routes.episode_relevance import EpisodeRelevance, classify_event_anchor_episode
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    classify_evidence_relevance,
    event_relevance,
    narrative_subject_proper_nouns,
    normalized_narrative_subjects,
    relation_admission_allowed,
)
from backend.app.routes.temporal import EvidenceTemporalResolver
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g6ag_query_subject_scope_normalization import (
    POSSESSIVE_QUERY,
    evidence,
)

BCE_PREFIX = "In 200 BCE Ariston marched from Rome to Capua."
BCE_SUFFIX = "Ariston marched from Rome to Capua in 200 BCE."
SECOND_PREFIX = "In 100 BCE Ariston sailed from Capua to Brundisium."
CAESAR_PREFIX = "In 49 BCE Julius Caesar marched from Rome to Brundisium."
CE_PREFIX = "In 200 CE Ariston marched from Rome to Capua."
AD_PREFIX = "In AD 200 Ariston marched from Rome to Capua."
WRONG_SUBJECT = "In 200 BCE Bion marched from Rome to Capua."
QUERY_200 = "Trace Ariston's route in 200 BCE."
CONTRADICTORY = "In 100 BCE Ariston marched from Rome to Capua."

COORDS = {
    "Roma": (41.9, 12.5),
    "Capua": (41.08, 14.25),
    "Brundisium": (40.6, 17.9),
}


def place(name: str) -> HistoricalPlace:
    lat, lon = COORDS[name]
    return HistoricalPlace(
        id=name.lower(),
        canonical_name=name,
        latitude=lat,
        longitude=lon,
        source="test",
        confidence=0.8,
    )


def binding(name: str, role: EventPlaceRole, refs: list[str]) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=refs),
        place=place(name),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
    )


def movement_event_multi(
    identifier: str,
    statement: str,
    origin: str,
    destination: str,
    *,
    refs: list[str],
) -> HistoricalEvent:
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=statement,
        evidence_refs=refs,
        place_bindings=[
            binding(origin, EventPlaceRole.ORIGIN, refs),
            binding(destination, EventPlaceRole.DESTINATION, refs),
        ],
        source_statements=[statement],
    )


def classify_route(query: str, statement: str, *, origin: str = "Roma", dest: str = "Capua"):
    event = movement_event_multi("e1", statement, origin, dest, refs=["ev1"])
    ev = evidence("ev1", statement)
    rel = relation(origin, dest, OrderingRule.SAME_MOVEMENT_EVENT, refs=("ev1",), event_ids=(event.id,))
    tag = event_relevance(event, {ev.id: ev}, (query,))
    episode, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        {ev.id: ev},
        (query,),
        subject_relevance=tag,
    )
    admitted = relation_admission_allowed(
        rel, {event.id: event}, {ev.id: ev}, (query,), rule=rel.rule,
    )
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event],
        [ev],
        event_id="g6ak",
        name="Ariston",
        period="unknown",
        query_contexts=(query,),
    )
    route_points = len(outcome.route.ordered_points) if outcome.route else 0
    return tag, episode, detail, admitted, route_points


def test_a_bce_prefix_subject_is_ariston_not_bce():
    subjects = narrative_subject_proper_nouns(BCE_PREFIX)
    assert "bce" not in subjects
    assert "ariston" in subjects
    assert classify_evidence_relevance(BCE_PREFIX, (POSSESSIVE_QUERY,)) is EvidenceRelevance.DIRECT_SUBJECT
    tag, episode, detail, admitted, route_points = classify_route(POSSESSIVE_QUERY, BCE_PREFIX)
    assert tag is EvidenceRelevance.DIRECT_SUBJECT
    assert episode is not EpisodeRelevance.OTHER_CAMPAIGN
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2


def test_b_suffix_equivalence_matches_prefix_subject():
    prefix_subjects = normalized_narrative_subjects(BCE_PREFIX)
    suffix_subjects = normalized_narrative_subjects(BCE_SUFFIX)
    assert prefix_subjects == suffix_subjects
    assert "ariston" in prefix_subjects
    assert "bce" not in prefix_subjects


def test_c_two_date_prefixed_movements_remain_viable():
    e1 = movement_event_multi("e1", BCE_PREFIX, "Roma", "Capua", refs=["ev1"])
    e2 = movement_event_multi("e2", SECOND_PREFIX, "Capua", "Brundisium", refs=["ev2"])
    ev1 = evidence("ev1", BCE_PREFIX)
    ev2 = evidence("ev2", SECOND_PREFIX)
    evidence_by_id = {ev1.id: ev1, ev2.id: ev2}
    for event in (e1, e2):
        tag = event_relevance(event, evidence_by_id, (POSSESSIVE_QUERY,))
        assert tag is EvidenceRelevance.DIRECT_SUBJECT
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [e1, e2],
        [ev1, ev2],
        event_id="g6ak",
        name="Ariston",
        period="unknown",
        query_contexts=(POSSESSIVE_QUERY,),
    )
    assert outcome.diagnostics["retained_relation_count"] == 2
    assert outcome.diagnostics["rejected_relation_count"] == 0


def test_d_multi_token_subject_still_matches():
    subjects = normalized_narrative_subjects(CAESAR_PREFIX)
    assert "bce" not in subjects
    assert "julius" in subjects
    assert "caesar" in subjects
    query = "Trace Julius Caesar's route."
    assert classify_evidence_relevance(CAESAR_PREFIX, (query,)) is EvidenceRelevance.DIRECT_SUBJECT


def test_e_ce_prefix_subject_is_ariston():
    subjects = normalized_narrative_subjects(CE_PREFIX)
    assert "ce" not in subjects
    assert "ariston" in subjects


def test_f_ad_prefix_subject_is_ariston():
    subjects = normalized_narrative_subjects(AD_PREFIX)
    assert "ad" not in subjects
    assert "ariston" in subjects


def test_g_temporal_parsing_remains_intact():
    resolver = EvidenceTemporalResolver()
    readings, codes = resolver.resolve(BCE_PREFIX, "probe")
    primary = resolver.primary(readings, "probe")
    assert primary.precision is TemporalPrecision.YEAR
    assert primary.normalized_start == "-200"
    assert codes == ["TEMPORAL_RESOLVED"]
    assert "ariston" in normalized_narrative_subjects(BCE_PREFIX)


def test_wrong_subject_still_rejects():
    tag, episode, detail, admitted, route_points = classify_route(POSSESSIVE_QUERY, WRONG_SUBJECT)
    assert tag is EvidenceRelevance.OTHER_CAMPAIGN
    assert detail["admitted"] is False
    assert route_points < 2


def test_query_year_contradiction_still_rejects():
    tag, episode, detail, admitted, route_points = classify_route(QUERY_200, CONTRADICTORY)
    assert detail["admitted"] is False
    assert route_points < 2
