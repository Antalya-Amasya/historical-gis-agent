"""G6AG: possessive query subjects and explicit campaign scope must parse correctly."""

from __future__ import annotations

from backend.app.models import (
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalClaim,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventTemporalGrounding,
    HistoricalEventType,
    HistoricalPlace,
    TemporalGroundingStatus,
    TemporalPrecision,
)
from backend.app.routes.episode_relevance import (
    EpisodeRelevance,
    classify_event_anchor_episode,
    classify_legacy_claim_episode,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    classify_evidence_relevance,
    event_relevance,
    relation_admission_allowed,
)
from backend.tests.test_g4b_route_components import relation

COORDS = {
    "Roma": (41.9, 12.5),
    "Capua": (41.08, 14.25),
    "Athens": (37.98, 23.73),
    "Sparta": (37.08, 22.43),
}

POSSESSIVE_QUERY = "Trace Ariston's route."
NONPOSSESSIVE_QUERY = "Trace Ariston route."
CAMPAIGN_QUERY = "Trace Ariston during the Northern campaign."
YEAR_QUERY = "Trace Ariston's route in 200 BCE."
ENDPOINT_QUERY = "Trace Ariston from Rome to Capua."
MOVEMENT = "Ariston marched from Rome to Capua."
CAMPAIGN_MOVEMENT = "During the Northern campaign, Ariston marched from Rome to Capua."
CAESAR_MOVEMENT = "Julius Caesar marched from Rome to Capua."
BION_MOVEMENT = "Bion marched from Rome to Capua."
MISMATCH_MOVEMENT = "Ariston marched from Athens to Sparta."


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


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(
        id=identifier,
        author="Source",
        work="Work",
        locator="1",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": 10},
    )


def binding(name: str, role: EventPlaceRole, refs: list[str]) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=refs),
        place=place(name),
        role=role,
        resolution_status=EventPlaceResolutionStatus.RESOLVED,
        evidence_refs=refs,
        resolver_provenance="test",
    )


def movement_event(
    identifier: str,
    statement: str,
    origin: str,
    destination: str,
    *,
    year: str | None = None,
) -> HistoricalEvent:
    refs = ["ev1"]
    grounding = HistoricalEventTemporalGrounding()
    if year is not None:
        grounding = HistoricalEventTemporalGrounding(
            raw_expression=f"{year} BCE",
            normalized_start=f"-{year}",
            normalized_end=f"-{year}",
            precision=TemporalPrecision.YEAR,
            evidence_refs=refs,
            status=TemporalGroundingStatus.EVIDENCE_GROUNDED,
        )
    return HistoricalEvent(
        id=identifier,
        name=identifier,
        summary=statement,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        source_statements=[statement],
        place_bindings=[
            binding(origin, EventPlaceRole.ORIGIN, refs),
            binding(destination, EventPlaceRole.DESTINATION, refs),
        ],
        temporal_grounding=grounding,
    )


def movement_claim(statement: str, *, source: str = "Roma", dest: str = "Capua") -> HistoricalClaim:
    return HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text=statement,
        textual_basis=statement,
        source_place=source,
        destination_place=dest,
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )


def classify_route(query: str, statement: str = MOVEMENT):
    event = movement_event("e1", statement, "Roma", "Capua")
    ev = evidence("ev1", statement)
    rel = relation(
        "Roma",
        "Capua",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=(event.id,),
    )
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
        event_id="g6ag",
        name="Ariston",
        period="unknown",
        query_contexts=(query,),
    )
    route_points = len(outcome.route.ordered_points) if outcome.route else 0
    return tag, episode, detail, admitted, route_points


def test_a_possessive_subject_query_admits_grammatical_movement():
    tag, episode, detail, admitted, route_points = classify_route(POSSESSIVE_QUERY)
    assert tag is EvidenceRelevance.DIRECT_SUBJECT
    assert episode is not EpisodeRelevance.OTHER_CAMPAIGN
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2


def test_b_nonpossessive_query_matches_possessive_subject_identity():
    possessive = classify_route(POSSESSIVE_QUERY)
    nonpossessive = classify_route(NONPOSSESSIVE_QUERY)
    assert possessive[0] is nonpossessive[0]
    assert possessive[1] is nonpossessive[1]
    assert possessive[3] is nonpossessive[3]


def test_c_multi_token_possessive_subject_matches():
    query = "Trace Julius Caesar's route."
    event = movement_event("e1", CAESAR_MOVEMENT, "Roma", "Capua")
    ev = evidence("ev1", CAESAR_MOVEMENT)
    tag = event_relevance(event, {ev.id: ev}, (query,))
    assert tag is EvidenceRelevance.DIRECT_SUBJECT
    assert classify_evidence_relevance(CAESAR_MOVEMENT, (query,)) is EvidenceRelevance.DIRECT_SUBJECT


def test_d_explicit_campaign_without_support_not_admitted():
    tag, episode, detail, admitted, route_points = classify_route(CAMPAIGN_QUERY)
    assert tag is EvidenceRelevance.DIRECT_SUBJECT
    assert episode is EpisodeRelevance.UNKNOWN
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_e_explicit_campaign_with_support_may_admit():
    tag, episode, detail, admitted, _ = classify_route(CAMPAIGN_QUERY, CAMPAIGN_MOVEMENT)
    assert tag is EvidenceRelevance.DIRECT_SUBJECT
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True
    assert admitted is True


def test_f_explicit_year_unresolved_remains_fail_closed():
    tag, episode, detail, admitted, _ = classify_route(YEAR_QUERY)
    assert tag is EvidenceRelevance.DIRECT_SUBJECT
    assert episode is EpisodeRelevance.UNKNOWN
    assert detail["admitted"] is False
    assert admitted is False


def test_g_endpoint_mismatch_not_unrestricted_fallback():
    claim = movement_claim(MISMATCH_MOVEMENT, source="Athens", dest="Sparta")
    ev = evidence("ev1", MISMATCH_MOVEMENT)
    episode, detail = classify_legacy_claim_episode(claim, {ev.id: ev}, (ENDPOINT_QUERY,))
    assert detail["admitted"] is False
    assert episode is not EpisodeRelevance.DIRECT_QUERY_EPISODE


def test_h_different_subject_not_admitted():
    claim = movement_claim(BION_MOVEMENT)
    ev = evidence("ev1", BION_MOVEMENT)
    episode, detail = classify_legacy_claim_episode(claim, {ev.id: ev}, (POSSESSIVE_QUERY,))
    assert classify_evidence_relevance(BION_MOVEMENT, (POSSESSIVE_QUERY,)) is EvidenceRelevance.OTHER_CAMPAIGN
    assert detail["admitted"] is False
    assert episode is EpisodeRelevance.OTHER_CAMPAIGN


def test_route_level_possessive_query_builds_fragment():
    _, _, detail, _, route_points = classify_route(POSSESSIVE_QUERY)
    assert detail["admitted"] is True
    assert route_points >= 2


def test_route_level_unsupported_campaign_does_not_fabricate_route():
    _, _, detail, _, route_points = classify_route(CAMPAIGN_QUERY)
    assert detail["admitted"] is False
    assert route_points < 2
