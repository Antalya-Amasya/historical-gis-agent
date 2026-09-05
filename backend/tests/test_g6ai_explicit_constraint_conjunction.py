"""G6AI: explicit query constraints are conjunctive; campaign support must be structural."""

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
    _statement_supports_query_campaign,
    classify_event_anchor_episode,
    classify_legacy_claim_episode,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.evidence_relevance import EvidenceRelevance, event_relevance, relation_admission_allowed
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g6ag_query_subject_scope_normalization import (
    CAMPAIGN_MOVEMENT,
    MOVEMENT,
    POSSESSIVE_QUERY,
    classify_route,
    movement_event,
)

CAMPAIGN_YEAR_QUERY = "Trace Ariston during the Northern campaign in 200 BCE."
CAMPAIGN_QUERY = "Trace Ariston during the Northern campaign."
CAMPAIGN_YEAR_MOVEMENT = "In 200 BCE, during the Northern campaign, Ariston marched from Rome to Capua."
WRONG_YEAR_MOVEMENT = "In 100 BCE, during the Northern campaign, Ariston marched from Rome to Capua."
YEAR_ONLY_MOVEMENT = "In 200 BCE Ariston marched from Rome to Capua."
ROAD_MOVEMENT = "Ariston marched from Rome to Capua along the northern road."
GATE_MOVEMENT = "Ariston marched from Rome to Capua through the northern gate."
BION_CAMPAIGN_YEAR = "In 200 BCE, during the Northern campaign, Bion marched from Rome to Capua."
ENDPOINT_QUERY = "Trace Ariston from Rome to Capua during the Northern campaign in 200 BCE."
MISMATCH_MOVEMENT = "In 200 BCE, during the Northern campaign, Ariston marched from Athens to Sparta."


def place(name: str) -> HistoricalPlace:
    coords = {
        "Roma": (41.9, 12.5),
        "Capua": (41.08, 14.25),
        "Athens": (37.98, 23.73),
        "Sparta": (37.08, 22.43),
    }
    lat, lon = coords[name]
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


def test_a_campaign_supported_missing_year_not_admitted():
    tag, episode, detail, admitted, route_points = classify_route(CAMPAIGN_YEAR_QUERY, CAMPAIGN_MOVEMENT)
    assert _statement_supports_query_campaign(CAMPAIGN_MOVEMENT, (CAMPAIGN_YEAR_QUERY,))
    assert tag is EvidenceRelevance.DIRECT_SUBJECT
    assert episode is not EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_b_northern_road_is_not_campaign_support():
    assert not _statement_supports_query_campaign(ROAD_MOVEMENT, (CAMPAIGN_QUERY,))
    _, episode, detail, admitted, route_points = classify_route(CAMPAIGN_QUERY, ROAD_MOVEMENT)
    assert episode is not EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_b2_northern_gate_is_not_campaign_support():
    assert not _statement_supports_query_campaign(GATE_MOVEMENT, (CAMPAIGN_QUERY,))


def test_c_campaign_and_year_supported_may_admit():
    tag, episode, detail, admitted, route_points = classify_route(
        CAMPAIGN_YEAR_QUERY,
        CAMPAIGN_YEAR_MOVEMENT,
    )
    assert _statement_supports_query_campaign(CAMPAIGN_YEAR_MOVEMENT, (CAMPAIGN_YEAR_QUERY,))
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2


def test_d_campaign_supported_year_contradictory_rejected():
    _, episode, detail, admitted, route_points = classify_route(
        CAMPAIGN_YEAR_QUERY,
        WRONG_YEAR_MOVEMENT,
    )
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_e_campaign_unsupported_year_supported_not_admitted():
    _, episode, detail, admitted, route_points = classify_route(
        CAMPAIGN_YEAR_QUERY,
        YEAR_ONLY_MOVEMENT,
    )
    assert not _statement_supports_query_campaign(YEAR_ONLY_MOVEMENT, (CAMPAIGN_YEAR_QUERY,))
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_f_campaign_unsupported_year_missing_not_admitted():
    _, episode, detail, admitted, route_points = classify_route(CAMPAIGN_YEAR_QUERY, MOVEMENT)
    assert detail["admitted"] is False
    assert admitted is False
    assert route_points < 2


def test_g_different_subject_not_admitted():
    claim = movement_claim(BION_CAMPAIGN_YEAR)
    ev = evidence("ev1", BION_CAMPAIGN_YEAR)
    episode, detail = classify_legacy_claim_episode(claim, {ev.id: ev}, (CAMPAIGN_YEAR_QUERY,))
    assert detail["admitted"] is False
    assert episode is EpisodeRelevance.OTHER_CAMPAIGN


def test_h_unrestricted_subject_route_remains_viable():
    tag, episode, detail, admitted, route_points = classify_route(POSSESSIVE_QUERY, MOVEMENT)
    assert tag is EvidenceRelevance.DIRECT_SUBJECT
    assert detail["admitted"] is True
    assert admitted is True
    assert route_points >= 2


def test_i_supported_campaign_without_year_constraint_still_works():
    tag, episode, detail, admitted, route_points = classify_route(CAMPAIGN_QUERY, CAMPAIGN_MOVEMENT)
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True
    assert route_points >= 2


def test_j_endpoint_mismatch_not_rescued_by_campaign_and_year():
    claim = movement_claim(MISMATCH_MOVEMENT, source="Athens", dest="Sparta")
    ev = evidence("ev1", MISMATCH_MOVEMENT)
    episode, detail = classify_legacy_claim_episode(claim, {ev.id: ev}, (ENDPOINT_QUERY,))
    assert detail["admitted"] is False
    assert episode is not EpisodeRelevance.DIRECT_QUERY_EPISODE
