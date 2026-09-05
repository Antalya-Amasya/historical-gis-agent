"""G6AC: unrestricted subject route requests must not fail solely on UNKNOWN episode."""

from __future__ import annotations

from backend.app.models import (
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalClaim,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalPlace,
)
from backend.app.routes.episode_relevance import (
    EpisodeRelevance,
    classify_event_anchor_episode,
    classify_legacy_claim_episode,
    episode_route_admission_allowed,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule, _filter_relations_for_query
from backend.app.routes.evidence_relevance import EvidenceRelevance, relation_admission_allowed
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g5e2_relation_admission import _event, _evidence, _place
from backend.tests.test_g6w_episode_unknown_vs_other import ARISTON_CAMPAIGN_A_QUERY

UNRESTRICTED_QUERY = "Trace Ariston's route."
QUERY_200 = "Trace Ariston's route in 200 BCE."
QUERY_ENDPOINTS = "Trace Ariston's route from Rome to Capua."
MOVEMENT_ROME_CAPUA = "Ariston marched from Rome to Capua."
MOVEMENT_ROME_CAPUA_PIPELINE = "Ariston's marched from Rome to Capua."
MOVEMENT_100 = "In 100 BCE Ariston marched from Rome to Capua."
MOVEMENT_200 = "In 200 BCE Ariston marched from Rome to Capua."
CONSUL_ROME = "Ariston was consul in Rome."
MOVEMENT_CORFINIUM_SICILY = "Ariston marched from Corfinium to Sicily."
BION_MOVEMENT = "Bion marched from Rome to Capua."


def _movement_claim_from(statement: str, *, source: str = "Roma", dest: str = "Capua") -> HistoricalClaim:
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


def _ariston_event(statement: str, *, source: str = "Roma", dest: str = "Capua") -> HistoricalEvent:
    refs = ["ev1"]
    bindings = [
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(
                raw_text=source, role=EventPlaceRole.ORIGIN, evidence_refs=refs,
            ),
            place=_place(source),
            role=EventPlaceRole.ORIGIN,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=refs,
            resolver_provenance="test",
        ),
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(
                raw_text=dest, role=EventPlaceRole.DESTINATION, evidence_refs=refs,
            ),
            place=_place(dest),
            role=EventPlaceRole.DESTINATION,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=refs,
            resolver_provenance="test",
        ),
    ]
    return _event("ariston-move", statement, source_statements=[statement], refs=refs, bindings=bindings)


def test_a_unrestricted_subject_movement_not_rejected_for_unknown_episode():
    event = _ariston_event(MOVEMENT_ROME_CAPUA)
    evidence = _evidence("ev1", MOVEMENT_ROME_CAPUA)
    rel = relation(
        "Roma",
        "Capua",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=(event.id,),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (UNRESTRICTED_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.UNKNOWN
    assert episode_route_admission_allowed(episode) is True
    assert detail["admitted"] is True
    assert detail["subject_relevance"] == EvidenceRelevance.DIRECT_SUBJECT.value
    assert detail["admission_reason"] == "EPISODE_RELEVANT"


def test_b_subject_only_non_movement_not_admitted():
    claim = HistoricalClaim(
        id="c1",
        claim_type="OFFICE",
        text=CONSUL_ROME,
        textual_basis=CONSUL_ROME,
        source_place=None,
        destination_place=None,
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )
    evidence = _evidence("ev1", CONSUL_ROME)
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (UNRESTRICTED_QUERY,))
    assert detail["admitted"] is False


def test_c_explicit_year_unresolved_remains_fail_closed():
    event = _ariston_event(MOVEMENT_ROME_CAPUA)
    evidence = _evidence("ev1", MOVEMENT_ROME_CAPUA)
    rel = relation(
        "Roma",
        "Capua",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=(event.id,),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (QUERY_200,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.UNKNOWN
    assert detail["admitted"] is False
    assert detail["admission_reason"] == "REJECT_UNKNOWN"


def test_d_explicit_contradictory_year_rejects_other_episode():
    claim = _movement_claim_from(MOVEMENT_100)
    evidence = _evidence("ev1", MOVEMENT_100)
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (QUERY_200,))
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False
    assert detail["admission_reason"] == "REJECT_SAME_SUBJECT_OTHER_EPISODE"


def test_e_explicit_endpoint_mismatch_not_subject_only_fallback():
    claim = _movement_claim_from(
        MOVEMENT_CORFINIUM_SICILY,
        source="Corfinium",
        dest="Sicily",
    )
    evidence = _evidence("ev1", MOVEMENT_CORFINIUM_SICILY)
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (QUERY_ENDPOINTS,))
    assert detail["admitted"] is False
    assert episode is not EpisodeRelevance.DIRECT_QUERY_EPISODE


def test_f_constrained_positive_match_remains_direct_query_episode():
    query = "Trace Ariston's route in 200 BCE."
    claim = _movement_claim_from(MOVEMENT_200)
    evidence = _evidence("ev1", MOVEMENT_200)
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_g_different_subject_not_admitted():
    claim = _movement_claim_from(BION_MOVEMENT)
    evidence = _evidence("ev1", BION_MOVEMENT)
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (UNRESTRICTED_QUERY,))
    assert detail["admitted"] is False


def test_g6w_regression_campaign_alpha_unknown_still_not_admitted():
    movement = "Ariston marched from Port Helios to Port Selene."
    event = _event("ariston", movement, source_statements=[movement], refs=["ev1"])
    evidence_by_id = {"ev1": _evidence("ev1", movement)}
    rel = relation(
        "Port Helios",
        "Port Selene",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("ariston",),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {"ariston": event},
        evidence_by_id,
        (ARISTON_CAMPAIGN_A_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.UNKNOWN
    assert detail["admitted"] is False


def test_route_level_unrestricted_subject_movement_builds_fragment():
    event = _ariston_event(MOVEMENT_ROME_CAPUA_PIPELINE)
    evidence = _evidence("ev1", MOVEMENT_ROME_CAPUA_PIPELINE)
    rel = relation(
        "Roma",
        "Capua",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=(event.id,),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (UNRESTRICTED_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode in {EpisodeRelevance.UNKNOWN, EpisodeRelevance.DIRECT_QUERY_EPISODE}
    assert detail["admitted"] is True
    assert relation_admission_allowed(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (UNRESTRICTED_QUERY,),
        rule=rel.rule,
    )
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event],
        [evidence],
        event_id="g6ac",
        name="Ariston",
        period="unknown",
        query_contexts=(UNRESTRICTED_QUERY,),
    )
    assert outcome.route is not None
    assert len(outcome.route.ordered_points) >= 2


def test_route_level_constrained_year_does_not_fabricate_route():
    event = _ariston_event(MOVEMENT_ROME_CAPUA)
    evidence = _evidence("ev1", MOVEMENT_ROME_CAPUA)
    rel = relation(
        "Roma",
        "Capua",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=(event.id,),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (QUERY_200,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert episode is EpisodeRelevance.UNKNOWN
    assert detail["admitted"] is False
    kept, rejected = _filter_relations_for_query(
        [rel],
        {event.id: event},
        {evidence.id: evidence},
        (QUERY_200,),
    )
    assert kept == []
    assert rejected
    assert not relation_admission_allowed(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (QUERY_200,),
        rule=rel.rule,
    )
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event],
        [evidence],
        event_id="g6ac-fail-closed",
        name="Ariston",
        period="200 BCE",
        query_contexts=(QUERY_200,),
    )
    assert outcome.route is None or len(outcome.route.ordered_points) < 2
