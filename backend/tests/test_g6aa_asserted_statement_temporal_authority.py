"""G6AA: asserted movement statement time must outrank neighboring context time."""

from __future__ import annotations

from backend.app.models import (
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalClaim,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
)
from backend.app.routes.episode_relevance import (
    EpisodeRelevance,
    classify_event_anchor_episode,
    classify_legacy_claim_episode,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule, _filter_relations_for_query
from backend.app.routes.evidence_relevance import EvidenceRelevance, relation_admission_allowed
from backend.app.routes.temporal import EvidenceTemporalResolver
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g5e2_relation_admission import _event, _evidence, _place
from backend.tests.test_g6r_temporal_authority import _movement_claim

QUERY_200 = "Trace Ariston's route in 200 BCE."
MOVEMENT_100 = "In 100 BCE Ariston marched from Rome to Capua."
CONTEXT_200 = "In 200 BCE Ariston was consul."


def _normalized_year(text: str) -> str | None:
    readings, _ = EvidenceTemporalResolver().resolve(text, "probe")
    if not readings:
        return None
    return EvidenceTemporalResolver().primary(readings, "probe").normalized_start


def _chunk(*sentences: str) -> str:
    return " ".join(sentences)


def _movement_claim_from(statement: str) -> HistoricalClaim:
    return HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text=statement,
        textual_basis=statement,
        source_place="Roma",
        destination_place="Capua",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )


def _ariston_event(statement: str) -> HistoricalEvent:
    refs = ["ev1"]
    bindings = [
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(
                raw_text="Rome", role=EventPlaceRole.ORIGIN, evidence_refs=refs,
            ),
            place=_place("Roma"),
            role=EventPlaceRole.ORIGIN,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=refs,
            resolver_provenance="test",
        ),
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(
                raw_text="Capua", role=EventPlaceRole.DESTINATION, evidence_refs=refs,
            ),
            place=_place("Capua"),
            role=EventPlaceRole.DESTINATION,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=refs,
            resolver_provenance="test",
        ),
    ]
    return _event("ariston-move", statement, source_statements=[statement], refs=refs, bindings=bindings)


def _classify_chunked_movement(*sentences: str):
    chunk = _chunk(*sentences)
    claim = _movement_claim_from(MOVEMENT_100)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=chunk, text=chunk,
    )
    return classify_legacy_claim_episode(claim, {evidence.id: evidence}, (QUERY_200,))


def test_a_exact_gpt6_counterexample_rejects_context_override():
    episode, detail = _classify_chunked_movement(CONTEXT_200, MOVEMENT_100)
    assert _normalized_year(MOVEMENT_100) == "-100"
    assert _normalized_year(CONTEXT_200) == "-200"
    assert _normalized_year(QUERY_200) == "-200"
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_b_reverse_sentence_order_still_rejects():
    episode, detail = _classify_chunked_movement(MOVEMENT_100, CONTEXT_200)
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_c_matching_movement_time_remains_admissible():
    movement_200 = "In 200 BCE Ariston marched from Rome to Capua."
    context_100 = "In 100 BCE Ariston was consul."
    query = "Trace Ariston's route from Rome to Capua in 200 BCE."
    chunk = _chunk(context_100, movement_200)
    claim = _movement_claim_from(movement_200)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=chunk, text=chunk,
    )
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))
    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_d_context_supplements_undated_movement():
    context = "In 100 BCE Ariston began the campaign."
    movement = "Ariston marched from Rome to Capua."
    query = "Trace Ariston's route in 100 BCE."
    claim = _movement_claim_from(movement)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1",
        excerpt=_chunk(context, movement), text=_chunk(context, movement),
    )
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))
    assert detail["admitted"] is True
    assert episode is not EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE


def test_e_query_favorable_context_does_not_override_explicit_movement_year():
    trailing = "In 50 BCE Ariston later governed elsewhere."
    episode, detail = _classify_chunked_movement(CONTEXT_200, MOVEMENT_100, trailing)
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_f_no_time_anywhere_stays_unknown_not_contradiction():
    movement = "Ariston marched from Rome to Capua."
    query = "Trace Ariston's route from Rome to Capua."
    claim = _movement_claim_from(movement)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=movement, text=movement,
    )
    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))
    assert episode is not EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is True


def test_route_level_gpt6_case_blocks_same_movement_for_200_bce_query():
    chunk = _chunk(CONTEXT_200, MOVEMENT_100)
    event = _ariston_event(MOVEMENT_100)
    evidence = _evidence("ev1", chunk)
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
    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False
    assert not relation_admission_allowed(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (QUERY_200,),
        rule=OrderingRule.SAME_MOVEMENT_EVENT,
    )
    kept, rejected = _filter_relations_for_query(
        [rel], {event.id: event}, {evidence.id: evidence}, (QUERY_200,),
    )
    assert kept == []
    assert rejected
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event],
        [evidence],
        event_id="g6aa",
        name="G6AA",
        period="200 BCE",
        query_contexts=(QUERY_200,),
    )
    same = [item for item in outcome.relations if item.rule is OrderingRule.SAME_MOVEMENT_EVENT]
    assert same == []
    assert outcome.route is None or len(outcome.route.ordered_points) < 2
