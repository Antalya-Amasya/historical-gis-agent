"""G6R: temporal normalization consistency and explicit contradiction authority."""
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
from backend.app.routes.episode_relevance import EpisodeRelevance, classify_legacy_claim_episode
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder, OrderingRule
from backend.app.routes.evidence_relevance import relation_admission_allowed
from backend.app.routes.temporal import EvidenceTemporalResolver

COORDINATES = {
    "Alpha City": (45.0, 5.0),
    "Beta Province": (45.5, 5.5),
}


def _normalized_year(text: str) -> str | None:
    readings, _ = EvidenceTemporalResolver().resolve(text, "probe")
    if not readings:
        return None
    return EvidenceTemporalResolver().primary(readings, "probe").normalized_start


def test_f08_suffix_and_prefix_bce_forms_normalize_equivalently():
    suffix = _normalized_year("200 BCE")
    prefix = _normalized_year("BCE 200")
    suffix_bc = _normalized_year("200 BC")
    prefix_bc = _normalized_year("BC 200")

    assert suffix == "-200"
    assert prefix == suffix
    assert suffix_bc == suffix
    assert prefix_bc == suffix


def test_f08_positive_era_controls_preserved():
    assert _normalized_year("200 CE") == "200"
    assert _normalized_year("AD 200") == "200"


def test_f08_plain_year_without_era_remains_unresolved():
    readings, codes = EvidenceTemporalResolver().resolve("The campaign in 200.", "probe")
    assert readings == []
    assert codes == ["TEMPORAL_UNRESOLVED"]


def _movement_claim(statement: str) -> HistoricalClaim:
    return HistoricalClaim(
        id="c1",
        claim_type="MOVEMENT",
        text=statement,
        textual_basis=statement,
        source_place="Alpha City",
        destination_place="Beta Province",
        movement_relation="from_to",
        sequence_status="explicit",
        supporting_evidence_ids=["ev1"],
        source_documents=["doc"],
        confidence=0.9,
    )


def _movement_event(statement: str, *, year_text: str) -> HistoricalEvent:
    refs = ["ev1"]
    readings, _ = EvidenceTemporalResolver().resolve(year_text, "ev1")
    temporal = EvidenceTemporalResolver().primary(readings, "ev1")
    mention_origin = HistoricalEventPlaceMention(
        raw_text="Alpha City", role=EventPlaceRole.ORIGIN, evidence_refs=refs,
    )
    mention_destination = HistoricalEventPlaceMention(
        raw_text="Beta Province", role=EventPlaceRole.DESTINATION, evidence_refs=refs,
    )
    bindings = []
    for mention in (mention_origin, mention_destination):
        name = mention.raw_text
        latitude, longitude = COORDINATES[name]
        bindings.append(
            HistoricalEventPlaceBinding(
                mention=mention,
                place=HistoricalPlace(
                    id=name.lower().replace(" ", "-"),
                    canonical_name=name,
                    latitude=latitude,
                    longitude=longitude,
                    source="test",
                    confidence=0.8,
                ),
                role=mention.role,
                resolution_status=EventPlaceResolutionStatus.RESOLVED,
                evidence_refs=refs,
                resolver_provenance="test",
            )
        )
    return HistoricalEvent(
        id="move-1",
        name="move-1",
        summary=statement,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=refs,
        place_bindings=bindings,
        source_statements=[statement],
        temporal_grounding=temporal,
    )


def test_f07_contradictory_year_blocks_endpoint_admission():
    query = "Trace the route from Alpha City to Beta Province in 200 BCE."
    statement = "Commander marched from Alpha City to Beta Province in 100 BCE."
    claim = _movement_claim(statement)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=statement, text=statement,
    )

    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))

    assert episode is EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE
    assert detail["admitted"] is False


def test_f07_same_year_with_matching_endpoints_remains_admissible():
    query = "Trace the route from Alpha City to Beta Province in 200 BCE."
    statement = "Commander marched from Alpha City to Beta Province in 200 BCE."
    claim = _movement_claim(statement)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=statement, text=statement,
    )

    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))

    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_f07_query_year_without_event_year_stays_admissible_not_rejected():
    query = "Trace the route from Alpha City to Beta Province in 200 BCE."
    statement = "Commander marched from Alpha City to Beta Province."
    claim = _movement_claim(statement)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=statement, text=statement,
    )

    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))

    assert detail["admitted"] is True
    assert episode is not EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE


def test_f07_event_year_without_query_year_stays_admissible_not_rejected():
    query = "Trace the route from Alpha City to Beta Province."
    statement = "Commander marched from Alpha City to Beta Province in 100 BCE."
    claim = _movement_claim(statement)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=statement, text=statement,
    )

    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))

    assert detail["admitted"] is True
    assert episode is not EpisodeRelevance.SAME_SUBJECT_OTHER_EPISODE


def test_era_equivalence_bce_200_vs_200_bce_not_contradictory():
    query = "Trace the route from Alpha City to Beta Province in BCE 200."
    statement = "Commander marched from Alpha City to Beta Province in 200 BCE."
    claim = _movement_claim(statement)
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=statement, text=statement,
    )

    episode, detail = classify_legacy_claim_episode(claim, {evidence.id: evidence}, (query,))

    assert episode is EpisodeRelevance.DIRECT_QUERY_EPISODE
    assert detail["admitted"] is True


def test_route_level_contradictory_year_is_blocked():
    query = "Trace the route from Alpha City to Beta Province in 200 BCE."
    statement = "Commander marched from Alpha City to Beta Province in 100 BCE."
    event = _movement_event(statement, year_text="100 BCE")
    evidence = Evidence(
        id="ev1", author="Source", work="Work", locator="1", excerpt=statement, text=statement,
    )

    from backend.tests.test_g5f_relation_pipeline_integrity import relation

    same_movement = relation(
        "Alpha City",
        "Beta Province",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=(event.id,),
    )

    assert not relation_admission_allowed(
        same_movement,
        {event.id: event},
        {evidence.id: evidence},
        (query,),
        rule=OrderingRule.SAME_MOVEMENT_EVENT,
    )

    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        [event],
        [evidence],
        event_id="g6r",
        name="G6R",
        period="200 BCE",
        query_contexts=(query,),
    )
    same = [rel for rel in outcome.relations if rel.rule is OrderingRule.SAME_MOVEMENT_EVENT]
    assert same == []
    assert outcome.route is None or len(outcome.route.ordered_points) < 2
