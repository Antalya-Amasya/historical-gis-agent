"""V1B3: admit direct same-movement relations when episode is otherwise UNKNOWN."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app.models import (
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
)
from backend.app.routes.episode_relevance import (
    EpisodeRelevance,
    classify_event_anchor_episode,
)
from backend.app.routes.event_route_orchestration import (
    EventAnchorRouteBuilder,
    OrderingRule,
)
from backend.app.routes.evidence_relevance import (
    EvidenceRelevance,
    relation_admission_allowed,
)
from backend.tests.test_g4b_route_components import relation
from backend.tests.test_g5e2_relation_admission import _event, _evidence, _place
from backend.tests.test_g6w_episode_unknown_vs_other import ARISTON_CAMPAIGN_A_QUERY

QUERY = "Trace the route from Alpha Port to Beta Harbor."
MOVEMENT = (
    "The commander sailed from Alpha Port with the fleet and came to Beta Harbor "
    "to secure the harbor mouth."
)
OTHER_CAMPAIGN = "Bion marched from Alpha Port to Beta Harbor."
MISMATCH = "The commander marched from Gamma Bay to Delta Sound."

EID = "9d81ae3d968b243004d968dd420cd9d6:915:1212"
LIBO_QUERY = "Trace the route from Oricum to Brundisium."


def _movement_event(
    statement: str,
    *,
    origin: str,
    destination: str,
    evidence_id: str = "ev1",
) -> tuple[object, Evidence]:
    refs = [evidence_id]
    bindings = [
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(
                raw_text=origin, role=EventPlaceRole.ORIGIN, evidence_refs=refs,
            ),
            place=_place(origin),
            role=EventPlaceRole.ORIGIN,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=refs,
            resolver_provenance="test",
        ),
        HistoricalEventPlaceBinding(
            mention=HistoricalEventPlaceMention(
                raw_text=destination, role=EventPlaceRole.DESTINATION, evidence_refs=refs,
            ),
            place=_place(destination),
            role=EventPlaceRole.DESTINATION,
            resolution_status=EventPlaceResolutionStatus.RESOLVED,
            evidence_refs=refs,
            resolver_provenance="test",
        ),
    ]
    event = _event(
        "movement",
        statement,
        source_statements=[statement],
        refs=refs,
        bindings=bindings,
    )
    evidence = _evidence(evidence_id, statement)
    return event, evidence


def test_same_movement_direct_subject_unknown_episode_is_admitted():
    event, evidence = _movement_event(
        MOVEMENT,
        origin="Alpha Port",
        destination="Beta Harbor",
    )
    rel = relation(
        "Alpha Port",
        "Beta Harbor",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=(evidence.id,),
        event_ids=(event.id,),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert detail["admitted"] is True
    assert detail["admission_reason"] == "EPISODE_RELEVANT"
    assert episode in {EpisodeRelevance.UNKNOWN, EpisodeRelevance.DIRECT_QUERY_EPISODE}
    assert relation_admission_allowed(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (QUERY,),
        rule=rel.rule,
    )


def test_explicit_other_campaign_still_rejected():
    query = "Trace Ariston's route from Alpha Port to Beta Harbor."
    event, evidence = _movement_event(
        OTHER_CAMPAIGN,
        origin="Alpha Port",
        destination="Beta Harbor",
    )
    rel = relation(
        "Alpha Port",
        "Beta Harbor",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=(evidence.id,),
        event_ids=(event.id,),
    )
    episode, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (query,),
        subject_relevance=EvidenceRelevance.OTHER_CAMPAIGN,
    )
    assert detail["admitted"] is False
    assert not relation_admission_allowed(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (query,),
        rule=rel.rule,
    )


def test_endpoint_mismatch_still_rejected():
    event, evidence = _movement_event(
        MISMATCH,
        origin="Gamma Bay",
        destination="Delta Sound",
    )
    rel = relation(
        "Gamma Bay",
        "Delta Sound",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=(evidence.id,),
        event_ids=(event.id,),
    )
    _, detail = classify_event_anchor_episode(
        rel,
        {event.id: event},
        {evidence.id: evidence},
        (QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert detail["admitted"] is False


def test_cross_event_unknown_campaign_control_still_rejected():
    movement = "Ariston marched from Port Helios to Port Selene."
    event = _event("ariston", movement, source_statements=[movement], refs=["ev1"])
    evidence = _evidence("ev1", movement)
    rel = relation(
        "Port Helios",
        "Port Selene",
        OrderingRule.SAME_MOVEMENT_EVENT,
        refs=("ev1",),
        event_ids=("ariston",),
    )
    _, detail = classify_event_anchor_episode(
        rel,
        {"ariston": event},
        {evidence.id: evidence},
        (ARISTON_CAMPAIGN_A_QUERY,),
        subject_relevance=EvidenceRelevance.DIRECT_SUBJECT,
    )
    assert detail["admitted"] is False


def test_libo_oricum_brundisium_real_control_builds_route():
    from backend.app.rag.lexical_index import derive_passages
    from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
    from backend.app.routes.event_places import HistoricalEventPlaceResolver
    from geography_mcp.service import GeographyService

    parent = EID.split(":")[0]
    db = Path(r"C:/D/python/202608231533/data/chroma_server_roman_republic_v2/chroma.sqlite3")
    text = sqlite3.connect(db).execute(
        """
        SELECT em.string_value
        FROM embedding_metadata em
        JOIN embeddings e ON e.id = em.id
        WHERE em.key = 'chroma:document' AND e.embedding_id LIKE ?
        LIMIT 1
        """,
        (parent + "%",),
    ).fetchone()[0]
    passage = next(
        item
        for item in derive_passages(parent, text, {"source_chunk_id": parent, "document_id": parent})
        if item.id == EID
    )
    evidence = Evidence(
        id=passage.id,
        author="Caesar",
        work="Civil War",
        locator="XXIII",
        excerpt=passage.text[:500],
        text=passage.text,
        metadata=dict(passage.metadata),
    )

    class Geography:
        def call(self, tool: str, arguments: dict) -> dict:
            return GeographyService().resolve_ancient_place_payload(arguments["name"])

    extractor = EvidenceGroundedHistoricalEventExtractor()
    candidates, _ = extractor.extract([evidence], query_contexts=(LIBO_QUERY,))
    events, _ = HistoricalEventConsolidator().consolidate(candidates)
    events, _ = HistoricalEventPlaceResolver(Geography()).resolve(events)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        events,
        [evidence],
        event_id="v1b3-libo",
        name="Libo",
        period="49 BCE",
        query_contexts=(LIBO_QUERY,),
    )
    assert outcome.route is not None
    assert len(outcome.relations) == 1
    assert outcome.relations[0].rule is OrderingRule.SAME_MOVEMENT_EVENT
    assert outcome.relations[0].earlier == "Orikon"
    assert outcome.relations[0].later == "Brundisium"
    waypoint_names = [point.historical_place.canonical_name for point in outcome.route.ordered_points]
    assert waypoint_names[0] in {"Orikon", "Oricum"}
    assert waypoint_names[-1] == "Brundisium"
