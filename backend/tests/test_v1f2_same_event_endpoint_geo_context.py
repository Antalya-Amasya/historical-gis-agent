"""V1F2: opposite movement endpoints must not basin-veto each other."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.app.geography.place_disambiguation import (
    PlaceResolutionContext,
    ResolvedCoMention,
    filter_resolution_candidates,
)
from backend.app.models import (
    EventGroundingStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEvent,
    HistoricalEventPlaceMention,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
)
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.rag.lexical_index import derive_passages
from geography_mcp.service import GeographyService

LIBO_EID = "9d81ae3d968b243004d968dd420cd9d6:915:1212"
LIBO_QUERY = "Trace Libo's route from Oricum to Brundisium."


def _place(
    *,
    pleiades_id: str,
    name: str,
    latitude: float,
    longitude: float,
) -> HistoricalPlace:
    return HistoricalPlace(
        id=f"pleiades-{pleiades_id}",
        canonical_name=name,
        latitude=latitude,
        longitude=longitude,
        source="test",
        source_id=pleiades_id,
        confidence=0.9,
        coordinate_role="exact_site",
        spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
    )


class _FilterBackedGeography:
    """Generic distant-basin fixture without hard-coded Libo/Oricum names."""

    def __init__(self) -> None:
        self.places = {
            "Port Alpha": _place(pleiades_id="100001", name="Port Alpha", latitude=40.6, longitude=17.9),
            "Port Beta": _place(pleiades_id="100002", name="Port Beta", latitude=32.8, longitude=13.2),
            "Scene Alpha": _place(pleiades_id="100003", name="Scene Alpha", latitude=40.6, longitude=17.9),
            "Scene Beta": _place(pleiades_id="100004", name="Scene Beta", latitude=32.8, longitude=13.2),
            "Shared Origin": _place(pleiades_id="100005", name="Shared Origin", latitude=40.6, longitude=17.9),
            "Shared Destination": _place(pleiades_id="100006", name="Shared Destination", latitude=32.8, longitude=13.2),
            "Regional Alpha": _place(pleiades_id="100007", name="Regional Alpha", latitude=40.6, longitude=17.9),
        }
        self.calls: list[dict] = []

    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        self.calls.append(dict(arguments))
        place = self.places.get(arguments["name"])
        if place is None:
            return {"found": False, "status": "NOT_FOUND"}
        context = PlaceResolutionContext(
            period=arguments.get("period"),
            source_statement=arguments.get("source_statement"),
            place_role=arguments.get("place_role"),
            co_mentions=tuple(arguments.get("co_mentions") or ()),
            resolved_co_mentions=tuple(
                ResolvedCoMention(
                    name=str(item["name"]),
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    spatial_semantics=item.get("spatial_semantics"),
                )
                for item in (arguments.get("resolved_co_mentions") or [])
            ),
        )
        candidate = {
            "pleiades_id": place.source_id,
            "canonical_name": place.canonical_name,
            "place_types": ["settlement"],
            "coordinate_available": True,
            "matched_name_details": [],
        }
        status, places, _, diagnostics = filter_resolution_candidates(
            status="CURATED",
            places=(place,),
            candidates=(candidate,),
            context=context,
        )
        if status == "UNIQUE" and len(places) == 1:
            return {"found": True, **places[0].model_dump(mode="json")}
        return {
            "found": False,
            "status": "AMBIGUOUS",
            "ambiguous": True,
            "candidate_count": 1,
            "disambiguation_diagnostics": diagnostics,
        }


def _movement_event(
    *,
    summary: str,
    origin: str,
    destination: str,
    related: str | None = None,
) -> HistoricalEvent:
    mentions = [
        HistoricalEventPlaceMention(
            raw_text=origin,
            canonical_hint=origin,
            role=EventPlaceRole.ORIGIN,
            evidence_refs=["ev1"],
        ),
    ]
    if related is not None:
        mentions.append(
            HistoricalEventPlaceMention(
                raw_text=related,
                canonical_hint=related,
                role=EventPlaceRole.RELATED_PLACE,
                evidence_refs=["ev1"],
            )
        )
    mentions.append(
        HistoricalEventPlaceMention(
            raw_text=destination,
            canonical_hint=destination,
            role=EventPlaceRole.DESTINATION,
            evidence_refs=["ev1"],
        )
    )
    return HistoricalEvent(
        id="movement-1",
        name="movement-1",
        summary=summary,
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev1"],
        source_statements=[summary],
        grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
        place_mentions=mentions,
    )


def test_same_event_opposite_endpoint_is_not_used_as_geo_context():
    geo = _FilterBackedGeography()
    event = _movement_event(
        summary="Actor sailed from Port Alpha to Port Beta.",
        origin="Port Alpha",
        destination="Port Beta",
    )

    resolved, diagnostics = HistoricalEventPlaceResolver(geo).resolve([event])
    bindings = {binding.role: binding for binding in resolved[0].place_bindings}

    assert bindings[EventPlaceRole.ORIGIN].resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert bindings[EventPlaceRole.DESTINATION].resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert diagnostics["ambiguous_count"] == 0

    destination_call = next(call for call in geo.calls if call["name"] == "Port Beta")
    assert destination_call["resolved_co_mentions"] == []


def test_ordinary_non_movement_co_mention_context_unchanged():
    geo = _FilterBackedGeography()
    summary = "The battle occurred near Scene Alpha while forces gathered at Scene Beta."
    event = HistoricalEvent(
        id="battle-1",
        name="battle-1",
        summary=summary,
        event_type=HistoricalEventType.BATTLE,
        evidence_refs=["ev1"],
        source_statements=[summary],
        grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
        place_mentions=[
            HistoricalEventPlaceMention(
                raw_text="Scene Alpha",
                canonical_hint="Scene Alpha",
                role=EventPlaceRole.EVENT_SITE,
                evidence_refs=["ev1"],
            ),
            HistoricalEventPlaceMention(
                raw_text="Scene Beta",
                canonical_hint="Scene Beta",
                role=EventPlaceRole.RELATED_PLACE,
                evidence_refs=["ev1"],
            ),
        ],
    )

    resolved, _ = HistoricalEventPlaceResolver(geo).resolve([event])
    bindings = {binding.role: binding for binding in resolved[0].place_bindings}
    assert bindings[EventPlaceRole.EVENT_SITE].resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert bindings[EventPlaceRole.RELATED_PLACE].resolution_status is EventPlaceResolutionStatus.AMBIGUOUS


def test_same_role_co_mentions_remain_contextual():
    geo = _FilterBackedGeography()
    event = HistoricalEvent(
        id="multi-origin",
        name="multi-origin",
        summary="Forces moved from Shared Origin toward Shared Destination.",
        event_type=HistoricalEventType.MOVEMENT,
        evidence_refs=["ev1"],
        source_statements=["Forces moved from Shared Origin toward Shared Destination."],
        grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
        place_mentions=[
            HistoricalEventPlaceMention(
                raw_text="Shared Origin",
                canonical_hint="Shared Origin",
                role=EventPlaceRole.ORIGIN,
                evidence_refs=["ev1"],
            ),
            HistoricalEventPlaceMention(
                raw_text="Shared Destination",
                canonical_hint="Shared Destination",
                role=EventPlaceRole.ORIGIN,
                evidence_refs=["ev1"],
            ),
        ],
    )

    resolved, _ = HistoricalEventPlaceResolver(geo).resolve([event])
    second_call = geo.calls[1]
    assert second_call["resolved_co_mentions"]
    assert resolved[0].place_bindings[1].resolution_status is EventPlaceResolutionStatus.AMBIGUOUS


def test_related_place_may_still_constrain_destination():
    geo = _FilterBackedGeography()
    event = _movement_event(
        summary="Actor sailed from Port Alpha to Port Beta by way of Scene Alpha.",
        origin="Port Alpha",
        destination="Port Beta",
        related="Scene Alpha",
    )

    resolved, _ = HistoricalEventPlaceResolver(geo).resolve([event])
    destination_call = next(call for call in geo.calls if call["name"] == "Port Beta")
    bindings = {binding.role: binding for binding in resolved[0].place_bindings}
    assert destination_call["resolved_co_mentions"]
    assert destination_call["resolved_co_mentions"][0]["name"] == "Scene Alpha"
    assert bindings[EventPlaceRole.DESTINATION].resolution_status is EventPlaceResolutionStatus.AMBIGUOUS


def test_regional_semantics_are_not_promoted():
    geo = _FilterBackedGeography()
    geo.places["Regional Alpha"] = _place(
        pleiades_id="100007",
        name="Regional Alpha",
        latitude=40.6,
        longitude=17.9,
    ).model_copy(update={"coordinate_role": "regional_centroid", "spatial_semantics": PlaceSpatialSemantics.REGION})
    event = _movement_event(
        summary="The army marched from Regional Alpha to Port Beta.",
        origin="Regional Alpha",
        destination="Port Beta",
    )

    resolved, diagnostics = HistoricalEventPlaceResolver(geo).resolve([event])
    origin = resolved[0].place_bindings[0]
    assert origin.resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert origin.place and origin.place.coordinate_role == "regional_centroid"
    assert diagnostics["regional_count"] == 1


def test_libo_live_path_resolves_both_endpoints():
    class Geography:
        def call(self, tool: str, arguments: dict) -> dict:
            assert tool == "resolve_ancient_place"
            return GeographyService().resolve_ancient_place_payload(**arguments)

    db = Path(r"C:/D/python/202608231533/data/chroma_server_roman_republic_v2/chroma.sqlite3")
    text = sqlite3.connect(db).execute(
        """
        SELECT em.string_value
        FROM embedding_metadata em
        JOIN embeddings e ON e.id = em.id
        WHERE em.key = 'chroma:document' AND e.embedding_id LIKE ?
        LIMIT 1
        """,
        (LIBO_EID.split(":")[0] + "%",),
    ).fetchone()[0]
    passage = next(
        item
        for item in derive_passages(
            LIBO_EID.split(":")[0],
            text,
            {"source_chunk_id": LIBO_EID.split(":")[0], "document_id": LIBO_EID.split(":")[0]},
        )
        if item.id == LIBO_EID
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

    extractor = EvidenceGroundedHistoricalEventExtractor()
    events, _ = HistoricalEventConsolidator().consolidate(
        extractor.extract([evidence], query_contexts=(LIBO_QUERY,))[0]
    )
    resolved, _ = HistoricalEventPlaceResolver(Geography()).resolve(events)
    bindings = {binding.role: binding for binding in resolved[0].place_bindings}

    assert bindings[EventPlaceRole.ORIGIN].resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert bindings[EventPlaceRole.ORIGIN].place
    assert bindings[EventPlaceRole.ORIGIN].place.canonical_name == "Orikon"
    assert bindings[EventPlaceRole.ORIGIN].place.coordinate_role == "exact_site"

    assert bindings[EventPlaceRole.DESTINATION].resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert bindings[EventPlaceRole.DESTINATION].place
    assert bindings[EventPlaceRole.DESTINATION].place.canonical_name == "Brundisium"
    assert bindings[EventPlaceRole.DESTINATION].place.source_id == "442509"
    assert bindings[EventPlaceRole.DESTINATION].place.coordinate_role == "exact_site"

    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        resolved,
        [evidence],
        event_id="v1f2-libo",
        name="Libo",
        period="49 BCE",
        query_contexts=(LIBO_QUERY,),
    )
    assert outcome.route is not None
    assert [point.historical_place.canonical_name for point in outcome.route.ordered_points] == [
        "Orikon",
        "Brundisium",
    ]
