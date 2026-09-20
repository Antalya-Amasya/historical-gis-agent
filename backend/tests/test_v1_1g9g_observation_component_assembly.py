"""V1.1G9G: observation semantic component assembly."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.models import (
    EventActorStatus,
    EventPlaceRole,
    Evidence,
    GeographicFeatureKind,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
)
from backend.app.routes.event_constraints import project_transition_constraints
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.observation_components import assemble_observation_components
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.app.routes.route_observations import (
    ObservationOrderingAuthority,
    ObservationOrderingRelation,
    RouteObservation,
    RouteObservationKind,
    project_observation_ordering,
)

HANNIBAL_QUERY = "Trace Hannibal's route from New Carthage toward Italy during the Second Punic War."

ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("Island", ("island",), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage",), "audited"),
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(HistoricalPlaceMentionExtractor(ALIASES))


def _evidence(text: str, *, eid: str = "ev", offset: int = 100) -> Evidence:
    return Evidence(
        id=eid,
        author="Polybius",
        work="Histories",
        locator="III",
        excerpt=text,
        text=text,
        metadata={"document_id": "doc-1", "spine_index": 1, "start_offset": offset},
    )


def _observation(
    observation_id: str,
    *,
    kind: RouteObservationKind,
    label: str,
    actor_text: str | None,
    actor_status: EventActorStatus,
    event_id: str = "e1",
    evidence_refs: tuple[str, ...] = ("ev-1",),
    place_role: EventPlaceRole | None = None,
    coordinate_role: str | None = None,
) -> RouteObservation:
    return RouteObservation(
        observation_id=observation_id,
        kind=kind,
        event_id=event_id,
        label=label,
        actor_text=actor_text,
        actor_status=actor_status,
        evidence_refs=evidence_refs,
        place_role=place_role,
        coordinate_role=coordinate_role,
        feature_kind=GeographicFeatureKind.RIVER if kind is RouteObservationKind.TRANSITION else None,
    )


def _relation(
    earlier: str,
    later: str,
    *,
    event_ids: tuple[str, ...] = ("e1",),
    authority: ObservationOrderingAuthority = ObservationOrderingAuthority.AFTER_SUBORDINATE,
) -> ObservationOrderingRelation:
    return ObservationOrderingRelation(
        earlier_observation_id=earlier,
        later_observation_id=later,
        ordering_rule=authority,
        event_ids=event_ids,
        evidence_refs=("ev-1",),
        authority=authority.value,
    )


def _assemble(observations, relations, events=None, evidence=None):
    return assemble_observation_components(
        observations,
        relations,
        events or [],
        evidence or [_evidence("x", eid="ev-1")],
    )


def test_same_actor_linear_chain_produces_three_node_component():
    observations = [
        _observation("a", kind=RouteObservationKind.TRANSITION, label="Alpha", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("b", kind=RouteObservationKind.PLACE, label="Beta", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT, place_role=EventPlaceRole.DESTINATION),
        _observation("c", kind=RouteObservationKind.PLACE, label="Gamma", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT, place_role=EventPlaceRole.DESTINATION),
    ]
    relations = [_relation("a", "b"), _relation("b", "c")]
    assembly = _assemble(observations, relations)
    assert len(assembly.components) == 1
    assert assembly.components[0].observation_ids == ("a", "b", "c")
    assert assembly.components[0].actor_text == "Hannibal"
    assert assembly.components[0].route_complete is False


def test_different_explicit_actors_reject_component_edge():
    observations = [
        _observation("a", kind=RouteObservationKind.PLACE, label="Alpha", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("b", kind=RouteObservationKind.PLACE, label="Beta", actor_text="Scipio", actor_status=EventActorStatus.EXPLICIT),
    ]
    assembly = _assemble(observations, [_relation("a", "b")])
    assert not assembly.components
    assert any(item["reason"] == "ACTOR_AUTHORITY_REJECTED" for item in assembly.rejected_edges)


def test_unknown_actor_rejects_component_edge():
    observations = [
        _observation("a", kind=RouteObservationKind.TRANSITION, label="Rhodanus", actor_text=None, actor_status=EventActorStatus.UNKNOWN),
        _observation("b", kind=RouteObservationKind.PLACE, label="Island", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
    ]
    assembly = _assemble(observations, [_relation("a", "b")])
    assert not assembly.components
    assert any(item["reason"] == "ACTOR_AUTHORITY_REJECTED" for item in assembly.rejected_edges)


def test_disconnected_graph_produces_two_components():
    observations = [
        _observation("a", kind=RouteObservationKind.PLACE, label="Alpha", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("b", kind=RouteObservationKind.PLACE, label="Beta", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("c", kind=RouteObservationKind.PLACE, label="Charlie", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("d", kind=RouteObservationKind.PLACE, label="Delta", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
    ]
    relations = [_relation("a", "b"), _relation("c", "d")]
    assembly = _assemble(observations, relations)
    assert len(assembly.components) == 2
    chains = {component.observation_ids for component in assembly.components}
    assert ("a", "b") in chains
    assert ("c", "d") in chains


def test_branching_graph_is_not_linearized():
    observations = [
        _observation("a", kind=RouteObservationKind.PLACE, label="Alpha", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("b", kind=RouteObservationKind.PLACE, label="Beta", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("c", kind=RouteObservationKind.PLACE, label="Charlie", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
    ]
    relations = [_relation("a", "b"), _relation("c", "b")]
    assembly = _assemble(observations, relations)
    assert "OBSERVATION_ORDERING_BRANCH" in assembly.diagnostics
    assert all(len(component.observation_ids) == 2 for component in assembly.components)


def test_cycle_produces_no_ordered_component():
    observations = [
        _observation("a", kind=RouteObservationKind.PLACE, label="Alpha", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("b", kind=RouteObservationKind.PLACE, label="Beta", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
    ]
    relations = [_relation("a", "b"), _relation("b", "a")]
    assembly = _assemble(observations, relations)
    assert "OBSERVATION_ORDERING_CYCLE" in assembly.diagnostics
    assert not assembly.components


def test_duplicate_equivalent_edge_merges_into_one_component():
    observations = [
        _observation("a", kind=RouteObservationKind.TRANSITION, label="Rhodanus", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("b", kind=RouteObservationKind.PLACE, label="Island", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
    ]
    relations = [
        _relation("a", "b"),
        _relation("a", "b"),
    ]
    assembly = _assemble(observations, relations)
    assert len(assembly.components) == 1


def test_same_pair_conflicting_authority_rejects_component():
    observations = [
        _observation("a", kind=RouteObservationKind.PLACE, label="Alpha", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
        _observation("b", kind=RouteObservationKind.PLACE, label="Beta", actor_text="Hannibal", actor_status=EventActorStatus.EXPLICIT),
    ]
    relations = [
        _relation("a", "b", authority=ObservationOrderingAuthority.AFTER_SUBORDINATE),
        _relation("a", "b", authority=ObservationOrderingAuthority.TEMPORAL_ORDER),
    ]
    assembly = _assemble(observations, relations)
    assert any(item["reason"] == "CONFLICTING_ORDERING_AUTHORITY" for item in assembly.rejected_edges)


def test_mixed_place_and_transition_component():
    text = (
        "After four days' march from the passage of the Rhone, "
        "Hannibal arrived at the place called the Island."
    )
    events, _ = _EXTRACTOR.extract([_evidence(text)])
    resolved, _ = HistoricalEventPlaceResolver(_HannibalGeography()).resolve(events)
    constraints, _ = project_transition_constraints(resolved, [_evidence(text)])
    observations, relations, _ = project_observation_ordering(resolved, [], constraints, [_evidence(text)])
    assembly = assemble_observation_components(observations, relations, resolved, [_evidence(text)])
    assert len(assembly.components) == 1
    component = assembly.components[0]
    obs_by_id = {item.observation_id: item for item in observations}
    labels = [obs_by_id[oid].label for oid in component.observation_ids]
    roles = [obs_by_id[oid].place_role for oid in component.observation_ids]
    assert "Rhodanus" in labels
    assert "Island" in labels
    assert EventPlaceRole.ORIGIN in roles
    assert EventPlaceRole.DESTINATION in roles


def test_crossing_semantics_mixed_transition_and_place_component():
    text = "After crossing the Rhone, Hannibal arrived at the Island."
    events, _ = _EXTRACTOR.extract([_evidence(text)])
    resolved, _ = HistoricalEventPlaceResolver(_HannibalGeography()).resolve(events)
    constraints, _ = project_transition_constraints(resolved, [_evidence(text)])
    observations, relations, _ = project_observation_ordering(resolved, [], constraints, [_evidence(text)])
    assembly = assemble_observation_components(observations, relations, resolved, [_evidence(text)])
    assert len(assembly.components) == 1
    obs_by_id = {item.observation_id: item for item in observations}
    kinds = [obs_by_id[oid].kind for oid in assembly.components[0].observation_ids]
    assert RouteObservationKind.TRANSITION in kinds
    assert RouteObservationKind.PLACE in kinds


def test_regional_place_remains_non_exact_in_component():
    observations = [
        _observation(
            "a",
            kind=RouteObservationKind.TRANSITION,
            label="Rhodanus",
            actor_text="Hannibal",
            actor_status=EventActorStatus.EXPLICIT,
        ),
        _observation(
            "b",
            kind=RouteObservationKind.PLACE,
            label="Italia",
            actor_text="Hannibal",
            actor_status=EventActorStatus.EXPLICIT,
            place_role=EventPlaceRole.DESTINATION,
            coordinate_role="regional_centroid",
        ),
    ]
    assembly = _assemble(observations, [_relation("a", "b")])
    assert len(assembly.components) == 1
    assert observations[1].coordinate_role == "regional_centroid"


class _HannibalGeography:
    _PLACES = {
        "rhodanus": ("Rhodanus", PlaceSpatialSemantics.RIVER, "representative_point", 43.33, 4.84),
        "island": ("Island", PlaceSpatialSemantics.ISLAND, "feature_centroid", 39.0, 9.0),
        "italia": ("Italia", PlaceSpatialSemantics.REGION, "regional_centroid", 41.87, 12.57),
        "new carthage": ("New Carthage", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.59, -0.98),
        "alpes": ("Alpes", PlaceSpatialSemantics.MOUNTAIN_REGION, "regional_centroid", 43.74, 7.40),
    }

    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        key = arguments["name"].casefold()
        for alias, (name, semantics, role, lat, lon) in self._PLACES.items():
            if key in {alias, name.casefold()}:
                place = HistoricalPlace(
                    id=alias,
                    canonical_name=name,
                    latitude=lat,
                    longitude=lon,
                    source="fixture",
                    confidence=0.8,
                    spatial_semantics=semantics,
                    coordinate_role=role,
                )
                payload = place.model_dump()
                payload["found"] = True
                return payload
        return {"found": False}


def _live_pipeline():
    texts = _live_hannibal_evidence()
    if texts is None:
        pytest.skip("Chroma corpus unavailable for live Hannibal replay")
    items = [_evidence(text, eid=f"ev-{index}", offset=100 + index * 100) for index, text in enumerate(texts)]
    candidates, _ = _EXTRACTOR.extract(items, query=HANNIBAL_QUERY, query_contexts=(HANNIBAL_QUERY,))
    events, _ = HistoricalEventConsolidator().consolidate(candidates)
    resolved, _ = HistoricalEventPlaceResolver(_HannibalGeography()).resolve(events)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        resolved,
        items,
        event_id="hannibal-route",
        name="Hannibal",
        period="Second Punic War",
        query_contexts=(HANNIBAL_QUERY,),
    )
    return outcome, resolved, items


def test_live_hannibal_replay_does_not_assemble_isolated_rhodanus_island_component():
    """STALE_TEST_CONTRACT: isolated Rhodanus->Island must not form a route component alone."""
    outcome, _, _ = _live_pipeline()
    assert outcome.diagnostics["observation_relation_count"] >= 1
    obs_by = {item.observation_id: item for item in outcome.observations}
    for component in outcome.observation_components:
        labels = {obs_by[oid].label for oid in component.observation_ids}
        if "Rhodanus" in labels and "Island" in labels:
            pytest.fail("isolated Rhodanus->Island must not assemble without endpoint-anchored graph")


def _live_hannibal_evidence() -> list[str] | None:
    for path in (
        Path(r"C:\D\python\historical-gis-cursor\data\chroma_server_roman_republic_v2"),
        Path(r"C:\D\python\202608231533\data\chroma_server_roman_republic_v2"),
    ):
        if not path.exists():
            continue
        try:
            import chromadb
            from backend.app.core.config import settings
            from backend.app.rag.embeddings.provider import SentenceTransformerEmbeddingProvider
            from backend.app.rag.http_store import ChromaHttpEvidenceStore
            from backend.app.rag.query_bridge import HistoricalQueryBridge
            from backend.app.rag.retriever import ChromaHistoricalRetriever

            client = chromadb.PersistentClient(path=str(path))
            if settings.rag_collection not in [collection.name for collection in client.list_collections()]:
                continue
            collection = client.get_collection(settings.rag_collection)
            if collection.count() < 1000:
                continue
            provider = SentenceTransformerEmbeddingProvider(
                settings.rag_embedding_model,
                settings.rag_embedding_device,
                settings.rag_embedding_batch_size,
            )
            retriever = ChromaHistoricalRetriever(
                ChromaHttpEvidenceStore(collection, provider),
                HistoricalQueryBridge(settings.rag_query_bridge_enabled),
            )
            evidence = retriever.retrieve_with_coverage(HANNIBAL_QUERY)
        except Exception:
            continue
        if len(evidence) < 10:
            continue
        return [item.text or item.excerpt for item in evidence if item.text or item.excerpt]
    return None
