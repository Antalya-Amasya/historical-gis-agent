"""V1.1G9G-R2: query-relative episode admission for observation components."""
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
    HistoricalPlaceAlias("Ephesus", ("ephesus",), "audited"),
    HistoricalPlaceAlias("Graecia", ("greece", "graecia"), "audited"),
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


class _HannibalGeography:
    _PLACES = {
        "rhodanus": ("Rhodanus", PlaceSpatialSemantics.RIVER, "representative_point", 43.33, 4.84),
        "island": ("Island", PlaceSpatialSemantics.ISLAND, "feature_centroid", 39.0, 9.0),
        "italia": ("Italia", PlaceSpatialSemantics.REGION, "regional_centroid", 41.87, 12.57),
        "new carthage": ("New Carthage", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.59, -0.98),
        "alpes": ("Alpes", PlaceSpatialSemantics.MOUNTAIN_REGION, "regional_centroid", 43.74, 7.40),
        "ephesus": ("Ephesus", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.94, 27.34),
        "graecia": ("Graecia", PlaceSpatialSemantics.REGION, "regional_centroid", 39.0, 22.0),
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


def _observation(
    observation_id: str,
    *,
    kind: RouteObservationKind,
    label: str,
    actor_text: str | None,
    actor_status: EventActorStatus,
    event_id: str = "e1",
    evidence_refs: tuple[str, ...] = ("ev",),
    place_role: EventPlaceRole | None = None,
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
        feature_kind=GeographicFeatureKind.RIVER if kind is RouteObservationKind.TRANSITION else None,
    )


def _relation(
    earlier: str,
    later: str,
    *,
    event_ids: tuple[str, ...] = ("e1",),
    authority: ObservationOrderingAuthority = ObservationOrderingAuthority.AFTER_SUBORDINATE,
    evidence_refs: tuple[str, ...] = ("ev",),
) -> ObservationOrderingRelation:
    return ObservationOrderingRelation(
        earlier_observation_id=earlier,
        later_observation_id=later,
        ordering_rule=authority,
        event_ids=event_ids,
        evidence_refs=evidence_refs,
        authority=authority.value,
    )


def _assemble_from_text(text: str, *, query_contexts=HANNIBAL_QUERY):
    evidence = _evidence(text)
    events, _ = _EXTRACTOR.extract([evidence])
    resolved, _ = HistoricalEventPlaceResolver(_HannibalGeography()).resolve(events)
    constraints, _ = project_transition_constraints(resolved, [evidence])
    observations, relations, _ = project_observation_ordering(resolved, [], constraints, [evidence])
    assembly = assemble_observation_components(
        observations,
        relations,
        resolved,
        [evidence],
        query_contexts=(query_contexts,) if query_contexts else None,
    )
    return assembly, observations, relations, resolved, evidence


def _assemble_manual(
    text: str,
    earlier_label: str,
    later_label: str,
    *,
    query_contexts=HANNIBAL_QUERY,
):
    evidence = _evidence(text)
    events, _ = _EXTRACTOR.extract([evidence])
    resolved, _ = HistoricalEventPlaceResolver(_HannibalGeography()).resolve(events)
    event_id = resolved[0].id
    observations = [
        _observation(
            "a",
            kind=RouteObservationKind.PLACE,
            label=earlier_label,
            actor_text="Hannibal",
            actor_status=EventActorStatus.EXPLICIT,
            event_id=event_id,
            evidence_refs=(evidence.id,),
            place_role=EventPlaceRole.ORIGIN,
        ),
        _observation(
            "b",
            kind=RouteObservationKind.PLACE,
            label=later_label,
            actor_text="Hannibal",
            actor_status=EventActorStatus.EXPLICIT,
            event_id=event_id,
            evidence_refs=(evidence.id,),
            place_role=EventPlaceRole.DESTINATION,
        ),
    ]
    relations = [_relation("a", "b", event_ids=(event_id,), evidence_refs=(evidence.id,))]
    assembly = assemble_observation_components(
        observations,
        relations,
        resolved,
        [evidence],
        query_contexts=(query_contexts,) if query_contexts else None,
    )
    return assembly, observations, relations


def test_wrong_episode_exile_rejected():
    assembly, _, _ = _assemble_manual(
        "Later Hannibal left Italy for exile and went to Antiochus.",
        "Italia",
        "Ephesus",
    )
    assert not assembly.components
    assert any(item["reason"] == "QUERY_EPISODE_REJECTED" for item in assembly.rejected_edges)


def test_wrong_episode_antiochus_rejected():
    assembly, _, _ = _assemble_manual(
        "Hannibal with Antiochus marched from Ephesus toward Greece and Italy.",
        "Ephesus",
        "Graecia",
    )
    assert not assembly.components
    assert any(item["reason"] == "QUERY_EPISODE_REJECTED" for item in assembly.rejected_edges)


def test_positive_island_passage_admitted():
    text = (
        "After four days' march from the passage of the Rhone, "
        "Hannibal arrived at the place called the Island."
    )
    assembly, observations, relations, _, _ = _assemble_from_text(text)
    assert relations
    assert len(assembly.components) == 1
    obs_by_id = {item.observation_id: item for item in observations}
    labels = [obs_by_id[oid].label for oid in assembly.components[0].observation_ids]
    assert "Rhodanus" in labels
    assert "Island" in labels
    assert not any(item["reason"] == "QUERY_EPISODE_REJECTED" for item in assembly.rejected_edges)


def test_unknown_episode_fail_closed():
    assembly, _, _ = _assemble_manual(
        "Hannibal was at Rome during the period.",
        "Rome",
        "Capua",
    )
    assert not assembly.components
    assert any(
        item["reason"] in {
            "QUERY_EPISODE_REJECTED",
            "QUERY_EPISODE_UNKNOWN",
            "MOVEMENT_ASSERTION_REJECTED",
        }
        for item in assembly.rejected_edges
    )


def test_same_event_relation_requires_query_episode_authority():
    text = "Later Hannibal left Italy for exile and went to Antiochus."
    assembly, _, _ = _assemble_manual(text, "Italia", "Ephesus")
    assert all(len(relation.event_ids) == 1 for relation in _)
    assert not assembly.components


def test_intermediate_new_carthage_passage_still_admitted():
    text = "After leaving New Carthage, Hannibal crossed the Alps and came into Italy."
    assembly, observations, relations, _, _ = _assemble_from_text(text)
    assert relations
    assert assembly.components
    obs_by_id = {item.observation_id: item for item in observations}
    labels = {
        obs_by_id[oid].label
        for component in assembly.components
        for oid in component.observation_ids
    }
    assert "New Carthage" in labels
    assert "Alpes" in labels or "Italia" in labels


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


def test_live_hannibal_replay_keeps_rhodanus_island():
    outcome, _, _ = _live_pipeline()
    assert outcome.diagnostics["observation_relation_count"] >= 1
    assert outcome.diagnostics["observation_component_count"] >= 1
    observations_by_id = {item.observation_id: item for item in outcome.observations}
    accepted_labels = {
        observations_by_id[oid].label
        for component in outcome.observation_components
        for oid in component.observation_ids
    }
    assert "Rhodanus" in accepted_labels
    assert "Island" in accepted_labels


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
        if evidence:
            return [item.text for item in evidence[:20]]
    return None
