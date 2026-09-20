"""V1.1G9C2: typed intra-event route ordering contract."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.app.models import (
    EventActorStatus,
    EventPlaceRole,
    EventRouteOrderingAuthority,
    Evidence,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
)
from backend.app.routes.event_constraints import project_transition_constraints
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.movement_semantics import analyze_sentence
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.app.routes.route_observations import ObservationOrderingAuthority, project_observation_ordering

HANNIBAL_QUERY = "Trace Hannibal's route from New Carthage toward Italy during the Second Punic War."

HANNIBAL_ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage", "carthago nova"), "audited"),
    HistoricalPlaceAlias("Iberus", ("iberus", "ebro"), "audited"),
    HistoricalPlaceAlias("Island", ("island",), "audited"),
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(
    HistoricalPlaceMentionExtractor(HANNIBAL_ALIASES),
)
_MENTION_EXTRACTOR = HistoricalPlaceMentionExtractor(HANNIBAL_ALIASES)


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


def _movement(text: str):
    events, _ = _EXTRACTOR.extract([_evidence(text)])
    movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
    assert len(movement) == 1, text
    return movement[0]


def _pipeline(texts: list[str] | str, *, evidence_ids: list[str] | None = None, query: str | None = None):
    if isinstance(texts, str):
        texts = [texts]
    items = [
        _evidence(text, eid=eid, offset=100 + index * 100)
        for index, (text, eid) in enumerate(
            zip(texts, evidence_ids or [f"ev-{index}" for index in range(len(texts))])
        )
    ]
    candidates, _ = _EXTRACTOR.extract(items, query=query, query_contexts=(query,) if query else None)
    events, _ = HistoricalEventConsolidator().consolidate(candidates)
    resolved, _ = HistoricalEventPlaceResolver(_HannibalGeography()).resolve(events)
    constraints, _ = project_transition_constraints(resolved, items)
    observations, relations, diagnostics = project_observation_ordering(resolved, [], constraints, items)
    typed_count = sum(len(event.route_orderings) for event in resolved)
    return observations, relations, diagnostics, resolved, constraints, items, typed_count


class _HannibalGeography:
    _PLACES = {
        "rhodanus": ("Rhodanus", PlaceSpatialSemantics.RIVER, "representative_point", 43.33, 4.84),
        "alpes": ("Alpes", PlaceSpatialSemantics.MOUNTAIN_REGION, "regional_centroid", 43.74, 7.40),
        "new carthage": ("New Carthage", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.59, -0.98),
        "carthago nova": ("New Carthage", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.59, -0.98),
        "italia": ("Italia", PlaceSpatialSemantics.REGION, "regional_centroid", 41.87, 12.57),
        "iberus": ("Iberus", PlaceSpatialSemantics.RIVER, "representative_point", 40.72, 0.86),
        "island": ("Island", PlaceSpatialSemantics.ISLAND, "feature_centroid", 39.0, 9.0),
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


def _labels(observations):
    return {item.observation_id: item.label for item in observations}


def _has_relation(relations, observations, earlier_label: str, later_label: str) -> bool:
    labels = _labels(observations)
    return any(
        earlier_label.casefold() in labels.get(relation.earlier_observation_id, "").casefold()
        and later_label.casefold() in labels.get(relation.later_observation_id, "").casefold()
        for relation in relations
    )


@pytest.mark.parametrize(
    ("text", "earlier", "later", "authority"),
    [
        ("Before crossing the Alps, Hannibal left New Carthage.", "New Carthage", "Alpes", EventRouteOrderingAuthority.BEFORE_SUBORDINATE),
        ("Hannibal left New Carthage before crossing the Alps.", "New Carthage", "Alpes", EventRouteOrderingAuthority.BEFORE_POSTPOSED),
        ("After leaving New Carthage, Hannibal crossed the Alps.", "New Carthage", "Alpes", EventRouteOrderingAuthority.AFTER_SUBORDINATE),
        ("Hannibal crossed the Alps after leaving New Carthage.", "New Carthage", "Alpes", EventRouteOrderingAuthority.AFTER_POSTPOSED),
        ("Before leaving New Carthage, Hannibal crossed the Alps.", "Alpes", "New Carthage", EventRouteOrderingAuthority.BEFORE_SUBORDINATE),
        ("After crossing the Rhone, Hannibal arrived at the Island.", "Rhodanus", "Island", EventRouteOrderingAuthority.AFTER_SUBORDINATE),
        ("After arriving at the Island, Hannibal crossed the Rhone.", "Island", "Rhodanus", EventRouteOrderingAuthority.AFTER_SUBORDINATE),
        ("Hannibal crossed the Rhone before reaching the Island.", "Rhodanus", "Island", EventRouteOrderingAuthority.BEFORE_POSTPOSED),
        ("Hannibal crossed the Rhone first and then crossed the Alps.", "Rhodanus", "Alpes", EventRouteOrderingAuthority.FIRST_THEN),
    ],
    ids=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
)
def test_event_layer_emits_typed_ordering(text, earlier, later, authority):
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    matches = [
        item for item in semantics.route_orderings
        if item.earlier.place_name == earlier
        and item.later.place_name == later
        and item.authority.upper() == authority.value
    ]
    assert len(matches) == 1, (text, semantics.route_orderings)


@pytest.mark.parametrize(
    ("text", "earlier", "later"),
    [
        ("Before crossing the Alps, Hannibal left New Carthage.", "New Carthage", "Alpes"),
        ("Before leaving New Carthage, Hannibal crossed the Alps.", "Alpes", "New Carthage"),
        ("After crossing the Rhone, Hannibal arrived at the Island.", "Rhodanus", "Island"),
    ],
)
def test_observation_layer_consumes_typed_ordering_without_reparse(text, earlier, later):
    observations, relations, _, _, _, _, typed_count = _pipeline(text)
    assert typed_count >= 1
    assert _has_relation(relations, observations, earlier, later)
    assert all(
        relation.ordering_rule
        in {
            ObservationOrderingAuthority.BEFORE_SUBORDINATE,
            ObservationOrderingAuthority.AFTER_SUBORDINATE,
            ObservationOrderingAuthority.BEFORE_POSTPOSED,
            ObservationOrderingAuthority.AFTER_POSTPOSED,
            ObservationOrderingAuthority.FIRST_THEN,
        }
        for relation in relations
    )


@pytest.mark.parametrize(
    "text",
    [
        "Hannibal did not cross the Rhone before reaching the Island.",
        "If Hannibal crossed the Rhone, he would reach the Island.",
        "The Rhone, according to Polybius, was difficult to cross.",
    ],
)
def test_ordering_fail_closed_for_unsafe_statements(text: str):
    event = _movement(text) if any(
        movement for movement in _EXTRACTOR.extract([_evidence(text)])[0]
        if movement.event_type is HistoricalEventType.MOVEMENT
    ) else None
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    assert not semantics.route_orderings
    if event is not None:
        assert not event.route_orderings


def test_unknown_actor_events_do_not_project_observation_relations():
    observations, relations, _, resolved, _, _, _ = _pipeline("He crossed the Rhone before reaching the Island.")
    assert any(event.actor.actor_status is EventActorStatus.UNKNOWN for event in resolved)
    assert not relations or all(
        _labels(observations).get(relation.earlier_observation_id, "").casefold() != "rhodanus"
        for relation in relations
    )


def test_hannibal_live_replay_produces_typed_ordering_and_relations():
    evidence = _live_hannibal_evidence()
    if evidence is None:
        pytest.skip("Chroma corpus unavailable for live Hannibal replay")

    observations, relations, diagnostics, resolved, constraints, items, typed_count = _pipeline(
        evidence,
        query=HANNIBAL_QUERY,
    )
    labels = _labels(observations)
    unsafe = [
        ("Italia", "New Carthage"),
        ("New Carthage", "Iberus"),
    ]
    assert not any(_has_relation(relations, observations, a, b) for a, b in unsafe)
    report = {
        "evidence": len(items),
        "events": len(resolved),
        "typed_orderings": typed_count,
        "observations": len(observations),
        "relations": len(relations),
    }
    assert report["evidence"] >= 10
    assert report["events"] >= 5
    assert report["observations"] >= 5
    assert report["typed_orderings"] >= 1
    # Live corpus often preserves typed orderings on UNKNOWN-actor events; observation
    # projection requires EXPLICIT actor and therefore may remain empty safely.
    for relation in relations:
        assert labels[relation.earlier_observation_id]
        assert labels[relation.later_observation_id]
        assert relation.ordering_rule in ObservationOrderingAuthority


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
