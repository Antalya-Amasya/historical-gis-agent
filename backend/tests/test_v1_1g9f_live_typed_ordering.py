"""V1.1G9F: live typed ordering coverage for elapsed-march and first-then ellipsis."""
from __future__ import annotations

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

ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("Island", ("island",), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage",), "audited"),
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(HistoricalPlaceMentionExtractor(ALIASES))
_MENTION_EXTRACTOR = HistoricalPlaceMentionExtractor(ALIASES)


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


def _ordering(text: str, earlier: str, later: str):
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    matches = [
        item
        for item in semantics.route_orderings
        if item.earlier.place_name == earlier and item.later.place_name == later
    ]
    assert len(matches) == 1, (text, semantics.route_orderings)
    return matches[0]


class _HannibalGeography:
    _PLACES = {
        "rhodanus": ("Rhodanus", PlaceSpatialSemantics.RIVER, "representative_point", 43.33, 4.84),
        "alpes": ("Alpes", PlaceSpatialSemantics.MOUNTAIN_REGION, "regional_centroid", 43.74, 7.40),
        "new carthage": ("New Carthage", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.59, -0.98),
        "italia": ("Italia", PlaceSpatialSemantics.REGION, "regional_centroid", 41.87, 12.57),
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


def _pipeline(texts: list[str] | str, *, query: str | None = None):
    if isinstance(texts, str):
        texts = [texts]
    items = [
        _evidence(text, eid=f"ev-{index}", offset=100 + index * 100)
        for index, text in enumerate(texts)
    ]
    candidates, _ = _EXTRACTOR.extract(items, query=query, query_contexts=(query,) if query else None)
    events, _ = HistoricalEventConsolidator().consolidate(candidates)
    resolved, _ = HistoricalEventPlaceResolver(_HannibalGeography()).resolve(events)
    constraints, _ = project_transition_constraints(resolved, items)
    observations, relations, diagnostics = project_observation_ordering(resolved, [], constraints, items)
    typed_count = sum(len(event.route_orderings) for event in resolved)
    return observations, relations, diagnostics, resolved, constraints, items, typed_count, len(candidates)


def _labels(observations):
    return {item.observation_id: item.label for item in observations}


def _has_relation(relations, observations, earlier_label: str, later_label: str) -> bool:
    labels = _labels(observations)
    return any(
        earlier_label.casefold() in labels.get(relation.earlier_observation_id, "").casefold()
        and later_label.casefold() in labels.get(relation.later_observation_id, "").casefold()
        for relation in relations
    )


def test_elapsed_march_from_feature_orders_rhone_before_island():
    text = (
        "After four days' march from the passage of the Rhone, "
        "Hannibal arrived at the place called the Island."
    )
    event = _movement(text)
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    roles = {endpoint.place_name: endpoint.role for endpoint in semantics.endpoints}
    assert roles.get("Rhodanus") in {"origin", "traversal"}
    assert roles.get("Island") == "destination"
    match = _ordering(text, "Rhodanus", "Island")
    assert match.authority == "after_subordinate"
    observations, relations, _, _, _, _, typed_count, _ = _pipeline(text)
    assert typed_count >= 1
    assert _has_relation(relations, observations, "Rhodanus", "Island")


@pytest.mark.parametrize(
    "text",
    [
        "Hannibal was first in crossing the Rhone, then the Alps.",
        "Hannibal crossed the Rhone first, then the Alps.",
        "He was first in crossing the Rhone, then the Alps.",
    ],
)
def test_first_then_ellipsis_orders_rhone_before_alps(text: str):
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    assert any(
        item.earlier.place_name == "Rhodanus" and item.later.place_name == "Alpes"
        for item in semantics.route_orderings
    )
    event = _movement(text)
    assert any(
        ordering.earlier.canonical == "Rhodanus" and ordering.later.canonical == "Alpes"
        for ordering in event.route_orderings
    )


@pytest.mark.parametrize(
    ("text", "earlier", "later"),
    [
        (
            "After three days' march from the passage of Saguntum, Ariston arrived at Capua.",
            "Saguntum",
            "Capua",
        ),
        (
            "After a day's march from the passage of Saguntum, the army reached Capua.",
            "Saguntum",
            "Capua",
        ),
    ],
)
def test_elapsed_march_semantic_shape_variants(text: str, earlier: str, later: str):
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    assert semantics.route_orderings
    assert semantics.route_orderings[0].earlier.surface.casefold() == earlier.casefold()
    assert semantics.route_orderings[0].later.surface.casefold() == later.casefold()


@pytest.mark.parametrize(
    "text",
    [
        "If Hannibal crossed the Rhone, he would then reach the Alps.",
        "Hannibal did not cross the Rhone before reaching the Island.",
        "The Rhone was mentioned first, then the Alps.",
        "According to Polybius, the Rhone passage was described before the Alps.",
    ],
)
def test_ordering_fail_closed_for_discourse_and_modality(text: str):
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    assert not semantics.route_orderings
    events, _ = _EXTRACTOR.extract([_evidence(text)])
    movement = [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]
    if movement:
        assert not movement[0].route_orderings


def test_unknown_actor_first_then_does_not_project_observation_relation():
    observations, relations, _, resolved, _, _, _, _ = _pipeline(
        "Hannibal was first in crossing the Rhone, then the Alps."
    )
    event = next(item for item in resolved if item.event_type is HistoricalEventType.MOVEMENT)
    assert event.actor.actor_status is EventActorStatus.UNKNOWN
    assert event.route_orderings
    assert not _has_relation(relations, observations, "Rhodanus", "Alpes")


def test_live_hannibal_replay_reports_orderings_and_relations():
    texts = _live_hannibal_evidence()
    if texts is None:
        pytest.skip("Chroma corpus unavailable for live Hannibal replay")
    observations, relations, _, resolved, _, items, typed_count, pre_count = _pipeline(
        texts,
        query=HANNIBAL_QUERY,
    )
    explicit_hannibal = sum(
        1
        for event in resolved
        if event.event_type is HistoricalEventType.MOVEMENT
        and event.actor.actor_status is EventActorStatus.EXPLICIT
        and event.actor.actor_text
        and "hannibal" in event.actor.actor_text.casefold()
    )
    movement_count = sum(1 for event in resolved if event.event_type is HistoricalEventType.MOVEMENT)
    print(
        {
            "evidence": len(items),
            "pre": pre_count,
            "post": len(resolved),
            "movement": movement_count,
            "typed": typed_count,
            "explicit_hannibal": explicit_hannibal,
            "observations": len(observations),
            "relations": len(relations),
        }
    )
    assert len(items) >= 10
    assert typed_count >= 3
    assert explicit_hannibal >= 3
    island_events = [
        event
        for event in resolved
        if event.event_type is HistoricalEventType.MOVEMENT
        and "island" in " ".join(event.source_statements).casefold()
        and event.actor.actor_status is EventActorStatus.EXPLICIT
        and event.actor.actor_text
        and "hannibal" in event.actor.actor_text.casefold()
    ]
    assert island_events
    assert any(
        ordering.earlier.canonical == "Rhodanus" and ordering.later.canonical == "Island"
        for event in island_events
        for ordering in event.route_orderings
    )
    assert _has_relation(relations, observations, "Rhodanus", "Island")


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
