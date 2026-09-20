"""V1.1G9G-R1: live ordering endpoint selection — predicate-owned destinations."""
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
from backend.app.routes.movement_semantics import (
    MovementEndpoint,
    _destination_for_asserted_arrival,
    _extract_intra_event_route_orderings,
    analyze_sentence,
)
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.app.routes.route_observations import project_observation_ordering

HANNIBAL_QUERY = "Trace Hannibal's route from New Carthage toward Italy during the Second Punic War."

ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("Island", ("island",), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage",), "audited"),
    HistoricalPlaceAlias("Capua", ("capua",), "audited"),
    HistoricalPlaceAlias("Roma", ("rome", "roma"), "audited"),
)

LIVE_ISLAND_PASSAGE = (
    "Meanwhile, after four days\u2019 march from the passage of the Rhone, "
    "Hannibal arrived at the place called the Island, "
    "Hannibal\u2019s march to the foot of the Alps."
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


def _has_ordering(text: str, earlier: str, later: str) -> bool:
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    return any(
        item.earlier.place_name == earlier and item.later.place_name == later
        for item in semantics.route_orderings
    )


class _HannibalGeography:
    _PLACES = {
        "rhodanus": ("Rhodanus", PlaceSpatialSemantics.RIVER, "representative_point", 43.33, 4.84),
        "alpes": ("Alpes", PlaceSpatialSemantics.MOUNTAIN_REGION, "regional_centroid", 43.74, 7.40),
        "new carthage": ("New Carthage", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 37.59, -0.98),
        "italia": ("Italia", PlaceSpatialSemantics.REGION, "regional_centroid", 41.87, 12.57),
        "island": ("Island", PlaceSpatialSemantics.ISLAND, "feature_centroid", 39.0, 9.0),
        "capua": ("Capua", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 41.08, 14.25),
        "roma": ("Roma", PlaceSpatialSemantics.SETTLEMENT, "exact_site", 41.89, 12.49),
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


def test_live_island_passage_orders_rhone_before_island_not_alps():
    event = _movement(LIVE_ISLAND_PASSAGE)
    assert event.actor.actor_status is EventActorStatus.EXPLICIT
    assert event.actor.actor_text == "Hannibal"
    match = _ordering(LIVE_ISLAND_PASSAGE, "Rhodanus", "Island")
    assert match.authority == "after_subordinate"
    assert not _has_ordering(LIVE_ISLAND_PASSAGE, "Rhodanus", "Alpes")
    roles = {mention.raw_text: mention.role for mention in event.place_mentions}
    assert roles.get("Island") is EventPlaceRole.DESTINATION or any(
        mention.role is EventPlaceRole.DESTINATION and "island" in mention.raw_text.casefold()
        for mention in event.place_mentions
    )
    assert any("rhone" in name.casefold() for name in roles)
    assert any("alps" in name.casefold() for name in roles)


def test_asserted_arrival_wins_over_trailing_march_to_destination_in_ordering_extract():
    aliases = _MENTION_EXTRACTOR.aliases_in(LIVE_ISLAND_PASSAGE)
    merged = (
        MovementEndpoint("Rhone", "Rhodanus", "origin", 58),
        MovementEndpoint("Alps", "Alpes", "destination", 150),
    )
    orderings = _extract_intra_event_route_orderings(
        LIVE_ISLAND_PASSAGE,
        aliases,
        merged,
        should_abstain=False,
    )
    assert any(
        item.earlier.canonical == "Rhodanus" and item.later.canonical == "Island"
        for item in orderings
    )
    assert not any(
        item.earlier.canonical == "Rhodanus" and item.later.canonical == "Alpes"
        for item in orderings
    )
    assert _destination_for_asserted_arrival(LIVE_ISLAND_PASSAGE, aliases, search_start=0).canonical == "Island"


@pytest.mark.parametrize(
    "text",
    [
        "Hannibal arrived at the place called the Island, his march toward the Alps continuing.",
        "Hannibal reached Capua, the route to Rome being difficult.",
    ],
)
def test_appositive_route_context_does_not_override_arrival_destination(text: str):
    event = _movement(text)
    destinations = [
        mention.raw_text
        for mention in event.place_mentions
        if mention.role is EventPlaceRole.DESTINATION
    ]
    assert destinations
    assert not any("alps" in name.casefold() for name in destinations)
    assert not any("rome" in name.casefold() for name in destinations)
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    assert not any(
        item.later.canonical == "Alpes" or item.later.canonical == "Roma"
        for item in semantics.route_orderings
    )


def test_sequential_reached_and_then_marched_retains_second_destination():
    text = "Hannibal reached Capua and then marched to Rome."
    event = _movement(text)
    destination_names = {
        (mention.canonical_hint or mention.raw_text).casefold()
        for mention in event.place_mentions
        if mention.role is EventPlaceRole.DESTINATION
    }
    assert "capua" in destination_names
    assert "roma" in destination_names
    semantics = analyze_sentence(text, _MENTION_EXTRACTOR.aliases_in(text))
    assert any(
        item.earlier.canonical == "Capua" and item.later.canonical == "Roma"
        for item in semantics.route_orderings
    )


def test_live_hannibal_replay_prefers_island_over_alps_on_island_passage():
    texts = _live_hannibal_evidence()
    if texts is None:
        pytest.skip("Chroma corpus unavailable for live Hannibal replay")
    _, _, _, resolved, _, _, typed_count, _ = _pipeline(texts, query=HANNIBAL_QUERY)
    island_events = [
        event
        for event in resolved
        if event.event_type is HistoricalEventType.MOVEMENT
        and any("island" in statement.casefold() and "rhone" in statement.casefold() for statement in event.source_statements)
    ]
    assert island_events
    assert any(
        ordering.earlier.canonical == "Rhodanus" and ordering.later.canonical == "Island"
        for event in island_events
        for ordering in event.route_orderings
    )
    assert not any(
        ordering.earlier.canonical == "Rhodanus"
        and ordering.later.canonical == "Alpes"
        and any("island" in statement.casefold() for statement in event.source_statements)
        for event in island_events
        for ordering in event.route_orderings
    )
    assert typed_count >= 3


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
