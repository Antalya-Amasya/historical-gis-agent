"""G6DN: pin exact retained production steering evidence through the event pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.app.core.config import settings
from backend.app.geography.feature_semantics import exact_anchor_eligible
from backend.app.models import (
    EventActorStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEventType,
)
from backend.app.routes.event_anchors import project_event_anchors
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator

CANONICAL_EVIDENCE_ID = "0e7e813af196915fa9ae7245dc4a069e:2929:3171"
CANONICAL_PARENT_ID = "0e7e813af196915fa9ae7245dc4a069e"
CANONICAL_SPAN = (2929, 3171)
EXPECTED_EVENT_ID = "event-bcd445242ec8"
EXPECTED_PLACE_ID = "pleiades-727192"
SIBLING_EVIDENCE_ID = "7bc6c989e5326bec35a87de26066725e:2922:3164"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHROMA_PATHS = (
    _REPO_ROOT / "backend" / "data" / "chroma_server_roman_republic_v2",
    Path(r"C:\D\python\202608231533\data\chroma_server_roman_republic_v2"),
)


@dataclass(frozen=True)
class ProductionPassage:
    evidence_id: str
    parent_id: str
    span: tuple[int, int]
    exact_text: str
    evidence: Evidence


class OfflineGeography:
    def call(self, tool: str, arguments: dict) -> dict:
        from geography_mcp.service import GeographyService

        service = GeographyService()
        if tool == "resolve_ancient_place":
            return service.resolve_ancient_place_payload(arguments["name"])
        raise ValueError(f"Unsupported geography tool: {tool}")


def _open_chroma_collection():
    import chromadb

    for path in _CHROMA_PATHS:
        if not path.exists():
            continue
        client = chromadb.PersistentClient(path=str(path))
        if settings.rag_collection not in [collection.name for collection in client.list_collections()]:
            continue
        collection = client.get_collection(settings.rag_collection)
        if collection.count() > 1000:
            return collection
    return None


def load_production_passage(evidence_id: str) -> ProductionPassage:
    collection = _open_chroma_collection()
    if collection is None:
        pytest.skip("Canonical local Chroma corpus unavailable for G6DN production pin")

    parent_id, start_text, end_text = evidence_id.rsplit(":", 2)
    start, end = int(start_text), int(end_text)
    payload = collection.get(ids=[parent_id], include=["documents", "metadatas"])
    if not payload["documents"]:
        pytest.skip(f"Parent chunk {parent_id!r} missing from canonical corpus")

    parent_text = payload["documents"][0]
    parent_meta = dict(payload["metadatas"][0] or {})
    exact_text = parent_text[start:end]
    evidence = Evidence(
        id=evidence_id,
        author=parent_meta.get("author") or "Plutarch",
        work=parent_meta.get("work") or "Lives",
        locator=str(start),
        excerpt=exact_text[:500],
        text=exact_text,
        metadata=parent_meta,
    )
    return ProductionPassage(
        evidence_id=evidence_id,
        parent_id=parent_id,
        span=(start, end),
        exact_text=exact_text,
        evidence=evidence,
    )


def _production_event_pipeline(passage: ProductionPassage):
    extractor = EvidenceGroundedHistoricalEventExtractor()
    candidates, _extraction = extractor.extract([passage.evidence])
    consolidated, _consolidation = HistoricalEventConsolidator().consolidate(candidates)
    resolved, place_diagnostics = HistoricalEventPlaceResolver(OfflineGeography()).resolve(consolidated)
    return resolved, place_diagnostics


def _steering_movement(events):
    return next(
        event
        for event in events
        if event.event_type is HistoricalEventType.MOVEMENT and "steer" in event.summary.casefold()
    )


@pytest.fixture(scope="module")
def canonical_passage() -> ProductionPassage:
    passage = load_production_passage(CANONICAL_EVIDENCE_ID)
    assert passage.parent_id == CANONICAL_PARENT_ID
    assert passage.span == CANONICAL_SPAN
    assert "city of Pelusium" in passage.exact_text
    assert "steered his course that way" in passage.exact_text
    return passage


def test_exact_production_text_is_loaded_verbatim_from_corpus(canonical_passage: ProductionPassage):
    assert canonical_passage.evidence.id == CANONICAL_EVIDENCE_ID
    assert canonical_passage.evidence.text == canonical_passage.exact_text
    assert canonical_passage.exact_text.startswith(" But on hearing that king Ptolemy")


@pytest.mark.integration
def test_exact_production_steering_event_path(canonical_passage: ProductionPassage):
    events, _place_diagnostics = _production_event_pipeline(canonical_passage)
    steering = _steering_movement(events)

    assert steering.id == EXPECTED_EVENT_ID
    assert steering.actor.actor_status is EventActorStatus.UNKNOWN
    roles = {mention.raw_text: mention.role for mention in steering.place_mentions}
    assert roles.get("Pelusium") is EventPlaceRole.DESTINATION
    assert EventPlaceRole.ORIGIN not in roles.values()
    assert "Cyprus" not in roles


@pytest.mark.integration
def test_exact_production_steering_geography_pin(canonical_passage: ProductionPassage):
    events, _place_diagnostics = _production_event_pipeline(canonical_passage)
    steering = _steering_movement(events)
    pelusium_binding = next(
        binding
        for binding in steering.place_bindings
        if binding.role is EventPlaceRole.DESTINATION and binding.mention.raw_text == "Pelusium"
    )
    assert pelusium_binding.resolution_status is EventPlaceResolutionStatus.RESOLVED
    assert pelusium_binding.place is not None
    assert pelusium_binding.place.id == EXPECTED_PLACE_ID
    assert pelusium_binding.place.coordinate_role == "exact_site"
    assert exact_anchor_eligible(pelusium_binding.place, strong_role=True)

    anchors, diagnostics = project_event_anchors(events, [canonical_passage.evidence])
    pelusium_anchor = next(
        anchor
        for anchor in anchors
        if anchor.canonical_name == "Pelusium" and anchor.role is EventPlaceRole.DESTINATION
    )
    assert pelusium_anchor.event_id == EXPECTED_EVENT_ID
    assert pelusium_anchor.place.id == EXPECTED_PLACE_ID
    assert not diagnostics


@pytest.mark.integration
def test_duplicate_parent_sibling_steering_semantics_match():
    canonical = load_production_passage(CANONICAL_EVIDENCE_ID)
    sibling = load_production_passage(SIBLING_EVIDENCE_ID)
    assert sibling.exact_text == canonical.exact_text

    canonical_roles = {
        mention.raw_text: mention.role
        for mention in _steering_movement(_production_event_pipeline(canonical)[0]).place_mentions
    }
    sibling_roles = {
        mention.raw_text: mention.role
        for mention in _steering_movement(_production_event_pipeline(sibling)[0]).place_mentions
    }
    assert sibling_roles == canonical_roles == {"Pelusium": EventPlaceRole.DESTINATION}
