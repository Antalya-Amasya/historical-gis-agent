"""V1.1G9C-R legacy tests updated for G9C2 typed event ordering consumer."""
from __future__ import annotations

import pytest

from backend.app.geography.feature_semantics import exact_anchor_eligible
from backend.app.models import (
    EventActorStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEventType,
    HistoricalPlace,
    PlaceSpatialSemantics,
)
from backend.app.routes.event_constraints import project_transition_constraints
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor, HistoricalEventConsolidator
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.app.routes.route_observations import ObservationOrderingAuthority, RouteObservationKind, project_observation_ordering

HANNIBAL_ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage", "carthago nova"), "audited"),
    HistoricalPlaceAlias("Iberus", ("iberus", "ebro"), "audited"),
    HistoricalPlaceAlias("Padus", ("padus",), "audited"),
    HistoricalPlaceAlias("Island", ("island",), "audited"),
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(
    HistoricalPlaceMentionExtractor(HANNIBAL_ALIASES),
)


class HannibalGeography:
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


def _pipeline(texts: list[str] | str, *, evidence_ids: list[str] | None = None):
    if isinstance(texts, str):
        texts = [texts]
    items = [
        _evidence(text, eid=eid, offset=100 + index * 100)
        for index, (text, eid) in enumerate(
            zip(texts, evidence_ids or [f"ev-{index}" for index in range(len(texts))])
        )
    ]
    candidates, _ = _EXTRACTOR.extract(items)
    events, _ = HistoricalEventConsolidator().consolidate(candidates)
    resolved, _ = HistoricalEventPlaceResolver(HannibalGeography()).resolve(events)
    constraints, _ = project_transition_constraints(resolved, items)
    observations, relations, diagnostics = project_observation_ordering(resolved, [], constraints, items)
    return observations, relations, diagnostics, resolved, constraints, items


def _labels(observations):
    return {item.observation_id: item.label for item in observations}


def _relation_between(relations, observations, earlier_label: str, later_label: str):
    labels = _labels(observations)
    matches = [
        relation for relation in relations
        if earlier_label.casefold() in labels.get(relation.earlier_observation_id, "").casefold()
        and later_label.casefold() in labels.get(relation.later_observation_id, "").casefold()
    ]
    assert len(matches) == 1, (relations, earlier_label, later_label, labels)
    return matches[0]


def _has_relation(relations, observations, earlier_label: str, later_label: str) -> bool:
    labels = _labels(observations)
    return any(
        earlier_label.casefold() in labels.get(relation.earlier_observation_id, "").casefold()
        and later_label.casefold() in labels.get(relation.later_observation_id, "").casefold()
        for relation in relations
    )


def test_before_crossing_alps_left_new_carthage_semantic_order():
    observations, relations, _, _, _, _ = _pipeline(
        "Before crossing the Alps, Hannibal left New Carthage."
    )
    relation = _relation_between(relations, observations, "New Carthage", "Alpes")
    assert relation.ordering_rule is ObservationOrderingAuthority.BEFORE_SUBORDINATE


def test_before_leaving_new_carthage_crossed_alps_reversed_scope():
    observations, relations, _, _, _, _ = _pipeline(
        "Before leaving New Carthage, Hannibal crossed the Alps."
    )
    relation = _relation_between(relations, observations, "Alpes", "New Carthage")
    assert relation.ordering_rule is ObservationOrderingAuthority.BEFORE_SUBORDINATE


def test_after_crossing_rhone_arrived_at_island():
    observations, relations, _, _, _, _ = _pipeline(
        "After crossing the Rhone, Hannibal arrived at the Island."
    )
    relation = _relation_between(relations, observations, "Rhodanus", "Island")
    assert relation.ordering_rule is ObservationOrderingAuthority.AFTER_SUBORDINATE


def test_first_then_rhone_before_alps():
    observations, relations, _, _, _, _ = _pipeline(
        "Hannibal crossed the Rhone first and then crossed the Alps."
    )
    relation = _relation_between(relations, observations, "Rhodanus", "Alpes")
    assert relation.ordering_rule is ObservationOrderingAuthority.FIRST_THEN


def test_came_to_italy_after_leaving_new_carthage():
    observations, relations, _, _, _, _ = _pipeline(
        "Hannibal came to Italy after leaving New Carthage."
    )
    _relation_between(relations, observations, "New Carthage", "Italia")


def test_arrived_at_new_carthage_after_leaving_iberus():
    observations, relations, _, _, _, _ = _pipeline(
        "Hannibal arrived at New Carthage after leaving the Iberus."
    )
    _relation_between(relations, observations, "Iberus", "New Carthage")


def test_no_semantic_authority_produces_no_relation():
    _, relations, _, _, _, _ = _pipeline(
        "Hannibal was at New Carthage and crossed the Alps."
    )
    assert not relations


def test_place_before_transition_new_carthage_to_rhone():
    observations, relations, _, _, constraints, _ = _pipeline(
        "After leaving New Carthage, Hannibal crossed the Rhone."
    )
    assert any(item.kind is RouteObservationKind.PLACE and item.label == "New Carthage" for item in observations)
    assert any(item.kind is RouteObservationKind.TRANSITION and item.label == "Rhodanus" for item in observations)
    relation = _relation_between(relations, observations, "New Carthage", "Rhodanus")
    assert relation.ordering_rule is ObservationOrderingAuthority.AFTER_SUBORDINATE
    assert constraints[0].exact_transition_coordinate is None


def test_place_transition_place_new_carthage_alps_italy():
    observations, relations, _, resolved, _, _ = _pipeline(
        "After leaving New Carthage, Hannibal crossed the Alps and came into Italy."
    )
    movement = next(event for event in resolved if event.event_type is HistoricalEventType.MOVEMENT)
    roles = {binding.mention.raw_text: binding.role for binding in movement.place_bindings}
    assert roles["New Carthage"] is EventPlaceRole.ORIGIN
    assert roles["Alps"] is EventPlaceRole.RELATED_PLACE
    assert roles["Italy"] is EventPlaceRole.DESTINATION
    _relation_between(relations, observations, "New Carthage", "Alpes")
    italy = next(item for item in observations if item.label == "Italia")
    assert italy.kind is RouteObservationKind.PLACE
    assert not exact_anchor_eligible(
        HistoricalPlace(
            id="italia",
            canonical_name="Italia",
            latitude=41.87,
            longitude=12.57,
            source="fixture",
            confidence=0.8,
            spatial_semantics=PlaceSpatialSemantics.REGION,
            coordinate_role="regional_centroid",
        ),
        strong_role=True,
    )


def test_different_actors_do_not_receive_inter_event_ordering():
    observations, relations, _, _, _, _ = _pipeline([
        "Hannibal crossed the Rhone.",
        "Scipio crossed the Alps.",
    ])
    assert not _has_relation(relations, observations, "Rhodanus", "Alpes")


def test_unknown_actor_inter_event_ordering_is_fail_closed():
    observations, relations, _, _, _, _ = _pipeline([
        "He crossed the Rhone.",
        "Then he reached the Island.",
    ])
    assert not _has_relation(relations, observations, "Rhodanus", "Island")


def test_unrelated_constraints_without_authority_are_not_ordered():
    observations, relations, _, _, _, _ = _pipeline([
        "Hannibal crossed the Rhone.",
        "Hannibal crossed the Alps.",
    ], evidence_ids=["ev-a", "ev-b"])
    assert not _has_relation(relations, observations, "Rhodanus", "Alpes")


def test_hannibal_replay_does_not_emit_unsafe_reversed_edges():
    texts = [
        "After leaving New Carthage, Hannibal crossed the Rhone.",
        "After crossing the Rhone, Hannibal arrived at the place called the Island.",
        "Hannibal crossed the Rhone first and then crossed the Alps.",
        "After leaving New Carthage, Hannibal crossed the Alps and came into Italy.",
        "Hannibal arrived at New Carthage after leaving the Iberus.",
    ]
    observations, relations, _, _, _, _ = _pipeline(texts)
    assert not _has_relation(relations, observations, "Italia", "New Carthage")
    assert not _has_relation(relations, observations, "New Carthage", "Iberus")


def test_build_with_diagnostics_exposes_observation_ordering():
    items = [_evidence("Hannibal crossed the Rhone.")]
    events, _ = _EXTRACTOR.extract(items)
    resolved, _ = HistoricalEventPlaceResolver(HannibalGeography()).resolve(events)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        resolved,
        items,
        event_id="hannibal",
        name="Hannibal",
        period="218 BCE",
    )
    assert outcome.diagnostics["observation_relation_count"] >= 0
    if outcome.observation_relations:
        payload = outcome.diagnostics["observation_relations"][0]
        assert "earlier_observation_id" in payload
        assert "ordering_rule" in payload
