"""V1.1G9B: TransitionConstraint projection contract."""
from __future__ import annotations

import pytest

from backend.app.geography.feature_semantics import exact_anchor_eligible
from backend.app.models import (
    EventActorStatus,
    EventPlaceRole,
    Evidence,
    GeographicFeatureKind,
    HistoricalEventType,
    TransitionAction,
)
from backend.app.routes.event_anchors import project_event_anchors
from backend.app.routes.event_constraints import project_transition_constraints
from backend.app.routes.event_places import HistoricalEventPlaceResolver
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.extractor import HistoricalPlaceMentionExtractor
from backend.app.routes.place_aliases import HistoricalPlaceAlias
from backend.app.models import HistoricalPlace, PlaceSpatialSemantics

HANNIBAL_ALIASES = (
    HistoricalPlaceAlias("Rhodanus", ("rhodanus", "rhone"), "audited"),
    HistoricalPlaceAlias("Alpes", ("alps", "alpes"), "audited"),
    HistoricalPlaceAlias("Italia", ("italy", "italia"), "audited"),
    HistoricalPlaceAlias("New Carthage", ("new carthage", "carthago nova"), "audited"),
    HistoricalPlaceAlias("Padus", ("padus",), "audited"),
)

_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor(
    HistoricalPlaceMentionExtractor(HANNIBAL_ALIASES),
)


class HannibalGeography:
    _PLACES = {
        "rhodanus": HistoricalPlace(
            id="148168",
            canonical_name="Rhodanus",
            latitude=43.33167,
            longitude=4.84861,
            source="Pleiades",
            source_id="148168",
            confidence=0.8,
            coordinate_role="representative_point",
            spatial_semantics=PlaceSpatialSemantics.RIVER,
        ),
        "alpes": HistoricalPlace(
            id="783",
            canonical_name="Alpes",
            latitude=43.74465275,
            longitude=7.40183905,
            source="Pleiades",
            source_id="783",
            confidence=0.8,
            coordinate_role="regional_centroid",
            spatial_semantics=PlaceSpatialSemantics.MOUNTAIN_REGION,
        ),
        "new carthage": HistoricalPlace(
            id="265849",
            canonical_name="New Carthage",
            latitude=37.599896,
            longitude=-0.98452,
            source="Pleiades",
            source_id="265849",
            confidence=0.8,
            coordinate_role="exact_site",
            spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
        ),
        "italia": HistoricalPlace(
            id="981",
            canonical_name="Italia",
            latitude=41.87194,
            longitude=12.56738,
            source="Pleiades",
            source_id="981",
            confidence=0.8,
            coordinate_role="regional_centroid",
            spatial_semantics=PlaceSpatialSemantics.REGION,
        ),
    }

    def call(self, tool: str, arguments: dict) -> dict:
        assert tool == "resolve_ancient_place"
        key = arguments["name"].casefold()
        for alias, place in self._PLACES.items():
            if key == alias or key in {alias, place.canonical_name.casefold()}:
                payload = place.model_dump()
                payload["found"] = True
                return payload
        return {"found": False}


def _evidence(text: str, *, eid: str = "ev") -> Evidence:
    return Evidence(
        id=eid,
        author="Polybius",
        work="Histories",
        locator="III",
        excerpt=text,
        text=text,
    )


def _pipeline(text: str, *, eid: str = "ev"):
    item = _evidence(text, eid=eid)
    events, _ = _EXTRACTOR.extract([item])
    resolved, _ = HistoricalEventPlaceResolver(HannibalGeography()).resolve(events)
    constraints, diagnostics = project_transition_constraints(resolved, [item])
    return constraints, resolved, diagnostics, item


def _single_constraint(text: str):
    constraints, _, diagnostics, _ = _pipeline(text)
    assert len(constraints) == 1, (text, diagnostics)
    return constraints[0]


def test_hannibal_crossed_the_rhone_produces_transition_constraint():
    constraint = _single_constraint("Hannibal crossed the Rhone.")
    assert constraint.action is TransitionAction.CROSS
    assert constraint.feature_surface == "Rhone"
    assert constraint.feature_canonical == "Rhodanus"
    assert constraint.feature_kind is GeographicFeatureKind.RIVER
    assert constraint.actor_status is EventActorStatus.EXPLICIT
    assert constraint.actor_text == "Hannibal"
    assert constraint.evidence_refs == ["ev"]
    assert constraint.source_statement == "Hannibal crossed the Rhone."
    assert constraint.exact_transition_coordinate is None
    assert not hasattr(constraint, "latitude")


def test_hannibal_crossed_the_alps_produces_transition_constraint():
    constraint = _single_constraint("Hannibal crossed the Alps.")
    assert constraint.feature_surface == "Alps"
    assert constraint.feature_canonical == "Alpes"
    assert constraint.feature_kind is GeographicFeatureKind.MOUNTAIN_REGION
    assert constraint.actor_text == "Hannibal"
    assert constraint.exact_transition_coordinate is None


def test_compound_movement_preserves_new_carthage_and_alps_transition():
    constraints, events, _, _ = _pipeline(
        "After leaving New Carthage, Hannibal crossed the Alps and came into Italy."
    )
    movement = next(event for event in events if event.event_type is HistoricalEventType.MOVEMENT)
    roles = {binding.mention.raw_text: binding.role for binding in movement.place_bindings}
    assert roles.get("New Carthage") is EventPlaceRole.ORIGIN
    assert roles.get("Italy") is EventPlaceRole.DESTINATION
    assert roles.get("Alps") is EventPlaceRole.RELATED_PLACE
    assert len(constraints) == 1
    assert constraints[0].feature_surface == "Alps"
    assert constraints[0].exact_transition_coordinate is None
    anchors, projection = project_event_anchors(events, [_evidence(movement.summary)], allow_contextual_related_places=True)
    assert any(code.startswith("NON_EXACT_FEATURE_ANCHOR") for code in projection) or not any(
        anchor.canonical_name == "Alpes" and anchor.admission_type == "MOVEMENT_WAYPOINT" for anchor in anchors
    )


@pytest.mark.parametrize(
    "text",
    [
        "Hannibal did not cross the Rhone.",
        "If Hannibal crossed the Alps, he would enter Italy.",
        "Hannibal would cross the Rhone.",
    ],
)
def test_negated_and_hypothetical_crossings_emit_no_transition_constraint(text: str):
    constraints, _, _, _ = _pipeline(text)
    assert constraints == []


def test_context_only_river_mention_emits_no_transition_constraint():
    constraints, _, _, _ = _pipeline("The Rhone was an important river.")
    assert constraints == []


def test_pronoun_actor_preserves_unknown_authority():
    constraint = _single_constraint("He crossed the Rhone.")
    assert constraint.feature_surface == "Rhone"
    assert constraint.actor_status is not EventActorStatus.EXPLICIT
    assert constraint.actor_text is None


def test_rhone_and_alps_do_not_become_exact_route_waypoints():
    text = "After leaving New Carthage, Hannibal crossed the Alps and came into Italy."
    item = _evidence(text)
    events, _ = _EXTRACTOR.extract([item])
    resolved, _ = HistoricalEventPlaceResolver(HannibalGeography()).resolve(events)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        resolved,
        [item],
        event_id="hannibal",
        name="Hannibal",
        period="218 BCE",
        allow_contextual_related_places=True,
    )
    constraints, _ = project_transition_constraints(resolved, [item])
    feature_names = {constraint.feature_canonical for constraint in constraints}
    assert {"Rhodanus", "Alpes"} & feature_names or "Alpes" in feature_names
    if outcome.route is not None:
        waypoint_names = {
            point.historical_place.canonical_name
            for component in outcome.route.route_components
            for point in component.ordered_points
        }
        assert "Rhodanus" not in waypoint_names
        assert "Alpes" not in waypoint_names
        for point in outcome.route.ordered_points:
            assert point.coordinate_role == "exact_site"
    for binding in resolved[0].place_bindings:
        if binding.mention.raw_text in {"Rhone", "Alps"}:
            assert not exact_anchor_eligible(binding.place, strong_role=True)


def test_build_with_diagnostics_exposes_transition_constraints():
    text = "Hannibal crossed the Rhone."
    item = _evidence(text)
    events, _ = _EXTRACTOR.extract([item])
    resolved, _ = HistoricalEventPlaceResolver(HannibalGeography()).resolve(events)
    outcome = EventAnchorRouteBuilder().build_with_diagnostics(
        resolved,
        [item],
        event_id="hannibal",
        name="Hannibal",
        period="218 BCE",
    )
    assert outcome.diagnostics["transition_constraint_count"] == 1
    payload = outcome.diagnostics["transition_constraints"][0]
    assert payload["feature_surface"] == "Rhone"
    assert payload["exact_transition_coordinate"] is None
