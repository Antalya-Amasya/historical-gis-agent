"""V1.1G8A: waypoint authority hardening regressions."""
from __future__ import annotations

import pytest

from backend.app.models import (
    EventActorStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    Evidence,
    HistoricalEventActorGrounding,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
)
from backend.app.routes.event_route_orchestration import EventAnchorRouteBuilder
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.tests.test_v1_1g2_waypoint_observation_contract import (
    _binding,
    _build,
    _event,
    _evidence,
    _pair,
)

UNKNOWN = HistoricalEventActorGrounding(actor_status=EventActorStatus.UNKNOWN)
_EXTRACTOR = EvidenceGroundedHistoricalEventExtractor()


def _ambiguous_binding(name: str, role: EventPlaceRole, refs: list[str]) -> HistoricalEventPlaceBinding:
    return HistoricalEventPlaceBinding(
        mention=HistoricalEventPlaceMention(raw_text=name, role=role, evidence_refs=list(refs)),
        place=None,
        role=role,
        resolution_status=EventPlaceResolutionStatus.AMBIGUOUS,
        evidence_refs=list(refs),
    )


def _presence_events(text: str):
    events, _ = _EXTRACTOR.extract([Evidence(id="ev", author="Fixture", work="Test", locator="1", excerpt=text, text=text)])
    return [event for event in events if event.event_type is HistoricalEventType.PRESENCE]


def test_unknown_actors_with_dates_do_not_form_waypoint_ordering():
    events = [
        _event("e1", "He departed from Harbor One.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"], years="49", actor=UNKNOWN),
        _event("e2", "He reached Fort Two.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"], years="48", actor=UNKNOWN),
    ]
    outcome = _build(events, [_evidence("ev-a", events[0].summary), _evidence("ev-b", events[1].summary, offset=200)])
    assert outcome.route is None
    assert outcome.relations == ()
    assert not any(claim.claim_type == "WAYPOINT_ORDERING" for claim in (outcome.route.claims if outcome.route else ()))


@pytest.mark.parametrize(
    "text",
    [
        "The commander was at Harbor One.",
        "The king remained at Harbor One.",
        "The consul was at Harbor One.",
        "The Commander was at Harbor One.",
        "The King remained at Harbor One.",
        "The Consul was at Harbor One.",
    ],
)
def test_generic_role_titles_are_not_explicit_presence_actors(text: str):
    events = _presence_events(text)
    assert not any(event.actor.actor_status is EventActorStatus.EXPLICIT for event in events)


def test_titled_proper_name_presence_actor_remains_explicit():
    events = _presence_events("Commander Alpha was at Harbor One.")
    assert len(events) == 1 and events[0].actor.actor_status is EventActorStatus.EXPLICIT


def test_earlier_ambiguous_destination_does_not_fallback_to_origin():
    events = [
        _event(
            "e1",
            "Commander Alpha departed Harbor One toward Alt Harbor.",
            [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"]), _ambiguous_binding("Alt Harbor", EventPlaceRole.DESTINATION, ["ev-a"])],
            ["ev-a"],
            years="402",
        ),
        _event("e2", "Commander Alpha reached Fort Two.", [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"])], ["ev-b"], years="401"),
    ]
    outcome = _build(events, [_evidence("ev-a", "a", offset=100), _evidence("ev-b", "b", offset=200)])
    assert outcome.route is None and not any(rel.earlier == "Harbor One" and rel.later == "Fort Two" for rel in outcome.relations)


def test_later_ambiguous_origin_does_not_fallback_to_destination():
    events = [
        _event("e1", "Commander Alpha departed Harbor One.", [_binding("Harbor One", EventPlaceRole.ORIGIN, ["ev-a"])], ["ev-a"], years="402"),
        _event(
            "e2",
            "Commander Alpha reached Fort Two from Alt Origin.",
            [_binding("Fort Two", EventPlaceRole.DESTINATION, ["ev-b"]), _ambiguous_binding("Alt Origin", EventPlaceRole.ORIGIN, ["ev-b"])],
            ["ev-b"],
            years="401",
        ),
    ]
    outcome = _build(events, [_evidence("ev-a", "a", offset=100), _evidence("ev-b", "b", offset=200)])
    assert outcome.route is None and not any(rel.earlier == "Harbor One" and rel.later == "Fort Two" for rel in outcome.relations)


def test_valid_one_ended_waypoint_ordering_still_works():
    events, items = _pair(connector="Then")
    outcome = _build(events, items)
    assert outcome.route is not None
