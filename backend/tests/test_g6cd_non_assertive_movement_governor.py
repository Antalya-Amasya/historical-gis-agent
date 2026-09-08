"""G6CD: clause-local non-assertive movement governors."""
from __future__ import annotations

from backend.app.models import Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def extract(text: str):
    return EvidenceGroundedHistoricalEventExtractor().extract([evidence("ev", text)])


def movement_events(text: str):
    events, _ = extract(text)
    return [event for event in events if event.event_type is HistoricalEventType.MOVEMENT]


def test_discussed_crossing_rejected():
    assert movement_events("Ariston discussed crossing the river.") == []


def test_discussed_marching_rejected():
    assert movement_events("Ariston discussed marching to Capua.") == []


def test_prevented_from_crossing_rejected():
    assert movement_events("Ariston was prevented from crossing the river.") == []


def test_prevented_from_marching_rejected():
    assert movement_events("Ariston was prevented from marching to Capua.") == []


def test_asserted_crossing_eligible():
    assert movement_events("Ariston crossed the river.")


def test_crossing_after_discussion_noun_eligible():
    assert movement_events("After discussion, Ariston crossed the river.")


def test_prevented_clause_does_not_veto_later_crossing():
    text = "Ariston was prevented from marching, but later crossed the river."
    assert movement_events(text)


def test_discussed_plan_does_not_veto_later_march():
    text = "Ariston discussed the plan, then marched to Capua."
    assert movement_events(text)
