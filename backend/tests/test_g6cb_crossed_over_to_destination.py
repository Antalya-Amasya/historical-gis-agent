"""G6CB: crossed/crossing over to X must assign destination role."""
from __future__ import annotations

from backend.app.models import EventPlaceRole, Evidence, HistoricalEventType
from backend.app.routes.events import EvidenceGroundedHistoricalEventExtractor
from backend.app.routes.movement_semantics import analyze_sentence


def evidence(identifier: str, text: str) -> Evidence:
    return Evidence(id=identifier, author="Source", work="Work", locator="1", excerpt=text, text=text)


def extract(text: str):
    return EvidenceGroundedHistoricalEventExtractor().extract([evidence("ev", text)])


def semantic_roles(text: str) -> dict[str, str]:
    semantics = analyze_sentence(text, [])
    return {endpoint.surface: endpoint.role for endpoint in semantics.endpoints}


def movement_roles(text: str) -> dict[str, EventPlaceRole]:
    events, _ = extract(text)
    assert len(events) == 1
    assert events[0].event_type is HistoricalEventType.MOVEMENT
    return {mention.raw_text: mention.role for mention in events[0].place_mentions}


def test_crossed_over_to_simple_destination():
    roles = semantic_roles("Ariston crossed over to Cos.")
    assert roles.get("Cos") == "destination"
    assert movement_roles("Ariston crossed over to Cos.").get("Cos") is EventPlaceRole.DESTINATION


def test_crossed_over_to_island_of_destination():
    text = "Ariston crossed over to the island of Cos."
    assert semantic_roles(text).get("Cos") == "destination"
    assert movement_roles(text).get("Cos") is EventPlaceRole.DESTINATION


def test_crossed_over_the_bridge_is_not_destination():
    roles = semantic_roles("Ariston crossed over the Bridge Alpha.")
    assert roles.get("Bridge Alpha") == "traversal"
    assert "Bridge Alpha" not in {
        surface for surface, role in roles.items() if role == "destination"
    }


def test_crossed_over_the_river_is_not_destination():
    roles = semantic_roles("Ariston crossed over the River Beta.")
    assert roles.get("River Beta") == "traversal"
    assert all(role != "destination" for role in roles.values())


def test_crossed_the_river_to_preserves_destination():
    roles = movement_roles("Ariston crossed the river to Cos.")
    assert roles.get("Cos") is EventPlaceRole.DESTINATION


def test_passed_over_the_bridge_preserves_traversal():
    roles = semantic_roles("Ariston passed over the Bridge Alpha.")
    assert roles.get("Bridge Alpha") == "traversal"
    assert all(role != "destination" for role in roles.values())


def test_crossing_over_to_participial_destination():
    roles = semantic_roles("Ariston crossing over to Cyprus toward the harbor.")
    assert roles.get("Cyprus") == "destination"


def test_mithridates_control_crossed_over_to_island_of_cos():
    text = "Mithridates crossed over to the island of Cos."
    assert semantic_roles(text).get("Cos") == "destination"
    roles = movement_roles(text)
    assert roles.get("Cos") is EventPlaceRole.DESTINATION
    assert EventPlaceRole.ORIGIN not in roles.values()
