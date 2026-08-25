import pytest

from backend.app.candidate_routes import (
    HistoricalEventChain,
    HistoricalEventStep,
    HistoricalWaypointBuilder,
    HistoricalWaypointEvidenceError,
)


def step(identifier, name, order, evidence):
    return HistoricalEventStep(
        id=identifier, name=name, order=order, description=f"{name} event",
        evidence_refs=evidence, confidence=0.8,
    )


def chain():
    return HistoricalEventChain(steps=[
        step("a", "A", 1, ["e-a"]),
        step("b", "B", 2, ["e-b"]),
        step("c", "C", 3, ["e-c"]),
    ])


def test_complete_event_chain_builds_ordered_waypoints_and_segments():
    graph = HistoricalWaypointBuilder.from_event_chain(chain())
    assert [item.id for item in graph.waypoints] == ["a", "b", "c"]
    assert [item.role.value for item in graph.waypoints] == ["START", "WAYPOINT", "END"]
    assert [(item.from_waypoint.id, item.to_waypoint.id) for item in graph.segments] == [("a", "b"), ("b", "c")]
    assert graph.segments[0].evidence_refs == ["e-a", "e-b"]


def test_missing_evidence_is_rejected_even_if_invalid_model_was_constructed():
    invalid = HistoricalEventStep.model_construct(
        id="a", name="A", order=1, description="A", evidence_refs=[], confidence=0.8,
    )
    invalid_chain = HistoricalEventChain.model_construct(steps=[invalid])
    with pytest.raises(HistoricalWaypointEvidenceError):
        HistoricalWaypointBuilder.from_event_chain(invalid_chain)


def test_duplicate_waypoint_merges_only_evidence_union_and_never_inserts_nodes():
    graph = HistoricalWaypointBuilder.from_event_chain(HistoricalEventChain(steps=[
        step("a", "A", 1, ["e-a1"]),
        step("b", "B", 2, ["e-b"]),
        step("a", "A", 3, ["e-a2", "e-a1"]),
    ]))
    assert [item.id for item in graph.waypoints] == ["a", "b"]
    assert graph.waypoints[0].evidence_refs == ["e-a1", "e-a2"]
    assert {item.canonical_name for item in graph.waypoints} == {"A", "B"}


def test_event_chain_expansion_is_deterministic():
    first = HistoricalWaypointBuilder.from_event_chain(chain())
    second = HistoricalWaypointBuilder.from_event_chain(chain())
    assert first.model_dump() == second.model_dump()
