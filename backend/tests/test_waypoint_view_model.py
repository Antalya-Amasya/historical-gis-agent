from backend.app.candidate_routes.annotation import (
    HistoricalAnnotation,
    HistoricalEventType,
    HistoricalExternalReference,
    HistoricalExternalReferenceType,
)
from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.location import LocationConfidence
from backend.app.candidate_routes.models import ArmyProfile, CandidateRouteAnchor, CandidateRouteSet
from backend.app.candidate_routes.ranking import RouteRankingModel
from backend.app.candidate_routes.view_model import to_waypoint_view_model
from backend.app.candidate_routes.waypoint_graph import (
    HistoricalWaypoint,
    HistoricalWaypointBuilder,
    HistoricalWaypointRole,
    waypoints_to_geojson,
)


def annotated_waypoint():
    waypoint = HistoricalWaypoint(
        id="crossing", canonical_name="River Crossing", order=2, role=HistoricalWaypointRole.CROSSING,
        evidence_refs=["e-crossing"], location_confidence=LocationConfidence.APPROXIMATE,
        location_notes="Representative crossing area",
    )
    reference = HistoricalExternalReference(
        id="reading", reference_type=HistoricalExternalReferenceType.ENCYCLOPEDIA,
        title="Further reading", url="https://example.invalid/reading", language="en", description="Display link",
    )
    annotation = HistoricalAnnotation(
        id="crossing", title="Crossing event", event_type=HistoricalEventType.CROSSING,
        period="218 BCE", description="Evidence-backed crossing annotation", source_refs=["e-crossing"],
        external_references=[reference],
    )
    return HistoricalWaypointBuilder.attach_annotations([waypoint], [annotation])[0]


def test_annotation_converts_to_stable_display_view_model_with_references_and_evidence():
    view = to_waypoint_view_model(annotated_waypoint())
    assert view.id == "crossing" and view.name == "River Crossing"
    assert view.event_type is HistoricalEventType.CROSSING
    assert view.period == "218 BCE" and view.description == "Evidence-backed crossing annotation"
    assert view.evidence_refs == ["e-crossing"]
    assert view.location_confidence is LocationConfidence.APPROXIMATE
    assert view.location_notes == "Representative crossing area"
    assert view.external_references[0].id == "reading"


def test_geojson_display_properties_are_view_fields_without_route_state():
    properties = waypoints_to_geojson([annotated_waypoint()])["features"][0]["properties"]
    assert properties["event_type"] == "CROSSING"
    assert properties["period"] == "218 BCE"
    assert properties["description"] == "Evidence-backed crossing annotation"
    assert properties["evidence_refs"] == ["e-crossing"]
    assert properties["external_references"][0]["id"] == "reading"
    assert properties["location_confidence"] == "APPROXIMATE"
    assert not {"cost", "ranking_score", "algorithm_state"} & set(properties)


def test_view_model_preserves_missing_annotation_fields_without_generation():
    view = to_waypoint_view_model(HistoricalWaypoint(
        id="plain", canonical_name="Plain", order=1, role=HistoricalWaypointRole.START,
        evidence_refs=["e-plain"], location_confidence=LocationConfidence.UNKNOWN,
    ))
    assert view.event_type is None and view.period is None and view.description is None
    assert view.external_references == []


def test_view_model_conversion_does_not_affect_route_ranking():
    candidate = CandidateRouteEngine().build_route(
        from_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        to_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
        start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=SyntheticGrid.flat(2, 1), profile=ArmyProfile(),
    )
    before = RouteRankingModel().rank(CandidateRouteSet(routes=[candidate])).model_dump()
    to_waypoint_view_model(annotated_waypoint())
    after = RouteRankingModel().rank(CandidateRouteSet(routes=[candidate])).model_dump()
    assert before == after
