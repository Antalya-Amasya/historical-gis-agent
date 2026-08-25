from backend.app.candidate_routes.annotation import (
    HistoricalAnnotation,
    HistoricalEventType,
    HistoricalExternalReference,
    HistoricalExternalReferenceType,
)
from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.knowledge_panel import KnowledgePanelBuilder
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


def annotated_view():
    waypoint = HistoricalWaypoint(
        id="siege", canonical_name="Siege Site", order=1, role=HistoricalWaypointRole.BATTLE_SITE,
        evidence_refs=["e-siege"], location_confidence=LocationConfidence.EXACT,
    )
    reference = HistoricalExternalReference(
        id="paper", reference_type=HistoricalExternalReferenceType.PAPER,
        title="Paper", url="https://example.invalid/paper", language="en", description="Reading link",
    )
    annotation = HistoricalAnnotation(
        id="siege", title="Siege event", event_type=HistoricalEventType.SIEGE,
        period="52 BCE", description="Input annotation", source_refs=["e-siege"], external_references=[reference],
    )
    return to_waypoint_view_model(HistoricalWaypointBuilder.attach_annotations([waypoint], [annotation])[0])


def test_annotation_builds_panel_preserving_evidence_sources_and_external_references():
    panel = KnowledgePanelBuilder().build(annotated_view(), summary="Explicit supplied summary")
    assert panel.waypoint_id == "siege" and panel.title == "Siege event"
    assert panel.event_type is HistoricalEventType.SIEGE
    assert panel.summary == "Explicit supplied summary"
    assert panel.evidence_refs == ["e-siege"]
    assert panel.source_references == []  # internal evidence IDs are not user-facing source labels
    assert panel.external_references[0].id == "paper"


def test_missing_summary_stays_null_and_optional_provider_is_explicit():
    view = annotated_view()
    assert KnowledgePanelBuilder().build(view).summary is None
    assert KnowledgePanelBuilder().build(view, summary_provider=lambda value: f"Summary for {value.name}").summary == "Summary for Siege Site"


def test_geojson_contains_only_panel_id_not_full_panel_text():
    view = annotated_view()
    # Build graph-compatible waypoint separately to exercise the feature metadata adapter.
    waypoint = HistoricalWaypoint(id=view.id, canonical_name=view.name, order=1, role=HistoricalWaypointRole.START, evidence_refs=view.evidence_refs)
    properties = waypoints_to_geojson([waypoint])["features"][0]["properties"]
    assert properties["knowledge_panel_id"] == "siege"
    assert "summary" not in properties


def test_panel_creation_does_not_change_ranking():
    route = CandidateRouteEngine().build_route(
        from_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        to_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
        start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=SyntheticGrid.flat(2, 1), profile=ArmyProfile(),
    )
    before = RouteRankingModel().rank(CandidateRouteSet(routes=[route])).model_dump()
    KnowledgePanelBuilder().build(annotated_view(), summary="Explicit")
    after = RouteRankingModel().rank(CandidateRouteSet(routes=[route])).model_dump()
    assert before == after
