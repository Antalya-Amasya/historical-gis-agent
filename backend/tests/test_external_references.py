import pytest

from backend.app.candidate_routes.annotation import (
    HistoricalAnnotation,
    HistoricalEventType,
    HistoricalExternalReference,
    HistoricalExternalReferenceType,
)
from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.models import ArmyProfile, CandidateRouteAnchor, CandidateRouteSet
from backend.app.candidate_routes.ranking import RouteRankingModel
from backend.app.candidate_routes.waypoint_graph import (
    HistoricalWaypoint,
    HistoricalWaypointBuilder,
    HistoricalWaypointEvidenceError,
    HistoricalWaypointRole,
    waypoints_to_geojson,
)


def waypoint():
    return HistoricalWaypoint(id="battle", canonical_name="Battle Site", order=1, role=HistoricalWaypointRole.BATTLE_SITE, evidence_refs=["e-battle"])


def annotation(references):
    return HistoricalAnnotation(
        id="battle", title="Battle details", event_type=HistoricalEventType.BATTLE,
        period="218 BCE", description="Input-supplied display annotation", source_refs=["e-battle"],
        external_references=references,
    )


def test_wikipedia_and_google_maps_references_bind_as_display_metadata():
    wikipedia = HistoricalExternalReference(
        id="wiki", reference_type=HistoricalExternalReferenceType.WIKIPEDIA,
        title="Battle article", url="https://example.invalid/wiki/battle", language="en", description="Further reading",
    )
    maps = HistoricalExternalReference(
        id="maps", reference_type=HistoricalExternalReferenceType.GOOGLE_MAPS,
        title="Modern location", url="https://example.invalid/maps/battle", language="en", description="Modern map link",
    )
    bound = HistoricalWaypointBuilder.attach_annotations([waypoint()], [annotation([wikipedia, maps])])
    properties = waypoints_to_geojson(bound)["features"][0]["properties"]
    assert properties["waypoint_name"] == "Battle Site"
    assert properties["annotation_title"] == "Battle details"
    assert properties["annotation_description"] == "Input-supplied display annotation"
    assert [item["reference_type"] for item in properties["external_references"]] == ["WIKIPEDIA", "GOOGLE_MAPS"]


def test_external_references_do_not_relax_evidence_validation():
    reference = HistoricalExternalReference(id="wiki", reference_type=HistoricalExternalReferenceType.WIKIPEDIA, title="Article", url="https://example.invalid/wiki")
    missing_source = HistoricalAnnotation.model_construct(
        id="battle", title="Invalid", event_type=HistoricalEventType.BATTLE, period=None,
        description="Invalid", source_refs=[], external_links=[], external_references=[reference],
    )
    with pytest.raises(HistoricalWaypointEvidenceError):
        HistoricalWaypointBuilder.attach_annotations([waypoint()], [missing_source])


def test_removing_external_references_does_not_change_route_ranking_or_planning_data():
    candidate = CandidateRouteEngine().build_route(
        from_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        to_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
        start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=SyntheticGrid.flat(2, 1), profile=ArmyProfile(),
    )
    before = RouteRankingModel().rank(CandidateRouteSet(routes=[candidate])).model_dump()
    external = HistoricalExternalReference(id="wiki", reference_type=HistoricalExternalReferenceType.WIKIPEDIA, title="Article", url="https://example.invalid/wiki")
    with_references = HistoricalWaypointBuilder.attach_annotations([waypoint()], [annotation([external])])
    without_references = HistoricalWaypointBuilder.attach_annotations([waypoint()], [annotation([])])
    after = RouteRankingModel().rank(CandidateRouteSet(routes=[candidate])).model_dump()
    assert before == after
    assert waypoints_to_geojson(with_references)["features"][0]["properties"]["external_references"]
    assert waypoints_to_geojson(without_references)["features"][0]["properties"]["external_references"] == []
