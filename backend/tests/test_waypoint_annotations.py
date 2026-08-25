import pytest

from backend.app.candidate_routes.annotation import HistoricalAnnotation, HistoricalEventType
from backend.app.candidate_routes.grid import GridPoint, SyntheticGrid
from backend.app.candidate_routes.models import ArmyProfile, CandidateRouteAnchor, CandidateRouteSet
from backend.app.candidate_routes.engine import CandidateRouteEngine
from backend.app.candidate_routes.ranking import RouteRankingModel
from backend.app.candidate_routes.waypoint_graph import (
    HistoricalWaypoint,
    HistoricalWaypointBuilder,
    HistoricalWaypointEvidenceError,
    HistoricalWaypointRole,
    waypoints_to_geojson,
)


def waypoint():
    return HistoricalWaypoint(
        id="battle-site", canonical_name="Battle Site", order=1,
        role=HistoricalWaypointRole.BATTLE_SITE, evidence_refs=["e-battle", "e-context"],
    )


def annotation():
    return HistoricalAnnotation(
        id="battle-site", title="Battle annotation", event_type=HistoricalEventType.BATTLE,
        period="218 BCE", description="Input-supplied event note", source_refs=["e-battle"],
        external_links=[],
    )


def test_battle_annotation_binds_to_matching_waypoint():
    bound = HistoricalWaypointBuilder.attach_annotations([waypoint()], [annotation()])
    assert bound[0].annotation is not None
    assert bound[0].annotation.event_type is HistoricalEventType.BATTLE
    assert bound[0].annotation.source_refs == ["e-battle"]


def test_annotations_without_sources_or_matching_waypoint_are_rejected():
    missing_sources = HistoricalAnnotation.model_construct(
        id="battle-site", title="Missing", event_type=HistoricalEventType.OTHER,
        period=None, description="Missing", source_refs=[], external_links=[],
    )
    with pytest.raises(HistoricalWaypointEvidenceError):
        HistoricalWaypointBuilder.attach_annotations([waypoint()], [missing_sources])
    unknown = annotation().model_copy(update={"id": "unknown"})
    with pytest.raises(HistoricalWaypointEvidenceError):
        HistoricalWaypointBuilder.attach_annotations([waypoint()], [unknown])
    unrelated_source = annotation().model_copy(update={"source_refs": ["e-not-present"]})
    with pytest.raises(HistoricalWaypointEvidenceError):
        HistoricalWaypointBuilder.attach_annotations([waypoint()], [unrelated_source])


def test_waypoint_geojson_metadata_carries_annotation_without_geometry_guessing():
    bound = HistoricalWaypointBuilder.attach_annotations([waypoint()], [annotation()])
    feature = waypoints_to_geojson(bound)["features"][0]
    assert feature["type"] == "Feature" and feature["geometry"] is None
    assert feature["properties"]["annotation"]["title"] == "Battle annotation"
    assert feature["properties"]["annotation"]["source_refs"] == ["e-battle"]


def test_annotations_do_not_change_candidate_route_ranking():
    candidate = CandidateRouteEngine().build_route(
        from_anchor=CandidateRouteAnchor(historical_place_id="a", canonical_name="A", evidence_refs=["e-a"]),
        to_anchor=CandidateRouteAnchor(historical_place_id="b", canonical_name="B", evidence_refs=["e-b"]),
        start=GridPoint(0, 0), goal=GridPoint(1, 0), grid=SyntheticGrid.flat(2, 1), profile=ArmyProfile(),
    )
    before = RouteRankingModel().rank(CandidateRouteSet(routes=[candidate])).model_dump()
    HistoricalWaypointBuilder.attach_annotations([waypoint()], [annotation()])
    after = RouteRankingModel().rank(CandidateRouteSet(routes=[candidate])).model_dump()
    assert before == after
