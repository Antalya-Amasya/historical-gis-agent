from backend.app.candidate_routes.models import CandidateRoute, CandidateRouteAnchor, RouteCostBreakdown, RouteMetrics
from backend.app.candidate_routes.roman_road_orchestration import (
    RomanRoadRouteAggregate,
    RomanRoadRouteGeometrySegment,
    RomanRoadRouteLeg,
    RomanRoadRouteResult,
    RomanRoadRouteStatus,
)
from backend.app.candidate_routes.roman_road_presentation import RomanRoadPresentationService
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateStatus
from backend.app.models import GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, HistoricalTravelMode


def _point(name, lon):
    return HistoricalRoutePoint(
        sequence=1,
        historical_place=HistoricalPlace(id=name, canonical_name=name, longitude=lon, latitude=0, source="fixture", confidence=0.9),
        event_summary="attested", evidence_refs=[name], confidence=0.9,
    )


def _anchor(name):
    return CandidateRouteAnchor(historical_place_id=name, canonical_name=name, evidence_refs=[name])


def _gis_candidate(src, dst, coords, method="DIRECT_WATER_EDGE"):
    return CandidateRoute(
        id=f"{src}-{dst}",
        from_anchor=_anchor(src),
        to_anchor=_anchor(dst),
        geometry=GeoJsonLineString(coordinates=coords),
        metrics=RouteMetrics(distance_km=1, elevation_gain_m=0, elevation_loss_m=0, estimated_cost=1, cell_count=2, segment_count=1),
        cost_breakdown=RouteCostBreakdown(distance_cost=1, slope_cost=0, terrain_cost=0, barrier_cost=0, total_cost=1),
        confidence=1.0,
        assumptions=["Modern surface approximation; not a reconstructed historical coastline.", "Plausible GIS reconstruction between historical constraints; not an exact historical sailing track.", "direct_water_validated=true"],
        provenance="gis_reconstruction",
        generation_method=method,
        terrain_source="natural_earth_10m" if method == "DIRECT_WATER_EDGE" else "offline_dem",
    )


def _leg(*, index, src, dst, method, mode, status=RomanRoadCandidateStatus.AVAILABLE, failure=None, terrain=None, coords=None):
    return RomanRoadRouteLeg(
        leg_index=index, source_anchor_id=src, destination_anchor_id=dst,
        source_evidence_refs=[src], destination_evidence_refs=[dst],
        status=status, failure_status=failure, terrain_candidate=terrain,
        reconstruction_method=method, travel_mode=mode,
    )


def _segment(*, index, src, dst, role, coords, failure=None):
    return RomanRoadRouteGeometrySegment(
        segment_type=role, leg_index=index, coordinates=list(coords),
        source_anchor_id=src, destination_anchor_id=dst, failure_status=failure,
    )


def _aggregate(successes, failures=0):
    return RomanRoadRouteAggregate(
        successful_leg_count=successes, failed_leg_count=failures,
        total_network_distance_m=0, total_access_connector_distance_m=0,
        road_type_counts={}, segment_status_counts={}, chronology_counts={},
    )


def _present(points, legs, segments, status=RomanRoadRouteStatus.COMPLETE):
    route = HistoricalRoute(
        id="r", event_id="e", name="mixed", period="test",
        ordered_points=points,
        geometry=GeoJsonLineString(coordinates=[]),
        historical_confidence=0.9,
    )
    result = RomanRoadRouteResult(
        historical_route_id="r", status=status, legs=legs, geometry_segments=segments,
        aggregate=_aggregate(sum(1 for leg in legs if leg.failure_status is None), sum(1 for leg in legs if leg.failure_status)),
        limitations=["GIS reconstruction only."],
    )
    return RomanRoadPresentationService().present(route, result)


def _line_features(payload):
    return [item for item in payload.geojson["features"] if item["properties"].get("segment_role")]


def test_direct_water_edge_is_maritime_not_road_or_terrain():
    sea = _gis_candidate("P", "Q", [(-2.0, -2.0), (-5.0, -5.0)])
    payload = _present(
        [_point("P", -2), _point("Q", -5)],
        [_leg(index=1, src="P", dst="Q", method="DIRECT_WATER_EDGE", mode=HistoricalTravelMode.SEA, terrain=sea)],
        [_segment(index=1, src="P", dst="Q", role="direct_water_edge", coords=[(-2.0, -2.0), (-5.0, -5.0)])],
    )
    feature = _line_features(payload)[0]
    assert feature["properties"]["layer_type"] == "direct_water_edge"
    assert feature["properties"]["segment_role"] == "direct_water_edge"
    assert feature["properties"]["reconstruction_method"] == "DIRECT_WATER_EDGE"
    assert feature["properties"]["travel_mode"] == HistoricalTravelMode.SEA.value
    assert payload.road_network["terrain_fallback_used"] is False
    assert "terrain" not in payload.presentation_summary["route_interpretation"].lower()
    dumped = payload.road_network["legs"][0]
    assert dumped["reconstruction_method"] == "DIRECT_WATER_EDGE"
    assert dumped["travel_mode"] == HistoricalTravelMode.SEA.value
    assert "direct_water_validated=true" in dumped["terrain_candidate"]["assumptions"]


def test_actual_terrain_fallback_and_roman_road_labels_unchanged():
    terrain = _gis_candidate("A", "B", [(0.0, 0.0), (1.0, 0.0)], method="terrain_astar")
    terrain_payload = _present(
        [_point("A", 0), _point("B", 1)],
        [_leg(index=1, src="A", dst="B", method="TERRAIN_ASTAR_FALLBACK", mode=HistoricalTravelMode.LAND, terrain=terrain)],
        [_segment(index=1, src="A", dst="B", role="terrain_candidate", coords=[(0.0, 0.0), (1.0, 0.0)])],
    )
    assert terrain_payload.road_network["terrain_fallback_used"] is True
    assert _line_features(terrain_payload)[0]["properties"]["layer_type"] == "terrain_reconstruction_segment"
    assert "terrain A*" in terrain_payload.presentation_summary["route_interpretation"]

    road_payload = _present(
        [_point("A", 0), _point("B", 1)],
        [_leg(index=1, src="A", dst="B", method="ROMAN_ROAD_NETWORK", mode=HistoricalTravelMode.LAND)],
        [_segment(index=1, src="A", dst="B", role="roman_road", coords=[(0.0, 0.0), (1.0, 0.0)])],
    )
    assert road_payload.road_network["terrain_fallback_used"] is False
    assert _line_features(road_payload)[0]["properties"]["layer_type"] == "roman_road_segment"


def test_sea_gap_stays_failed_gap_not_direct_water():
    payload = _present(
        [_point("P", 1), _point("Q", 2)],
        [_leg(
            index=1, src="P", dst="Q", method="MARITIME_GEOMETRY_UNAVAILABLE",
            mode=HistoricalTravelMode.SEA, status=RomanRoadCandidateStatus.DISCONNECTED,
            failure="MARITIME_GEOMETRY_UNAVAILABLE",
        )],
        [_segment(index=1, src="P", dst="Q", role="failed_gap", coords=[], failure="MARITIME_GEOMETRY_UNAVAILABLE")],
        status=RomanRoadRouteStatus.UNAVAILABLE,
    )
    feature = _line_features(payload)[0]
    assert feature["geometry"] is None
    assert feature["properties"]["segment_role"] == "failed_gap"
    assert feature["properties"]["layer_type"] != "direct_water_edge"


def test_mixed_success_and_mixed_gap_preserve_order_without_concatenation():
    sea = _gis_candidate("B", "C", [(-2.0, -2.0), (-5.0, -5.0)])
    mixed = _present(
        [_point("A", 0), _point("B", 1), _point("C", 2), _point("D", 3)],
        [
            _leg(index=1, src="A", dst="B", method="ROMAN_ROAD_NETWORK", mode=HistoricalTravelMode.LAND),
            _leg(index=2, src="B", dst="C", method="DIRECT_WATER_EDGE", mode=HistoricalTravelMode.SEA, terrain=sea),
            _leg(index=3, src="C", dst="D", method="ROMAN_ROAD_NETWORK", mode=HistoricalTravelMode.LAND),
        ],
        [
            _segment(index=1, src="A", dst="B", role="roman_road", coords=[(0.0, 0.0), (1.0, 0.0)]),
            _segment(index=2, src="B", dst="C", role="direct_water_edge", coords=[(-2.0, -2.0), (-5.0, -5.0)]),
            _segment(index=3, src="C", dst="D", role="roman_road", coords=[(2.0, 0.0), (3.0, 0.0)]),
        ],
    )
    roles = [item["properties"]["segment_role"] for item in _line_features(mixed)]
    layers = [item["properties"]["layer_type"] for item in _line_features(mixed)]
    assert roles == ["roman_road", "direct_water_edge", "roman_road"]
    assert layers == ["roman_road_segment", "direct_water_edge", "roman_road_segment"]
    assert mixed.road_network["terrain_fallback_used"] is False

    gapped = _present(
        [_point("A", 0), _point("B", 1), _point("C", 2), _point("D", 3)],
        [
            _leg(index=1, src="A", dst="B", method="ROMAN_ROAD_NETWORK", mode=HistoricalTravelMode.LAND),
            _leg(index=2, src="B", dst="C", method="MARITIME_GEOMETRY_UNAVAILABLE", mode=HistoricalTravelMode.SEA, status=RomanRoadCandidateStatus.DISCONNECTED, failure="MARITIME_GEOMETRY_UNAVAILABLE"),
            _leg(index=3, src="C", dst="D", method="ROMAN_ROAD_NETWORK", mode=HistoricalTravelMode.LAND),
        ],
        [
            _segment(index=1, src="A", dst="B", role="roman_road", coords=[(0.0, 0.0), (1.0, 0.0)]),
            _segment(index=2, src="B", dst="C", role="failed_gap", coords=[], failure="MARITIME_GEOMETRY_UNAVAILABLE"),
            _segment(index=3, src="C", dst="D", role="roman_road", coords=[(2.0, 0.0), (3.0, 0.0)]),
        ],
        status=RomanRoadRouteStatus.PARTIAL,
    )
    features = _line_features(gapped)
    assert [item["properties"]["segment_role"] for item in features] == ["roman_road", "failed_gap", "roman_road"]
    assert features[1]["geometry"] is None
    assert features[0]["geometry"] is not None and features[2]["geometry"] is not None
    coords = [item["geometry"]["coordinates"] for item in features if item["geometry"]]
    assert coords == [[[0.0, 0.0], [1.0, 0.0]], [[2.0, 0.0], [3.0, 0.0]]]
