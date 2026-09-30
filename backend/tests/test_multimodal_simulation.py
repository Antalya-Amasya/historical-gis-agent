from backend.app.candidate_routes.geographic import GeographicCandidateRouteService
from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.candidate_routes.roman_road_presentation import RomanRoadPresentationService
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateService
from backend.app.gis.coastal_access import select_simulation_mode
from backend.app.gis.natural_earth_surface import LandSegmentValidation, NaturalEarthAvailability
from backend.app.gis.surface import SurfaceClassification, SurfaceType, WaterDomain
from backend.app.models import GeoJsonLineString, HistoricalClaim, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, HistoricalTravelMode, PlaceSpatialSemantics
from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology


def _road(identifier, geometry):
    return RomanRoadSegment(
        record_id=identifier, part_index=0, geometry=tuple(geometry), road_type="Main Road", route_type=1,
        segment_status="Certain", name=identifier, citation="c", bibliography="b",
        chronology=RoadChronology(None, None, None, None, {}), average_slope=None, passability=None, shape_length=None,
    )


def _point(identifier, longitude, latitude, claim_id=None):
    return HistoricalRoutePoint(
        sequence=1,
        historical_place=HistoricalPlace(
            id=identifier, canonical_name=identifier, longitude=longitude, latitude=latitude,
            source="fixture", confidence=0.9, spatial_semantics=PlaceSpatialSemantics.SETTLEMENT,
            coordinate_role="representative_point",
        ),
        event_summary="evidence-backed endpoint", evidence_refs=[identifier], confidence=0.9,
        claim_ids=[claim_id] if claim_id else [], coordinate_role="representative_point",
    )


def _route(points, claims=None):
    return HistoricalRoute(
        id="route", event_id="event", name="fixture", period="fixture", ordered_points=points,
        geometry=GeoJsonLineString(coordinates=[(point.historical_place.longitude, point.historical_place.latitude) for point in points]),
        historical_confidence=0.9, claims=claims or [],
    )


class OceanNorth:
    audit = type("Audit", (), {"availability": NaturalEarthAvailability.AVAILABLE})()

    def classify(self, latitude, longitude):
        if latitude >= 0.2:
            return SurfaceClassification(SurfaceType.WATER, "fixture", 1.0, "interior", {}, WaterDomain.OCEAN)
        return SurfaceClassification(SurfaceType.LAND, "fixture", 1.0, "interior", {}, WaterDomain.UNKNOWN)

    def validate_land_segment(self, _coordinates):
        return LandSegmentValidation(False, "clear_of_land")


class AlwaysLand(OceanNorth):
    def classify(self, latitude, longitude):
        return SurfaceClassification(SurfaceType.LAND, "fixture", 1.0, "interior", {}, WaterDomain.UNKNOWN)


class InlandWater(OceanNorth):
    def classify(self, latitude, longitude):
        if latitude >= 0.2:
            return SurfaceClassification(SurfaceType.WATER, "fixture", 1.0, "interior", {}, WaterDomain.INLAND_WATER)
        return SurfaceClassification(SurfaceType.LAND, "fixture", 1.0, "interior", {}, WaterDomain.UNKNOWN)


def _long_graph():
    return RomanRoadGraph.from_segments([
        _road("south-a", ((0, 0), (0, -2))),
        _road("south", ((0, -2), (0.5, -2))),
        _road("south-b", ((0.5, -2), (0.5, 0))),
    ], snap_tolerance_m=5)


def test_simulation_mode_does_not_override_explicit_historical_mode():
    assert select_simulation_mode(historical_mode="LAND", land_distance_m=1000, mixed_distance_m=10) == "LAND"
    assert select_simulation_mode(historical_mode="SEA", land_distance_m=1000, mixed_distance_m=10) == "SEA"
    assert select_simulation_mode(historical_mode="UNKNOWN", land_distance_m=1000, mixed_distance_m=700) == "LAND_SEA_MIXED"
    assert select_simulation_mode(historical_mode="UNKNOWN", land_distance_m=1000, mixed_distance_m=900) == "LAND"


def test_unknown_mode_can_select_shorter_mixed_simulation_without_new_historical_points():
    points = [_point("a", 0, 0), _point("b", 0.5, 0)]
    route = _route(points)
    before = [point.historical_place.id for point in route.ordered_points]
    result = RomanRoadRouteOrchestrator(
        RomanRoadCandidateService(_long_graph(), max_access_distance_m=2_000),
        terrain_route_service=GeographicCandidateRouteService(),
        maritime_surface=OceanNorth(),
    ).build_roman_road_candidates(route)
    assert [point.historical_place.id for point in route.ordered_points] == before
    assert result.legs[0].travel_mode is HistoricalTravelMode.UNKNOWN
    assert result.legs[0].simulation_route_mode == "LAND_SEA_MIXED"
    roles = [item.segment_type for item in result.geometry_segments]
    assert "direct_water_edge" in roles
    assert "simulated_coastal_access" in roles
    assert "failed_gap" not in roles
    payload = RomanRoadPresentationService().present(route, result)
    coastal = [item for item in payload.geojson["features"] if item["properties"].get("segment_role") == "simulated_coastal_access"]
    assert coastal
    assert coastal[0]["properties"]["simulation_provenance"] == "simulated_coastal_access"
    assert all(item["properties"].get("layer_type") != "historical_anchor" or item["properties"]["name"] in {"a", "b"} for item in payload.geojson["features"])


def test_inland_land_route_is_not_forced_to_sea():
    graph = RomanRoadGraph.from_segments([_road("ab", ((0, 0), (0.01, 0)))], snap_tolerance_m=5)
    route = _route([_point("a", 0, 0), _point("b", 0.01, 0)])
    result = RomanRoadRouteOrchestrator(
        RomanRoadCandidateService(graph, max_access_distance_m=2_000),
        maritime_surface=AlwaysLand(),
    ).build_roman_road_candidates(route)
    assert result.legs[0].simulation_route_mode == "LAND"
    assert result.legs[0].travel_mode is HistoricalTravelMode.UNKNOWN
    assert {item.segment_type for item in result.geometry_segments} <= {"roman_road", "access_connector"}


def test_explicit_land_claim_rejects_a_shorter_sea_option():
    claim = HistoricalClaim(id="land-leg", claim_type="ORDERING", text="marched", confidence=0.9, travel_mode=HistoricalTravelMode.LAND)
    route = _route([_point("a", 0, 0, claim.id), _point("b", 0.5, 0, claim.id)], [claim])
    result = RomanRoadRouteOrchestrator(
        RomanRoadCandidateService(_long_graph(), max_access_distance_m=2_000),
        maritime_surface=OceanNorth(),
    ).build_roman_road_candidates(route)
    assert result.legs[0].travel_mode is HistoricalTravelMode.LAND
    assert result.legs[0].simulation_route_mode == "LAND"
    assert "direct_water_edge" not in {item.segment_type for item in result.geometry_segments}


def test_explicit_sea_keeps_historical_mode_and_does_not_call_the_road_service():
    claim = HistoricalClaim(id="sea-leg", claim_type="ORDERING", text="sailed", confidence=0.9, travel_mode=HistoricalTravelMode.SEA)

    class Roads:
        def build(self, *_args):
            raise AssertionError("explicit SEA must not enter road planning")

    route = _route([_point("a", 0, 0, claim.id), _point("b", 0.5, 0, claim.id)], [claim])
    result = RomanRoadRouteOrchestrator(Roads(), maritime_surface=OceanNorth()).build_roman_road_candidates(route)
    assert result.legs[0].travel_mode is HistoricalTravelMode.SEA
    assert result.legs[0].simulation_route_mode == "SEA"
    assert "simulated_coastal_access" in {item.segment_type for item in result.geometry_segments}


def test_lakes_are_not_used_as_ocean_access():
    route = _route([_point("a", 0, 0), _point("b", 0.5, 0)])
    result = RomanRoadRouteOrchestrator(
        RomanRoadCandidateService(_long_graph(), max_access_distance_m=2_000),
        maritime_surface=InlandWater(),
    ).build_roman_road_candidates(route)
    assert result.legs[0].simulation_route_mode == "LAND"
    assert "direct_water_edge" not in {item.segment_type for item in result.geometry_segments}


def test_access_connector_distance_is_its_own_geometry():
    graph = RomanRoadGraph.from_segments([_road("ab", ((0, 0), (1, 0)))], snap_tolerance_m=5)
    route = _route([_point("a", 0.001, 0), _point("b", 0.999, 0)])
    result = RomanRoadRouteOrchestrator(RomanRoadCandidateService(graph, max_access_distance_m=500)).build_roman_road_candidates(route)
    payload = RomanRoadPresentationService().present(route, result)
    connectors = [item for item in payload.geojson["features"] if item["properties"].get("segment_role") == "access_connector"]
    road = next(item for item in payload.geojson["features"] if item["properties"].get("segment_role") == "roman_road")
    assert connectors
    assert connectors[0]["properties"]["segment_distance_m"] < road["properties"]["segment_distance_m"]
    assert connectors[0]["properties"]["segment_distance_m"] < 1_000
