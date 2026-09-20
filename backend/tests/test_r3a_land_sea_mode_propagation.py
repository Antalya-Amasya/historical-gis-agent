from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.candidate_routes.roman_roads import RomanRoadCandidateStatus
from backend.app.models import EventActorStatus, GeoJsonLineString, HistoricalClaim, HistoricalPlace, HistoricalRoute, HistoricalRoutePoint, HistoricalTravelMode
from backend.app.routes.canonical_route_adapter import _anchor_relation_for_observation
from backend.app.routes.movement_semantics import classify_historical_travel_mode
from backend.app.routes.route_observations import ObservationOrderingAuthority, ObservationOrderingRelation, RouteObservation, RouteObservationKind
from backend.app.route_orchestrator import HistoricalRouteOrchestrator, MaritimePlannerUnavailableError


def test_evidence_local_mode_classifier_is_conservative():
    assert classify_historical_travel_mode(("Commander Alpha marched from City A to City B.",)) is HistoricalTravelMode.LAND
    assert classify_historical_travel_mode(("Commander Alpha sailed from Port A to Island B.",)) is HistoricalTravelMode.SEA
    assert classify_historical_travel_mode(("Commander Alpha moved from A to B.",)) is HistoricalTravelMode.UNKNOWN


def test_mode_classifier_fails_closed_for_conflict_and_unasserted_language():
    assert classify_historical_travel_mode(("Commander Alpha marched from A to B and sailed to C.",)) is HistoricalTravelMode.UNKNOWN
    assert classify_historical_travel_mode(("Commander Alpha did not sail from A to B.",)) is HistoricalTravelMode.UNKNOWN
    assert classify_historical_travel_mode(("Commander Alpha planned to sail from A to B.",)) is HistoricalTravelMode.UNKNOWN


def _point(identifier: str, claim_id: str) -> HistoricalRoutePoint:
    return HistoricalRoutePoint(
        sequence=1,
        historical_place=HistoricalPlace(id=identifier, canonical_name=identifier, latitude=1.0, longitude=1.0, confidence=0.9, source="test"),
        event_summary="attested movement", confidence=0.9, claim_ids=[claim_id],
    )


def test_sea_claim_never_enters_roman_road_or_terrain_planning():
    claim = HistoricalClaim(
        id="sea-leg", claim_type="ORDERING", text="sailed", confidence=0.9,
        travel_mode=HistoricalTravelMode.SEA,
    )
    route = HistoricalRoute(
        id="r", event_id="e", name="Alpha", period="200 BCE", geometry=GeoJsonLineString(coordinates=[]),
        historical_confidence=0.9, ordered_points=[_point("A", claim.id), _point("B", claim.id)], claims=[claim],
    )

    class CandidateService:
        def build(self, *_args):
            raise AssertionError("SEA must not enter road planning")

    class TerrainService:
        def build_between(self, *_args):
            raise AssertionError("SEA must not enter terrain planning")

    result = RomanRoadRouteOrchestrator(CandidateService(), terrain_route_service=TerrainService()).build_roman_road_candidates(route)
    assert result.legs[0].failure_status == "MARITIME_PLANNER_UNAVAILABLE"
    assert result.legs[0].travel_mode is HistoricalTravelMode.SEA

    try:
        HistoricalRouteOrchestrator().present(type("Intent", (), {"intent": "historical_route"})(), route, [])
    except MaritimePlannerUnavailableError as exc:
        assert str(exc) == "MARITIME_PLANNER_UNAVAILABLE"
    else:
        raise AssertionError("SEA must not enter the terrain planner")


def test_unknown_claim_preserves_roman_road_compatibility_path():
    claim = HistoricalClaim(id="unknown-leg", claim_type="ORDERING", text="moved", confidence=0.9)
    route = HistoricalRoute(
        id="r", event_id="e", name="Alpha", period="200 BCE", geometry=GeoJsonLineString(coordinates=[]),
        historical_confidence=0.9, ordered_points=[_point("A", claim.id), _point("B", claim.id)], claims=[claim],
    )

    calls = []

    class CandidateService:
        def build(self, *_args):
            calls.append(True)
            return type("Result", (), {"candidate": None, "status": RomanRoadCandidateStatus.DISCONNECTED, "limitation": "gap"})()

    result = RomanRoadRouteOrchestrator(CandidateService()).build_roman_road_candidates(route)
    assert calls == [True]
    assert result.legs[0].travel_mode is HistoricalTravelMode.UNKNOWN


def test_canonical_adapter_preserves_per_relation_mode_without_mode_borrowing():
    observations = {
        "a": RouteObservation("a", RouteObservationKind.PLACE, "e1", "A", "Alpha", EventActorStatus.EXPLICIT, ("ev",)),
        "b": RouteObservation("b", RouteObservationKind.PLACE, "e1", "B", "Alpha", EventActorStatus.EXPLICIT, ("ev",)),
    }
    direct = ObservationOrderingRelation("a", "b", ObservationOrderingAuthority.AFTER_SUBORDINATE, ("e1",), ("ev",), "SAME_MOVEMENT_EVENT", HistoricalTravelMode.SEA)
    structural = ObservationOrderingRelation("a", "b", ObservationOrderingAuthority.SOURCE_STRUCTURAL_ORDER, ("e1", "e2"), ("ev",), "SOURCE_STRUCTURAL_ORDER")
    assert _anchor_relation_for_observation(direct, observations).travel_mode is HistoricalTravelMode.SEA
    assert _anchor_relation_for_observation(structural, observations).travel_mode is HistoricalTravelMode.UNKNOWN
