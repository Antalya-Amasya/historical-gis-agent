from backend.app.candidate_routes.roman_road_orchestration import RomanRoadRouteOrchestrator
from backend.app.gis.natural_earth_surface import LandSegmentValidation, NaturalEarthAvailability
from backend.app.gis.sea import plan_direct_water_edge
from backend.app.gis.surface import SurfaceClassification, SurfaceType, WaterDomain
from backend.app.models import HistoricalTravelMode
from backend.tests.test_r3b2_direct_water_edge import _point, _provider, _RoadService, _route


class _Capable:
    def __init__(self, audit="omitted"):
        if audit != "omitted":
            self.audit = audit

    def classify(self, latitude, longitude):
        return SurfaceClassification(
            surface_type=SurfaceType.WATER, source="unauthoritative",
            status="interior", water_domain=WaterDomain.OCEAN,
        )

    def validate_land_segment(self, coordinates):
        return LandSegmentValidation(False, "clear_of_land")


def _reject(provider):
    plan = plan_direct_water_edge(-2.0, -2.0, -5.0, -5.0, provider)
    assert plan.available is False
    assert plan.direct_water_validated is False
    assert plan.coordinates == ()
    assert plan.reason == "maritime_surface_unavailable"
    return plan


def test_missing_audit_none_audit_and_missing_availability_fail_closed():
    _reject(_Capable())
    _reject(_Capable(audit=None))
    _reject(_Capable(audit=type("Audit", (), {})()))


def test_forged_string_available_is_not_authoritative():
    _reject(_Capable(audit=type("Audit", (), {"availability": "AVAILABLE"})()))
    _reject(_Capable(audit=type("Audit", (), {"availability": True})()))


def test_non_available_enum_states_fail_closed():
    for state in (
        NaturalEarthAvailability.MISSING,
        NaturalEarthAvailability.HASH_MISMATCH,
        NaturalEarthAvailability.INVALID,
        NaturalEarthAvailability.UNAVAILABLE,
    ):
        _reject(_Capable(audit=type("Audit", (), {"availability": state})()))


def test_approved_available_from_manifest_still_accepts_open_ocean(tmp_path):
    _, provider = _provider(tmp_path)
    assert provider.audit.availability is NaturalEarthAvailability.AVAILABLE
    plan = plan_direct_water_edge(-2.0, -2.0, -5.0, -5.0, provider)
    assert plan.available is True
    assert plan.reason == "direct_water_validated"
    assert plan.coordinates


def test_orchestration_rejects_unaudited_injected_provider():
    sea = _route(
        [_point("P", "leg", -2.0, -2.0), _point("Q", "leg", -5.0, -5.0)],
        HistoricalTravelMode.SEA,
    )
    result = RomanRoadRouteOrchestrator(_RoadService(), maritime_surface=_Capable()).build_roman_road_candidates(sea)
    assert result.legs[0].failure_status == "MARITIME_PLANNER_UNAVAILABLE"
    assert result.legs[0].terrain_candidate is None
    assert result.geometry_segments[0].coordinates == []
    assert result.legs[0].reconstruction_method != "DIRECT_WATER_EDGE"
