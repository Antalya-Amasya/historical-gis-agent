import pytest

from backend.app.gis.dem import ElevationSample, ElevationStatus
from backend.app.gis.ports import HistoricalPort, PortSurfaceContext, validate_port_surface_context
from backend.app.gis.surface import MockSurfaceClassifier, SurfaceType
from backend.app.gis.transport import (
    FixedTransitionCostModel,
    PortTransitionRequest,
    SearchState,
    TransitionEligibilityStatus,
    TransitionEvidenceStatus,
    TransportMode,
    evaluate_port_transition,
)


def port(*, land=True, sea=True):
    return HistoricalPort(
        port_id="PORT_A", canonical_name="Synthetic Port", latitude=40.0, longitude=10.0,
        historical_status_source="synthetic_fixture", coordinate_source="synthetic_fixture",
        supporting_evidence_ids=("port-evidence",), supports_land_access=land, supports_sea_access=sea,
    )


def request(*, current_port=None, from_mode=TransportMode.LAND, to_mode=TransportMode.SEA, status=TransitionEvidenceStatus.SUPPORTED, diagnostic=None):
    return PortTransitionRequest(
        port=port() if current_port is None else current_port,
        from_mode=from_mode, to_mode=to_mode, transition_evidence_status=status,
        supporting_evidence_ids=("transition-evidence",), event_id="synthetic-event", surface_diagnostic=diagnostic,
    )


def test_land_and_sea_are_distinct_search_states_at_the_same_location():
    land = SearchState("same-place", 40.0, 10.0, TransportMode.LAND)
    sea = SearchState("same-place", 40.0, 10.0, TransportMode.SEA)

    assert land != sea
    assert {land, sea} == {land, sea}


def test_explicit_supported_port_allows_both_directions_and_preserves_provenance():
    land_to_sea = evaluate_port_transition(request())
    sea_to_land = evaluate_port_transition(request(from_mode=TransportMode.SEA, to_mode=TransportMode.LAND))

    assert land_to_sea.status is TransitionEligibilityStatus.ALLOWED
    assert sea_to_land.status is TransitionEligibilityStatus.ALLOWED
    assert land_to_sea.supporting_evidence_ids == ("transition-evidence",)
    assert land_to_sea.port_evidence_ids == ("port-evidence",)
    assert land_to_sea.event_id == "synthetic-event"


def test_no_port_or_coastal_context_or_water_never_creates_a_transition():
    no_port = evaluate_port_transition(PortTransitionRequest(None, TransportMode.LAND, TransportMode.SEA))
    classifier = MockSurfaceClassifier()
    classifier.register(40.0, 10.0, SurfaceType.WATER)
    diagnostic = validate_port_surface_context(port(), classifier)
    water_only = evaluate_port_transition(
        PortTransitionRequest(None, TransportMode.LAND, TransportMode.SEA, surface_diagnostic=diagnostic)
    )

    assert no_port.status is TransitionEligibilityStatus.DENIED
    assert water_only.status is TransitionEligibilityStatus.DENIED
    assert diagnostic.context is PortSurfaceContext.COMPATIBLE


def test_port_existence_without_event_transition_evidence_is_unknown():
    result = evaluate_port_transition(request(status=TransitionEvidenceStatus.UNKNOWN))

    assert result.status is TransitionEligibilityStatus.UNKNOWN
    assert "does not establish" in result.reason


def test_surface_and_dem_diagnostics_do_not_change_transition_permission():
    classifier = MockSurfaceClassifier()
    classifier.register(40.0, 10.0, SurfaceType.LAND)
    diagnostic = validate_port_surface_context(port(), classifier)
    with_diagnostic = evaluate_port_transition(request(diagnostic=diagnostic))
    no_diagnostic = evaluate_port_transition(request())
    valid_zero = ElevationSample(0.0, ElevationStatus.VALID, "fixture_dem")
    missing = ElevationSample(None, ElevationStatus.MISSING, "fixture_dem")

    assert diagnostic.context is PortSurfaceContext.INCOMPATIBLE
    assert with_diagnostic.status is no_diagnostic.status is TransitionEligibilityStatus.ALLOWED
    assert valid_zero.status is ElevationStatus.VALID and missing.status is ElevationStatus.MISSING


def test_transition_without_both_explicit_access_capabilities_is_denied_and_cost_is_deterministic():
    denied = evaluate_port_transition(request(current_port=port(land=True, sea=False)))
    allowed = evaluate_port_transition(request())
    cost = FixedTransitionCostModel(2.5)

    assert denied.status is TransitionEligibilityStatus.DENIED
    assert cost.cost_for(allowed) == cost.cost_for(allowed) == 2.5
    with pytest.raises(ValueError):
        cost.cost_for(denied)
    with pytest.raises(ValueError):
        FixedTransitionCostModel(-1)
