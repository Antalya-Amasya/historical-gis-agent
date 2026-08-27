import pytest

from backend.app.gis.ports import (
    HistoricalPort,
    InMemoryPortRegistry,
    PortStatus,
    PortSurfaceContext,
    validate_port_surface_context,
)
from backend.app.gis.dem import ElevationSample, ElevationStatus
from backend.app.gis.surface import MockSurfaceClassifier, NoSurfaceClassifier, SurfaceType


def synthetic_port(port_id="PORT_A", *, latitude=40.0, longitude=10.0):
    return HistoricalPort(
        port_id=port_id,
        canonical_name=port_id,
        latitude=latitude,
        longitude=longitude,
        historical_names=(f"Old {port_id}",),
        status=PortStatus.SUPPORTED,
        confidence=0.8,
        supporting_evidence_ids=("evidence-1",),
        source_documents=("fixture-source",),
        historical_status_source="synthetic_fixture",
        coordinate_source="synthetic_fixture",
        coordinate_role="fixture_coordinate",
        coordinate_uncertainty_km=1.0,
        supports_land_access=True,
        supports_sea_access=True,
    )


def test_explicit_registration_and_read_only_lookups_preserve_provenance():
    port = synthetic_port()
    registry = InMemoryPortRegistry([port])

    assert registry.get_by_id("PORT_A") is port
    assert registry.find_by_name("old port_a") == (port,)
    assert port.supporting_evidence_ids == ("evidence-1",)
    assert port.historical_status_source == "synthetic_fixture"
    assert port.coordinate_source == "synthetic_fixture"
    assert registry.list_ports(status=PortStatus.CONFIRMED) == ()
    with pytest.raises(TypeError):
        port.metadata["mutate"] = "no"


def test_near_lookup_uses_haversine_and_respects_tolerance():
    nearby = synthetic_port("PORT_A", latitude=40.0, longitude=10.0)
    distant = synthetic_port("PORT_B", latitude=42.0, longitude=10.0)
    registry = InMemoryPortRegistry([distant, nearby])

    assert registry.find_near(40.01, 10.01, tolerance_km=5) == (nearby,)
    assert registry.find_near(40.01, 10.01, tolerance_km=0.01) == ()
    with pytest.raises(ValueError):
        registry.find_near(40, 10, tolerance_km=-1)


def test_unknown_provenance_remains_unknown_and_surface_cannot_create_ports():
    port = HistoricalPort(port_id="PORT_UNKNOWN", canonical_name="Unknown", latitude=0, longitude=0)
    registry = InMemoryPortRegistry()
    classifier = MockSurfaceClassifier()
    classifier.register(0, 0, SurfaceType.WATER)

    assert port.historical_status_source == "UNKNOWN"
    assert port.coordinate_source == "UNKNOWN"
    assert registry.list_ports() == ()
    assert registry.find_near(0, 0, 10) == ()
    assert validate_port_surface_context(port, classifier).context is PortSurfaceContext.COMPATIBLE
    assert registry.list_ports() == ()


def test_dem_availability_states_cannot_create_ports_or_sea_semantics():
    registry = InMemoryPortRegistry()
    valid_zero = ElevationSample(0.0, ElevationStatus.VALID, "fixture_dem")
    missing = ElevationSample(None, ElevationStatus.MISSING, "fixture_dem")

    assert valid_zero.status is ElevationStatus.VALID
    assert missing.status is ElevationStatus.MISSING
    assert registry.list_ports() == ()


def test_surface_diagnostic_is_non_mutating_and_handles_unavailable_context():
    port = synthetic_port()
    original = port

    diagnostic = validate_port_surface_context(port, NoSurfaceClassifier())

    assert diagnostic.context is PortSurfaceContext.UNAVAILABLE
    assert diagnostic.surface.surface_type is SurfaceType.UNKNOWN
    assert port is original
    assert port.status is PortStatus.SUPPORTED
