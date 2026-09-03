"""G5I typed non-settlement geography coverage tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.geography.feature_semantics import exact_anchor_eligible, place_limitations
from backend.app.models import HistoricalPlace, PlaceSpatialSemantics
from backend.app.routes.event_anchors import project_event_anchors
from backend.app.models import (
    EventGroundingStatus,
    EventPlaceResolutionStatus,
    EventPlaceRole,
    HistoricalEvent,
    HistoricalEventPlaceBinding,
    HistoricalEventPlaceMention,
    HistoricalEventType,
)
from geography_mcp.service import GeographyService


def _service() -> GeographyService:
    return GeographyService()


def test_sea_entity_resolves_typed_without_exact_anchor():
    payload = _service().resolve_ancient_place_payload(
        "Adriatic", period="48 BCE", source_statement="crossed the Adriatic", place_role="TRAVERSAL",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "sea"
    assert payload["coordinate_role"] == "feature_reference"
    assert any("not an exact" in item.casefold() for item in place_limitations(HistoricalPlace.model_validate(payload)))


@pytest.mark.parametrize("name", ["Adriatic", "Aegean", "Ionian", "Mediterranean", "Atlantic"])
def test_common_seas_resolve_typed(name: str):
    payload = _service().resolve_ancient_place_payload(
        name, period="48 BCE", source_statement=f"crossed the {name}", place_role="TRAVERSAL",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "sea"
    assert payload["coordinate_role"] == "feature_reference"


def test_strait_typed_traversal():
    payload = _service().resolve_ancient_place_payload(
        "Hellespont", period="480 BCE", source_statement="crossed the Hellespont", place_role="TRAVERSAL",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "strait"


def test_river_traversal_not_crossing_point():
    payload = _service().resolve_ancient_place_payload(
        "Rhodanus", period="58 BCE", source_statement="crossed the Rhodanus", place_role="TRAVERSAL",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "river"
    assert payload["coordinate_role"] == "representative_point"


def test_mountain_region_not_pass():
    payload = _service().resolve_ancient_place_payload(
        "Alpes", period="218 BCE", source_statement="crossed the Alps", place_role="TRAVERSAL",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "mountain_region"
    assert payload["coordinate_role"] == "regional_centroid"


def test_named_pass_may_anchor():
    payload = _service().resolve_ancient_place_payload(
        "Thermopylae", period="480 BCE", source_statement="at Thermopylae", place_role="EVENT_SITE",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "pass"
    assert payload["coordinate_role"] == "exact_site"


def test_region_endpoint_preserves_centroid_limitation():
    payload = _service().resolve_ancient_place_payload(
        "Italia", period="218 BCE", source_statement="entered Italy", place_role="DESTINATION",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "region"
    assert payload["coordinate_role"] == "regional_centroid"
    assert place_limitations(HistoricalPlace.model_validate(payload))


def test_island_destination_coarse():
    payload = _service().resolve_ancient_place_payload(
        "Sicily", period="264 BCE", source_statement="landed on Sicily", place_role="DESTINATION",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "island"
    assert payload["coordinate_role"] == "feature_centroid"


def test_named_port_exactish_anchor():
    payload = _service().resolve_ancient_place_payload(
        "Ostia", period="50 BCE", source_statement="sailed from Ostia", place_role="ORIGIN",
    )
    assert payload["found"] is True
    assert payload["spatial_semantics"] == "port"
    assert payload["coordinate_role"] == "exact_site"


def test_generic_harbor_unresolved():
    payload = _service().resolve_ancient_place_payload(
        "the harbor", period="50 BCE", source_statement="left the harbor", place_role="ORIGIN",
    )
    assert payload["found"] is False


def test_sea_not_projected_as_route_anchor():
    place = HistoricalPlace(
        id="pleiades-1004",
        canonical_name="Adriatic Sea",
        latitude=44.9,
        longitude=13.7,
        source="test",
        confidence=0.65,
        coordinate_role="feature_reference",
        spatial_semantics=PlaceSpatialSemantics.SEA,
    )
    event = HistoricalEvent(
        id="evt-1",
        name="adriatic crossing",
        event_type=HistoricalEventType.MOVEMENT,
        summary="crossed the Adriatic",
        period="48 BCE",
        grounding_status=EventGroundingStatus.EVIDENCE_GROUNDED,
        evidence_refs=["e1"],
        source_statements=["crossed the Adriatic"],
        place_bindings=[
            HistoricalEventPlaceBinding(
                mention=HistoricalEventPlaceMention(raw_text="Adriatic", canonical_hint="Adriatic", role=EventPlaceRole.DESTINATION, evidence_refs=["e1"]),
                place=place,
                role=EventPlaceRole.DESTINATION,
                resolution_status=EventPlaceResolutionStatus.RESOLVED,
                evidence_refs=["e1"],
            )
        ],
    )
    anchors, diagnostics = project_event_anchors([event], [type("E", (), {"id": "e1"})()])
    assert anchors == []
    assert any("NON_EXACT_FEATURE_ANCHOR" in code for code in diagnostics)


def test_acropolis_regression():
    payload = _service().resolve_ancient_place_payload(
        "Acropolis",
        period="401 BCE",
        source_statement="garrison into the Acropolis",
        place_role="RELATED_PLACE",
        co_mentions=["Piraeus"],
    )
    assert payload["found"] is False
    assert payload["status"] in {"AMBIGUOUS", "NOT_FOUND"}


def test_asia_regression_without_context():
    payload = _service().resolve_ancient_place_payload("Asia", period="88 BCE")
    assert payload["found"] is False
    assert payload["status"] == "AMBIGUOUS"


def test_unsafe_exact_anchor_count_zero_for_seas():
    violations = 0
    for name in ["Adriatic", "Aegean", "Ionian", "Mediterranean", "Atlantic"]:
        payload = _service().resolve_ancient_place_payload(name, place_role="TRAVERSAL")
        place = HistoricalPlace.model_validate({k: v for k, v in payload.items() if k != "found"})
        if place.coordinate_role == "exact_site":
            violations += 1
    assert violations == 0


def test_g5i_benchmark_file_has_required_quotas():
    path = Path(__file__).with_name("g5i_geography_benchmark.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    mentions = data["mentions"]
    assert len(mentions) >= 160
    counts: dict[str, int] = {}
    for item in mentions:
        counts[item["feature_type"]] = counts.get(item["feature_type"], 0) + 1
    assert counts.get("SEA", 0) >= 15
    assert counts.get("STRAIT", 0) >= 8
    assert counts.get("RIVER", 0) >= 15
    assert counts.get("MOUNTAIN", 0) + counts.get("PASS", 0) >= 15
    assert counts.get("REGION", 0) >= 20
    assert counts.get("ISLAND", 0) >= 10
    assert counts.get("PORT", 0) >= 10
    assert counts.get("SETTLEMENT", 0) >= 30


def test_g5i_benchmark_entity_resolution_rates():
    path = Path(__file__).with_name("g5i_geography_benchmark.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    service = _service()
    sea_total = sea_resolved = 0
    unsafe = 0
    for item in data["mentions"]:
        payload = service.resolve_ancient_place_payload(
            item["raw_mention"],
            period=item.get("period"),
            source_statement=item.get("source_statement"),
            place_role=item.get("role") or item.get("place_role"),
        )
        if item["feature_type"] == "SEA":
            sea_total += 1
            if payload.get("found"):
                sea_resolved += 1
            if payload.get("coordinate_role") == "exact_site":
                unsafe += 1
        if payload.get("found") and payload.get("spatial_semantics") in {"sea", "strait", "river", "mountain_region"}:
            if payload.get("coordinate_role") == "exact_site":
                unsafe += 1
    assert sea_total >= 15
    assert sea_resolved / sea_total >= 0.8
    assert unsafe == 0
