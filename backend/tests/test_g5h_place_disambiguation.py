"""G5H context-aware historical place disambiguation tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.geography.place_disambiguation import (
    PlaceResolutionContext,
    ResolvedCoMention,
    evaluate_candidate,
    filter_resolution_candidates,
)
from backend.app.models import HistoricalPlace, PlaceSpatialSemantics
from geography_mcp.service import GeographyService


def _ctx(**kwargs) -> PlaceResolutionContext:
    return PlaceResolutionContext(**kwargs)


def test_sole_candidate_context_conflict_not_resolved():
    service = GeographyService()
    statement = (
        "He proceeded also at once to alter the government, placing thirty rulers in the city, "
        "and ten in the Piraeus: he put, also, a garrison into the Acropolis"
    )
    piraeus = service.resolve_ancient_place_payload(
        "Piraeus", period="401 BCE", source_statement=statement, place_role="RELATED_PLACE",
    )
    assert piraeus["found"] is True
    resolved = [{
        "name": piraeus["canonical_name"],
        "latitude": piraeus["latitude"],
        "longitude": piraeus["longitude"],
        "spatial_semantics": piraeus.get("spatial_semantics"),
    }]
    payload = service.resolve_ancient_place_payload(
        "Acropolis",
        period="401 BCE",
        source_statement=statement,
        place_role="RELATED_PLACE",
        co_mentions=["Piraeus"],
        resolved_co_mentions=resolved,
    )
    assert payload["found"] is False
    assert payload["status"] == "AMBIGUOUS"
    assert payload["disambiguation_diagnostics"][0]["passed"] is False


def test_sole_candidate_compatible_context_resolves():
    service = GeographyService()
    payload = service.resolve_ancient_place_payload(
        "Brundisium",
        period="48 BCE",
        source_statement="He crossed from Brundisium toward Epirus.",
        place_role="ORIGIN",
    )
    assert payload["found"] is True
    assert payload["canonical_name"] == "Brundisium"


def test_multiple_candidates_period_excludes_to_one():
    service = GeographyService()
    payload = service.resolve_ancient_place_payload(
        "Asia",
        period="88 BCE",
        source_statement="Thence he passed to Asia after fleeing Pontus.",
        place_role="DESTINATION",
        co_mentions=["Pontus"],
    )
    assert payload["found"] is True
    assert "province" in payload["canonical_name"].casefold() or payload["spatial_semantics"] == "region"


def test_multiple_candidates_period_insufficient_stays_ambiguous():
    payload = GeographyService().resolve_ancient_place_payload("Asia", period="88 BCE")
    assert payload["found"] is False
    assert payload["status"] == "AMBIGUOUS"
    assert payload["candidate_count"] >= 2


def test_type_mismatch_filters_plaza_endpoint():
    candidate = {
        "pleiades_id": "1",
        "canonical_name": "Plaza Site",
        "place_types": ["plaza"],
        "coordinate_available": True,
        "matched_name_details": [],
    }
    result = evaluate_candidate(
        candidate,
        None,
        _ctx(place_role="ORIGIN"),
        event_window=(-100, -50),
    )
    assert result.passed is False
    assert "type_incompatible" in result.failed_filters


def test_co_mention_broad_basin_conflict_excludes_candidate():
    candidate = {
        "pleiades_id": "540663113",
        "canonical_name": "Triangular Forum",
        "place_types": ["plaza"],
        "coordinate_available": True,
        "matched_name_details": [{"name_start": 1700, "name_end": 2100, "attestations": [{"time_period": "modern"}]}],
    }
    place = HistoricalPlace(
        id="pleiades-540663113",
        canonical_name="Triangular Forum",
        latitude=40.7488316,
        longitude=14.4877038,
        source="test",
        confidence=0.5,
        coordinate_role="representative_point",
        spatial_semantics=PlaceSpatialSemantics.UNKNOWN,
        source_id="540663113",
    )
    result = evaluate_candidate(
        candidate,
        place,
        _ctx(
            period="401 BCE",
            place_role="RELATED_PLACE",
            resolved_co_mentions=(ResolvedCoMention("Peiraieus/Piraeus", 37.937222, 23.644609),),
        ),
        event_window=(-401, -401),
    )
    assert result.passed is False
    assert "geo_basin_incompatible" in result.failed_filters


def test_co_mention_ambiguity_remains_ambiguous():
    payload = GeographyService().resolve_ancient_place_payload(
        "Apollonia",
        period="48 BCE",
        source_statement="He marched through Apollonia.",
        place_role="RELATED_PLACE",
    )
    assert payload["found"] is False
    assert payload["status"] == "AMBIGUOUS"


def test_correct_candidate_absent_fail_closed():
    payload = GeographyService().resolve_ancient_place_payload(
        "Acropolis",
        period="401 BCE",
        source_statement="garrison into the Acropolis",
        place_role="RELATED_PLACE",
    )
    assert payload["found"] is False
    assert payload["status"] in {"AMBIGUOUS", "NOT_FOUND"}


def test_region_candidate_preserves_centroid_limitation():
    payload = GeographyService().resolve_ancient_place_payload(
        "Hispania",
        period="80 BCE",
        source_statement="He campaigned in Hispania.",
        place_role="DESTINATION",
    )
    assert payload["found"] is True
    assert payload["coordinate_role"] == "regional_centroid"
    assert payload["spatial_semantics"] == "region"


def test_unlocated_candidate_not_auto_promoted_for_anchor_role():
    province_place = HistoricalPlace(
        id="pleiades-981509",
        canonical_name="Asia (Roman province)",
        latitude=38.46,
        longitude=27.77,
        source="test",
        confidence=0.55,
        coordinate_role="regional_centroid",
        spatial_semantics=PlaceSpatialSemantics.REGION,
        source_id="981509",
    )
    status, places, candidates, diagnostics = filter_resolution_candidates(
        status="AMBIGUOUS",
        places=(province_place,),
        candidates=(
            {
                "pleiades_id": "915832",
                "canonical_name": "Asia",
                "place_types": ["unlocated"],
                "coordinate_available": False,
                "matched_name_details": [],
            },
            {
                "pleiades_id": "981509",
                "canonical_name": "Asia (Roman province)",
                "place_types": ["province-2"],
                "coordinate_available": True,
                "matched_name_details": [],
            },
        ),
        context=_ctx(period="88 BCE", place_role="DESTINATION"),
    )
    assert status == "UNIQUE"
    assert len(places) == 1
    assert places[0].canonical_name == "Asia (Roman province)"
    assert any(item["pleiades_id"] == "915832" and not item["passed"] for item in diagnostics)


def test_period_metadata_absent_does_not_exclude():
    candidate = {
        "pleiades_id": "981509",
        "canonical_name": "Asia (Roman province)",
        "place_types": ["province-2"],
        "coordinate_available": True,
        "matched_name_details": [{"original_name": "Asia", "name_start": None, "name_end": None, "attestations": []}],
    }
    result = evaluate_candidate(candidate, None, _ctx(period="88 BCE"), event_window=(-88, -88))
    assert result.passed is True


def test_curated_alias_still_resolves_with_compatible_context():
    payload = GeographyService().resolve_ancient_place_payload(
        "Athens",
        period="401 BCE",
        source_statement="He left Athens for the Peloponnesus.",
        place_role="ORIGIN",
    )
    assert payload["found"] is True
    assert payload["canonical_name"] == "Athenae"


@pytest.mark.parametrize(
    "name,period,statement,role,expect_found",
    [
        ("Gades", "218 BCE", "through the straits of Cadiz", "ORIGIN", True),
        ("Rhodanus", "58 BCE", "crossed the Rhodanus", "TRAVERSAL", True),
        ("Alpes", "218 BCE", "crossed the Alps", "TRAVERSAL", True),
        ("Adriatic", "48 BCE", "crossed the Adriatic", "TRAVERSAL", True),
        ("Atlantic", "218 BCE", "sailed the Atlantic", "TRAVERSAL", True),
        ("Syria", "60 BCE", "marched into Syria", "DESTINATION", False),
        ("Armenia", "66 BCE", "retreated through Armenia", "TRAVERSAL", False),
        ("Pontus", "88 BCE", "fled out of Pontus", "ORIGIN", True),
        ("Hydaspes", "326 BCE", "crossed the Hydaspes", "TRAVERSAL", True),
        ("Bactria", "327 BCE", "advanced into Bactria", "DESTINATION", True),
    ],
)
def test_benchmark_controls(name, period, statement, role, expect_found):
    payload = GeographyService().resolve_ancient_place_payload(
        name, period=period, source_statement=statement, place_role=role,
    )
    assert payload.get("found") is expect_found


def test_benchmark_file_has_at_least_100_mentions():
    path = Path(__file__).with_name("g5h_geography_benchmark.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["mentions"]) >= 100
