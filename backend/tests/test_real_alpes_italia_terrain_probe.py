"""Controlled local acceptance for a real DEM; skipped when the local dataset is absent."""

from pathlib import Path

import pytest

from backend.app.candidate_routes.historical_reconstruction import RealTerrainGraphProvider
from backend.app.candidate_routes.terrain import MosaicDEMProvider
from backend.app.core.config import settings
from backend.app.models import Evidence, GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRouteIntent, HistoricalRoutePoint
from backend.app.route_orchestrator import HistoricalRouteOrchestrator


DEM_DIR = Path(settings.dem_hgt_dir) if settings.dem_hgt_dir else None


@pytest.mark.skipif(DEM_DIR is None or not DEM_DIR.is_dir(), reason="controlled local SRTM acceptance requires DEM_HGT_DIR")
def test_real_alpes_to_italia_is_an_auditable_algorithmic_candidate():
    evidence = Evidence(id="e54", author="Livy", work="History of Rome", locator="Book 21", excerpt="came to Italy having crossed the Alps")
    alpes = HistoricalPlace(id="alpes", canonical_name="Alpes", longitude=7.0, latitude=44.0, source="fixture", confidence=.7, uncertain=True, coordinate_role="regional_centroid")
    italia = HistoricalPlace(id="italia", canonical_name="Italia", longitude=12.5, latitude=42.5, source="fixture", confidence=.7, uncertain=True, coordinate_role="regional_centroid")
    historical_route = HistoricalRoute(
        id="alpes-italia", event_id="hannibal", name="Alpes to Italia", period="218 BCE",
        ordered_points=[
            HistoricalRoutePoint(sequence=1, historical_place=alpes, event_summary="Crossed the Alps then came to Italy.", evidence_refs=[evidence.id], confidence=.7, claim_ids=["movement-e54"]),
            HistoricalRoutePoint(sequence=2, historical_place=italia, event_summary="Crossed the Alps then came to Italy.", evidence_refs=[evidence.id], confidence=.7, claim_ids=["movement-e54"]),
        ],
        geometry=GeoJsonLineString(coordinates=[(7.0, 44.0), (12.5, 42.5)]), evidence_refs=[evidence.id], historical_confidence=.7,
    )
    response = HistoricalRouteOrchestrator(
        terrain_graph_provider=RealTerrainGraphProvider(MosaicDEMProvider(DEM_DIR)), cell_size_m=5_000,
    ).present(HistoricalRouteIntent(campaign_id="hannibal_italy_campaign", entity="hannibal_alpine_crossing", route_type="movement"), historical_route, [evidence])
    properties = response.route_geojson["properties"]
    ledger = properties["route_quality"]["segment_ledger"]

    assert response.route_geojson["geometry"]["coordinates"][0] == [7.0, 44.0]
    assert response.route_geojson["geometry"]["coordinates"][-1] == [12.5, 42.5]
    assert properties["terrain_source"] == "offline_srtm_hgt_mosaic"
    assert properties["applied_constraints"] == ["dem_availability_blocking", "dem_nodata_blocking", "slope_cost"]
    assert response.route.score.total_cost == ledger[0]["search_cost_total"]
    assert ledger[0]["physical_distance_km"] > 300
    assert ledger[0]["physical_distance_km"] > response.route.score.distance_cost
    assert ledger[0]["geometry_role"] == "algorithmic_candidate"
    assert ledger[0]["source_anchor_id"] == "alpes" and ledger[0]["target_anchor_id"] == "italia"
    assert ledger[0]["nodata_or_missing_count"] == 0
