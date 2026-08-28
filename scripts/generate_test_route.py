"""Generate a Lyon-to-Milan terrain-route GeoJSON for visual inspection.

Run from the repository root:
    .venv\\Scripts\\python.exe scripts\\generate_test_route.py

Set DEM_HGT_DIR to a directory containing SRTM .hgt tiles for real terrain.
"""
from __future__ import annotations

import json
import sys
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.candidate_routes.engine import NoPathError
from backend.app.candidate_routes.historical_reconstruction import RealTerrainGraphProvider
from backend.app.gis.srtm import SrtmElevationService
from backend.app.core.config import settings
from backend.app.models import Evidence, GeoJsonLineString, HistoricalPlace, HistoricalRoute, HistoricalRouteIntent, HistoricalRoutePoint
from backend.app.route_orchestrator import HistoricalRouteOrchestrator

CELL_SIZE_M = 5_000.0
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "test_route.geojson"


def haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Return the geographic length of one WGS84 line segment."""
    lon_a, lat_a = first
    lon_b, lat_b = second
    lat_a, lon_a, lat_b, lon_b = map(radians, (lat_a, lon_a, lat_b, lon_b))
    value = 2 * 6_371.0088 * asin(sqrt(
        sin((lat_b - lat_a) / 2) ** 2
        + cos(lat_a) * cos(lat_b) * sin((lon_b - lon_a) / 2) ** 2
    ))
    return value


def route_inputs() -> tuple[HistoricalRouteIntent, HistoricalRoute, list[Evidence]]:
    evidence = Evidence(
        id="visual-validation-fixture",
        author="Project fixture",
        work="GIS visual validation",
        locator="local test input",
        excerpt="Lyon and Milan are supplied endpoints for terrain-routing visual validation.",
    )
    lyon = HistoricalPlace(
        id="visual-lyon", canonical_name="Lyon", longitude=4.8357, latitude=45.7640,
        source="visual-validation fixture", confidence=1.0,
    )
    milan = HistoricalPlace(
        id="visual-milan", canonical_name="Milan", longitude=9.1900, latitude=45.4642,
        source="visual-validation fixture", confidence=1.0,
    )
    route = HistoricalRoute(
        id="visual-lyon-milan", event_id="visual-validation", name="Lyon to Milan terrain test",
        period="test fixture",
        ordered_points=[
            HistoricalRoutePoint(sequence=1, historical_place=lyon, event_summary="Supplied western endpoint.", evidence_refs=[evidence.id], confidence=1.0),
            HistoricalRoutePoint(sequence=2, historical_place=milan, event_summary="Supplied eastern endpoint.", evidence_refs=[evidence.id], confidence=1.0),
        ],
        geometry=GeoJsonLineString(coordinates=[(lyon.longitude, lyon.latitude), (milan.longitude, milan.latitude)]),
        evidence_refs=[evidence.id], historical_confidence=0.0,
        limitations=["Synthetic endpoints supplied solely for GIS visual validation."],
    )
    return HistoricalRouteIntent(campaign_id="visual_terrain_validation"), route, [evidence]


def real_provider_from_settings() -> RealTerrainGraphProvider | None:
    if not settings.dem_hgt_dir:
        return None
    hgt_dir = Path(settings.dem_hgt_dir)
    if not hgt_dir.is_dir() or not any(hgt_dir.glob("*.hgt")):
        return None
    return RealTerrainGraphProvider(SrtmElevationService(hgt_dir))


def build_geojson(response, *, real_terrain: bool) -> dict[str, object]:
    coordinates = [tuple(point) for point in response.route_geojson["geometry"]["coordinates"]]
    cumulative_distance_km = [0.0]
    for first, second in zip(coordinates, coordinates[1:]):
        cumulative_distance_km.append(cumulative_distance_km[-1] + haversine_km(first, second))

    total_distance_km = cumulative_distance_km[-1]
    total_cost = response.route.score.total_cost
    vertex_metrics = [
        {
            "index": index,
            "coordinate": coordinate,
            "cumulative_distance_km": distance_km,
            # A* stores total route cost, not a per-vertex cost ledger. This makes
            # the visualization estimate explicit instead of implying exact values.
            "cumulative_cost_estimate": total_cost * distance_km / total_distance_km if total_distance_km else 0.0,
        }
        for index, (coordinate, distance_km) in enumerate(zip(coordinates, cumulative_distance_km))
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coordinates},
        "properties": {
            "route_name": "Lyon to Milan terrain test",
            "cell_size_m": CELL_SIZE_M,
            "terrain_source": response.route_geojson["properties"].get("terrain_source"),
            "real_terrain": real_terrain,
            "total_cost": total_cost,
            # The current real-terrain graph fixes terrain_multiplier at 1.0, so
            # the presentation terrain component is entirely calibrated slope cost.
            "slope_cost": response.route.score.terrain_cost,
            "total_distance_km": total_distance_km,
            "vertex_metrics": vertex_metrics,
        },
    }


def main() -> None:
    intent, route, evidence = route_inputs()
    terrain_graph_provider = real_provider_from_settings()
    using_mock = terrain_graph_provider is None
    if using_mock:
        print("WARNING: Using mock terrain. Visual validation requires real DEM data.")

    orchestrator = HistoricalRouteOrchestrator(
        terrain_graph_provider=terrain_graph_provider,
        cell_size_m=CELL_SIZE_M,
    )
    try:
        response = orchestrator.present(intent, route, evidence)
    except NoPathError:
        if using_mock:
            raise
        print("WARNING: Using mock terrain. Visual validation requires real DEM data.")
        using_mock = True
        response = HistoricalRouteOrchestrator(cell_size_m=CELL_SIZE_M).present(intent, route, evidence)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(build_geojson(response, real_terrain=not using_mock), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {OUTPUT_PATH}")
    print(f"terrain_source={response.route_geojson['properties'].get('terrain_source')} cell_size_m={CELL_SIZE_M}")


if __name__ == "__main__":
    main()
