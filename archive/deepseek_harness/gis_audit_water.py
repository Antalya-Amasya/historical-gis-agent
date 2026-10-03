"""READ-ONLY probe: does the LAND LAND-planner cross water?
Builds real grids from the production MosaicDEMProvider and inspects blocked cells
and the actual A* path. Writes nothing; never touches Chroma."""
import sys, math
sys.path.insert(0, r"C:\D\python\historical-gis-cursor")
from backend.app.candidate_routes.terrain import MosaicDEMProvider
from backend.app.candidate_routes.geographic import GeographicGridSpec, GeographicCandidateRouteService
from backend.app.candidate_routes.models import ArmyProfile
from backend.app.models import HistoricalRoutePoint, HistoricalPlace

HGT = r"C:\data\srtm-hgt"
prov = MosaicDEMProvider(HGT)
print("DEM provider:", prov.source, "| hgt_dir:", prov.hgt_dir)

def place(pid, name, lon, lat):
    return HistoricalPlace(id=pid, canonical_name=name, longitude=lon, latitude=lat,
                           source="audit_probe", confidence=0.5)

def point(p):
    return HistoricalRoutePoint(sequence=1, historical_place=p, event_summary="audit",
                                evidence_refs=["ev"], confidence=0.5,
                                coordinate_role="representative_point")

CASES = [
    # (label, lonA, latA, lonB, latB, expectation)
    ("A Rome -> Carthage (open sea)",        12.4964, 41.9028,  10.3231, 36.8531, "should NOT be LAND-traversable"),
    ("B Oricum -> Brundisium (Adriatic)",    19.4833, 40.3167,  17.9429, 40.6387, "should NOT be LAND-traversable"),
    ("C Athens -> Corinth (across isthmus)", 23.7275, 37.9838,  22.9289, 37.9386, "land detour exists"),
    ("D mainland -> Sicily (Messina strait)",15.6500, 38.2500,  15.5500, 38.1900, "narrow strait"),
    ("E Lesbos island -> Turkey mainland",   26.5500, 39.1000,  26.3500, 39.0000, "island -> mainland"),
    ("F Rome -> Capua (clear land)",         12.4964, 41.9028,  14.2500, 41.0833, "clear land, should work"),
    ("G across Lake Trasimene",              12.0333, 43.1333,  12.1500, 43.1000, "lake"),
]

for label, lonA, latA, lonB, latB, note in CASES:
    print("\n" + "=" * 92)
    print(f"{label}   [{note}]")
    try:
        spec = GeographicGridSpec.from_anchor_coordinates(
            (lonA, latA), (lonB, latB), padding_km=5.0, cell_size_m=2_000.0, max_grid_cells=200_000)
    except Exception as e:
        print("  grid spec failed:", type(e).__name__, e); continue
    grid = prov.build_grid(spec, resolution_m=spec.cell_size_m)
    total = spec.width * spec.height
    blocked = sum(1 for p in grid._cells if grid.cell(p).blocked)
    nodata = sum(1 for p in grid._cells if grid.cell(p).terrain == "no_data")
    zero_elev = sum(1 for p in grid._cells if not grid.cell(p).blocked and abs(grid.cell(p).elevation_m) < 1.0)
    surfaces = set(grid.cell(p).surface.type if hasattr(grid.cell(p).surface, "type") else str(grid.cell(p).surface) for p in list(grid._cells)[:1])
    print(f"  grid {spec.width}x{spec.height} = {total} cells | blocked={blocked} ({100.0*blocked/total:.1f}%) "
          f"| no_data={nodata} | unblocked elevation<1m={zero_elev} ({100.0*zero_elev/total:.1f}%)")
    print(f"  cell.surface populated? {surfaces}")
    svc = GeographicCandidateRouteService(prov)
    try:
        route = svc.build_between(point(place("a", "A", lonA, latA)), point(place("b", "B", lonB, latB)),
                                  ArmyProfile(name="audit"), padding_km=5.0, cell_size_m=2_000.0,
                                  max_grid_cells=200_000)
        coords = route.geometry.coordinates
        straight_km = spec.projection.distance_m((lonA, latA), (lonB, latB)) / 1000.0
        print(f"  PATH OK: points={len(coords)} distance={route.metrics.distance_km:.1f} km "
              f"(straight-line {straight_km:.1f} km, ratio {route.metrics.distance_km/straight_km:.3f})")
        # how much of the path lies over near-zero elevation (likely water on this DEM)
        lows = 0
        for lon, lat in coords:
            try:
                e = prov.get_elevation(lon, lat)
            except Exception:
                continue
            if abs(e) < 1.0: lows += 1
        print(f"  path samples at elevation<1m: {lows}/{len(coords)} ({100.0*lows/len(coords):.1f}%)  <- water proxy")
        print(f"  first={coords[0]} last={coords[-1]}")
    except Exception as e:
        print(f"  PATH FAILED: {type(e).__name__}: {e}")
print("\nDONE")
