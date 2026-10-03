"""READ-ONLY: production-default A* envelope + CRS error + corner-cutting probe."""
import sys, time, math, inspect
sys.path.insert(0, r"C:\D\python\historical-gis-cursor")
from backend.app.candidate_routes.terrain import MosaicDEMProvider
from backend.app.candidate_routes.geographic import (GeographicGridSpec, GeographicCandidateRouteService,
                                                     DEFAULT_MAX_GRID_CELLS, GridTooLargeError)
from backend.app.candidate_routes.models import ArmyProfile
from backend.app.candidate_routes.grid import SyntheticGrid, GridPoint
from backend.app.candidate_routes.engine import CandidateRouteEngine, NoPathError
from backend.app.models import HistoricalRoutePoint, HistoricalPlace

HGT = r"C:\data\srtm-hgt"
print("DEFAULT_MAX_GRID_CELLS =", DEFAULT_MAX_GRID_CELLS)
print("build_between signature:", inspect.signature(GeographicCandidateRouteService.build_between))
prov = MosaicDEMProvider(HGT)
svc = GeographicCandidateRouteService(prov)

def pt(pid, lon, lat):
    return HistoricalRoutePoint(sequence=1,
        historical_place=HistoricalPlace(id=pid, canonical_name=pid, longitude=lon, latitude=lat,
                                         source="audit", confidence=0.5),
        event_summary="audit", evidence_refs=["ev"], confidence=0.5,
        coordinate_role="representative_point")

print("\n=== A* ENVELOPE AT PRODUCTION DEFAULTS (padding_km=2.0, cell_size_m=1000) ===")
print(f"{'route':34} {'km':>7} {'grid':>12} {'cells':>8} {'sec':>7} {'pts':>5} {'result':>12}")
CASES = [
    ("short  Rome->Ostia",       12.4964, 41.9028, 12.2333, 41.7333),
    ("short  Capua->Naples",     14.2500, 41.0833, 14.2500, 40.8500),
    ("medium Rome->Capua",       12.4964, 41.9028, 14.2500, 41.0833),
    ("medium Rome->Brundisium",  12.4964, 41.9028, 17.9429, 40.6387),
    ("long   Rome->Carthage",    12.4964, 41.9028, 10.3231, 36.8531),
    ("long   Rome->Alexandria",  12.4964, 41.9028, 29.9187, 31.2001),
    ("vlong  Rome->Antioch",     12.4964, 41.9028, 36.2021, 36.2021),
]
for label, a1, b1, a2, b2 in CASES:
    km = GeographicGridSpec.from_anchor_coordinates
    # straight-line distance for reference (haversine)
    R = 6371.0
    p1, p2 = math.radians(b1), math.radians(b2); dl = math.radians(a2 - a1)
    h = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    dist = 2*R*math.asin(min(1, math.sqrt(h)))
    t0 = time.perf_counter()
    try:
        r = svc.build_between(pt("a", a1, b1), pt("b", a2, b2), ArmyProfile(name="audit"))
        el = time.perf_counter() - t0
        print(f"{label:34} {dist:>7.0f} {str(r.grid_width)+'x'+str(r.grid_height):>12} "
              f"{r.grid_width*r.grid_height:>8} {el:>7.2f} {len(r.geometry.coordinates):>5} {'OK':>12}")
    except GridTooLargeError as e:
        print(f"{label:34} {dist:>7.0f} {'-':>12} {'-':>8} {time.perf_counter()-t0:>7.2f} {'-':>5} {'GRID_TOO_LARGE':>12}")
    except NoPathError:
        print(f"{label:34} {dist:>7.0f} {'-':>12} {'-':>8} {time.perf_counter()-t0:>7.2f} {'-':>5} {'NO_PATH':>12}")
    except Exception as e:
        print(f"{label:34} {dist:>7.0f} {'-':>12} {'-':>8} {time.perf_counter()-t0:>7.2f} {'-':>5} {type(e).__name__:>12}")

print("\n=== COST MODEL DEFAULT WEIGHTS ===")
from backend.app.candidate_routes.models import ArmyProfile as AP
p = AP(name="default")
for f in ("distance_weight", "slope_weight", "terrain_weight", "max_slope_ratio", "name"):
    print(f"  {f:20} {getattr(p, f, '<absent>')}")

print("\n=== CORNER-CUTTING (diagonal between two blocked orthogonal cells) ===")
g = SyntheticGrid(3, 3)
# block the two orthogonal cells around the centre diagonal
g.set_cell(GridPoint(1, 0), blocked=True)
g.set_cell(GridPoint(0, 1), blocked=True)
eng = CandidateRouteEngine()
try:
    path = eng.find_path(g, GridPoint(0, 0), GridPoint(1, 1), p)
    print(f"  path found through blocked-corner diagonal: {[ (q.x,q.y) for q in path.points ]}")
    print("  -> corner cutting IS permitted" if len(path.points) == 2 else "  -> detour taken")
except NoPathError:
    print("  NoPathError -> corner cutting prevented")

print("\n=== CRS ERROR: single-reference equirectangular over long spans ===")
for label, a1, b1, a2, b2 in [("Rome->Capua", 12.4964,41.9028,14.2500,41.0833),
                              ("Rome->Carthage", 12.4964,41.9028,10.3231,36.8531),
                              ("Rome->Antioch", 12.4964,41.9028,36.2021,36.2021)]:
    spec = GeographicGridSpec.from_anchor_coordinates((a1,b1),(a2,b2), padding_km=2.0, cell_size_m=1000.0, max_grid_cells=10**9)
    # true haversine vs projection distance
    R = 6371.0
    p1, p2 = math.radians(b1), math.radians(b2); dl = math.radians(a2-a1)
    h = math.sin((p2-p1)/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    true_km = 2*R*math.asin(min(1, math.sqrt(h)))
    proj_km = spec.projection.distance_m((a1,b1),(a2,b2))/1000.0
    print(f"  {label:16} true={true_km:8.1f} km  projected={proj_km:8.1f} km  error={100.0*(proj_km-true_km)/true_km:+6.2f}%")
print("\nDONE")
