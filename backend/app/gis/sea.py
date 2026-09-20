"""Direct computational sea-edge validation; not historical travel or graph search."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import acos, asin, atan2, ceil, cos, isfinite, radians, sin, sqrt
from typing import Protocol
from .natural_earth_surface import MODERN_GEOGRAPHY_WARNING, NaturalEarthAvailability
from .surface import SurfaceClassifier, SurfaceType, WaterDomain
from .transport import SearchState, TransportMode

DIRECT_WATER_SAMPLING_INTERVAL_M = 5000.0
GIS_RECONSTRUCTION_LIMITATION = (
    "Plausible GIS reconstruction between historical constraints; not an exact historical sailing track."
)
_DIRECT_WATER_LIMITATIONS = (MODERN_GEOGRAPHY_WARNING, GIS_RECONSTRUCTION_LIMITATION)

class SeaEdgeStatus(str, Enum): AVAILABLE = "AVAILABLE"; UNAVAILABLE = "UNAVAILABLE"
class CostComponentStatus(str, Enum): ACTIVE="ACTIVE"; NOT_MODELED="NOT_MODELED"
@dataclass(frozen=True)
class MaritimeCostBreakdown:
    distance_cost: float; season_cost: float = 0.; wind_cost: float = 0.; current_cost: float = 0.; hazard_cost: float = 0.
    @property
    def total_cost(self): return self.distance_cost + self.season_cost + self.wind_cost + self.current_cost + self.hazard_cost
    component_status: tuple[tuple[str,CostComponentStatus],...]=(("distance",CostComponentStatus.ACTIVE),("season",CostComponentStatus.NOT_MODELED),("wind",CostComponentStatus.NOT_MODELED),("current",CostComponentStatus.NOT_MODELED),("hazard",CostComponentStatus.NOT_MODELED))
class MaritimeCostModel(Protocol):
    def cost_for(self, distance_m: float) -> MaritimeCostBreakdown: ...
@dataclass(frozen=True)
class BaselineMaritimeCostModel:
    distance_weight: float = 1.
    def cost_for(self, distance_m: float) -> MaritimeCostBreakdown: return MaritimeCostBreakdown(distance_m * self.distance_weight)
@dataclass(frozen=True)
class SeaEdge:
    source: SearchState; target: SearchState; physical_distance_m: float; status: SeaEdgeStatus; reason: str; sample_count: int; sampling_interval_m: float; cost_breakdown: MaritimeCostBreakdown | None; surface_counts: tuple[int,int,int,int]=(0,0,0,0); metadata: tuple[tuple[str,str],...]=()
    @property
    def physical_distance_km(self): return self.physical_distance_m / 1000
def _distance(a,b):
    la,oa,lb,ob=map(radians,(a.latitude,a.longitude,b.latitude,b.longitude)); return 2*6371008.8*asin(sqrt(sin((lb-la)/2)**2+cos(la)*cos(lb)*sin((ob-oa)/2)**2))
def build_sea_edge(source, target, classifier: SurfaceClassifier, *, max_sampling_interval_m=5000., cost_model=None):
    if max_sampling_interval_m<=0: raise ValueError("max_sampling_interval_m must be positive")
    distance=_distance(source,target); common=dict(source=source,target=target,physical_distance_m=distance,sampling_interval_m=max_sampling_interval_m)
    meta=(("geometry_role","computational_maritime_candidate"),("historical_compatibility","NOT_EVALUATED"),("wind","not_modeled"),("current","not_modeled"))
    if source.mode is not TransportMode.SEA or target.mode is not TransportMode.SEA: return SeaEdge(status=SeaEdgeStatus.UNAVAILABLE,reason="source and target must be SEA",sample_count=0,cost_breakdown=None,metadata=meta,**common)
    count=max(2,ceil(distance/max_sampling_interval_m)+1)
    counts={SurfaceType.WATER:0,SurfaceType.LAND:0,SurfaceType.UNKNOWN:0,SurfaceType.BLOCKED:0}
    for n in range(count):
        try: lat, lon = _slerp(source.latitude,source.longitude,target.latitude,target.longitude,n/(count-1))
        except ValueError: return SeaEdge(status=SeaEdgeStatus.UNAVAILABLE,reason="great-circle interpolation unavailable",sample_count=n,sampling_interval_m=max_sampling_interval_m,cost_breakdown=None,**{k:v for k,v in common.items() if k!='sampling_interval_m'})
        surface=classifier.classify(lat,lon); counts[surface.surface_type]+=1
        if surface.surface_type is not SurfaceType.WATER: return SeaEdge(status=SeaEdgeStatus.UNAVAILABLE,reason=f"surface sample {n} is {surface.surface_type.value}",sample_count=n+1,cost_breakdown=None,surface_counts=tuple(counts[x] for x in (SurfaceType.WATER,SurfaceType.LAND,SurfaceType.UNKNOWN,SurfaceType.BLOCKED)),metadata=meta,**common)
    return SeaEdge(status=SeaEdgeStatus.AVAILABLE,reason="all direct samples WATER; heuristic only, wind/current not modeled",sample_count=count,cost_breakdown=(cost_model or BaselineMaritimeCostModel()).cost_for(distance),surface_counts=tuple(counts[x] for x in (SurfaceType.WATER,SurfaceType.LAND,SurfaceType.UNKNOWN,SurfaceType.BLOCKED)),metadata=meta,**common)

@dataclass(frozen=True)
class DirectWaterEdgePlan:
    available: bool
    reason: str
    coordinates: tuple[tuple[float, float], ...] = ()
    physical_distance_m: float = 0.0
    sample_count: int = 0
    sampling_interval_m: float = DIRECT_WATER_SAMPLING_INTERVAL_M
    planner: str = "DIRECT_WATER_EDGE"
    direct_water_validated: bool = False
    detour_used: bool = False
    provenance: tuple[tuple[str, str], ...] = ()
    limitations: tuple[str, ...] = _DIRECT_WATER_LIMITATIONS


def _gap(reason: str, **kwargs) -> DirectWaterEdgePlan:
    kwargs.setdefault("limitations", _DIRECT_WATER_LIMITATIONS)
    return DirectWaterEdgePlan(available=False, reason=reason, **kwargs)


def _ocean_interior(surface) -> bool:
    return (
        surface.surface_type is SurfaceType.WATER
        and getattr(surface, "water_domain", None) is WaterDomain.OCEAN
        and surface.status == "interior"
    )


def plan_direct_water_edge(lat1, lon1, lat2, lon2, classifier, *, max_sampling_interval_m=DIRECT_WATER_SAMPLING_INTERVAL_M):
    """Validate a densified geodesic as ocean-only GIS geometry. Not historical authority."""
    if getattr(getattr(classifier, "audit", None), "availability", None) is not NaturalEarthAvailability.AVAILABLE:
        return _gap("maritime_surface_unavailable")
    values = (lat1, lon1, lat2, lon2)
    if any(not isfinite(value) for value in values):
        return _gap("invalid_coordinate")
    if not -90 <= lat1 <= 90 or not -90 <= lat2 <= 90 or not -180 <= lon1 <= 180 or not -180 <= lon2 <= 180:
        return _gap("invalid_coordinate")
    if not _ocean_interior(classifier.classify(lat1, lon1)) or not _ocean_interior(classifier.classify(lat2, lon2)):
        return _gap("endpoint_not_ocean_interior")
    source = type("P", (), {"latitude": lat1, "longitude": lon1})()
    target = type("P", (), {"latitude": lat2, "longitude": lon2})()
    distance = _distance(source, target)
    if distance < 1.0:
        return _gap("zero_length_leg")
    count = max(2, ceil(distance / max_sampling_interval_m) + 1)
    samples: list[tuple[float, float]] = []
    try:
        for n in range(count):
            lat, lon = _slerp(lat1, lon1, lat2, lon2, n / (count - 1))
            if not _ocean_interior(classifier.classify(lat, lon)):
                return _gap("sample_not_ocean_interior", physical_distance_m=distance, sample_count=n + 1)
            samples.append((lon, lat))
    except ValueError as exc:
        return _gap(str(exc), physical_distance_m=distance)
    samples[0], samples[-1] = (lon1, lat1), (lon2, lat2)
    land = classifier.validate_land_segment(tuple(samples))
    if getattr(land, "intersects_land", None) is not False:
        return _gap(getattr(land, "status", "land_intersection"), physical_distance_m=distance, sample_count=count)
    metadata = dict(getattr(classifier.classify(lat1, lon1), "metadata", {}) or {})
    provenance = tuple((key, metadata[key]) for key in ("dataset_name", "dataset_version", "source_sha256") if key in metadata)
    return DirectWaterEdgePlan(
        available=True,
        reason="direct_water_validated",
        coordinates=tuple(samples),
        physical_distance_m=distance,
        sample_count=count,
        sampling_interval_m=max_sampling_interval_m,
        direct_water_validated=True,
        provenance=provenance,
        limitations=_DIRECT_WATER_LIMITATIONS,
    )


def _slerp(lat1,lon1,lat2,lon2,f):
    a,b,c,d=map(radians,(lat1,lon1,lat2,lon2)); dot=sin(a)*sin(c)+cos(a)*cos(c)*cos(d-b); omega=acos(max(-1,min(1,dot)))
    if abs(omega-3.141592653589793)<1e-10: raise ValueError("near-antipodal")
    if omega<1e-12: return lat1,lon1
    left,right=sin((1-f)*omega)/sin(omega),sin(f*omega)/sin(omega)
    x=left*cos(a)*cos(b)+right*cos(c)*cos(d); y=left*cos(a)*sin(b)+right*cos(c)*sin(d); z=left*sin(a)+right*sin(c)
    lat,lon=atan2(z,sqrt(x*x+y*y)),atan2(y,x)
    if not isfinite(lat+lon): raise ValueError("non-finite")
    return round(lat*180/3.141592653589793,12),round(lon*180/3.141592653589793,12)
