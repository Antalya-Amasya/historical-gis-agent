from math import asin, cos, radians, sin, sqrt
from pydantic import BaseModel, Field
from backend.app.models import HistoricalPlace
from backend.app.geography.place_registry import resolve_with_status


class GeoPoint(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class DistanceResult(BaseModel):
    meters: float
    kilometers: float
    method: str = "haversine_geodesic"
    source: str = "local"
    note: str = "Geodesic distance; not road distance."


class ElevationResult(BaseModel):
    point: GeoPoint
    elevation_m: float
    provider: str
    source: str


class ElevationProfile(BaseModel):
    points: list[ElevationResult]
    minimum_m: float
    maximum_m: float
    provider: str
    source: str


class MockElevationProvider:
    def get(self, point: GeoPoint) -> ElevationResult:
        return ElevationResult(point=point, elevation_m=round(abs(point.latitude) * 7 + abs(point.longitude) * 3, 1), provider="mock", source="mock")


class GeographyService:
    def __init__(self, elevation_provider=None, provider_mode: str = "mock"):
        if elevation_provider is not None:
            self.elevation_provider = elevation_provider
        elif provider_mode == "open_meteo":
            from geography_mcp.providers import OpenMeteoElevationProvider
            self.elevation_provider = OpenMeteoElevationProvider()
        else:
            self.elevation_provider = MockElevationProvider()
    def resolve_ancient_place(self, name: str, period: str | None = None) -> HistoricalPlace | None:
        resolution = resolve_with_status(name)
        return resolution.places[0] if resolution.status in {"CURATED", "UNIQUE"} else None
    def resolve_ancient_place_payload(self, name: str, period: str | None = None) -> dict:
        resolution = resolve_with_status(name)
        if resolution.status in {"CURATED", "UNIQUE"} and len(resolution.places) == 1:
            return {"found": True, **resolution.places[0].model_dump(mode="json")}
        if resolution.status == "AMBIGUOUS":
            return {
                "found": False,
                "ambiguous": True,
                "candidate_count": resolution.candidate_count,
                "candidates": list(resolution.candidates),
            }
        if resolution.status == "UNLOCATED":
            return {
                "found": False,
                "authority_status": "UNLOCATED",
                "candidate_count": resolution.candidate_count,
                "candidates": list(resolution.candidates),
            }
        if resolution.status == "UNAVAILABLE":
            # Keep the established MCP response contract while the internal
            # resolver retains the explicit UNAVAILABLE diagnostic.
            return {"found": False}
        return {"found": False}
    def calculate_distance(self, point_a: GeoPoint, point_b: GeoPoint) -> DistanceResult:
        lat1,lon1,lat2,lon2=map(radians,(point_a.latitude,point_a.longitude,point_b.latitude,point_b.longitude))
        value=2*6371008.8*asin(sqrt(sin((lat2-lat1)/2)**2+cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2))
        return DistanceResult(meters=round(value,2),kilometers=round(value/1000,3))
    def get_elevation(self, point: GeoPoint) -> ElevationResult:
        return self.elevation_provider.get(point)
    def get_elevation_profile(self, points: list[GeoPoint]) -> ElevationProfile:
        if not points: raise ValueError("points must not be empty")
        values = self.elevation_provider.get_many(points) if hasattr(self.elevation_provider, "get_many") else [self.get_elevation(point) for point in points]
        return ElevationProfile(points=values,minimum_m=min(x.elevation_m for x in values),maximum_m=max(x.elevation_m for x in values),provider=values[0].provider,source=values[0].source)
