from math import asin, cos, radians, sin, sqrt
from pydantic import BaseModel, Field
from backend.app.models import HistoricalPlace
from backend.app.geography.place_registry import resolve_with_status
from backend.app.geography.place_disambiguation import PlaceResolutionContext, ResolvedCoMention


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


def _build_resolution_context(
    *,
    period: str | None = None,
    source_statement: str | None = None,
    place_role: str | None = None,
    co_mentions: list[str] | None = None,
    resolved_co_mentions: list[dict] | None = None,
) -> PlaceResolutionContext | None:
    resolved = tuple(
        ResolvedCoMention(
            name=str(item["name"]),
            latitude=float(item["latitude"]),
            longitude=float(item["longitude"]),
            spatial_semantics=item.get("spatial_semantics"),
        )
        for item in (resolved_co_mentions or [])
    )
    if not any([period, source_statement, place_role, co_mentions, resolved]):
        return None
    return PlaceResolutionContext(
        period=period,
        source_statement=source_statement,
        place_role=place_role,
        co_mentions=tuple(co_mentions or ()),
        resolved_co_mentions=resolved,
    )


class GeographyService:
    def __init__(self, elevation_provider=None, provider_mode: str = "mock"):
        if elevation_provider is not None:
            self.elevation_provider = elevation_provider
        elif provider_mode == "open_meteo":
            from geography_mcp.providers import OpenMeteoElevationProvider
            self.elevation_provider = OpenMeteoElevationProvider()
        else:
            self.elevation_provider = MockElevationProvider()

    def resolve_ancient_place(self, name: str, period: str | None = None, **context_kwargs) -> HistoricalPlace | None:
        resolution = resolve_with_status(name, _build_resolution_context(period=period, **context_kwargs))
        return resolution.places[0] if resolution.status in {"CURATED", "UNIQUE"} and len(resolution.places) == 1 else None

    def resolve_ancient_place_payload(self, name: str, period: str | None = None, **context_kwargs) -> dict:
        resolution = resolve_with_status(name, _build_resolution_context(period=period, **context_kwargs))
        if resolution.status in {"CURATED", "UNIQUE"} and len(resolution.places) == 1:
            return {"found": True, **resolution.places[0].model_dump(mode="json")}
        if resolution.status == "AMBIGUOUS":
            return {
                "found": False,
                "status": "AMBIGUOUS",
                "ambiguous": True,
                "candidate_count": resolution.candidate_count,
                "candidates": list(resolution.candidates),
                "disambiguation_diagnostics": list(resolution.disambiguation_diagnostics),
            }
        if resolution.status == "UNLOCATED":
            return {
                "found": False,
                "status": "UNLOCATED",
                "authority_status": "UNLOCATED",
                "candidate_count": resolution.candidate_count,
                "candidates": list(resolution.candidates),
                "disambiguation_diagnostics": list(resolution.disambiguation_diagnostics),
            }
        if resolution.status == "UNAVAILABLE":
            return {"found": False, "status": "UNAVAILABLE", "reason": resolution.reason}
        return {"found": False, "status": "NOT_FOUND"}

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
