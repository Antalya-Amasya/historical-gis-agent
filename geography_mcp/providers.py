from abc import ABC, abstractmethod
import httpx
from geography_mcp.service import GeoPoint, ElevationResult


class ElevationProvider(ABC):
    @abstractmethod
    def get(self, point: GeoPoint) -> ElevationResult: ...


class OpenMeteoElevationProvider(ElevationProvider):
    endpoint = "https://api.open-meteo.com/v1/elevation"
    source = "Copernicus DEM GLO-90 via Open-Meteo"

    def __init__(self, client=None):
        self.client = client or httpx.Client()

    def _fetch(self, points: list[GeoPoint]) -> list[float]:
        response = self.client.get(self.endpoint, params={"latitude": ",".join(str(p.latitude) for p in points), "longitude": ",".join(str(p.longitude) for p in points)}, timeout=15)
        response.raise_for_status()
        values = response.json().get("elevation")
        if not isinstance(values, list) or len(values) != len(points):
            raise RuntimeError("Open-Meteo elevation response count mismatch")
        return [float(value) for value in values]

    def get(self, point: GeoPoint) -> ElevationResult:
        return ElevationResult(point=point, elevation_m=self._fetch([point])[0], provider="open_meteo", source=self.source)

    def get_many(self, points: list[GeoPoint]) -> list[ElevationResult]:
        values=[]
        for index in range(0, len(points), 100): values.extend(self._fetch(points[index:index+100]))
        return [ElevationResult(point=point,elevation_m=value,provider="open_meteo",source=self.source) for point,value in zip(points,values)]
