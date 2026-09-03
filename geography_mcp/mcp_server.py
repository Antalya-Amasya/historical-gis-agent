from mcp.server.fastmcp import FastMCP
from geography_mcp.service import GeographyService, GeoPoint
from backend.app.core.config import settings
mcp=FastMCP("Historical Military GIS Geography")
service=GeographyService(provider_mode=settings.geography_provider_mode)
@mcp.tool()
def resolve_ancient_place(
    name: str,
    period: str | None = None,
    source_statement: str | None = None,
    place_role: str | None = None,
    co_mentions: list[str] | None = None,
    resolved_co_mentions: list[dict] | None = None,
) -> dict:
    return service.resolve_ancient_place_payload(
        name,
        period,
        source_statement=source_statement,
        place_role=place_role,
        co_mentions=co_mentions,
        resolved_co_mentions=resolved_co_mentions,
    )
@mcp.tool()
def calculate_distance(point_a: GeoPoint, point_b: GeoPoint) -> dict:
    return service.calculate_distance(point_a,point_b).model_dump()
@mcp.tool()
def get_elevation(latitude: float, longitude: float) -> dict:
    return service.get_elevation(GeoPoint(latitude=latitude,longitude=longitude)).model_dump()
@mcp.tool()
def get_elevation_profile(points: list[GeoPoint]) -> dict:
    return service.get_elevation_profile(points).model_dump()
if __name__ == "__main__": mcp.run(transport="stdio")
