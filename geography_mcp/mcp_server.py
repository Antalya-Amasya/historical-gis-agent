"""Standard MCP stdio entry point for Geography tools.

Run with: ``python -m geography_mcp.mcp_server``.
It intentionally has no web server lifecycle or container dependency.
"""

from mcp.server.fastmcp import FastMCP

from geography_mcp.tools import resolve_demo_place

mcp = FastMCP("Historical Military GIS Geography")


@mcp.tool()
def resolve_ancient_place(name: str, period: str | None = None) -> dict:
    """Resolve a historical place from the Phase 0 demo repository."""
    place = resolve_demo_place(name)
    if place is None:
        return {"found": False, "message": "Place is not in the Phase 0 demo repository"}
    return {"found": True, **place.model_dump(mode="json")}


if __name__ == "__main__":
    mcp.run(transport="stdio")
