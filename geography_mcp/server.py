from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.app.models import HistoricalPlace
from geography_mcp.tools import resolve_demo_place

app = FastAPI(title="Historical Military GIS Geography MCP", version="0.1.0")

class PlaceRequest(BaseModel):
    name: str
    period: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "geography-mcp-demo"}


@app.post("/tools/resolve_ancient_place", response_model=HistoricalPlace)
def resolve_ancient_place(request: PlaceRequest) -> HistoricalPlace:
    place = resolve_demo_place(request.name)
    if place is None:
        raise HTTPException(status_code=404, detail="Place is not in the Phase 0 demo repository")
    return place
