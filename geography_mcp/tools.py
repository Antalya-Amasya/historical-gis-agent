"""Shared, local and auditable geography data used by both MCP transports and Mock Agent."""

from backend.app.models import HistoricalPlace

PLEIADES_SOURCE = "Pleiades: A Gazetteer of Past Places"

# This intentionally small repository is curated by hand for the Phase 1 demo.
# Coordinates are copied from the cited Pleiades representative points; no model
# or geocoder is used to create or infer them.
DEMO_PLACES = {
    "carthago": HistoricalPlace(
        id="pleiades-314921",
        canonical_name="Carthago",
        modern_name="Carthage, Tunisia",
        latitude=36.853056,
        longitude=10.323056,
        period="750 BCE–640 CE",
        source=PLEIADES_SOURCE,
        source_id="314921",
        source_url="https://pleiades.stoa.org/places/314921",
        confidence=0.95,
        uncertain=False,
    ),
    "carthago nova": HistoricalPlace(
        id="pleiades-265849",
        canonical_name="Carthago Nova",
        modern_name="Cartagena, Spain",
        latitude=37.599896,
        longitude=-0.98452,
        period="228 BCE–640 CE",
        source=PLEIADES_SOURCE,
        source_id="265849",
        source_url="https://pleiades.stoa.org/places/265849",
        confidence=0.90,
        uncertain=False,
    ),
}


def resolve_demo_place(name: str) -> HistoricalPlace | None:
    return DEMO_PLACES.get(name.strip().lower())


def hannibal_demo_places() -> list[HistoricalPlace]:
    """Return fixed campaign-context places; this does not assert a route."""
    return [DEMO_PLACES["carthago"], DEMO_PLACES["carthago nova"]]
