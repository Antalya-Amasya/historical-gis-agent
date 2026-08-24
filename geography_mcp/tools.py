"""Small, audited Pleiades-backed ancient-place repository for MCP resolution."""

from backend.app.models import HistoricalPlace

PLEIADES_SOURCE = "Pleiades: A Gazetteer of Past Places"

# Representative coordinates are copied from the cited Pleiades place resources.
# River representatives are schematic reference points, not crossing locations.
DEMO_PLACES = {
    "carthago": HistoricalPlace(id="pleiades-314921", canonical_name="Carthago", modern_name="Carthage, Tunisia", latitude=36.853056, longitude=10.323056, period="750 BCE–640 CE", source=PLEIADES_SOURCE, source_id="314921", source_url="https://pleiades.stoa.org/places/314921", confidence=0.95),
    "carthago nova": HistoricalPlace(id="pleiades-265849", canonical_name="Carthago Nova", modern_name="Cartagena, Spain", latitude=37.599896, longitude=-0.98452, period="228 BCE–640 CE", source=PLEIADES_SOURCE, source_id="265849", source_url="https://pleiades.stoa.org/places/265849", confidence=0.90),
    "rhodanus": HistoricalPlace(id="pleiades-148168", canonical_name="Rhodanus", modern_name="Rhône, France", latitude=43.33167, longitude=4.84861, period="750 BCE–640 CE", source=PLEIADES_SOURCE, source_id="148168", source_url="https://pleiades.stoa.org/places/148168", confidence=0.70, uncertain=True),
    "padus": HistoricalPlace(id="pleiades-393469", canonical_name="Padus", modern_name="Po, Italy", latitude=44.952389, longitude=12.432028, period="750 BCE–640 CE", source=PLEIADES_SOURCE, source_id="393469", source_url="https://pleiades.stoa.org/places/393469", confidence=0.70, uncertain=True),
}

ALIASES = {
    "new carthage": "carthago nova",
    "cartagena": "carthago nova",
    "rhone": "rhodanus",
    "rhône": "rhodanus",
    "po": "padus",
}


def resolve_demo_place(name: str) -> HistoricalPlace | None:
    key = name.strip().lower()
    return DEMO_PLACES.get(ALIASES.get(key, key))


def hannibal_demo_places() -> list[HistoricalPlace]:
    return [DEMO_PLACES["carthago"], DEMO_PLACES["carthago nova"]]
