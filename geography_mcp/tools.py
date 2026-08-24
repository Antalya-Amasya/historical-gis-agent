"""Small, audited Pleiades-backed ancient-place repository for MCP resolution."""

from backend.app.models import HistoricalPlace

PLEIADES_SOURCE = "Pleiades: A Gazetteer of Past Places"

# Coordinates are copied from cited Pleiades resources. Representative coordinates
# identify a river or mountain region only; they never claim a crossing or pass.
DEMO_PLACES = {
    "carthago": HistoricalPlace(id="pleiades-314921", canonical_name="Carthago", modern_name="Carthage, Tunisia", latitude=36.853056, longitude=10.323056, period="750 BCE–640 CE", source=PLEIADES_SOURCE, source_id="314921", source_url="https://pleiades.stoa.org/places/314921", confidence=0.95),
    "carthago nova": HistoricalPlace(id="pleiades-265849", canonical_name="Carthago Nova", modern_name="Cartagena, Spain", latitude=37.599896, longitude=-0.98452, period="228 BCE–640 CE", source=PLEIADES_SOURCE, source_id="265849", source_url="https://pleiades.stoa.org/places/265849", confidence=0.90),
    "rhodanus": HistoricalPlace(id="pleiades-148168", canonical_name="Rhodanus", modern_name="Rhône, France", latitude=43.33167, longitude=4.84861, period="750 BCE–640 CE", source=PLEIADES_SOURCE, source_id="148168", source_url="https://pleiades.stoa.org/places/148168", confidence=0.70, uncertain=True, coordinate_role="representative_point"),
    "iberus": HistoricalPlace(id="pleiades-246418", canonical_name="Iberus", modern_name="Ebro, Spain", latitude=40.72, longitude=0.863056, period="750 BCE–640 CE", source=PLEIADES_SOURCE, source_id="246418", source_url="https://pleiades.stoa.org/places/246418", confidence=0.70, uncertain=True, coordinate_role="representative_point"),
    "massalia": HistoricalPlace(id="pleiades-148127", canonical_name="Massalia", modern_name="Marseille, France", latitude=43.296854, longitude=5.382499, period="600 BCE–640 CE", source=PLEIADES_SOURCE, source_id="148127", source_url="https://pleiades.stoa.org/places/148127", confidence=0.90),
    "alpes": HistoricalPlace(id="pleiades-783", canonical_name="Alpes", modern_name="Alps", latitude=43.74465275, longitude=7.40183905, period="Roman period", source=PLEIADES_SOURCE, source_id="783", source_url="https://pleiades.stoa.org/places/783", confidence=0.45, uncertain=True, coordinate_role="regional_centroid"),
    "padus": HistoricalPlace(id="pleiades-393469", canonical_name="Padus", modern_name="Po, Italy", latitude=44.952389, longitude=12.432028, period="750 BCE–640 CE", source=PLEIADES_SOURCE, source_id="393469", source_url="https://pleiades.stoa.org/places/393469", confidence=0.70, uncertain=True, coordinate_role="representative_point"),
}

ALIASES = {
    "new carthage": "carthago nova",
    "cartagena": "carthago nova",
    "rhone": "rhodanus",
    "rhône": "rhodanus",
    "iber": "iberus",
    "ebro": "iberus",
    "massilia": "massalia",
    "marseilles": "massalia",
    "alps": "alpes",
    "po": "padus",
}


def resolve_demo_place(name: str) -> HistoricalPlace | None:
    key = name.strip().lower()
    return DEMO_PLACES.get(ALIASES.get(key, key))


def hannibal_demo_places() -> list[HistoricalPlace]:
    return [DEMO_PLACES["carthago"], DEMO_PLACES["carthago nova"]]
