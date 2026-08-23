from backend.app.models import HistoricalPlace

DEMO_PLACES = {
    "carthago": HistoricalPlace(id="pleiades-314936", canonical_name="Carthago", modern_name="Carthage", latitude=36.8529, longitude=10.3233, period="-200", source="Pleiades demo record", confidence=0.93),
    "alpes": HistoricalPlace(id="demo-alps", canonical_name="Alpes", modern_name="Alps", latitude=45.85, longitude=7.35, period="-218", source="Phase 0 demo place repository", confidence=0.55, uncertain=True),
}


def resolve_demo_place(name: str) -> HistoricalPlace | None:
    return DEMO_PLACES.get(name.strip().lower())

