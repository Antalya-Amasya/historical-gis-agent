"""Versioned Pleiades-backed Roman Republican registry loader."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from backend.app.models import HistoricalPlace, PlaceSpatialSemantics

PLEIADES_SOURCE = "Pleiades: A Gazetteer of Past Places"
_DATA = Path(__file__).with_name("data") / "roman_republic_places.json"

@lru_cache(maxsize=1)
def records() -> list[dict]:
    return json.loads(_DATA.read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def places() -> tuple[HistoricalPlace, ...]:
    return tuple(HistoricalPlace(id=f"pleiades-{r['pleiades_id']}", canonical_name=r["canonical_name"], modern_name=r.get("modern_name"), latitude=r["latitude"], longitude=r["longitude"], period=r.get("period"), source=PLEIADES_SOURCE, source_id=str(r["pleiades_id"]), source_url=f"https://pleiades.stoa.org/places/{r['pleiades_id']}", confidence=r["confidence"], uncertain=r.get("uncertain", False), coordinate_role=r["coordinate_role"], spatial_semantics=PlaceSpatialSemantics(r["spatial_semantics"]), spatial_semantics_provenance=r["spatial_semantics_provenance"]) for r in records())

@lru_cache(maxsize=1)
def aliases() -> dict[str, tuple[HistoricalPlace, ...]]:
    by_id = {place.id: place for place in places()}; result = {}
    for r in records():
        for alias in r["aliases"]: result.setdefault(alias.casefold(), []).append(by_id[f"pleiades-{r['pleiades_id']}"])
    return {key: tuple(value) for key, value in result.items()}

def resolve(name: str) -> tuple[HistoricalPlace, ...]: return aliases().get(name.strip().casefold(), ())
def alias_records() -> tuple[tuple[str, tuple[str, ...], str], ...]: return tuple((r["canonical_name"], tuple(r["aliases"]), "pleiades_registry_snapshot_2026_08") for r in records())
