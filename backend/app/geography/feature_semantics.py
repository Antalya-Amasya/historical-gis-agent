"""Coordinate roles, limitations, and anchor policy for typed geography features."""
from __future__ import annotations

from backend.app.models import HistoricalPlace, PlaceSpatialSemantics

_NON_EXACT_COORDINATE_ROLES = frozenset({
    "representative_point",
    "regional_centroid",
    "feature_reference",
    "feature_centroid",
    "unlocated_entity",
})

_TRAVERSAL_ONLY_SEMANTICS = frozenset({
    PlaceSpatialSemantics.SEA,
    PlaceSpatialSemantics.STRAIT,
    PlaceSpatialSemantics.RIVER,
    PlaceSpatialSemantics.MOUNTAIN_REGION,
})


def coordinate_role_for_semantics(
    semantics: PlaceSpatialSemantics,
    *,
    place_types: tuple[str, ...] = (),
    title: str = "",
) -> str:
    if semantics is PlaceSpatialSemantics.SEA:
        return "feature_reference"
    if semantics is PlaceSpatialSemantics.STRAIT:
        return "feature_reference"
    if semantics is PlaceSpatialSemantics.RIVER:
        return "representative_point"
    if semantics in {PlaceSpatialSemantics.MOUNTAIN_REGION, PlaceSpatialSemantics.REGION}:
        return "regional_centroid"
    if semantics is PlaceSpatialSemantics.ISLAND:
        return "feature_centroid"
    if semantics is PlaceSpatialSemantics.PASS:
        return "exact_site"
    if semantics is PlaceSpatialSemantics.PORT:
        return "exact_site"
    return "representative_point"


def place_limitations(place: HistoricalPlace) -> list[str]:
    semantics = place.spatial_semantics
    if semantics is PlaceSpatialSemantics.SEA:
        return ["Resolved sea entity is not an exact maritime traversal or route waypoint location."]
    if semantics is PlaceSpatialSemantics.STRAIT:
        return ["Resolved strait entity is not an exact crossing location."]
    if semantics is PlaceSpatialSemantics.RIVER:
        return ["Resolved river location is a representative entity point, not an exact crossing or event site."]
    if semantics is PlaceSpatialSemantics.MOUNTAIN_REGION:
        return ["Resolved mountain-region centroid identifies a region, not an exact event site or pass."]
    if semantics is PlaceSpatialSemantics.REGION:
        return ["Resolved regional centroid is not an exact event site."]
    if semantics is PlaceSpatialSemantics.ISLAND:
        return ["Resolved island centroid is not an exact landing or event site."]
    if semantics is PlaceSpatialSemantics.PASS and place.coordinate_role != "exact_site":
        return ["Resolved pass reference is not asserted as an exact crossing location."]
    if place.coordinate_role in _NON_EXACT_COORDINATE_ROLES:
        return [f"Resolved coordinate role is {place.coordinate_role}; it is not asserted as an exact event site."]
    return []


def exact_anchor_eligible(place: HistoricalPlace, *, strong_role: bool) -> bool:
    if not strong_role:
        return True
    return place.spatial_semantics not in {
        PlaceSpatialSemantics.SEA,
        PlaceSpatialSemantics.STRAIT,
        PlaceSpatialSemantics.RIVER,
        PlaceSpatialSemantics.MOUNTAIN_REGION,
    }


def traversal_eligible(place: HistoricalPlace) -> bool:
    return place.spatial_semantics in _TRAVERSAL_ONLY_SEMANTICS | {
        PlaceSpatialSemantics.STRAIT,
        PlaceSpatialSemantics.PASS,
    }
