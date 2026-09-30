"""Bounded simulated coastal access. Not a historical port or waypoint."""
from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, radians, sin

from backend.app.gis.sea import _distance, plan_direct_water_edge
from backend.app.gis.surface import SurfaceType, WaterDomain

SEARCH_RADIUS_M = 100_000.0
STEP_M = 5_000.0
BEARING_COUNT = 16
MAX_WATER_PAIR_CHECKS = 4
# A mixed simulation replaces a land candidate only when it is substantially shorter.
MIXED_DISTANCE_RATIO = 0.75


@dataclass(frozen=True)
class CoastalAccess:
    """Land/ocean transition generated for simulation only."""

    land_latitude: float
    land_longitude: float
    ocean_latitude: float
    ocean_longitude: float
    distance_from_origin_m: float


def offset_coordinate(latitude: float, longitude: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    radius = 6_371_008.8
    bearing = radians(bearing_deg)
    lat1 = radians(latitude)
    lon1 = radians(longitude)
    step = distance_m / radius
    lat2 = asin(sin(lat1) * cos(step) + cos(lat1) * sin(step) * cos(bearing))
    lon2 = lon1 + atan2(sin(bearing) * sin(step) * cos(lat1), cos(step) - sin(lat1) * sin(lat2))
    return degrees(lat2), (degrees(lon2) + 540) % 360 - 180


def _ocean(surface) -> bool:
    return (
        surface.surface_type is SurfaceType.WATER
        and getattr(surface, "water_domain", None) is WaterDomain.OCEAN
        and surface.status == "interior"
    )


def _land(surface) -> bool:
    return surface.surface_type is SurfaceType.LAND and surface.status == "interior"


def _blocked_water(surface) -> bool:
    return surface.surface_type is SurfaceType.WATER and not _ocean(surface)


def find_coastal_accesses(latitude: float, longitude: float, classifier, *, radius_m: float = SEARCH_RADIUS_M) -> list[CoastalAccess]:
    """Walk fixed bearings until a land-to-ocean transition. Lakes and ambiguity are rejected."""
    origin = classifier.classify(latitude, longitude)
    if not _land(origin):
        return []
    found: list[CoastalAccess] = []
    steps = int(radius_m // STEP_M)
    for index in range(BEARING_COUNT):
        bearing = index * (360 / BEARING_COUNT)
        previous = (latitude, longitude)
        previous_surface = origin
        for step in range(1, steps + 1):
            current = offset_coordinate(latitude, longitude, bearing, step * STEP_M)
            surface = classifier.classify(current[0], current[1])
            if surface.status != "interior" or surface.surface_type is SurfaceType.UNKNOWN:
                break
            if _blocked_water(surface):
                break
            if _land(previous_surface) and _ocean(surface):
                found.append(CoastalAccess(
                    land_latitude=previous[0], land_longitude=previous[1],
                    ocean_latitude=current[0], ocean_longitude=current[1],
                    distance_from_origin_m=step * STEP_M,
                ))
                break
            if not _land(surface):
                break
            previous, previous_surface = current, surface
    found.sort(key=lambda item: item.distance_from_origin_m)
    return found


def select_simulation_mode(
    *,
    historical_mode: str,
    land_distance_m: float | None,
    mixed_distance_m: float | None,
) -> str:
    """Choose a simulation mode without changing historical travel authority."""
    if historical_mode == "LAND":
        return "LAND"
    if historical_mode == "SEA":
        return "SEA"
    if mixed_distance_m is None:
        return "LAND" if land_distance_m is not None else "UNAVAILABLE"
    if land_distance_m is None:
        return "LAND_SEA_MIXED"
    if mixed_distance_m <= land_distance_m * MIXED_DISTANCE_RATIO:
        return "LAND_SEA_MIXED"
    return "LAND"


def shortest_valid_water_pair(origin: list[CoastalAccess], destination: list[CoastalAccess], classifier):
    """Return the ocean-valid pair with the shortest land-proxy plus water distance."""
    ranked = []
    for left in origin:
        for right in destination:
            proxy = left.distance_from_origin_m + right.distance_from_origin_m
            proxy += _distance(
                type("P", (), {"latitude": left.ocean_latitude, "longitude": left.ocean_longitude})(),
                type("P", (), {"latitude": right.ocean_latitude, "longitude": right.ocean_longitude})(),
            )
            ranked.append((proxy, left, right))
    ranked.sort(key=lambda item: item[0])
    for _proxy, left, right in ranked[:MAX_WATER_PAIR_CHECKS]:
        plan = plan_direct_water_edge(
            left.ocean_latitude, left.ocean_longitude, right.ocean_latitude, right.ocean_longitude, classifier,
        )
        if plan.available:
            return left, right, plan
    return None
