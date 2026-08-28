"""Itiner-e road records -> deterministic endpoint-only topology.

This module deliberately does not attach roads to Evidence, chronology decisions,
terrain costs, or Agent output.  A graph edge is infrastructure data, not proof of
a historical movement.
"""
from __future__ import annotations

import heapq
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scipy.spatial import cKDTree


EARTH_RADIUS_M = 6_371_008.8
UNKNOWN_DATE = 9999
Coordinate = tuple[float, float]  # GeoJSON order: longitude, latitude
WEB_MERCATOR_SEMIMAJOR_M = 6_378_137.0
WGS84_ECCENTRICITY = 0.08181919084262149


def haversine_m(first: Coordinate, second: Coordinate) -> float:
    lon1, lat1, lon2, lat2 = map(math.radians, (*first, *second))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def polyline_length_m(geometry: tuple[Coordinate, ...]) -> float:
    return sum(haversine_m(first, second) for first, second in zip(geometry, geometry[1:]))


def _date(value: object) -> float | None:
    if value in (None, "", UNKNOWN_DATE, float(UNKNOWN_DATE)):
        return None
    return float(value)


@dataclass(frozen=True)
class RoadChronology:
    lower_date: float | None
    lower_date_error: float | None
    upper_date: float | None
    upper_date_error: float | None
    raw_values: dict[str, object]

    @property
    def status(self) -> str:
        return "known" if self.lower_date is not None and self.upper_date is not None else "unknown"


@dataclass(frozen=True)
class RomanRoadSegment:
    record_id: str
    part_index: int
    geometry: tuple[Coordinate, ...]
    road_type: str | None
    route_type: object | None
    segment_status: str | None
    name: str | None
    citation: str | None
    bibliography: str | None
    chronology: RoadChronology
    average_slope: float | None
    passability: float | None
    shape_length: float | None
    source_name: str = "Itiner-e — The Digital Atlas of Ancient Roads"
    source_format: str = "geojson"
    source_properties: dict[str, object] | None = None
    source_geometry: tuple[tuple[float, ...], ...] | None = None
    coordinate_reference_system: str = "EPSG:4326"

    @property
    def edge_id(self) -> str:
        return f"itiner-e:{self.record_id}:part:{self.part_index}"

    @property
    def length_m(self) -> float:
        return polyline_length_m(self.geometry)


def parse_itiner_e_geojson(path: str | Path) -> list[RomanRoadSegment]:
    """Parse actual Itiner-e FeatureCollection records, safely exploding lines."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise ValueError("expected an Itiner-e GeoJSON FeatureCollection")
    coordinate_reference_system = _coordinate_reference_system(payload)
    result: list[RomanRoadSegment] = []
    for feature_index, feature in enumerate(payload["features"]):
        geometry = feature.get("geometry") or {}
        properties = dict(feature.get("properties") or {})
        geometry_type = geometry.get("type")
        if geometry_type == "LineString":
            components = [geometry.get("coordinates") or []]
        elif geometry_type == "MultiLineString":
            components = geometry.get("coordinates") or []
        else:
            raise ValueError(f"unsupported Itiner-e geometry at feature {feature_index}: {geometry_type!r}")
        record_id = str(properties.get("InLine_FID", feature_index))
        chronology = RoadChronology(
            lower_date=_date(properties.get("Lower_Date")),
            lower_date_error=_date(properties.get("Low_Date_E")),
            upper_date=_date(properties.get("Upper_Date")),
            upper_date_error=_date(properties.get("Up_Date_E")),
            raw_values={key: properties.get(key) for key in ("Lower_Date", "Low_Date_E", "Upper_Date", "Up_Date_E")},
        )
        for part_index, component in enumerate(components):
            source_geometry = tuple(tuple(float(value) for value in point) for point in component)
            normalized = tuple(_normalise_coordinate(point, coordinate_reference_system) for point in source_geometry)
            if len(source_geometry) < 2:
                raise ValueError(f"Itiner-e feature {record_id} part {part_index} has fewer than two coordinates")
            result.append(RomanRoadSegment(
                record_id=record_id, part_index=part_index, geometry=normalized,
                road_type=properties.get("Type"), route_type=properties.get("Route_Type"),
                segment_status=properties.get("Segment_s"), name=properties.get("Name"),
                citation=properties.get("Citation"), bibliography=properties.get("Bibliograp"),
                chronology=chronology, average_slope=_number(properties.get("Avg_Slope")),
                passability=_number(properties.get("passabilit")), shape_length=_number(properties.get("Shape_Leng")),
                source_properties=properties, source_geometry=source_geometry,
                coordinate_reference_system=coordinate_reference_system,
            ))
    return result


def _number(value: object) -> float | None:
    return None if value in (None, "") else float(value)


def _coordinate_reference_system(payload: dict[str, object]) -> str:
    """Return the only source CRS variants supported by this topology loader.

    Itiner-e's Zenodo snapshot declares EPSG:3395.  GeoJSON with no legacy CRS
    member is interpreted as WGS84 longitude/latitude.  Rejecting other CRS
    values is safer than silently treating projected metres as degrees.
    """
    crs = payload.get("crs")
    if crs is None:
        return "EPSG:4326"
    if not isinstance(crs, dict):
        raise ValueError("invalid GeoJSON CRS declaration")
    name = ((crs.get("properties") or {}).get("name") if isinstance(crs.get("properties"), dict) else None)
    if not isinstance(name, str):
        raise ValueError("GeoJSON CRS declaration has no name")
    normalized = name.upper().replace("::", ":")
    if normalized.endswith("EPSG:3395"):
        return "EPSG:3395"
    if normalized.endswith("EPSG:4326"):
        return "EPSG:4326"
    raise ValueError(f"unsupported Itiner-e coordinate reference system: {name}")


def _normalise_coordinate(point: tuple[float, ...], coordinate_reference_system: str) -> Coordinate:
    if len(point) < 2:
        raise ValueError("Itiner-e coordinate has fewer than two ordinates")
    x, y = point[:2]
    if coordinate_reference_system == "EPSG:4326":
        if not (-180 <= x <= 180 and -90 <= y <= 90):
            raise ValueError("EPSG:4326 coordinate is outside longitude/latitude bounds")
        return x, y
    if coordinate_reference_system != "EPSG:3395":  # defensive; CRS was validated above
        raise ValueError(f"unsupported coordinate reference system: {coordinate_reference_system}")
    longitude = math.degrees(x / WEB_MERCATOR_SEMIMAJOR_M)
    t = math.exp(-y / WEB_MERCATOR_SEMIMAJOR_M)
    latitude = math.pi / 2 - 2 * math.atan(t)
    for _ in range(8):
        latitude = math.pi / 2 - 2 * math.atan(
            t * ((1 - WGS84_ECCENTRICITY * math.sin(latitude)) / (1 + WGS84_ECCENTRICITY * math.sin(latitude))) ** (WGS84_ECCENTRICITY / 2)
        )
    return longitude, math.degrees(latitude)


@dataclass(frozen=True)
class RomanRoadNode:
    id: str
    coordinate: Coordinate


@dataclass(frozen=True)
class RomanRoadEdge:
    id: str
    segment: RomanRoadSegment
    source_node_id: str
    target_node_id: str
    source_original_coordinate: Coordinate
    target_original_coordinate: Coordinate
    source_snap_distance_m: float
    target_snap_distance_m: float
    length_m: float


@dataclass(frozen=True)
class RomanRoadPath:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    geometry: tuple[Coordinate, ...]
    distance_m: float


@dataclass(frozen=True)
class RomanRoadGraphStats:
    tolerance_m: float
    node_count: int
    edge_count: int
    component_count: int
    largest_component_nodes: int
    snapped_endpoint_count: int
    maximum_snap_distance_m: float
    self_loop_segment_count: int
    duplicate_geometry_count: int


class _EndpointSnapper:
    """Greedy, source-order-stable snapper with geodesic confirmation.

    A node's normalized coordinate is its first deterministically sorted endpoint.
    Later endpoints are compared to that fixed coordinate, avoiding transitive
    chain-merges of endpoints farther apart than the tolerance.
    """

    def __init__(self, tolerance_m: float):
        if tolerance_m <= 0:
            raise ValueError("snap tolerance must be positive")
        self.tolerance_m = tolerance_m
        self.cell_degrees = tolerance_m / 111_320.0
        self.nodes: dict[str, RomanRoadNode] = {}
        self._buckets: dict[tuple[int, int], list[str]] = defaultdict(list)

    def _cell(self, coordinate: Coordinate) -> tuple[int, int]:
        longitude, latitude = coordinate
        return math.floor(longitude / self.cell_degrees), math.floor(latitude / self.cell_degrees)

    def snap(self, coordinate: Coordinate) -> tuple[RomanRoadNode, float]:
        longitude, latitude = coordinate
        cell_x, cell_y = self._cell(coordinate)
        longitude_span = math.ceil(1 / max(0.01, abs(math.cos(math.radians(latitude))))) + 1
        candidates: list[tuple[float, str]] = []
        for x in range(cell_x - longitude_span, cell_x + longitude_span + 1):
            for y in range(cell_y - 1, cell_y + 2):
                for node_id in self._buckets.get((x, y), ()):
                    distance = haversine_m(coordinate, self.nodes[node_id].coordinate)
                    if distance <= self.tolerance_m:
                        candidates.append((distance, node_id))
        if candidates:
            distance, node_id = min(candidates)
            return self.nodes[node_id], distance
        node = RomanRoadNode(id=f"road-node-{len(self.nodes) + 1}", coordinate=coordinate)
        self.nodes[node.id] = node
        self._buckets[self._cell(coordinate)].append(node.id)
        return node, 0.0


class _NodeSpatialIndex:
    """SciPy cKDTree over unit-sphere coordinates for exact nearest ordering."""

    def __init__(self, nodes: dict[str, RomanRoadNode]):
        self.nodes = nodes
        self.node_ids = tuple(sorted(nodes))
        self.tree = cKDTree([self._unit_vector(nodes[node_id].coordinate) for node_id in self.node_ids]) if self.node_ids else None

    @staticmethod
    def _unit_vector(coordinate: Coordinate) -> tuple[float, float, float]:
        longitude, latitude = map(math.radians, coordinate)
        return math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude)

    def nearest(self, coordinate: Coordinate) -> tuple[RomanRoadNode, float] | None:
        if not self.nodes:
            return None
        assert self.tree is not None
        _chord_distance, index = self.tree.query(self._unit_vector(coordinate), k=1)
        node = self.nodes[self.node_ids[int(index)]]
        return node, haversine_m(coordinate, node.coordinate)


class RomanRoadGraph:
    """Endpoint-only topology validation graph for Itiner-e road infrastructure."""

    def __init__(self, *, nodes: dict[str, RomanRoadNode], edges: dict[str, RomanRoadEdge], adjacency: dict[str, tuple[str, ...]], stats: RomanRoadGraphStats, component_ids: dict[str, str]):
        self.nodes, self.edges, self.adjacency, self.stats = nodes, edges, adjacency, stats
        self.component_ids = component_ids
        self._index = _NodeSpatialIndex(nodes)

    @classmethod
    def load(cls, path: str | Path, *, snap_tolerance_m: float = 25.0) -> "RomanRoadGraph":
        return cls.from_segments(parse_itiner_e_geojson(path), snap_tolerance_m=snap_tolerance_m)

    @classmethod
    def from_segments(cls, segments: Iterable[RomanRoadSegment], *, snap_tolerance_m: float = 25.0) -> "RomanRoadGraph":
        ordered = sorted(segments, key=lambda segment: (segment.record_id, segment.part_index))
        snapper = _EndpointSnapper(snap_tolerance_m)
        edges: dict[str, RomanRoadEdge] = {}
        adjacency: dict[str, list[str]] = defaultdict(list)
        self_loops = 0
        geometry_counts: Counter[tuple[Coordinate, ...]] = Counter()
        for segment in ordered:
            source, source_distance = snapper.snap(segment.geometry[0])
            target, target_distance = snapper.snap(segment.geometry[-1])
            geometry = segment.geometry
            geometry_counts[min(geometry, geometry[::-1])] += 1
            is_self_loop = source.id == target.id
            if is_self_loop:
                # Controlled snapping can collapse a very short source segment.
                # Retain its record as an explicit self-loop rather than losing
                # road provenance; it cannot connect distinct graph nodes.
                self_loops += 1
            edge = RomanRoadEdge(
                id=segment.edge_id, segment=segment, source_node_id=source.id, target_node_id=target.id,
                source_original_coordinate=segment.geometry[0], target_original_coordinate=segment.geometry[-1],
                source_snap_distance_m=source_distance, target_snap_distance_m=target_distance,
                length_m=segment.length_m,
            )
            edges[edge.id] = edge
            adjacency[source.id].append(edge.id)
            if not is_self_loop:
                adjacency[target.id].append(edge.id)
        frozen_adjacency = {node_id: tuple(sorted(edge_ids)) for node_id, edge_ids in adjacency.items()}
        components = cls._components(snapper.nodes, frozen_adjacency, edges)
        snap_distances = [distance for edge in edges.values() for distance in (edge.source_snap_distance_m, edge.target_snap_distance_m)]
        stats = RomanRoadGraphStats(
            tolerance_m=snap_tolerance_m, node_count=len(snapper.nodes), edge_count=len(edges),
            component_count=len(components), largest_component_nodes=max((len(component) for component in components), default=0),
            snapped_endpoint_count=sum(distance > 0 for distance in snap_distances),
            maximum_snap_distance_m=max(snap_distances, default=0.0), self_loop_segment_count=self_loops,
            duplicate_geometry_count=sum(count - 1 for count in geometry_counts.values() if count > 1),
        )
        component_ids = {
            node_id: f"road-component-{index}"
            for index, component in enumerate(sorted(components, key=lambda item: min(item)), start=1)
            for node_id in component
        }
        return cls(nodes=snapper.nodes, edges=edges, adjacency=frozen_adjacency, stats=stats, component_ids=component_ids)

    @staticmethod
    def _components(nodes: dict[str, RomanRoadNode], adjacency: dict[str, tuple[str, ...]], edges: dict[str, RomanRoadEdge]) -> list[set[str]]:
        unseen = set(nodes)
        components: list[set[str]] = []
        while unseen:
            root = unseen.pop()
            component = {root}
            queue = deque([root])
            while queue:
                node_id = queue.popleft()
                for edge_id in adjacency.get(node_id, ()):
                    edge = edges[edge_id]
                    other = edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
                    if other in unseen:
                        unseen.remove(other)
                        component.add(other)
                        queue.append(other)
            components.append(component)
        return components

    def nearest_node(self, coordinate: Coordinate) -> tuple[RomanRoadNode, float] | None:
        return self._index.nearest(coordinate)

    def component_id(self, node_id: str) -> str:
        try:
            return self.component_ids[node_id]
        except KeyError as error:
            raise KeyError("unknown Roman-road node") from error

    def connected(self, source_node_id: str, target_node_id: str) -> bool:
        return self.shortest_path(source_node_id, target_node_id) is not None

    def shortest_path(self, source_node_id: str, target_node_id: str) -> RomanRoadPath | None:
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            raise KeyError("unknown Roman-road node")
        distances = {source_node_id: 0.0}
        previous: dict[str, tuple[str, str]] = {}
        queue: list[tuple[float, str]] = [(0.0, source_node_id)]
        while queue:
            distance, node_id = heapq.heappop(queue)
            if distance != distances.get(node_id):
                continue
            if node_id == target_node_id:
                break
            for edge_id in self.adjacency.get(node_id, ()):
                edge = self.edges[edge_id]
                other = edge.target_node_id if edge.source_node_id == node_id else edge.source_node_id
                candidate = distance + edge.length_m
                if candidate < distances.get(other, math.inf):
                    distances[other] = candidate
                    previous[other] = (node_id, edge_id)
                    heapq.heappush(queue, (candidate, other))
        if target_node_id not in distances:
            return None
        reversed_steps: list[tuple[str, str]] = []
        cursor = target_node_id
        while cursor != source_node_id:
            prior, edge_id = previous[cursor]
            reversed_steps.append((prior, edge_id))
            cursor = prior
        steps = list(reversed(reversed_steps))
        node_ids = [source_node_id]
        edge_ids: list[str] = []
        geometry: list[Coordinate] = []
        cursor = source_node_id
        for prior, edge_id in steps:
            assert prior == cursor
            edge = self.edges[edge_id]
            directed = edge.segment.geometry if edge.source_node_id == cursor else tuple(reversed(edge.segment.geometry))
            geometry.extend(directed if not geometry else directed[1:])
            cursor = edge.target_node_id if edge.source_node_id == cursor else edge.source_node_id
            node_ids.append(cursor)
            edge_ids.append(edge_id)
        return RomanRoadPath(tuple(node_ids), tuple(edge_ids), tuple(geometry), distances[target_node_id])
