"""Offline Natural Earth 1:10m land/water surface classification."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping

from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from .surface import SurfaceClassification, SurfaceType


MODERN_GEOGRAPHY_WARNING = (
    "Modern Natural Earth surface classification; not a reconstructed Roman-period shoreline."
)
_LAYER_FILES = {
    "land": "ne_10m_land.geojson",
    "ocean": "ne_10m_ocean.geojson",
    "lakes": "ne_10m_lakes.geojson",
    "minor_islands": "ne_10m_minor_islands.geojson",
}
_LAYER_VERSIONS = {
    "land": "5.1.1",
    "ocean": "5.1.1",
    "lakes": "5.0.0",
    "minor_islands": "4.1.0",
}


@dataclass(frozen=True)
class NaturalEarthDatasetAudit:
    dataset_name: str
    layers: Mapping[str, str]
    layer_versions: Mapping[str, str]
    file_hashes: Mapping[str, str]
    file_sizes: Mapping[str, int]
    geometry_counts: Mapping[str, int]
    invalid_geometry_counts: Mapping[str, int]
    raw_source_hashes: Mapping[str, str] | None
    load_time_ms: float
    availability_error: str | None


@dataclass(frozen=True)
class _LayerIndex:
    name: str
    source_file: str
    geometries: tuple[object, ...]
    tree: STRtree


class NaturalEarthSurfaceClassifier:
    """Read-only, offline Natural Earth classifier with fail-closed semantics.

    The constructor loads local GeoJSON once and indexes it.  No query reads
    files, accesses the network, samples DEMs, or changes routing eligibility.
    """

    dataset_name = "Natural Earth 1:10m Physical Vectors"
    classification_method = "WGS84 point-in-polygon with explicit boundary ambiguity"
    resolution_or_scale = "1:10m"

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        expected_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self._expected_hashes = {name: value.lower() for name, value in (expected_hashes or {}).items()}
        self._layers: dict[str, _LayerIndex] = {}
        self._file_hashes: dict[str, str] = {}
        self._file_sizes: dict[str, int] = {}
        self._geometry_counts: dict[str, int] = {}
        self._invalid_geometry_counts: dict[str, int] = {}
        self._raw_source_hashes: Mapping[str, str] | None = None
        self._availability_error: str | None = None
        started = perf_counter()
        self._load()
        self._load_time_ms = (perf_counter() - started) * 1_000

    @property
    def audit(self) -> NaturalEarthDatasetAudit:
        return NaturalEarthDatasetAudit(
            dataset_name=self.dataset_name,
            layers=dict(_LAYER_FILES),
            layer_versions=dict(_LAYER_VERSIONS),
            file_hashes=dict(self._file_hashes),
            file_sizes=dict(self._file_sizes),
            geometry_counts=dict(self._geometry_counts),
            invalid_geometry_counts=dict(self._invalid_geometry_counts),
            raw_source_hashes=self._raw_source_hashes,
            load_time_ms=self._load_time_ms,
            availability_error=self._availability_error,
        )

    def set_raw_source_hashes(self, hashes: Mapping[str, str]) -> None:
        """Attach audited preprocessing inputs without changing classifications."""
        self._raw_source_hashes = dict(hashes)

    def classify(self, latitude: float, longitude: float) -> SurfaceClassification:
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return self._unknown("invalid_coordinate", source_file="")
        if self._availability_error is not None:
            return self._unknown("dataset_unavailable", source_file="")

        point = Point(longitude, latitude)
        relations = {name: self._relation(index, point) for name, index in self._layers.items()}
        if any(boundary for _, boundary in relations.values()):
            return self._unknown("boundary_ambiguous", source_file="")

        lake_match = relations["lakes"][0]
        ocean_match = relations["ocean"][0]
        land_match = relations["land"][0] or relations["minor_islands"][0]
        if ocean_match and land_match:
            return self._unknown("conflicting_geometry", source_file="")
        if lake_match:
            return self._classified(SurfaceType.WATER, "lakes", "interior")
        if ocean_match:
            return self._classified(SurfaceType.WATER, "ocean", "interior")
        if land_match:
            layer = "land" if relations["land"][0] else "minor_islands"
            return self._classified(SurfaceType.LAND, layer, "interior")
        return self._unknown("outside_classified_geometry", source_file="")

    def _load(self) -> None:
        try:
            for layer, filename in _LAYER_FILES.items():
                path = self.dataset_dir / filename
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                self._file_hashes[layer] = digest
                self._file_sizes[layer] = len(raw)
                expected = self._expected_hashes.get(layer)
                if expected is not None and expected != digest:
                    raise ValueError(f"hash mismatch for {filename}")
                payload = json.loads(raw.decode("utf-8"))
                geometries = tuple(self._load_geometries(payload.get("features", ()), layer))
                if not geometries:
                    raise ValueError(f"no valid geometries in {filename}")
                self._layers[layer] = _LayerIndex(layer, filename, geometries, STRtree(geometries))
        except (OSError, UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._layers.clear()
            self._availability_error = f"{type(exc).__name__}: {exc}"

    def _load_geometries(self, features: Iterable[object], layer: str) -> Iterable[object]:
        geometry_count = invalid_count = 0
        for feature in features:
            geometry_count += 1
            try:
                geometry = shape(feature["geometry"])
            except (KeyError, TypeError, ValueError):
                invalid_count += 1
                continue
            if geometry.is_empty or not geometry.is_valid:
                invalid_count += 1
                continue
            yield geometry
        self._geometry_counts[layer] = geometry_count
        self._invalid_geometry_counts[layer] = invalid_count
        if invalid_count:
            raise ValueError(f"invalid geometry in {_LAYER_FILES[layer]}")

    @staticmethod
    def _relation(index: _LayerIndex, point: Point) -> tuple[bool, bool]:
        candidates = index.tree.query(point)
        interior = boundary = False
        for candidate_index in candidates:
            geometry = index.geometries[int(candidate_index)]
            if geometry.contains(point):
                interior = True
            elif geometry.covers(point):
                boundary = True
        return interior, boundary

    def _classified(self, surface_type: SurfaceType, layer: str, status: str) -> SurfaceClassification:
        return SurfaceClassification(
            surface_type=surface_type,
            source="natural_earth_10m",
            confidence=1.0,
            status=status,
            metadata=self._metadata(_LAYER_FILES[layer], status),
        )

    def _unknown(self, status: str, *, source_file: str) -> SurfaceClassification:
        return SurfaceClassification(
            surface_type=SurfaceType.UNKNOWN,
            source="natural_earth_10m",
            confidence=0.0,
            status=status,
            metadata=self._metadata(source_file, status),
        )

    def _metadata(self, source_file: str, boundary_status: str) -> dict[str, str]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": "; ".join(f"{name}={version}" for name, version in _LAYER_VERSIONS.items()),
            "source_file": source_file,
            "source_sha256": next((digest for layer, digest in self._file_hashes.items() if _LAYER_FILES[layer] == source_file), ""),
            "raw_source_sha256": next((digest for layer, digest in (self._raw_source_hashes or {}).items() if _LAYER_FILES[layer] == source_file), ""),
            "classification_method": self.classification_method,
            "resolution_or_scale": self.resolution_or_scale,
            "modern_geography_warning": MODERN_GEOGRAPHY_WARNING,
            "boundary_status": boundary_status,
        }
