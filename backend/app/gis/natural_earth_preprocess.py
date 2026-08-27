"""Explicit preprocessing for locally stored Natural Earth GeoJSON assets."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from shapely.geometry import mapping, shape
from shapely.validation import make_valid

from .natural_earth_surface import _LAYER_FILES


@dataclass(frozen=True)
class NaturalEarthPreprocessAudit:
    raw_hashes: Mapping[str, str]
    normalized_hashes: Mapping[str, str]
    feature_counts: Mapping[str, int]
    repaired_geometry_counts: Mapping[str, int]


def normalize_natural_earth_geojson(
    source_dir: str | Path,
    output_dir: str | Path,
) -> NaturalEarthPreprocessAudit:
    """Create deterministic, valid GeoJSON for the runtime classifier.

    Source files are never modified.  Any GEOS-invalid geometry is repaired
    with Shapely ``make_valid`` before the normalized file is atomically
    replaced.  Runtime code remains fail-closed if these outputs are missing
    or invalid.
    """
    source_root = Path(source_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_hashes: dict[str, str] = {}
    normalized_hashes: dict[str, str] = {}
    feature_counts: dict[str, int] = {}
    repaired_geometry_counts: dict[str, int] = {}
    for layer, filename in _LAYER_FILES.items():
        source_path = source_root / filename
        raw = source_path.read_bytes()
        raw_hashes[layer] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw.decode("utf-8"))
        features = payload.get("features")
        if not isinstance(features, list):
            raise ValueError(f"{filename} must contain a feature list")
        normalized_features = []
        repaired = 0
        for feature in features:
            geometry = shape(feature["geometry"])
            if geometry.is_empty:
                raise ValueError(f"{filename} contains an empty geometry")
            if not geometry.is_valid:
                geometry = make_valid(geometry)
                repaired += 1
            if geometry.is_empty or not geometry.is_valid:
                raise ValueError(f"{filename} contains an unrepaired invalid geometry")
            normalized_features.append({**feature, "geometry": mapping(geometry)})
        normalized_payload = {**payload, "features": normalized_features}
        encoded = (json.dumps(normalized_payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        target_path = output_root / filename
        temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
        temporary_path.write_bytes(encoded)
        temporary_path.replace(target_path)
        normalized_hashes[layer] = hashlib.sha256(encoded).hexdigest()
        feature_counts[layer] = len(normalized_features)
        repaired_geometry_counts[layer] = repaired
    return NaturalEarthPreprocessAudit(raw_hashes, normalized_hashes, feature_counts, repaired_geometry_counts)
