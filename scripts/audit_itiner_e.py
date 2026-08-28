"""Read-only schema and connectivity pre-audit for an Itiner-e GeoJSON export."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def audit_geojson(path: Path) -> dict:
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    features = dataset.get("features")
    if dataset.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise ValueError("expected a GeoJSON FeatureCollection")
    property_keys = Counter(key for feature in features for key in (feature.get("properties") or {}))
    geometry_types = Counter((feature.get("geometry") or {}).get("type") for feature in features)
    def counter_value(value: object) -> object:
        return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value

    values = {
        key: Counter(counter_value((feature.get("properties") or {}).get(key)) for feature in features)
        for key in property_keys
    }
    chronology_keys = [key for key in property_keys if any(token in key.casefold() for token in ("date", "period", "chron", "year"))]
    bibliography_keys = [key for key in property_keys if any(token in key.casefold() for token in ("citation", "biblio", "source", "reference", "uri"))]
    certainty_keys = [key for key in property_keys if any(token in key.casefold() for token in ("certain", "confidence", "quality"))]
    segment_classification_keys = [key for key in property_keys if "segment" in key.casefold()]
    route_type_keys = [key for key in property_keys if any(token in key.casefold() for token in ("type", "category", "route"))]
    pleiades_keys = [key for key in property_keys if "pleiad" in key.casefold()]
    passability_keys = [key for key in property_keys if any(token in key.casefold() for token in ("passab", "transport"))]
    explicit_network_keys = [key for key in property_keys if re.search(r"(^|_)(node|edge|network|connect|from|to)($|_)", key.casefold())]
    exact_endpoints: Counter[tuple[float, float]] = Counter()
    rounded_endpoints: Counter[tuple[float, float]] = Counter()
    geometries: Counter[tuple] = Counter()
    vertex_counts: list[int] = []
    endpoint_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    invalid_records = 0
    for feature in features:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") == "LineString":
            lines = [coordinates]
        elif geometry.get("type") == "MultiLineString":
            lines = coordinates
        else:
            invalid_records += 1
            continue
        valid_lines = 0
        for line in lines:
            if len(line) < 2:
                continue
            valid_lines += 1
            source, target = tuple(line[0]), tuple(line[-1])
            exact_endpoints.update((source, target))
            endpoint_edges.append((source, target))
            rounded_endpoints.update(((round(source[0], 5), round(source[1], 5)), (round(target[0], 5), round(target[1], 5))))
            normalized = tuple(tuple(point) for point in line)
            geometries[min(normalized, normalized[::-1])] += 1
            vertex_counts.append(len(line))
        if not valid_lines:
            invalid_records += 1
    parent = {point: point for point in exact_endpoints}
    def find(point: tuple[float, float]) -> tuple[float, float]:
        while parent[point] != point:
            parent[point] = parent[parent[point]]
            point = parent[point]
        return point
    for source, target in endpoint_edges:
        source_root, target_root = find(source), find(target)
        if source_root != target_root:
            parent[target_root] = source_root
    component_sizes = Counter(find(point) for point in parent)
    def field_summary(key: str) -> dict:
        populated = Counter({value: count for value, count in values[key].items() if value not in (None, "")})
        return {
            "non_null_count": sum(populated.values()),
            "unique_non_null": len(populated),
            "common_values": populated.most_common(12),
            "unknown_9999_count": populated.get(9999, 0),
        }
    return {
        "root_type": dataset["type"],
        "top_level_keys": sorted(dataset),
        "record_count": len(features),
        "geometry_types": dict(geometry_types),
        "property_keys": sorted(property_keys),
        "route_type_fields": {key: field_summary(key) for key in route_type_keys},
        "chronology_fields": {key: field_summary(key) for key in chronology_keys},
        "bibliography_source_fields": bibliography_keys,
        "certainty_fields": certainty_keys,
        "segment_classification_fields": {key: field_summary(key) for key in segment_classification_keys},
        "pleiades_fields": pleiades_keys,
        "passability_fields": {key: field_summary(key) for key in passability_keys},
        "null_or_empty_rate": {
            key: sum((feature.get("properties") or {}).get(key) in (None, "") for feature in features) / len(features)
            for key in sorted(property_keys)
        },
        "connectivity": {
            "invalid_or_non_line_records": invalid_records,
            "vertices_per_line": {"min": min(vertex_counts) if vertex_counts else None, "max": max(vertex_counts) if vertex_counts else None, "mean": sum(vertex_counts) / len(vertex_counts) if vertex_counts else None},
            "exact_shared_endpoint_keys": sum(count > 1 for count in exact_endpoints.values()),
            "exact_endpoint_keys": len(exact_endpoints),
            "near_1e5_shared_endpoint_keys": sum(count > 1 for count in rounded_endpoints.values()),
            "near_1e5_endpoint_keys": len(rounded_endpoints),
            "duplicate_geometries": sum(count - 1 for count in geometries.values() if count > 1),
            "explicit_network_fields": explicit_network_keys,
            "exact_endpoint_components": len(component_sizes),
            "largest_exact_endpoint_component_nodes": max(component_sizes.values()) if component_sizes else 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_geojson(args.path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
