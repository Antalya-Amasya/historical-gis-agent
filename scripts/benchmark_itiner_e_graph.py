"""Local, read-only topology benchmark for the acquired Itiner-e GeoJSON."""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).parents[1]))

from backend.app.roads.itiner_e import RomanRoadGraph, parse_itiner_e_geojson


AUDITED_PLACES = {
    "Carthago Nova": (-0.98452, 37.599896),
    "Massalia": (5.382499, 43.296854),
    "Rhodanus": (4.84861, 43.33167),
    "Alpes": (7.40183905, 43.74465275),
    "Padus": (12.432028, 44.952389),
}
PROBE_PAIRS = (("Carthago Nova", "Massalia"), ("Massalia", "Rhodanus"), ("Rhodanus", "Padus"))


def graph_summary(graph: RomanRoadGraph) -> dict:
    edges = list(graph.edges.values())
    return {
        "stats": graph.stats.__dict__,
        "road_types": dict(Counter(edge.segment.road_type for edge in edges)),
        "segment_statuses": dict(Counter(edge.segment.segment_status for edge in edges)),
        "chronology_statuses": dict(Counter(edge.segment.chronology.status for edge in edges)),
    }


def probe(graph: RomanRoadGraph) -> dict:
    nearest = {}
    for name, coordinate in AUDITED_PLACES.items():
        result = graph.nearest_node(coordinate)
        assert result is not None
        node, distance = result
        nearest[name] = {"node_id": node.id, "distance_m": round(distance, 3)}
    pairs = {}
    for source, target in PROBE_PAIRS:
        path = graph.shortest_path(nearest[source]["node_id"], nearest[target]["node_id"])
        pairs[f"{source} -> {target}"] = {
            "same_component": path is not None,
            "network_distance_m": round(path.distance_m, 3) if path else None,
            "edge_count": len(path.edge_ids) if path else 0,
        }
    return {"nearest_nodes": nearest, "pairs": pairs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[5, 10, 25, 50, 100])
    parser.add_argument("--trace-memory", action="store_true", help="Use tracemalloc; slower but reports Python allocation peak.")
    parser.add_argument("--output", type=Path, help="Optional local JSON result path.")
    args = parser.parse_args()
    if args.trace_memory:
        tracemalloc.start()
    started = time.perf_counter()
    segments = parse_itiner_e_geojson(args.path)
    parse_seconds = time.perf_counter() - started
    runs = []
    selected_graph = None
    for tolerance in args.tolerances:
        started = time.perf_counter()
        graph = RomanRoadGraph.from_segments(segments, snap_tolerance_m=tolerance)
        runs.append({"tolerance_m": tolerance, "build_seconds": round(time.perf_counter() - started, 4), **graph_summary(graph)})
        if tolerance == 25:
            selected_graph = graph
    if selected_graph is None:
        selected_graph = RomanRoadGraph.from_segments(segments, snap_tolerance_m=args.tolerances[0])
    peak = tracemalloc.get_traced_memory()[1] if args.trace_memory else None
    result = {
        "parse_seconds": round(parse_seconds, 4), "parsed_segments": len(segments),
        "tracemalloc_peak_bytes": peak, "tolerance_runs": runs,
        "selected_probe_tolerance_m": selected_graph.stats.tolerance_m,
        "connectivity_probes": probe(selected_graph),
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
