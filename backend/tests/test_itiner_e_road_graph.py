import json

import pytest

from backend.app.roads.itiner_e import RomanRoadGraph, RomanRoadSegment, RoadChronology, parse_itiner_e_geojson


def chronology(lower=9999, upper=9999):
    return RoadChronology(
        lower_date=None if lower == 9999 else lower,
        lower_date_error=None,
        upper_date=None if upper == 9999 else upper,
        upper_date_error=None,
        raw_values={"Lower_Date": lower, "Low_Date_E": 9999, "Upper_Date": upper, "Up_Date_E": 9999},
    )


def segment(identifier, geometry, *, status="Certain", lower=9999, upper=9999):
    return RomanRoadSegment(
        record_id=identifier, part_index=0, geometry=tuple(geometry), road_type="Main Road", route_type=2,
        segment_status=status, name=f"road-{identifier}", citation="citation", bibliography="bibliography",
        chronology=chronology(lower, upper), average_slope=1.0, passability=1.0, shape_length=100.0,
        source_properties={"InLine_FID": identifier},
    )


def test_geojson_parser_explodes_multiline_and_preserves_properties_and_geometry(tmp_path):
    path = tmp_path / "itiner.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": [[[1, 2], [2, 2]], [[2, 2], [3, 3]]]}, "properties": {"InLine_FID": 5, "Type": "Secondary Road", "Route_Type": 1, "Segment_s": "Hypothetical", "Name": "test", "Citation": "c", "Bibliograp": "b", "Lower_Date": 9999, "Low_Date_E": 9999, "Upper_Date": 50, "Up_Date_E": 1, "Avg_Slope": 2, "passabilit": 0.5, "Shape_Leng": 10}}]}), encoding="utf-8")
    segments = parse_itiner_e_geojson(path)
    assert len(segments) == 2
    assert segments[0].geometry == ((1.0, 2.0), (2.0, 2.0))
    assert segments[1].edge_id == "itiner-e:5:part:1"
    assert segments[0].road_type == "Secondary Road"
    assert segments[0].chronology.status == "unknown"
    assert segments[0].chronology.raw_values["Lower_Date"] == 9999


def test_geojson_parser_normalises_declared_epsg_3395_and_retains_source_geometry(tmp_path):
    path = tmp_path / "projected.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3395"}}, "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0, 5], [111319.490793, 0, 6]]}, "properties": {"InLine_FID": 6}}]}), encoding="utf-8")
    segment = parse_itiner_e_geojson(path)[0]
    assert segment.coordinate_reference_system == "EPSG:3395"
    assert segment.source_geometry == ((0.0, 0.0, 5.0), (111319.490793, 0.0, 6.0))
    assert segment.geometry[0] == (0.0, 0.0)
    assert segment.geometry[1][0] == pytest.approx(1.0, abs=0.001)


def test_endpoint_topology_is_deterministic_and_tolerance_controlled():
    segments = [segment("b", ((0, 0), (1, 0))), segment("a", ((1.00005, 0), (2, 0)))]
    five_m = RomanRoadGraph.from_segments(segments, snap_tolerance_m=5)
    ten_m = RomanRoadGraph.from_segments(segments, snap_tolerance_m=10)
    assert five_m.stats.component_count == 2
    assert ten_m.stats.component_count == 1
    assert ten_m.stats.snapped_endpoint_count == 1
    assert RomanRoadGraph.from_segments(list(reversed(segments)), snap_tolerance_m=10).stats == ten_m.stats


def test_disconnected_components_nearest_lookup_and_path_geometry_reconstruction():
    graph = RomanRoadGraph.from_segments([
        segment("one", ((0, 0), (1, 0), (2, 0))),
        segment("two", ((2, 0), (3, 0))),
        segment("far", ((20, 0), (21, 0))),
    ], snap_tolerance_m=5)
    start, _ = graph.nearest_node((0.01, 0))
    end, _ = graph.nearest_node((3.01, 0))
    far, _ = graph.nearest_node((20.01, 0))
    path = graph.shortest_path(start.id, end.id)
    assert path is not None
    assert path.geometry == ((0, 0), (1, 0), (2, 0), (3, 0))
    assert [graph.edges[edge_id].segment.record_id for edge_id in path.edge_ids] == ["one", "two"]
    assert graph.connected(start.id, far.id) is False


def test_crossing_interiors_are_not_implicitly_split_or_connected():
    graph = RomanRoadGraph.from_segments([
        segment("horizontal", ((0, 0), (2, 0))),
        segment("vertical", ((1, -1), (1, 1))),
    ], snap_tolerance_m=5)
    horizontal_start, _ = graph.nearest_node((0, 0))
    vertical_start, _ = graph.nearest_node((1, -1))
    assert graph.stats.component_count == 2
    assert graph.shortest_path(horizontal_start.id, vertical_start.id) is None


def test_duplicate_geometries_keep_parallel_provenance_edges():
    graph = RomanRoadGraph.from_segments([segment("first", ((0, 0), (1, 0))), segment("second", ((1, 0), (0, 0)))], snap_tolerance_m=5)
    assert graph.stats.duplicate_geometry_count == 1
    assert {edge.segment.record_id for edge in graph.edges.values()} == {"first", "second"}


def test_chronology_and_segment_statuses_are_preserved_without_inference():
    known = segment("known", ((0, 0), (1, 0)), status="Certain", lower=-200, upper=50)
    unknown = segment("unknown", ((1, 0), (2, 0)), status="Conjectured")
    hypothetical = segment("hyp", ((2, 0), (3, 0)), status="Hypothetical")
    graph = RomanRoadGraph.from_segments([known, unknown, hypothetical], snap_tolerance_m=5)
    assert graph.edges[known.edge_id].segment.chronology.status == "known"
    assert graph.edges[unknown.edge_id].segment.chronology.status == "unknown"
    assert {edge.segment.segment_status for edge in graph.edges.values()} == {"Certain", "Conjectured", "Hypothetical"}


def test_snapping_preserves_a_collapsed_short_segment_as_an_explicit_self_loop():
    graph = RomanRoadGraph.from_segments([segment("tiny", ((0, 0), (0.00001, 0)))], snap_tolerance_m=5)
    assert graph.stats.self_loop_segment_count == 1
    assert graph.stats.edge_count == 1
    edge = graph.edges["itiner-e:tiny:part:0"]
    assert edge.source_node_id == edge.target_node_id
