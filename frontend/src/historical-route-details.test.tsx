import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RouteDetails } from "./historical-route-details";
import {
  drawableRouteSegments,
  failedRouteSegments,
  fetchAgentHistoricalRoutePresentation,
  markerFeatures,
  routeSegments,
  toLeafletLineCoordinates,
  type HistoricalRoutePresentationPayload,
} from "./phase10-contract";

const point = (id: string, name: string, coordinates: [number, number]) => ({
  type: "Feature" as const,
  geometry: { type: "Point" as const, coordinates },
  properties: { layer_type: "historical_anchor", waypoint_id: id, knowledge_panel_id: id, name, evidence_refs: [`e-${id}`], coordinate_role: "representative_point" },
});

const terrain: HistoricalRoutePresentationPayload = {
  route: { route_id: "terrain-route", route_name: "Evidence-constrained route", confidence: 0.72 },
  waypoints: [
    { id: "a", name: "Anchor A", location_confidence: "approximate", evidence_refs: ["e-a"] },
    { id: "b", name: "Anchor B", location_confidence: "approximate", evidence_refs: ["e-b"] },
  ],
  geojson: { type: "FeatureCollection", features: [point("a", "Anchor A", [4, 43]), point("b", "Anchor B", [7, 45])] },
  route_geojson: { type: "Feature", geometry: { type: "LineString", coordinates: [[4, 43], [5, 44], [7, 45]] }, properties: { route_type: "terrain_aware_historical_reconstruction", route_quality: { terrain_source: "offline_dem", segment_ledger: [{ segment_id: "a-b", source_anchor_id: "a", target_anchor_id: "b", physical_distance_km: 223.64, search_cost_total: 44.73, terrain_source: "offline_dem" }] } } },
  knowledge_panels: [{ waypoint_id: "a", title: "Anchor A", evidence_refs: ["e-a"], source_references: ["Primary source"], external_references: [], confidence: "approximate" }],
  presentation_summary: { title: "Route", route_interpretation: "Algorithmic connection between historical anchors.", limitations: ["Not an exact documented track."] },
};

const road: HistoricalRoutePresentationPayload = {
  route: { route_id: "road-route", confidence: 0.7, generation_method: "ROMAN_ROAD_NETWORK" },
  geojson: { type: "FeatureCollection", features: [point("a", "Anchor A", [1, 2]), point("b", "Anchor B", [2, 3]), { type: "Feature", geometry: { type: "LineString", coordinates: [[1, 2], [2, 3]] }, properties: { layer_type: "roman_road_segment", segment_role: "roman_road", leg_index: 1 } }, { type: "Feature", geometry: null, properties: { layer_type: "roman_road_segment", segment_role: "failed_gap", leg_index: 2, failure_status: "NO_PATH" } }] },
  road_network: { source: "Itiner-e", route_status: "PARTIAL", aggregate: { successful_leg_count: 1, failed_leg_count: 1, total_network_distance_m: 12000, total_access_connector_distance_m: 0, road_type_counts: {}, segment_status_counts: {}, chronology_counts: {} }, limitations: ["Infrastructure candidate only."], legs: [{ leg_index: 1, source_anchor_id: "a", destination_anchor_id: "b", reconstruction_method: "ROMAN_ROAD_NETWORK", candidate: { network_distance_m: 12000 } }, { leg_index: 2, source_anchor_id: "b", destination_anchor_id: "c", reconstruction_method: "UNAVAILABLE", failure_status: "NO_PATH" }] },
};

describe("Phase 33A historical route presentation", () => {
  it("keeps an ordinary non-route response presentation-free", async () => {
    const result = await fetchAgentHistoricalRoutePresentation("ordinary question", "s", async () => ({ ok: true, status: 200, json: async () => ({ reply: "Normal answer", state: {} }) }));
    expect(result).toMatchObject({ reply: "Normal answer", payload: null });
  });

  it("creates waypoint markers and a drawable candidate polyline", () => {
    expect(markerFeatures(terrain)).toHaveLength(2);
    expect(drawableRouteSegments(terrain)).toHaveLength(1);
  });

  it("converts backend longitude/latitude to Leaflet latitude/longitude", () => {
    expect(toLeafletLineCoordinates(terrain.route_geojson ?? undefined)).toEqual([[43, 4], [44, 5], [45, 7]]);
  });

  it("identifies terrain and Roman-road reconstruction independently", () => {
    expect(routeSegments(terrain)[0]).toMatchObject({ kind: "terrain", distanceKm: 223.64, terrainSource: "offline_dem" });
    expect(routeSegments(road)[0]).toMatchObject({ kind: "roman_road", distanceKm: 12 });
  });

  it("withholds failed gaps from drawable route geometry", () => {
    expect(failedRouteSegments(road)).toHaveLength(1);
    expect(drawableRouteSegments(road)).toHaveLength(1);
  });

  it("renders historical authority, reconstruction semantics, provenance, and limitations", () => {
    const html = renderToStaticMarkup(<RouteDetails payload={terrain} routeSource="legacy_movement_claims" />);
    expect(html).toContain("历史依据");
    expect(html).toContain("算法重建");
    expect(html).toContain("史料中的移动关系");
    expect(html).not.toContain("legacy_movement_claims");
    expect(html).toContain("Primary source");
    expect(html).toContain("Not an exact documented track.");
  });

  it("renders failed-gap warnings and Roman-road semantics as text, not color alone", () => {
    const html = renderToStaticMarkup(<RouteDetails payload={road} />);
    expect(html).toContain("古罗马道路优先重建");
    expect(html).toContain("未能可靠重建");
    expect(html).toContain("NO_PATH");
  });

  it("remains stable when optional presentation fields and geometry are absent", () => {
    const sparse: HistoricalRoutePresentationPayload = { route: { route_id: "sparse", confidence: 0.4 }, geojson: { type: "FeatureCollection", features: [point("only", "Only waypoint", [1, 2])] } };
    expect(() => renderToStaticMarkup(<RouteDetails payload={sparse} />)).not.toThrow();
    expect(drawableRouteSegments(sparse)).toEqual([]);
  });

  it("does not retain a previous route in a later non-route response", async () => {
    const first = await fetchAgentHistoricalRoutePresentation("route", "s", async () => ({ ok: true, status: 200, json: async () => ({ reply: "route", state: { historical_route_presentation: terrain } }) }));
    const second = await fetchAgentHistoricalRoutePresentation("ordinary", "s", async () => ({ ok: true, status: 200, json: async () => ({ reply: "ordinary", state: { historical_route_presentation: null } }) }));
    expect(first.payload).not.toBeNull();
    expect(second.payload).toBeNull();
  });
});
