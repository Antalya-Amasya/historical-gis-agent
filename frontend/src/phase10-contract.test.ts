import { describe, expect, it } from "vitest";
import { fetchAgentHistoricalRoutePresentation, fetchHistoricalRoutePresentation, markerFeatures, panelForFeature, routeDirectionArrows, routeFeature, visibleMapLayers, toLeafletLineCoordinates, uncertaintyCorridorFeatures, waypointPopupMetadata } from "./phase10-contract";

const response = { route: { route_id: "r1", route_name: "Route", confidence: 0.7 }, waypoints: [{ id: "a", name: "Anchor", event_type: "CITY", period: "218 BCE", description: "Supplied", location_confidence: "EXACT", evidence_refs: ["e1"] }], geojson: { type: "FeatureCollection" as const, features: [{ type: "Feature" as const, geometry: { type: "LineString" as const, coordinates: [[1, 2], [3, 4]] as [number, number][] }, properties: {} }, { type: "Feature" as const, geometry: { type: "Point" as const, coordinates: [1, 2] as [number, number] }, properties: { waypoint_id: "a", knowledge_panel_id: "a" } }, { type: "Feature" as const, geometry: null, properties: { waypoint_id: "missing" } }] }, knowledge_panels: [{ waypoint_id: "a", title: "Anchor", summary: "Bound summary", evidence_refs: ["e1"], source_references: ["e1"], external_references: [{ id: "ref", title: "Reading", url: "https://example.invalid", reference_type: "PAPER" }], confidence: "EXACT" }] };

describe("Phase 10.1 presentation API contract", () => {
  it("fetches and loads backend route JSON", async () => { const payload = await fetchHistoricalRoutePresentation("r1", async () => ({ ok: true, status: 200, json: async () => response })); expect(routeFeature(payload)?.geometry?.type).toBe("LineString"); });
  it("preserves backend waypoint metadata for popups", () => { const feature = response.geojson.features[1]; expect(waypointPopupMetadata(response, feature)).toMatchObject({ name: "Anchor", evidenceCount: 1, knowledgePanelId: "a" }); });
  it("does not select a marker for null geometry", () => { expect(markerFeatures(response)).toHaveLength(1); expect(response.geojson.features[2].geometry).toBeNull(); });
  it("renders supplied external references through the linked knowledge panel", () => { expect(panelForFeature(response, response.geojson.features[1])?.external_references[0].title).toBe("Reading"); });


  it("keeps internal evidence identifiers out of popup metadata", () => {
    const metadata = waypointPopupMetadata(response, response.geojson.features[1]);
    expect(JSON.stringify(metadata)).not.toContain("e1");
    expect(metadata.evidenceCount).toBe(1);
  });

  it("prefers the standalone schematic route feature when supplied", () => {
    const payload = { ...response, route_geojson: { type: "Feature" as const, geometry: { type: "LineString" as const, coordinates: [[5, 6], [7, 8]] as [number, number][] }, properties: { route_type: "schematic_historical_route" } } };
    expect(routeFeature(payload)?.properties.route_type).toBe("schematic_historical_route");
  });



  it("converts GeoJSON longitude/latitude into Leaflet latitude/longitude", () => {
    expect(toLeafletLineCoordinates(routeFeature(response))).toEqual([[2, 1], [4, 3]]);
  });

  it("loads the Agent-returned terrain presentation without frontend coordinate inference", async () => {
    const agentResponse = { reply: "Ready", state: { route_intent: { intent: "historical_route", campaign_id: "hannibal_italy_campaign" }, historical_route_presentation: response } };
    const result = await fetchAgentHistoricalRoutePresentation("展示汉尼拔路线", "test", async () => ({ ok: true, status: 200, json: async () => agentResponse }));
    expect(result.campaignId).toBe("hannibal_italy_campaign");
    expect(routeFeature(result.payload)?.geometry?.type).toBe("LineString");
  });

});


it("returns only explicitly labeled uncertainty-corridor polygon features", () => {
  const payload = { ...response, geojson: { ...response.geojson, features: [...response.geojson.features, { type: "Feature" as const, geometry: { type: "Polygon" as const, coordinates: [[[1, 2], [3, 2], [3, 4], [1, 2]]] as [number, number][][] }, properties: { layer_type: "uncertainty_corridor", label: "Reviewed corridor" } }] } };
  expect(uncertaintyCorridorFeatures(payload)).toHaveLength(1);
});


it("derives display-only direction arrows from backend LineString coordinates", () => {
  const payload = { ...response, route_geojson: { type: "Feature" as const, geometry: { type: "LineString" as const, coordinates: [[1, 2], [2, 3], [3, 4]] as [number, number][] }, properties: {} } };
  const arrows = routeDirectionArrows(payload);
  expect(arrows.length).toBeGreaterThan(0);
  expect(arrows[0].position).toEqual([3, 2]);
});


it("selects each map layer independently for the UI toggles", () => {
  const payload = { ...response, geojson: { ...response.geojson, features: [...response.geojson.features, { type: "Feature" as const, geometry: { type: "Polygon" as const, coordinates: [[[1, 2], [3, 2], [3, 4], [1, 2]]] as [number, number][][] }, properties: { layer_type: "uncertainty_corridor" } }] } };
  expect(visibleMapLayers(payload, { evidence: true, route: false, corridor: true })).toMatchObject({ route: undefined, evidence: [expect.anything()], corridor: [expect.anything()] });
  expect(visibleMapLayers(payload, { evidence: false, route: true, corridor: false }).evidence).toEqual([]);
  expect(visibleMapLayers(payload, { evidence: false, route: true, corridor: false }).corridor).toEqual([]);
});
