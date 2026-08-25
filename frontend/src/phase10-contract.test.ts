import { describe, expect, it } from "vitest";
import { fetchHistoricalRoutePresentation, markerFeatures, panelForFeature, routeFeature, waypointPopupMetadata } from "./phase10-contract";

const response = { route: { route_id: "r1", route_name: "Route", confidence: 0.7 }, waypoints: [{ id: "a", name: "Anchor", event_type: "CITY", period: "218 BCE", description: "Supplied", location_confidence: "EXACT", evidence_refs: ["e1"] }], geojson: { type: "FeatureCollection" as const, features: [{ type: "Feature" as const, geometry: { type: "LineString" as const, coordinates: [[1, 2], [3, 4]] as [number, number][] }, properties: {} }, { type: "Feature" as const, geometry: { type: "Point" as const, coordinates: [1, 2] as [number, number] }, properties: { waypoint_id: "a", knowledge_panel_id: "a" } }, { type: "Feature" as const, geometry: null, properties: { waypoint_id: "missing" } }] }, knowledge_panels: [{ waypoint_id: "a", title: "Anchor", summary: "Bound summary", evidence_refs: ["e1"], source_references: ["e1"], external_references: [{ id: "ref", title: "Reading", url: "https://example.invalid", reference_type: "PAPER" }], confidence: "EXACT" }] };

describe("Phase 10.1 presentation API contract", () => {
  it("fetches and loads backend route JSON", async () => { const payload = await fetchHistoricalRoutePresentation("r1", async () => ({ ok: true, status: 200, json: async () => response })); expect(routeFeature(payload)?.geometry?.type).toBe("LineString"); });
  it("preserves backend waypoint metadata for popups", () => { const feature = response.geojson.features[1]; expect(waypointPopupMetadata(response, feature)).toMatchObject({ name: "Anchor", evidenceCount: 1, knowledgePanelId: "a" }); });
  it("does not select a marker for null geometry", () => { expect(markerFeatures(response)).toHaveLength(1); expect(response.geojson.features[2].geometry).toBeNull(); });
  it("renders supplied external references through the linked knowledge panel", () => { expect(panelForFeature(response, response.geojson.features[1])?.external_references[0].title).toBe("Reading"); });
});
