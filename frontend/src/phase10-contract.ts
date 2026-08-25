export type ExternalReference = { id: string; title: string; url: string; reference_type: string; language?: string | null; description?: string | null; };
export type HistoricalKnowledgePanel = { waypoint_id: string; title: string; period?: string | null; event_type?: string | null; summary?: string | null; evidence_refs: string[]; source_references: string[]; external_references: ExternalReference[]; confidence: string; };
export type HistoricalWaypoint = { id: string; name: string; event_type?: string | null; period?: string | null; description?: string | null; location_confidence: string; evidence_refs: string[]; };
export type GeoJsonFeature = { type: "Feature"; geometry: { type: "LineString"; coordinates: [number, number][] } | { type: "Point"; coordinates: [number, number] } | null; properties: Record<string, unknown>; };
export type HistoricalRoutePresentationPayload = { route: { route_id: string; route_name?: string | null; period?: string | null; confidence: number }; waypoints: HistoricalWaypoint[]; geojson: { type: "FeatureCollection"; features: GeoJsonFeature[] }; knowledge_panels: HistoricalKnowledgePanel[]; };
export type WaypointPopupMetadata = { waypointId: string; name: string; eventType: string | null; period: string | null; description: string | null; confidence: string | null; evidenceCount: number; knowledgePanelId: string | null; };

type FetchLike = (input: string) => Promise<{ ok: boolean; status: number; json: () => Promise<unknown> }>;
const BACKEND_ORIGIN = import.meta.env.VITE_BACKEND_BASE_URL ?? "http://127.0.0.1:8000";

export function loadHistoricalRoutePresentation(payload: HistoricalRoutePresentationPayload): HistoricalRoutePresentationPayload {
  if (payload.geojson.type !== "FeatureCollection") throw new Error("Expected a GeoJSON FeatureCollection");
  return payload;
}

export async function fetchHistoricalRoutePresentation(routeId: string, request: FetchLike = fetch): Promise<HistoricalRoutePresentationPayload> {
  const response = await request(`${BACKEND_ORIGIN}/api/v1/historical-routes/${encodeURIComponent(routeId)}/presentation`);
  if (!response.ok) throw new Error(response.status === 404 ? "Historical route not found" : "Historical route presentation is unavailable");
  return loadHistoricalRoutePresentation(await response.json() as HistoricalRoutePresentationPayload);
}

export function routeFeature(payload: HistoricalRoutePresentationPayload): GeoJsonFeature | undefined { return payload.geojson.features.find((feature) => feature.geometry?.type === "LineString"); }
export function markerFeatures(payload: HistoricalRoutePresentationPayload): GeoJsonFeature[] { return payload.geojson.features.filter((feature) => feature.geometry?.type === "Point"); }

export function waypointPopupMetadata(payload: HistoricalRoutePresentationPayload, feature: GeoJsonFeature): WaypointPopupMetadata {
  const waypointId = String(feature.properties.waypoint_id ?? "");
  const waypoint = payload.waypoints.find((item) => item.id === waypointId);
  return { waypointId, name: waypoint?.name ?? "Unnamed waypoint", eventType: waypoint?.event_type ?? null, period: waypoint?.period ?? null, description: waypoint?.description ?? null, confidence: waypoint?.location_confidence ?? null, evidenceCount: waypoint?.evidence_refs.length ?? 0, knowledgePanelId: typeof feature.properties.knowledge_panel_id === "string" ? feature.properties.knowledge_panel_id : null };
}

export function panelForFeature(payload: HistoricalRoutePresentationPayload, feature: GeoJsonFeature): HistoricalKnowledgePanel | null {
  const panelId = waypointPopupMetadata(payload, feature).knowledgePanelId;
  return panelId ? payload.knowledge_panels.find((panel) => panel.waypoint_id === panelId) ?? null : null;
}
