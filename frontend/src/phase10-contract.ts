export type ExternalReference = { id: string; title: string; url: string; reference_type: string; language?: string | null; description?: string | null; };
export type HistoricalKnowledgePanel = { waypoint_id: string; title: string; period?: string | null; event_type?: string | null; summary?: string | null; evidence_refs: string[]; source_references: string[]; external_references: ExternalReference[]; confidence: string; };
export type HistoricalWaypoint = { id: string; name: string; event_type?: string | null; period?: string | null; description?: string | null; location_confidence: string; evidence_refs: string[]; source_book?: string | null; source_chapter?: string | null; external_references?: ExternalReference[]; };
export type GeoJsonFeature = { type: "Feature"; geometry: { type: "LineString"; coordinates: [number, number][] } | { type: "Point"; coordinates: [number, number] } | { type: "Polygon"; coordinates: [number, number][][] } | null; properties: Record<string, unknown>; };
export type PresentationTimelineStep = { order: number; title: string; description?: string | null; period?: string | null; evidence_count: number; confidence: number; };
export type PresentationSummary = { title: string; campaign_id?: string | null; campaign?: string | null; operation_id?: string | null; operation?: string | null; date?: string | null; historical_context: string; route_method: string; evidence_basis: string[]; route_interpretation: string; route_stages: string[]; sources: string[]; geographic_constraints: string[]; uncertainty_notes: string[]; limitations: string[]; timeline: PresentationTimelineStep[]; };
export type RomanRoadNetwork = { source: string; route_status: "COMPLETE" | "PARTIAL" | "UNAVAILABLE"; aggregate: { successful_leg_count: number; failed_leg_count: number; total_network_distance_m: number; total_access_connector_distance_m: number; road_type_counts: Record<string, number>; segment_status_counts: Record<string, number>; chronology_counts: Record<string, number> }; limitations: string[]; legs: unknown[]; };
export type HistoricalRoutePresentationPayload = { route: { route_id: string; route_name?: string | null; period?: string | null; confidence: number; generation_method?: string; route_status?: string }; waypoints: HistoricalWaypoint[]; geojson: { type: "FeatureCollection"; features: GeoJsonFeature[] }; route_geojson?: GeoJsonFeature; knowledge_panels: HistoricalKnowledgePanel[]; presentation_summary?: PresentationSummary | null; road_network?: RomanRoadNetwork; };
export type WaypointPopupMetadata = { waypointId: string; name: string; eventType: string | null; period: string | null; description: string | null; confidence: string | null; evidenceCount: number; sourceBook: string | null; sourceChapter: string | null; externalReferenceCount: number; knowledgePanelId: string | null; };

type FetchLike = (input: string, init?: RequestInit) => Promise<{ ok: boolean; status: number; json: () => Promise<unknown> }>;
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

export function routeFeature(payload: HistoricalRoutePresentationPayload): GeoJsonFeature | undefined { return payload.route_geojson?.geometry?.type === "LineString" ? payload.route_geojson : payload.geojson.features.find((feature) => feature.geometry?.type === "LineString"); }

/** GeoJSON is longitude/latitude; Leaflet consumes latitude/longitude. */
export function toLeafletLineCoordinates(feature: GeoJsonFeature | undefined): [number, number][] {
  if (feature?.geometry?.type !== "LineString") return [];
  return feature.geometry.coordinates.map(([longitude, latitude]) => {
    if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) throw new Error("Route contains an invalid coordinate");
    return [latitude, longitude];
  });
}


export type RouteDirectionArrow = { position: [number, number]; rotationDeg: number };
export function routeDirectionArrows(payload: HistoricalRoutePresentationPayload, maxArrows = 3): RouteDirectionArrow[] {
  const coordinates = toLeafletLineCoordinates(routeFeature(payload));
  if (coordinates.length < 3) return [];
  const count = Math.min(maxArrows, coordinates.length - 2);
  return Array.from({ length: count }, (_, index) => {
    const currentIndex = Math.max(0, Math.floor(((index + 1) * (coordinates.length - 1)) / (count + 1)));
    const current = coordinates[currentIndex];
    const next = coordinates[Math.min(currentIndex + 1, coordinates.length - 1)];
    return { position: current, rotationDeg: Math.atan2(next[0] - current[0], next[1] - current[1]) * 180 / Math.PI };
  });
}

export function markerFeatures(payload: HistoricalRoutePresentationPayload): GeoJsonFeature[] { return payload.geojson.features.filter((feature) => feature.geometry?.type === "Point"); }
export function romanRoadSegmentFeatures(payload: HistoricalRoutePresentationPayload): GeoJsonFeature[] { return payload.geojson.features.filter((feature) => feature.properties.layer_type === "roman_road_segment"); }

export function uncertaintyCorridorFeatures(payload: HistoricalRoutePresentationPayload): GeoJsonFeature[] { return payload.geojson.features.filter((feature) => feature.geometry?.type === "Polygon" && feature.properties.layer_type === "uncertainty_corridor"); }

export function timelineWaypoint(payload: HistoricalRoutePresentationPayload, order: number): HistoricalWaypoint | null {
  return payload.waypoints[order - 1] ?? null;
}

export type MapLayerVisibility = { evidence: boolean; route: boolean; corridor: boolean };
export function visibleMapLayers(payload: HistoricalRoutePresentationPayload, visibility: MapLayerVisibility) {
  return {
    route: visibility.route ? routeFeature(payload) : undefined,
    evidence: visibility.evidence ? markerFeatures(payload) : [],
    corridor: visibility.corridor ? uncertaintyCorridorFeatures(payload) : [],
  };
}

export function waypointPopupMetadata(payload: HistoricalRoutePresentationPayload, feature: GeoJsonFeature): WaypointPopupMetadata {
  const waypointId = String(feature.properties.waypoint_id ?? "");
  const waypoint = payload.waypoints.find((item) => item.id === waypointId);
  return { waypointId, name: waypoint?.name ?? "Unnamed waypoint", eventType: waypoint?.event_type ?? null, period: waypoint?.period ?? null, description: waypoint?.description ?? null, confidence: waypoint?.location_confidence ?? null, evidenceCount: waypoint?.evidence_refs.length ?? 0, sourceBook: waypoint?.source_book ?? null, sourceChapter: waypoint?.source_chapter ?? null, externalReferenceCount: waypoint?.external_references?.length ?? 0, knowledgePanelId: typeof feature.properties.knowledge_panel_id === "string" ? feature.properties.knowledge_panel_id : null };
}

export function panelForFeature(payload: HistoricalRoutePresentationPayload, feature: GeoJsonFeature): HistoricalKnowledgePanel | null {
  const panelId = waypointPopupMetadata(payload, feature).knowledgePanelId;
  return panelId ? payload.knowledge_panels.find((panel) => panel.waypoint_id === panelId) ?? null : null;
}


export type AgentRouteChatResponse = {
  reply: string;
  state: { route_intent?: { intent: string; campaign_id: string } | null; historical_route_presentation?: HistoricalRoutePresentationPayload | null; };
};

export async function fetchAgentHistoricalRoutePresentation(
  message: string,
  sessionId: string,
  request: FetchLike = fetch,
): Promise<{ reply: string; payload: HistoricalRoutePresentationPayload | null; campaignId: string | null }> {
  const response = await request(`${BACKEND_ORIGIN}/api/v1/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!response.ok) throw new Error("Historical route request is unavailable");
  const body = await response.json() as AgentRouteChatResponse;
  const payload = body.state.historical_route_presentation;
  return {
    reply: body.reply,
    payload: payload ? loadHistoricalRoutePresentation(payload) : null,
    campaignId: body.state.route_intent?.campaign_id ?? null,
  };
}
