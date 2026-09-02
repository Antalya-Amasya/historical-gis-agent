export type ExternalReference = { id: string; title: string; url: string; reference_type: string; language?: string | null; description?: string | null; };
export type HistoricalKnowledgePanel = { waypoint_id: string; title: string; period?: string | null; event_type?: string | null; summary?: string | null; evidence_refs: string[]; source_references: string[]; external_references: ExternalReference[]; confidence: string; };
export type HistoricalWaypoint = { id: string; name: string; event_type?: string | null; period?: string | null; description?: string | null; location_confidence?: string; location_notes?: string | null; evidence_refs?: string[]; source_book?: string | null; source_chapter?: string | null; external_references?: ExternalReference[]; };
export type GeoJsonFeature = { type: "Feature"; geometry: { type: "LineString"; coordinates: [number, number][] } | { type: "Point"; coordinates: [number, number] } | { type: "Polygon"; coordinates: [number, number][][] } | null; properties: Record<string, unknown>; };
export type PresentationTimelineStep = { order: number; title: string; description?: string | null; period?: string | null; evidence_count: number; confidence: number; };
export type PresentationSummary = { title: string; campaign_id?: string | null; campaign?: string | null; operation_id?: string | null; operation?: string | null; date?: string | null; historical_context?: string; route_method?: string; route_status?: string; evidence_basis?: string[]; route_interpretation?: string; route_stages?: string[]; sources?: string[]; geographic_constraints?: string[]; uncertainty_notes?: string[]; limitations?: string[]; timeline?: PresentationTimelineStep[]; };
export type CandidateRouteScore = { total_cost?: number; distance_cost?: number; slope_cost?: number; terrain_cost?: number; barrier_cost?: number; historical_cost?: number; };
export type CandidateSegmentLedger = { segment_id?: string; source_anchor_id?: string; target_anchor_id?: string; physical_distance_km?: number; search_cost_total?: number; cost_breakdown?: CandidateRouteScore; terrain_source?: string; geometry_role?: string; applied_constraints?: string[]; };
export type RouteQuality = { segment_ledger?: CandidateSegmentLedger[]; waypoint_order_preserved?: boolean; terrain_source?: string; terrain_constrained?: boolean; applied_constraints?: string[]; coordinate_count?: number; };
export type CrossingCandidate = { coordinate: [number, number]; barrier_id: string; barrier_name: string; source: "ANCIENT_ROAD_BARRIER_CANDIDATE" | "TERRAIN_DERIVED_CROSSING"; road_support: boolean; terrain_support: boolean; reconstruction_cost: number; authority: "algorithmic_gis_candidate"; limitations: string[]; };
export type RomanRoadLeg = { leg_index?: number; source_anchor_id?: string; destination_anchor_id?: string; barrier_anchor_id?: string | null; reconstruction_method?: string; limitation?: string | null; failure_status?: string | null; crossing_candidate?: CrossingCandidate | null; terrain_candidate?: { metrics?: { distance_m?: number }; cost_breakdown?: CandidateRouteScore } | null; candidate?: { network_distance_m?: number } | null; };
export type RomanRoadNetwork = { source: string; route_status: "COMPLETE" | "PARTIAL" | "UNAVAILABLE"; terrain_fallback_used?: boolean; aggregate: { successful_leg_count: number; failed_leg_count: number; total_network_distance_m: number; total_access_connector_distance_m: number; road_type_counts: Record<string, number>; segment_status_counts: Record<string, number>; chronology_counts: Record<string, number> }; limitations: string[]; legs: RomanRoadLeg[]; };
export type HistoricalRoutePresentationPayload = { route: { route_id: string; route_name?: string | null; period?: string | null; confidence: number; generation_method?: string; route_status?: string; score?: CandidateRouteScore; explanations?: { distance_reason?: string; terrain_reason?: string; historical_reason?: string }; }; waypoints?: HistoricalWaypoint[]; geojson: { type: "FeatureCollection"; features: GeoJsonFeature[] }; route_geojson?: GeoJsonFeature | null; knowledge_panels?: HistoricalKnowledgePanel[]; presentation_summary?: PresentationSummary | null; road_network?: RomanRoadNetwork; location_warnings?: string[]; fragments?: RoutePresentationFragment[]; };
export type RoutePresentationFragment = { component_id: string; status: "COMPLETE" | "FAILED" | "SKIPPED"; evidence_refs?: string[]; waypoints?: HistoricalWaypoint[]; route_geojson?: GeoJsonFeature | null; reason_code?: string | null; };
export type WaypointPopupMetadata = { waypointId: string; name: string; eventType: string | null; period: string | null; description: string | null; confidence: string | null; evidenceCount: number; sourceBook: string | null; sourceChapter: string | null; externalReferenceCount: number; knowledgePanelId: string | null; };
export type HistoricalRouteSegmentPresentation = { id: string; kind: "roman_road" | "terrain" | "connector" | "failed_gap"; feature?: GeoJsonFeature; from?: string; to?: string; distanceKm?: number; cost?: number; terrainSource?: string; limitation?: string | null; failureStatus?: string | null; };

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

export function routeFragmentFeatures(payload: HistoricalRoutePresentationPayload): GeoJsonFeature[] {
  const tagged = payload.geojson.features.filter((feature) => feature.geometry?.type === "LineString" && feature.properties.layer_type === "route_fragment");
  if (tagged.length) return tagged;
  const single = routeFeature(payload);
  return single ? [single] : [];
}

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
  const coordinates = routeFragmentFeatures(payload).flatMap((feature) => toLeafletLineCoordinates(feature));
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
export function romanRoadSegmentFeatures(payload: HistoricalRoutePresentationPayload): GeoJsonFeature[] { return payload.geojson.features.filter((feature) => feature.properties.layer_type === "roman_road_segment" || feature.properties.layer_type === "terrain_reconstruction_segment"); }

export function routeSegments(payload: HistoricalRoutePresentationPayload): HistoricalRouteSegmentPresentation[] {
  if (payload.road_network) {
    return romanRoadSegmentFeatures(payload).map((feature, index) => {
      const role = String(feature.properties.segment_role ?? "");
      const legIndex = Number(feature.properties.leg_index ?? index + 1);
      const leg = payload.road_network?.legs.find((item) => item.leg_index === legIndex);
      const kind = role === "failed_gap" ? "failed_gap" : role === "terrain_candidate" ? "terrain" : role === "roman_road" ? "roman_road" : "connector";
      const distanceM = kind === "terrain" ? leg?.terrain_candidate?.metrics?.distance_m : leg?.candidate?.network_distance_m;
      const featureFailure = String(feature.properties.failure_status ?? "") || null;
      return { id: `${legIndex}-${role || kind}-${index}`, kind, feature, from: leg?.source_anchor_id, to: leg?.destination_anchor_id, distanceKm: typeof distanceM === "number" ? distanceM / 1000 : undefined, cost: leg?.terrain_candidate?.cost_breakdown?.total_cost, limitation: leg?.limitation, failureStatus: leg?.failure_status ?? featureFailure };
    });
  }
  const feature = routeFeature(payload);
  const fragmentFeatures = routeFragmentFeatures(payload);
  const quality = feature?.properties.route_quality as RouteQuality | undefined;
  const ledgers = quality?.segment_ledger ?? [];
  if (ledgers.length) return ledgers.map((ledger, index) => ({ id: ledger.segment_id ?? `terrain-${index + 1}`, kind: "terrain", feature, from: ledger.source_anchor_id, to: ledger.target_anchor_id, distanceKm: ledger.physical_distance_km, cost: ledger.search_cost_total ?? ledger.cost_breakdown?.total_cost, terrainSource: ledger.terrain_source ?? quality?.terrain_source }));
  if (fragmentFeatures.length > 1) {
    return fragmentFeatures.map((fragment, index) => ({
      id: String(fragment.properties.component_id ?? `fragment-${index + 1}`),
      kind: "terrain" as const,
      feature: fragment,
      from: String(fragment.properties.source_anchor_id ?? fragment.properties.component_id ?? `fragment-${index + 1}`),
      to: String(fragment.properties.target_anchor_id ?? fragment.properties.component_id ?? `fragment-${index + 1}`),
      terrainSource: String((fragment.properties.route_quality as RouteQuality | undefined)?.terrain_source ?? ""),
    }));
  }
  return feature?.geometry?.type === "LineString" ? [{ id: "candidate-route", kind: "terrain", feature, terrainSource: quality?.terrain_source }] : [];
}

export function drawableRouteSegments(payload: HistoricalRoutePresentationPayload): HistoricalRouteSegmentPresentation[] { return routeSegments(payload).filter((segment) => segment.kind !== "failed_gap" && segment.feature?.geometry?.type === "LineString"); }
export function failedRouteSegments(payload: HistoricalRoutePresentationPayload): HistoricalRouteSegmentPresentation[] { return routeSegments(payload).filter((segment) => segment.kind === "failed_gap"); }
export function failedRouteFragments(payload: HistoricalRoutePresentationPayload): RoutePresentationFragment[] {
  return (payload.fragments ?? []).filter((fragment) => fragment.status === "FAILED");
}

export function uncertaintyCorridorFeatures(payload: HistoricalRoutePresentationPayload): GeoJsonFeature[] { return payload.geojson.features.filter((feature) => feature.geometry?.type === "Polygon" && feature.properties.layer_type === "uncertainty_corridor"); }

export function timelineWaypoint(payload: HistoricalRoutePresentationPayload, order: number): HistoricalWaypoint | null {
  return payload.waypoints?.[order - 1] ?? null;
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
  const waypoint = payload.waypoints?.find((item) => item.id === waypointId);
  const evidenceRefs = waypoint?.evidence_refs ?? (Array.isArray(feature.properties.evidence_refs) ? feature.properties.evidence_refs : []);
  return { waypointId, name: waypoint?.name ?? String(feature.properties.name ?? "Unnamed waypoint"), eventType: waypoint?.event_type ?? null, period: waypoint?.period ?? null, description: waypoint?.description ?? (typeof feature.properties.coordinate_role === "string" ? `Coordinate role: ${feature.properties.coordinate_role}` : null), confidence: waypoint?.location_confidence ?? (typeof feature.properties.confidence === "string" ? feature.properties.confidence : null), evidenceCount: evidenceRefs.length, sourceBook: waypoint?.source_book ?? null, sourceChapter: waypoint?.source_chapter ?? null, externalReferenceCount: waypoint?.external_references?.length ?? 0, knowledgePanelId: typeof feature.properties.knowledge_panel_id === "string" ? feature.properties.knowledge_panel_id : null };
}

export function panelForFeature(payload: HistoricalRoutePresentationPayload, feature: GeoJsonFeature): HistoricalKnowledgePanel | null {
  const panelId = waypointPopupMetadata(payload, feature).knowledgePanelId;
  return panelId ? payload.knowledge_panels?.find((panel) => panel.waypoint_id === panelId) ?? null : null;
}


export type AgentRouteChatResponse = {
  reply: string;
  state: { route_intent?: { intent: string; campaign_id: string } | null; historical_route_presentation?: HistoricalRoutePresentationPayload | null; historical_route_diagnostics?: { route_source?: string } | null; };
};

export async function fetchAgentHistoricalRoutePresentation(
  message: string,
  sessionId: string,
  request: FetchLike = fetch,
): Promise<{ reply: string; payload: HistoricalRoutePresentationPayload | null; campaignId: string | null; routeSource: string | null }> {
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
    routeSource: body.state.historical_route_diagnostics?.route_source ?? null,
  };
}
