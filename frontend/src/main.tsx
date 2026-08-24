import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";

const API = "http://127.0.0.1:8000/api/v1/agent/chat";
const ISOLATION_BASEMAP = import.meta.env.DEV && new URLSearchParams(window.location.search).get("basemap") === "diagnostic";
const DIAGNOSTIC_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: { diagnostic_raster: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, maxzoom: 19, attribution: "© OpenStreetMap contributors" } },
  layers: [{ id: "diagnostic-background", type: "background", paint: { "background-color": "#d8f2ff" } }, { id: "diagnostic-raster", type: "raster", source: "diagnostic_raster" }],
};
// This browser-exposed tile key is intentionally injected from the ignored root .env at build/dev time.
const GEOAPIFY_MAP_KEY = import.meta.env.VITE_GEOAPIFY_API_KEY;
const GEOAPIFY_RASTER_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    geoapify_raster: {
      type: "raster",
      tiles: GEOAPIFY_MAP_KEY ? [`https://maps.geoapify.com/v1/tile/osm-bright/{z}/{x}/{y}.png?apiKey=${encodeURIComponent(GEOAPIFY_MAP_KEY)}`] : [],
      tileSize: 256,
      maxzoom: 20,
      attribution: "© OpenStreetMap contributors, © Geoapify",
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#d8f2ff" } },
    { id: "geoapify-raster", type: "raster", source: "geoapify_raster" },
  ],
};
const BASEMAP_STYLE = GEOAPIFY_MAP_KEY ? GEOAPIFY_RASTER_STYLE : DIAGNOSTIC_STYLE;
const ROUTE_SOURCE = "historical-route";
const ROUTE_LAYER = "historical-route-line";

declare global {
  interface Window { __historicalMap?: maplibregl.Map; }
}

function safeMapDiagnostic(value: string | undefined) {
  return value?.replace(/apiKey=[^&\s]+/gi, "apiKey=[redacted]");
}

function mapDebug(message: string, details?: Record<string, unknown>) {
  if (import.meta.env.DEV) console.debug("[HistoricalMap]", message, details ?? {});
}

type HistoricalPlace = { id: string; canonical_name: string; modern_name?: string | null; latitude: number; longitude: number; period?: string | null; source: string; source_id?: string | null; source_url?: string | null; confidence: number; uncertain: boolean; coordinate_role?: string; };
type Evidence = { id: string; author: string; work: string; locator: string; excerpt: string; source_file?: string | null; };
type HistoricalEvent = { id: string; name: string; period: string; summary: string; places: HistoricalPlace[]; evidence: Evidence[]; uncertainty_note?: string | null; };
type HistoricalRoutePoint = { sequence: number; historical_place: HistoricalPlace; event_summary: string; date_or_period?: string | null; evidence_refs: string[]; confidence: number; coordinate_role?: string; source_support?: string[]; };
type HistoricalRoute = { id: string; name: string; period: string; ordered_points: HistoricalRoutePoint[]; geometry: { type: "LineString"; coordinates: [number, number][] }; evidence_refs: string[]; assumptions: string[]; limitations: string[]; historical_confidence: number; };
type ChatResponse = { reply: string; state: { current_event?: HistoricalEvent | null; historical_route?: HistoricalRoute | null }; };

function placePopup(place: HistoricalPlace, event: HistoricalEvent, routePoint?: HistoricalRoutePoint) {
  const element = document.createElement("section");
  element.className = "place-popup";
  const source = place.source_url ? `<a href="${place.source_url}" target="_blank" rel="noreferrer">${place.source}</a>` : place.source;
  const evidence = routePoint?.evidence_refs.length ? `<p><strong>史料节点：</strong>${routePoint.evidence_refs.join(", ")}</p>` : "";
  const coordinateRole = routePoint?.coordinate_role ?? place.coordinate_role;
  const coordinateNote = coordinateRole && coordinateRole !== "exact_site" ? `<p><strong>坐标说明：</strong>${coordinateRole === "regional_centroid" ? "区域代表点，不代表具体山口或行军通道。" : "代表性参考点，不代表精确经过位置。"}</p>` : "";
  element.innerHTML = `<h3>${place.canonical_name}</h3><p><strong>现代名称：</strong>${place.modern_name ?? "未提供"}</p><p><strong>时间/时期：</strong>${routePoint?.date_or_period ?? place.period ?? event.period}</p><p><strong>事件：</strong>${routePoint?.event_summary ?? event.summary}</p>${evidence}${coordinateNote}<p><strong>坐标来源：</strong>${source}${place.source_id ? `（${place.source_id}）` : ""}</p><p><strong>置信度：</strong>${Math.round((routePoint?.confidence ?? place.confidence) * 100)}%${place.uncertain ? "；代表点存在不确定性" : ""}</p>`;
  return element;
}

function App() {
  const [message, setMessage] = useState("根据史料展示汉尼拔公元前218年进入意大利前的行动路线。");
  const [reply, setReply] = useState("等待提问。Mock Agent 无需 API Key。");
  const [event, setEvent] = useState<HistoricalEvent | null>(null);
  const [route, setRoute] = useState<HistoricalRoute | null>(null);
  const [routeVisible, setRouteVisible] = useState(true);
  const [mapReady, setMapReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const markers = useRef<maplibregl.Marker[]>([]);

  useEffect(() => {
    if (!mapContainer.current || map.current) return;
    const instance = new maplibregl.Map({ container: mapContainer.current, style: ISOLATION_BASEMAP ? DIAGNOSTIC_STYLE : BASEMAP_STYLE, center: [5, 41], zoom: 3 });
    instance.addControl(new maplibregl.NavigationControl(), "top-right");
    map.current = instance;
    if (import.meta.env.DEV) window.__historicalMap = instance;
    mapDebug("map constructed", { isolationBasemap: ISOLATION_BASEMAP });
    mapDebug("geoapify key diagnostics", { geoapifyKeyPresent: Boolean(GEOAPIFY_MAP_KEY), geoapifyKeyLength: GEOAPIFY_MAP_KEY.length });
    instance.once("load", () => { mapDebug("load event fired"); setMapReady(true); });
    instance.on("style.load", () => mapDebug("style.load event fired"));
    instance.on("styledata", (event) => mapDebug("styledata", { dataType: event.dataType }));
    instance.on("sourcedata", (event) => mapDebug("sourcedata", { sourceId: event.sourceId, sourceDataType: event.sourceDataType, isSourceLoaded: event.isSourceLoaded }));
    instance.on("data", (event) => { const diagnostic = event as unknown as { dataType?: string; sourceId?: string }; mapDebug("data", { dataType: diagnostic.dataType, sourceId: diagnostic.sourceId }); });
    instance.on("idle", () => mapDebug("idle"));
    instance.on("error", (event) => {
      const diagnostic = event as unknown as { error?: { message?: string; url?: string }; sourceId?: string; tile?: unknown };
      mapDebug("map error", { message: safeMapDiagnostic(diagnostic.error?.message) ?? "unknown MapLibre error", sourceId: diagnostic.sourceId, tile: diagnostic.tile ? String(diagnostic.tile) : undefined, resourceUrl: safeMapDiagnostic(diagnostic.error?.url) });
    });
    return () => { markers.current.forEach((marker) => marker.remove()); markers.current = []; instance.remove(); map.current = null; };
  }, []);

  useEffect(() => {
    const instance = map.current;
    mapDebug("route effect executed", { mapReady, routePointCount: route?.ordered_points.length ?? 0 });
    if (ISOLATION_BASEMAP) { mapDebug("route effect skipped", { reason: "diagnostic basemap mode" }); return; }
    if (!instance) { mapDebug("route effect skipped", { reason: "map unavailable" }); return; }
    if (!mapReady) { mapDebug("route effect skipped", { reason: "map not ready" }); return; }
    if (instance.getLayer(ROUTE_LAYER)) instance.removeLayer(ROUTE_LAYER);
    if (instance.getSource(ROUTE_SOURCE)) instance.removeSource(ROUTE_SOURCE);
    if (!route) { mapDebug("route effect skipped", { reason: "no route" }); return; }
    instance.addSource(ROUTE_SOURCE, { type: "geojson", data: { type: "Feature", properties: { label: "Historical reconstruction / schematic connection" }, geometry: route.geometry } });
    mapDebug("addSource executed", { source: ROUTE_SOURCE });
    instance.addLayer({ id: ROUTE_LAYER, type: "line", source: ROUTE_SOURCE, layout: { visibility: routeVisible ? "visible" : "none", "line-join": "round", "line-cap": "round" }, paint: { "line-color": "#2f6f9f", "line-width": 4, "line-opacity": 0.78, "line-dasharray": [2, 1] } });
    mapDebug("addLayer executed", { layer: ROUTE_LAYER });
  }, [mapReady, route]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !mapReady || !instance.getLayer(ROUTE_LAYER)) return;
    instance.setLayoutProperty(ROUTE_LAYER, "visibility", routeVisible ? "visible" : "none");
  }, [mapReady, route, routeVisible]);

  useEffect(() => {
    const instance = map.current;
    if (ISOLATION_BASEMAP) { mapDebug("marker effect skipped", { reason: "diagnostic basemap mode" }); return; }
    if (!instance) { mapDebug("marker effect skipped", { reason: "map unavailable" }); return; }
    if (!mapReady) { mapDebug("marker effect skipped", { reason: "map not ready" }); return; }
    markers.current.forEach((marker) => marker.remove());
    markers.current = [];
    const points = route?.ordered_points ?? event?.places.map((historical_place, index) => ({ sequence: index + 1, historical_place, event_summary: event.summary, evidence_refs: [], confidence: historical_place.confidence })) ?? [];
    if (!event) { mapDebug("marker effect skipped", { reason: "no event" }); return; }
    if (!points.length) { mapDebug("marker effect skipped", { reason: "no points" }); return; }
    mapDebug("current route point count", { count: points.length });
    const bounds = new maplibregl.LngLatBounds();
    points.forEach((point) => {
      const place = point.historical_place;
      const marker = new maplibregl.Marker({ color: route ? "#2f6f9f" : "#9c3f22" }).setLngLat([place.longitude, place.latitude]).setPopup(new maplibregl.Popup({ offset: 24 }).setDOMContent(placePopup(place, event, point))).addTo(instance);
      marker.getElement().setAttribute("aria-label", `${place.canonical_name} marker`);
      markers.current.push(marker);
      bounds.extend([place.longitude, place.latitude]);
    });
    mapDebug("marker creation count", { count: markers.current.length });
    instance.fitBounds(bounds, { padding: 72, maxZoom: 6, duration: 0 });
    mapDebug("fitBounds executed", { pointCount: points.length });
  }, [mapReady, event, route]);

  async function send() {
    setLoading(true);
    try {
      const response = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: "browser-demo", message }) });
      if (!response.ok) throw new Error("Agent API request failed");
      const body: ChatResponse = await response.json(); setReply(body.reply); setEvent(body.state.current_event ?? null); setRoute(body.state.historical_route ?? null);
    } catch { setReply("后端不可达。请先启动 FastAPI 服务。"); setEvent(null); setRoute(null); } finally { setLoading(false); }
  }

  return <main><header><h1>Historical Military GIS Agent</h1><p>Phase 4 · Evidence-driven Historical Route · MapLibre</p></header><section><label>历史问题<input value={message} onChange={(e) => setMessage(e.target.value)} /></label><button onClick={send} disabled={loading}>{loading ? "分析中…" : "发送"}</button></section><article><h2>Agent</h2><p>{reply}</p></article><article className="map-panel"><div className="map-heading"><div><h2>历史行动路线</h2><p>{route ? `${route.name} · ${route.period}` : event ? `${event.name} · ${event.period}` : "发送汉尼拔问题以加载地点"}</p></div><div className="map-actions">{route && <label><input type="checkbox" checked={routeVisible} onChange={(e) => setRouteVisible(e.target.checked)} /> 显示路线</label>}<span data-testid="marker-count">{route?.ordered_points.length ?? event?.places.length ?? 0} markers</span></div></div><div ref={mapContainer} className="map" aria-label="Historical places and route map" />{route && <div className="route-note"><strong>Historical reconstruction / schematic connection</strong><p>{route.assumptions.join(" ")}</p><p>{route.limitations.join(" ")}</p></div>}{event?.uncertainty_note && <p className="uncertainty">{event.uncertainty_note}</p>}</article></main>;
}

createRoot(document.getElementById("root")!).render(<App />);
