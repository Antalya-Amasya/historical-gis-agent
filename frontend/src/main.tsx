import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";

const API = "http://127.0.0.1:8000/api/v1/agent/chat";
const BASEMAP_STYLE = "https://demotiles.maplibre.org/style.json";
const ROUTE_SOURCE = "historical-route";
const ROUTE_LAYER = "historical-route-line";

type HistoricalPlace = { id: string; canonical_name: string; modern_name?: string | null; latitude: number; longitude: number; period?: string | null; source: string; source_id?: string | null; source_url?: string | null; confidence: number; uncertain: boolean; };
type Evidence = { id: string; author: string; work: string; locator: string; excerpt: string; source_file?: string | null; };
type HistoricalEvent = { id: string; name: string; period: string; summary: string; places: HistoricalPlace[]; evidence: Evidence[]; uncertainty_note?: string | null; };
type HistoricalRoutePoint = { sequence: number; historical_place: HistoricalPlace; event_summary: string; date_or_period?: string | null; evidence_refs: string[]; confidence: number; };
type HistoricalRoute = { id: string; name: string; period: string; ordered_points: HistoricalRoutePoint[]; geometry: { type: "LineString"; coordinates: [number, number][] }; evidence_refs: string[]; assumptions: string[]; limitations: string[]; historical_confidence: number; };
type ChatResponse = { reply: string; state: { current_event?: HistoricalEvent | null; historical_route?: HistoricalRoute | null }; };

function placePopup(place: HistoricalPlace, event: HistoricalEvent, routePoint?: HistoricalRoutePoint) {
  const element = document.createElement("section");
  element.className = "place-popup";
  const source = place.source_url ? `<a href="${place.source_url}" target="_blank" rel="noreferrer">${place.source}</a>` : place.source;
  const evidence = routePoint?.evidence_refs.length ? `<p><strong>史料节点：</strong>${routePoint.evidence_refs.join(", ")}</p>` : "";
  element.innerHTML = `<h3>${place.canonical_name}</h3><p><strong>现代名称：</strong>${place.modern_name ?? "未提供"}</p><p><strong>时间/时期：</strong>${routePoint?.date_or_period ?? place.period ?? event.period}</p><p><strong>事件：</strong>${routePoint?.event_summary ?? event.summary}</p>${evidence}<p><strong>坐标来源：</strong>${source}${place.source_id ? `（${place.source_id}）` : ""}</p><p><strong>置信度：</strong>${Math.round((routePoint?.confidence ?? place.confidence) * 100)}%${place.uncertain ? "；代表点存在不确定性" : ""}</p>`;
  return element;
}

function App() {
  const [message, setMessage] = useState("根据史料展示汉尼拔公元前218年进入意大利前的行动路线。");
  const [reply, setReply] = useState("等待提问。Mock Agent 无需 API Key。");
  const [event, setEvent] = useState<HistoricalEvent | null>(null);
  const [route, setRoute] = useState<HistoricalRoute | null>(null);
  const [routeVisible, setRouteVisible] = useState(true);
  const [loading, setLoading] = useState(false);
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const markers = useRef<maplibregl.Marker[]>([]);

  useEffect(() => {
    if (!mapContainer.current || map.current) return;
    const instance = new maplibregl.Map({ container: mapContainer.current, style: BASEMAP_STYLE, center: [5, 41], zoom: 3 });
    instance.addControl(new maplibregl.NavigationControl(), "top-right");
    map.current = instance;
    return () => { markers.current.forEach((marker) => marker.remove()); instance.remove(); map.current = null; };
  }, []);

  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    const render = () => {
      markers.current.forEach((marker) => marker.remove()); markers.current = [];
      const points = route?.ordered_points ?? event?.places.map((historical_place, index) => ({ sequence: index + 1, historical_place, event_summary: event.summary, evidence_refs: [], confidence: historical_place.confidence })) ?? [];
      if (!event || !points.length) return;
      const bounds = new maplibregl.LngLatBounds();
      points.forEach((point) => {
        const place = point.historical_place;
        const marker = new maplibregl.Marker({ color: route ? "#2f6f9f" : "#9c3f22" }).setLngLat([place.longitude, place.latitude]).setPopup(new maplibregl.Popup({ offset: 24 }).setDOMContent(placePopup(place, event, point))).addTo(instance);
        marker.getElement().setAttribute("aria-label", `${place.canonical_name} marker`); markers.current.push(marker); bounds.extend([place.longitude, place.latitude]);
      });
      instance.fitBounds(bounds, { padding: 72, maxZoom: 6, duration: 0 });
    };
    if (instance.loaded()) render(); else instance.once("load", render);
  }, [event, route]);

  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    const renderRoute = () => {
      if (instance.getLayer(ROUTE_LAYER)) instance.removeLayer(ROUTE_LAYER);
      if (instance.getSource(ROUTE_SOURCE)) instance.removeSource(ROUTE_SOURCE);
      if (!route) return;
      instance.addSource(ROUTE_SOURCE, { type: "geojson", data: { type: "Feature", properties: { label: "Historical reconstruction / schematic connection" }, geometry: route.geometry } });
      instance.addLayer({ id: ROUTE_LAYER, type: "line", source: ROUTE_SOURCE, layout: { visibility: routeVisible ? "visible" : "none", "line-join": "round", "line-cap": "round" }, paint: { "line-color": "#2f6f9f", "line-width": 4, "line-opacity": 0.78, "line-dasharray": [2, 1] } });
    };
    if (instance.loaded()) renderRoute(); else instance.once("load", renderRoute);
  }, [route, routeVisible]);

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
