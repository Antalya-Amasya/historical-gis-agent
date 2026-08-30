import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./phase17.css";
import { type GeoJsonFeature, type HistoricalRoutePresentationPayload, drawableRouteSegments, fetchAgentHistoricalRoutePresentation, markerFeatures, routeFeature, toLeafletLineCoordinates, waypointPopupMetadata } from "./phase10-contract";
import { AnswerResult } from "./answer-result";
import { RouteDetails } from "./historical-route-details";

function HistoricalMap({ payload }: { payload: HistoricalRoutePresentationPayload }) {
  const container = useRef<HTMLDivElement | null>(null); const map = useRef<L.Map | null>(null); const layer = useRef<L.LayerGroup | null>(null);
  useEffect(() => { if (!container.current || map.current) return; map.current = L.map(container.current).setView([42, 5], 4); L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap contributors" }).addTo(map.current); layer.current = L.layerGroup().addTo(map.current); return () => { map.current?.remove(); map.current = null; layer.current = null; }; }, []);
  useEffect(() => { if (!map.current || !layer.current) return; layer.current.clearLayers(); const bounds = L.latLngBounds([]); const draw = (feature: GeoJsonFeature | undefined, style: L.PolylineOptions, label: string) => { const coordinates = toLeafletLineCoordinates(feature); if (coordinates.length < 2) return; const line = L.polyline(coordinates, style).bindTooltip(label).addTo(layer.current!); bounds.extend(line.getBounds()); };
    if (payload.road_network) for (const segment of drawableRouteSegments(payload)) { draw(segment.feature, segment.kind === "roman_road" ? { color: "#9a4f2d", weight: 5 } : segment.kind === "terrain" ? { color: "#1e5f78", weight: 5, dashArray: "12 8" } : { color: "#786f65", weight: 2, dashArray: "3 7" }, segment.kind === "roman_road" ? "古罗马道路优先重建" : segment.kind === "terrain" ? "地形算法重建" : "道路接入连接"); }
    else draw(routeFeature(payload), { color: "#1e5f78", weight: 5, dashArray: "12 8" }, "地形算法重建候选路线");
    for (const [index, feature] of markerFeatures(payload).entries()) { if (feature.geometry?.type !== "Point") continue; const [longitude, latitude] = feature.geometry.coordinates; const metadata = waypointPopupMetadata(payload, feature); const reconstructedCrossing = feature.properties.layer_type === "reconstructed_crossing"; const popup = document.createElement("section"); popup.className = "route-popup"; const title = document.createElement("h3"); title.textContent = reconstructedCrossing ? metadata.name : `${index + 1}. ${metadata.name}`; const evidence = document.createElement("p"); evidence.textContent = reconstructedCrossing ? "算法推定穿越点 · 非历史路点" : `历史依据：${metadata.evidenceCount} 条证据引用`; const note = document.createElement("p"); note.textContent = reconstructedCrossing ? "由古代道路/地形约束生成；不代表史料直接证明。" : metadata.description ?? "史料约束的历史路点"; popup.append(title, evidence, note); const marker = L.marker([latitude, longitude], { icon: L.divIcon({ className: reconstructedCrossing ? "reconstructed-crossing" : `waypoint-number waypoint-tone-${index % 4}`, html: reconstructedCrossing ? "△" : String(index + 1), iconSize: [28, 28], iconAnchor: [14, 14] }) }).bindPopup(popup); marker.addTo(layer.current); bounds.extend(marker.getLatLng()); }
    if (bounds.isValid()) map.current.fitBounds(bounds, { padding: [42, 42], maxZoom: 7 });
  }, [payload]);
  return <div ref={container} className="historical-map" aria-label="Historical GIS map" />;
}

function QueryApp() {
  const [query, setQuery] = useState("");
  const [reply, setReply] = useState("");
  const [payload, setPayload] = useState<HistoricalRoutePresentationPayload | null>(null);
  const [routeSource, setRouteSource] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef(`historical-gis-${crypto.randomUUID()}`);

  const submit = async () => {
    setLoading(true);
    setError(null);
    setReply("");
    setPayload(null);
    setRouteSource(null);
    try {
      const result = await fetchAgentHistoricalRoutePresentation(query, sessionId.current);
      setReply(result.reply);
      setPayload(result.payload);
      setRouteSource(result.routeSource);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return <main className="demo-shell">
    <header className="demo-header"><p className="eyebrow">Historical GIS Agent</p><h1>Historical GIS Agent</h1><p>Ask about a historical campaign, movement, or place.</p></header>
    <section className="controls"><label>Historical question<textarea value={query} onChange={(event) => setQuery(event.target.value)} /></label><button type="button" disabled={loading || !query.trim()} onClick={() => void submit()}>{loading ? "Loading…" : "Ask"}</button></section>
    {error && <p className="error" role="alert">{error}</p>}
    {reply && <AnswerResult reply={reply} routeStatus={payload?.road_network?.route_status} />}
    {payload && <div className="route-layout"><section className="map-card"><div className="map-card-header"><div><p className="panel-kicker">Historical Route / 历史路线</p><h2>{payload.presentation_summary?.title ?? payload.route.route_name ?? "历史路线重建"}</h2></div><span className="candidate-badge">候选路线 · 非精确史实轨迹</span></div><HistoricalMap payload={payload} /><div className="map-legend" aria-label="Route map legend"><span><i className="legend-marker" /> 史料约束路点</span><span><i className="legend-crossing">△</i> 算法推定穿越点</span><span><i className="legend-line legend-road" /> 古罗马道路重建</span><span><i className="legend-line legend-terrain" /> 地形算法重建</span><span><i className="legend-gap" /> 未重建区段（不连线）</span></div></section><RouteDetails payload={payload} routeSource={routeSource} /></div>}
  </main>;
}

createRoot(document.getElementById("root")!).render(<QueryApp />);
