import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./phase17.css";
import { type HistoricalRoutePresentationPayload, fetchAgentHistoricalRoutePresentation, markerFeatures, romanRoadSegmentFeatures, routeFeature, toLeafletLineCoordinates } from "./phase10-contract";
import { AnswerResult } from "./answer-result";

function HistoricalMap({ payload }: { payload: HistoricalRoutePresentationPayload }) {
  const container = useRef<HTMLDivElement | null>(null); const map = useRef<L.Map | null>(null); const layer = useRef<L.LayerGroup | null>(null);
  useEffect(() => { if (!container.current || map.current) return; map.current = L.map(container.current).setView([42, 5], 4); L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap contributors" }).addTo(map.current); layer.current = L.layerGroup().addTo(map.current); return () => { map.current?.remove(); map.current = null; layer.current = null; }; }, []);
  useEffect(() => { if (!map.current || !layer.current) return; layer.current.clearLayers(); const bounds = L.latLngBounds([]); const draw = (coordinates: [number, number][], style: L.PolylineOptions) => { if (coordinates.length < 2) return; const line = L.polyline(coordinates, style).addTo(layer.current!); bounds.extend(line.getBounds()); };
    if (payload.road_network) for (const feature of romanRoadSegmentFeatures(payload)) { if (feature.geometry?.type !== "LineString") continue; const role = String(feature.properties.segment_role); draw(toLeafletLineCoordinates(feature), role === "roman_road" ? { color: "#a34422", weight: 4 } : { color: "#786f65", weight: 2, dashArray: "5 7" }); }
    else draw(toLeafletLineCoordinates(routeFeature(payload)), { color: "#1e4f79", weight: 5, dashArray: "11 8" });
    for (const [index, feature] of markerFeatures(payload).entries()) { if (feature.geometry?.type !== "Point") continue; const [longitude, latitude] = feature.geometry.coordinates; const marker = L.marker([latitude, longitude], { icon: L.divIcon({ className: `waypoint-number waypoint-tone-${index % 4}`, html: String(index + 1), iconSize: [26, 26], iconAnchor: [13, 13] }) }).bindPopup(String(feature.properties.name ?? "Historical anchor")); marker.addTo(layer.current); bounds.extend(marker.getLatLng()); }
    if (bounds.isValid()) map.current.fitBounds(bounds, { padding: [42, 42], maxZoom: 7 });
  }, [payload]);
  return <div ref={container} className="historical-map" aria-label="Historical GIS map" />;
}

function KnowledgePanel({ payload }: { payload: HistoricalRoutePresentationPayload }) { const road = payload.road_network; if (!road) return <aside className="knowledge-panel"><h2>Historical evidence</h2><p>Historical anchors and their evidence are shown on the map when available.</p></aside>; const stats = road.aggregate; return <aside className="knowledge-panel"><h2>Roman-road candidate</h2><p><strong>Status:</strong> {road.route_status}</p><p><strong>Network distance:</strong> {(stats.total_network_distance_m / 1000).toFixed(3)} km</p><p><strong>Access connectors:</strong> {(stats.total_access_connector_distance_m / 1000).toFixed(3)} km</p><p><strong>Legs:</strong> {stats.successful_leg_count} successful / {stats.failed_leg_count} unresolved</p><p><strong>Road source:</strong> {road.source}</p><p>Roman-road infrastructure candidate. Not proof of the exact historical track.</p></aside>; }

function QueryApp() {
  const [query, setQuery] = useState("");
  const [reply, setReply] = useState("");
  const [payload, setPayload] = useState<HistoricalRoutePresentationPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useRef(`historical-gis-${crypto.randomUUID()}`);

  const submit = async () => {
    setLoading(true);
    setError(null);
    setReply("");
    setPayload(null);
    try {
      const result = await fetchAgentHistoricalRoutePresentation(query, sessionId.current);
      setReply(result.reply);
      setPayload(result.payload);
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
    {payload && <div className="route-layout"><section className="map-card"><HistoricalMap payload={payload} /><div className="map-legend"><span><i className="legend-marker" /> Historical anchor</span>{payload.road_network && <><span><i className="legend-line" /> Roman-road candidate</span><span>Dashed line: access connector</span><span>Unresolved leg: no line drawn</span></>}</div></section><aside className="right-panels"><KnowledgePanel payload={payload} /></aside></div>}
  </main>;
}

createRoot(document.getElementById("root")!).render(<QueryApp />);
