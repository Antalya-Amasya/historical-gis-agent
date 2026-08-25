import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./phase10.css";
import {
  type HistoricalKnowledgePanel,
  type HistoricalRoutePresentationPayload,
  fetchHistoricalRoutePresentation,
  markerFeatures,
  panelForFeature,
  routeFeature,
  toLeafletLineCoordinates,
  waypointPopupMetadata,
} from "./phase10-contract";

function popupElement(metadata: ReturnType<typeof waypointPopupMetadata>, onDetails: () => void) {
  const element = document.createElement("section");
  element.className = "phase10-popup";
  const title = document.createElement("h3");
  title.textContent = metadata.name;
  element.append(title);
  for (const [label, value] of [["Event type", metadata.eventType], ["Period", metadata.period], ["Description", metadata.description], ["Source book", metadata.sourceBook], ["Source chapter", metadata.sourceChapter], ["Confidence", metadata.confidence], ["Reviewed evidence", `${metadata.evidenceCount} record(s)`], ["External references", String(metadata.externalReferenceCount)]] as Array<[string, string | null]>) {
    const row = document.createElement("p");
    row.textContent = `${label}: ${value ?? "Not supplied"}`;
    element.append(row);
  }
  if (metadata.knowledgePanelId) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "View details";
    button.addEventListener("click", onDetails);
    element.append(button);
  }
  return element;
}

function KnowledgePanel({ panel }: { panel: HistoricalKnowledgePanel | null }) {
  if (!panel) return <aside className="knowledge-panel"><h2>Knowledge panel</h2><p>Select a mapped waypoint and choose “View details”.</p></aside>;
  return <aside className="knowledge-panel"><h2>{panel.title}</h2><p><strong>Summary:</strong> {panel.summary ?? "No supplied summary."}</p><p><strong>Reviewed evidence:</strong> {panel.evidence_refs.length} record(s)</p><p><strong>Source references:</strong> {panel.source_references.join(", ") || "Not supplied"}</p><h3>External references</h3>{panel.external_references.length ? <ul>{panel.external_references.map((reference) => <li key={reference.id}><a href={reference.url} target="_blank" rel="noreferrer">{reference.title}</a> <small>({reference.reference_type})</small></li>)}</ul> : <p>None supplied.</p>}</aside>;
}

function HistoricalRouteMap({ payload, onSelectPanel }: { payload: HistoricalRoutePresentationPayload; onSelectPanel: (panel: HistoricalKnowledgePanel | null) => void }) {
  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<L.Map | null>(null);
  const layer = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!container.current || map.current) return;
    map.current = L.map(container.current, { zoomControl: true }).setView([42, 5], 4);
    layer.current = L.layerGroup().addTo(map.current);
    return () => { map.current?.remove(); map.current = null; layer.current = null; };
  }, []);

  useEffect(() => {
    if (!map.current || !layer.current) return;
    layer.current.clearLayers();
    const bounds = L.latLngBounds([]);
    const line = routeFeature(payload);
    const lineCoordinates = toLeafletLineCoordinates(line);
    if (lineCoordinates.length >= 2) {
      const lineLayer = L.polyline(lineCoordinates, { color: "#163d63", weight: 5, opacity: 0.9, dashArray: "10 7" });
      lineLayer.addTo(layer.current);
      bounds.extend(lineLayer.getBounds());
    }
    for (const feature of markerFeatures(payload)) {
      if (feature.geometry?.type !== "Point") continue;
      const metadata = waypointPopupMetadata(payload, feature);
      const [longitude, latitude] = feature.geometry.coordinates;
      const marker = L.circleMarker([latitude, longitude], { radius: 8, color: "#8a3c24", fillColor: "#c76a45", fillOpacity: 0.95, weight: 2 });
      marker.bindPopup(popupElement(metadata, () => onSelectPanel(panelForFeature(payload, feature))));
      marker.addTo(layer.current);
      bounds.extend(marker.getLatLng());
    }
    if (bounds.isValid()) map.current.fitBounds(bounds, { padding: [36, 36], maxZoom: 7 });
  }, [payload, onSelectPanel]);

  return <div className="phase10-map" ref={container} aria-label="Historical route presentation map" />;
}

function App() {
  const [payload, setPayload] = useState<HistoricalRoutePresentationPayload | null>(null);
  const [selectedPanel, setSelectedPanel] = useState<HistoricalKnowledgePanel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selectPanel = useCallback((panel: HistoricalKnowledgePanel | null) => setSelectedPanel(panel), []);
  useEffect(() => { fetchHistoricalRoutePresentation("caesar-gallic-campaign").then(setPayload).catch((cause: Error) => setError(cause.message)); }, []);
  return <main className="phase10"><header><h1>Historical Route Presentation</h1><p>Leaflet consumes the read-only backend presentation API; no coordinates are inferred.</p></header>{error ? <p role="alert">{error}</p> : !payload ? <p>Loading historical route presentation…</p> : <div className="phase10-layout"><section><h2>{payload.route.route_name ?? payload.route.route_id}</h2><HistoricalRouteMap payload={payload} onSelectPanel={selectPanel} /><p className="phase10-note">Historical reconstruction / schematic connection. Waypoints without supplied Point geometry are deliberately not drawn.</p></section><KnowledgePanel panel={selectedPanel} /></div>}</main>;
}

createRoot(document.getElementById("root")!).render(<App />);
