import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./phase17.css";
import { DEMO_CAMPAIGNS, DEMO_EVALUATION, type DemoCampaign } from "./demo-fixtures";
import {
  type HistoricalKnowledgePanel,
  type HistoricalRoutePresentationPayload,
  fetchAgentHistoricalRoutePresentation,
  fetchHistoricalRoutePresentation,
  markerFeatures,
  uncertaintyCorridorFeatures,
  routeDirectionArrows,
  timelineWaypoint,
  visibleMapLayers,
  panelForFeature,
  routeFeature,
  toLeafletLineCoordinates,
  waypointPopupMetadata,
} from "./phase10-contract";

function sourceText(panel: HistoricalKnowledgePanel) {
  return panel.source_references.length ? panel.source_references.join(" · ") : "Reviewed evidence metadata";
}

function popupElement(payload: HistoricalRoutePresentationPayload, feature: ReturnType<typeof markerFeatures>[number], onDetails: () => void) {
  const metadata = waypointPopupMetadata(payload, feature);
  const panel = panelForFeature(payload, feature);
  const element = document.createElement("section");
  element.className = "route-popup";
  const rows: Array<[string, string | null]> = [
    ["时间", metadata.period],
    ["事件类型", metadata.eventType],
    ["事件描述", metadata.description],
    ["史料来源", panel ? sourceText(panel) : metadata.sourceBook ? `Book ${metadata.sourceBook}` : null],
    ["证据记录", `${metadata.evidenceCount}`],
    ["置信度", metadata.confidence],
  ];
  element.innerHTML = `<h3>${metadata.name}</h3>`;
  for (const [label, value] of rows) {
    if (!value) continue;
    const row = document.createElement("p");
    row.textContent = `${label}：${value}`;
    element.append(row);
  }
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "查看资料";
  button.addEventListener("click", onDetails);
  element.append(button);
  return element;
}

function HistoricalMap({ payload, showRoute, showEvidence, showCorridor, highlightedWaypointId, onSelectPanel, onSelectWaypoint }: { payload: HistoricalRoutePresentationPayload; showRoute: boolean; showEvidence: boolean; showCorridor: boolean; highlightedWaypointId: string | null; onSelectPanel: (panel: HistoricalKnowledgePanel | null) => void; onSelectWaypoint: (waypointId: string) => void }) {
  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<L.Map | null>(null);
  const layer = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!container.current || map.current) return;
    map.current = L.map(container.current, { zoomControl: true }).setView([42, 5], 4);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap contributors", maxZoom: 19 }).addTo(map.current);
    layer.current = L.layerGroup().addTo(map.current);
    return () => { map.current?.remove(); map.current = null; layer.current = null; };
  }, []);

  useEffect(() => {
    if (!map.current || !layer.current) return;
    layer.current.clearLayers();
    const bounds = L.latLngBounds([]);
    const visibleLayers = visibleMapLayers(payload, { evidence: showEvidence, route: showRoute, corridor: showCorridor });
    for (const feature of visibleLayers.corridor) {
      if (feature.geometry?.type !== "Polygon") continue;
      const coordinates = feature.geometry.coordinates.map((ring) => ring.map(([longitude, latitude]) => [latitude, longitude] as [number, number]));
      L.polygon(coordinates, { color: "#8f6b2e", weight: 1.5, dashArray: "5 6", fillColor: "#d4a24c", fillOpacity: 0.12 }).bindTooltip(String(feature.properties.label ?? "Uncertainty corridor")).addTo(layer.current);
    }
    const coordinates = toLeafletLineCoordinates(visibleLayers.route);
    if (coordinates.length >= 2) {
      const polyline = L.polyline(coordinates, { color: "#1e4f79", weight: 5, opacity: 0.9, dashArray: "11 8", lineCap: "round" });
      polyline.addTo(layer.current);
      for (const arrow of routeDirectionArrows(payload)) {
        L.marker(arrow.position, { interactive: false, icon: L.divIcon({ className: "route-direction-arrow", html: `<span style="transform:rotate(${arrow.rotationDeg}deg)">➜</span>`, iconSize: [22, 22], iconAnchor: [11, 11] }) }).addTo(layer.current);
      }
      bounds.extend(polyline.getBounds());
    }
    for (const [index, feature] of visibleLayers.evidence.entries()) {
      if (feature.geometry?.type !== "Point") continue;
      const [longitude, latitude] = feature.geometry.coordinates;
      const waypointId = String(feature.properties.waypoint_id ?? "");
      const className = `waypoint-number waypoint-tone-${index % 4}${highlightedWaypointId === waypointId ? " is-highlighted" : ""}`;
      const marker = L.marker([latitude, longitude], { icon: L.divIcon({ className, html: String(index + 1), iconSize: [26, 26], iconAnchor: [13, 13] }) });
      marker.bindPopup(popupElement(payload, feature, () => { onSelectWaypoint(waypointId); onSelectPanel(panelForFeature(payload, feature)); }));
      marker.addTo(layer.current);
      if (highlightedWaypointId === waypointId) map.current.panTo(marker.getLatLng());
      bounds.extend(marker.getLatLng());
    }
    if (bounds.isValid()) map.current.fitBounds(bounds, { padding: [42, 42], maxZoom: 7 });
  }, [payload, showRoute, showEvidence, showCorridor, highlightedWaypointId, onSelectPanel, onSelectWaypoint]);

  return <div ref={container} className="historical-map" aria-label="Historical route map" />;
}

function KnowledgePanel({ panel, payload }: { panel: HistoricalKnowledgePanel | null; payload: HistoricalRoutePresentationPayload | null }) {
  if (!panel) return <aside className="knowledge-panel"><h2>历史资料</h2><p>点击一个历史节点，然后选择“查看资料”。</p>{payload && <p>路线节点均保留已审核的史料来源；不会展示内部 evidence ID。</p>}</aside>;
  return <aside className="knowledge-panel"><h2>{panel.title}</h2><dl><dt>时间</dt><dd>{panel.period ?? "未提供"}</dd><dt>阶段</dt><dd>{panel.event_type ?? "未提供"}</dd><dt>史料来源</dt><dd>{sourceText(panel)}</dd><dt>证据记录</dt><dd>{panel.evidence_refs.length}</dd></dl><h3>资料摘要</h3><p>{panel.summary ?? "未提供独立摘要。"}</p><h3>相关资料</h3>{panel.external_references.length ? <ul>{panel.external_references.map((reference) => <li key={reference.id}><a href={reference.url} target="_blank" rel="noreferrer">{reference.title}</a> <small>({reference.reference_type})</small></li>)}</ul> : <p>未提供。</p>}</aside>;
}

function TimelinePanel({ summary, payload, highlightedWaypointId, onSelect }: { summary: HistoricalRoutePresentationPayload["presentation_summary"]; payload: HistoricalRoutePresentationPayload; highlightedWaypointId: string | null; onSelect: (waypointId: string) => void }) {
  if (!summary?.timeline.length) return null;
  return <section className="timeline-panel" aria-label="Historical route timeline"><h2>Timeline</h2><ol>{summary.timeline.map((step) => { const waypoint = timelineWaypoint(payload, step.order); const selected = waypoint?.id === highlightedWaypointId; return <li key={step.order} className={selected ? "is-selected" : ""}><button type="button" onClick={() => waypoint && onSelect(waypoint.id)}><strong>{step.order}. {step.title}</strong>{step.period && <span> · {step.period}</span>}</button>{step.description && <p>{step.description}</p>}<small>Evidence records: {step.evidence_count} · Confidence: {Math.round(step.confidence * 100)}%</small></li>; })}</ol></section>;
}

function DemoApp() {
  const [campaignId, setCampaignId] = useState<DemoCampaign["id"]>("hannibal");
  const [question, setQuestion] = useState(DEMO_CAMPAIGNS[0].prompt);
  const [payload, setPayload] = useState<HistoricalRoutePresentationPayload | null>(null);
  const [presentationSummary, setPresentationSummary] = useState<HistoricalRoutePresentationPayload["presentation_summary"]>(null);
  const [selectedPanel, setSelectedPanel] = useState<HistoricalKnowledgePanel | null>(null);
  const [showRoute, setShowRoute] = useState(true);
  const [showEvidence, setShowEvidence] = useState(true);
  const [showCorridor, setShowCorridor] = useState(true);
  const [highlightedWaypointId, setHighlightedWaypointId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const campaign = DEMO_CAMPAIGNS.find((item) => item.id === campaignId)!;
  const selectPanel = useCallback((panel: HistoricalKnowledgePanel | null) => setSelectedPanel(panel), []);

  function selectCampaign(nextId: DemoCampaign["id"]) {
    const next = DEMO_CAMPAIGNS.find((item) => item.id === nextId)!;
    setCampaignId(nextId); setQuestion(next.prompt); setPayload(null); setPresentationSummary(null); setSelectedPanel(null); setHighlightedWaypointId(null); setError(null);
  }

  async function loadPresentationFor(target: DemoCampaign) {
    setLoading(true); setError(null); setPresentationSummary(null); setSelectedPanel(null);
    try {
      if (target.presentationRouteId) {
        const next = await fetchHistoricalRoutePresentation(target.presentationRouteId);
        setPayload(next); setPresentationSummary(next.presentation_summary ?? null);
      } else {
        const result = await fetchAgentHistoricalRoutePresentation(question, `demo-${Date.now()}`);
        setPayload(result.payload); setPresentationSummary(result.payload.presentation_summary ?? null);
      }
    } catch (cause) {
      setPayload(null); setPresentationSummary(null); setError(cause instanceof Error ? cause.message : "无法加载历史路线。");
    } finally { setLoading(false); }
  }

  function loadPresentation() { return loadPresentationFor(campaign); }
  function activatePreset(next: DemoCampaign) { selectCampaign(next.id); void loadPresentationFor(next); }
  function selectTimelineWaypoint(waypointId: string) {
    setHighlightedWaypointId(waypointId);
    if (!payload) return;
    const pointFeature = markerFeatures(payload).find((feature) => feature.properties.waypoint_id === waypointId);
    if (pointFeature) setSelectedPanel(panelForFeature(payload, pointFeature));
  }

  return <main className="demo-shell"><header className="demo-header"><div><p className="eyebrow">Historical GIS Agent · Phase 20</p><h1>Historical Route Reconstruction Agent</h1><p>Evidence-grounded historical GIS reconstruction using RAG, MCP and terrain-aware path planning.</p></div><a href="/evaluation">查看评估链路</a></header><section className="demo-presets" aria-label="Demo presets">{DEMO_CAMPAIGNS.map((item) => <button key={item.id} type="button" className={item.id === campaignId ? "preset is-active" : "preset"} onClick={() => activatePreset(item)} disabled={loading}><strong>{item.id === "hannibal" ? "Hannibal" : "Caesar"}</strong><span>{item.id === "hannibal" ? "Second Punic War · 218 BCE" : "Gallic War · 58–51 BCE"}</span></button>)}</section><section className="controls"><label>Campaign<select value={campaignId} onChange={(event) => selectCampaign(event.target.value as DemoCampaign["id"])}>{DEMO_CAMPAIGNS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>{campaign.id === "hannibal" && <label>历史问题<input value={question} onChange={(event) => setQuestion(event.target.value)} /></label>}<button type="button" onClick={loadPresentation} disabled={loading}>{loading ? "加载中…" : campaign.id === "hannibal" ? "请求路线" : "加载路线"}</button></section>{error && <p className="error" role="alert">{error}</p>}{presentationSummary && <section className="presentation-summary" aria-label="Historical route summary"><h2>{presentationSummary.title}</h2>{presentationSummary.campaign && <p><strong>Campaign:</strong> {presentationSummary.campaign}</p>}{presentationSummary.operation && <p><strong>Operation:</strong> {presentationSummary.operation}</p>}{presentationSummary.date && <p><strong>Date:</strong> {presentationSummary.date}</p>}<p><strong>Sources:</strong> {presentationSummary.sources.join(" · ") || "Reviewed evidence metadata"}</p><section className="why-route"><h3>Why this route?</h3><p><strong>Historical evidence:</strong> {presentationSummary.evidence_basis.join(" · ") || "Reviewed evidence metadata"}</p><p><strong>Geographic constraints:</strong> {presentationSummary.geographic_constraints.join(" ")}</p><p><strong>Algorithm method:</strong> {presentationSummary.route_method}</p><p>{presentationSummary.route_interpretation}</p></section><p><strong>Uncertainty:</strong> {presentationSummary.uncertainty_notes.join(" ")}</p><p><strong>Limitations:</strong> {presentationSummary.limitations.join(" ")}</p></section>}{payload && <div className="route-layout"><section className="map-card"><div className="map-card-header"><div><h2>{payload.route.route_name ?? payload.route.route_id}</h2><p>{payload.route.period ?? "时期未提供"} · {payload.waypoints.length} 个历史节点</p></div><div className="layer-toggles"><label className="route-toggle"><input type="checkbox" checked={showEvidence} onChange={(event) => setShowEvidence(event.target.checked)} /> Evidence points</label><label className="route-toggle"><input type="checkbox" checked={showRoute} onChange={(event) => setShowRoute(event.target.checked)} /> Reconstructed route</label><label className="route-toggle"><input type="checkbox" checked={showCorridor} onChange={(event) => setShowCorridor(event.target.checked)} /> Uncertainty corridor</label></div></div><HistoricalMap payload={payload} showRoute={showRoute} showEvidence={showEvidence} showCorridor={showCorridor} highlightedWaypointId={highlightedWaypointId} onSelectPanel={selectPanel} onSelectWaypoint={setHighlightedWaypointId} /><div className="map-legend" aria-label="Map legend"><span><i className="legend-line" /> Reconstructed route</span><span><i className="legend-marker" /> Historical evidence points</span></div><p className="route-notice">该路线为基于史料节点与地理约束生成的示意路线，不代表真实逐日行军轨迹。</p></section><aside className="right-panels"><KnowledgePanel panel={selectedPanel} payload={payload} /><TimelinePanel summary={presentationSummary} payload={payload} highlightedWaypointId={highlightedWaypointId} onSelect={selectTimelineWaypoint} /><section className="architecture-panel"><h2>Architecture</h2><ol>{["Natural language", "Agent Intent", "RAG Evidence", "Historical Registry", "Geographic MCP", "Terrain constrained A*", "GeoJSON"].map((step) => <li key={step}>{step}</li>)}</ol></section><section className="uncertainty-panel"><h2>Uncertainty guide</h2><p><strong>Known:</strong> Evidence-supported location</p><p><strong>Estimated:</strong> Algorithm reconstruction</p><p><strong>Unknown:</strong> Historical uncertainty</p></section></aside></div>}</main>;
}

function EvaluationPage() {
  return <main className="evaluation"><a href="/">← 返回 Demo</a><p className="eyebrow">Phase 17</p><h1>Demo evaluation</h1><section><h2>当前支持的语料</h2><ul>{DEMO_EVALUATION.corpora.map((item) => <li key={item}>{item}</li>)}</ul></section><section><h2>路线展示链路</h2><ol>{DEMO_EVALUATION.pipeline.map((item) => <li key={item}>{item}</li>)}</ol></section><section><h2>最近离线验证</h2><ul>{DEMO_EVALUATION.verification.map((item) => <li key={item}>{item}</li>)}</ul><p>测试统计是发布时的 demo metadata；请以本地 CI/pytest 输出为最终依据。</p></section></main>;
}

createRoot(document.getElementById("root")!).render(window.location.pathname === "/evaluation" ? <EvaluationPage /> : <DemoApp />);
