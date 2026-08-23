import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";

const API = "http://127.0.0.1:8000/api/v1/agent/chat";
const BASEMAP_STYLE = "https://demotiles.maplibre.org/style.json";

type HistoricalPlace = {
  id: string;
  canonical_name: string;
  modern_name?: string | null;
  latitude: number;
  longitude: number;
  period?: string | null;
  source: string;
  source_id?: string | null;
  source_url?: string | null;
  confidence: number;
  uncertain: boolean;
};

type HistoricalEvent = {
  id: string;
  name: string;
  period: string;
  summary: string;
  places: HistoricalPlace[];
  uncertainty_note?: string | null;
};

type ChatResponse = {
  reply: string;
  state: { current_event?: HistoricalEvent | null };
};

function popupContent(place: HistoricalPlace, event: HistoricalEvent) {
  const element = document.createElement("section");
  element.className = "place-popup";
  const source = place.source_url
    ? `<a href="${place.source_url}" target="_blank" rel="noreferrer">${place.source}</a>`
    : place.source;
  element.innerHTML = `
    <h3>${place.canonical_name}</h3>
    <p><strong>现代名称：</strong>${place.modern_name ?? "未提供"}</p>
    <p><strong>时间/时期：</strong>${place.period ?? event.period}</p>
    <p><strong>事件：</strong>${event.summary}</p>
    <p><strong>坐标来源：</strong>${source}${place.source_id ? `（${place.source_id}）` : ""}</p>
    <p><strong>置信度：</strong>${Math.round(place.confidence * 100)}%${place.uncertain ? "；存在不确定性" : ""}</p>
  `;
  return element;
}

function App() {
  const [message, setMessage] = useState("分析公元前218年汉尼拔翻越阿尔卑斯。");
  const [reply, setReply] = useState("等待提问。Mock Agent 无需 API Key。");
  const [event, setEvent] = useState<HistoricalEvent | null>(null);
  const [loading, setLoading] = useState(false);
  const mapContainer = useRef<HTMLDivElement | null>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const markers = useRef<maplibregl.Marker[]>([]);

  useEffect(() => {
    if (!mapContainer.current || map.current) return;
    const instance = new maplibregl.Map({
      container: mapContainer.current,
      style: BASEMAP_STYLE,
      center: [5, 41],
      zoom: 3,
    });
    instance.addControl(new maplibregl.NavigationControl(), "top-right");
    map.current = instance;
    return () => {
      markers.current.forEach((marker) => marker.remove());
      instance.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    const instance = map.current;
    if (!instance) return;
    markers.current.forEach((marker) => marker.remove());
    markers.current = [];
    if (!event?.places.length) return;

    const addMarkersAndFit = () => {
      const bounds = new maplibregl.LngLatBounds();
      event.places.forEach((place) => {
        const popup = new maplibregl.Popup({ offset: 24 }).setDOMContent(popupContent(place, event));
        const marker = new maplibregl.Marker({ color: "#9c3f22" })
          .setLngLat([place.longitude, place.latitude])
          .setPopup(popup)
          .addTo(instance);
        marker.getElement().setAttribute("aria-label", `${place.canonical_name} marker`);
        markers.current.push(marker);
        bounds.extend([place.longitude, place.latitude]);
      });
      instance.fitBounds(bounds, { padding: 72, maxZoom: 6, duration: 0 });
    };

    if (instance.loaded()) addMarkersAndFit();
    else instance.once("load", addMarkersAndFit);
  }, [event]);

  async function send() {
    setLoading(true);
    try {
      const response = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: "browser-demo", message }),
      });
      if (!response.ok) throw new Error("Agent API request failed");
      const body: ChatResponse = await response.json();
      setReply(body.reply);
      setEvent(body.state.current_event ?? null);
    } catch {
      setReply("后端不可达。请先启动 FastAPI 服务。");
      setEvent(null);
    } finally {
      setLoading(false);
    }
  }

  return <main>
    <header><h1>Historical Military GIS Agent</h1><p>Phase 1 · Mock Agent · HistoricalPlace → MapLibre markers</p></header>
    <section><label>历史问题<input value={message} onChange={(e) => setMessage(e.target.value)} /></label><button onClick={send} disabled={loading}>{loading ? "分析中…" : "发送"}</button></section>
    <article><h2>Agent</h2><p>{reply}</p></article>
    <article className="map-panel">
      <div className="map-heading"><div><h2>事件地点</h2><p>{event ? `${event.name} · ${event.period}` : "发送汉尼拔演示问题以加载地点"}</p></div><span data-testid="marker-count">{event?.places.length ?? 0} markers</span></div>
      <div ref={mapContainer} className="map" aria-label="Historical places map" />
      {event?.uncertainty_note && <p className="uncertainty">{event.uncertainty_note}</p>}
    </article>
  </main>;
}

createRoot(document.getElementById("root")!).render(<App />);
