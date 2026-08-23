import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = "http://127.0.0.1:8000/api/v1/agent/chat";

function App() {
  const [message, setMessage] = useState("分析公元前218年汉尼拔翻越阿尔卑斯。");
  const [reply, setReply] = useState("等待提问。Mock Agent 无需 API Key。");
  const [place, setPlace] = useState("尚未识别地点");
  const [loading, setLoading] = useState(false);
  async function send() {
    setLoading(true);
    try {
      const response = await fetch(API, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: "browser-demo", message }) });
      const body = await response.json();
      setReply(body.reply);
      const event = body.state.current_event;
      setPlace(event?.places?.[0] ? `${event.places[0].canonical_name} (${event.places[0].latitude}, ${event.places[0].longitude})` : "未识别地点");
    } catch { setReply("后端不可达。请先启动 FastAPI 服务。"); }
    finally { setLoading(false); }
  }
  return <main><h1>Historical Military GIS Agent</h1><p>Phase 0 · Mock Agent · HTTP contract validation</p><section><label>历史问题<input value={message} onChange={(e) => setMessage(e.target.value)} /></label><button onClick={send} disabled={loading}>{loading ? "分析中…" : "发送"}</button></section><article><h2>Agent</h2><p>{reply}</p></article><article><h2>地图占位</h2><p>{place}</p><small>MapLibre Marker 与地图渲染将在 Phase 1 实现。</small></article></main>;
}

createRoot(document.getElementById("root")!).render(<App />);

