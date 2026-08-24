import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

const key = import.meta.env.VITE_GEOAPIFY_API_KEY;
const root = document.querySelector<HTMLDivElement>("#map");
if (!root) throw new Error("Missing map container");
root.style.cssText = "position:fixed;inset:0";

function redact(value: string | undefined) {
  return value?.replace(/apiKey=[^&\s]+/gi, "apiKey=[redacted]");
}
function debug(name: string, details: Record<string, unknown> = {}) {
  console.debug("[GeoapifyMapSmoke]", name, details);
}
function styleSummary(map: maplibregl.Map) {
  const style = map.getStyle();
  return { sourceIds: Object.keys(style.sources), layerCount: style.layers?.length ?? 0, sourceTypes: Object.fromEntries(Object.entries(style.sources).map(([id, source]) => [id, source.type])) };
}

if (!key) {
  root.textContent = "VITE_GEOAPIFY_API_KEY is unavailable";
  debug("key diagnostics", { geoapifyKeyPresent: false, geoapifyKeyLength: 0 });
} else {
  const map = new maplibregl.Map({ container: root, style: `https://maps.geoapify.com/v1/styles/osm-bright/style.json?apiKey=${encodeURIComponent(key)}`, center: [5, 41], zoom: 3 });
  window.__geoapifyMapSmoke = map;
  debug("map constructed", { geoapifyKeyPresent: true, geoapifyKeyLength: key.length });
  map.on("styledata", () => debug("styledata", styleSummary(map)));
  map.on("sourcedata", (event) => debug("sourcedata", { sourceId: event.sourceId, sourceDataType: event.sourceDataType, isSourceLoaded: event.isSourceLoaded }));
  map.on("data", (event) => { const detail = event as unknown as { dataType?: string; sourceId?: string }; debug("data", detail); });
  map.on("idle", () => debug("idle", { loaded: map.loaded(), styleLoaded: map.isStyleLoaded() }));
  map.once("load", () => debug("load", { loaded: map.loaded(), styleLoaded: map.isStyleLoaded(), ...styleSummary(map) }));
  map.on("error", (event) => { const detail = event as unknown as { error?: { message?: string; url?: string }; sourceId?: string; tile?: unknown }; debug("error", { message: redact(detail.error?.message), resourceUrl: redact(detail.error?.url), sourceId: detail.sourceId, tile: detail.tile ? String(detail.tile) : undefined }); });
}

declare global { interface Window { __geoapifyMapSmoke?: maplibregl.Map; } }
