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
  console.debug("[GeoapifyRasterSmoke]", name, details);
}

if (!key) {
  root.textContent = "VITE_GEOAPIFY_API_KEY is unavailable";
  debug("key diagnostics", { geoapifyKeyPresent: false, geoapifyKeyLength: 0 });
} else {
  const style: maplibregl.StyleSpecification = {
    version: 8,
    sources: {
      geoapify_raster: {
        type: "raster",
        tiles: [`https://maps.geoapify.com/v1/tile/osm-bright/{z}/{x}/{y}.png?apiKey=${encodeURIComponent(key)}`],
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
  const map = new maplibregl.Map({ container: root, style, center: [5, 41], zoom: 3 });
  window.__geoapifyRasterMapSmoke = map;
  debug("map constructed", { geoapifyKeyPresent: true, geoapifyKeyLength: key.length, sourceType: "raster" });
  map.once("load", () => debug("load", { loaded: map.loaded(), styleLoaded: map.isStyleLoaded() }));
  map.on("idle", () => debug("idle", { loaded: map.loaded(), styleLoaded: map.isStyleLoaded() }));
  map.on("error", (event) => { const detail = event as unknown as { error?: { message?: string; url?: string }; sourceId?: string; tile?: unknown }; debug("error", { message: redact(detail.error?.message), resourceUrl: redact(detail.error?.url), sourceId: detail.sourceId, tile: detail.tile ? String(detail.tile) : undefined }); });
}

declare global { interface Window { __geoapifyRasterMapSmoke?: maplibregl.Map; } }
