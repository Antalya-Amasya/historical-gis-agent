import { failedRouteFragments, failedRouteSegments, markerFeatures, routeSegments, type HistoricalRoutePresentationPayload, waypointPopupMetadata } from "./phase10-contract";

function formatMetric(value: number | undefined, suffix = "") {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}`
    : "未提供";
}

export function segmentLabel(kind: ReturnType<typeof routeSegments>[number]["kind"]) {
  return kind === "roman_road" ? "古罗马道路优先重建"
    : kind === "terrain" ? "地形 A* 算法重建"
      : kind === "direct_water_edge" ? "海上重建路线"
        : kind === "simulated_coastal_access" ? "模拟海岸接入"
          : kind === "connector" ? "道路接入连接"
            : "未能重建的区段";
}

function visibleEndpoint(payload: HistoricalRoutePresentationPayload, identifier: string | undefined) {
  if (!identifier) return null;
  const named = markerFeatures(payload).find((feature) => String(feature.properties.waypoint_id ?? "") === identifier);
  const name = named?.properties.name;
  if (typeof name === "string" && name.trim() && !/^(pleiades-|sim-coast-|itiner-e:)/.test(name)) return name;
  if (/^(pleiades-|sim-coast-|itiner-e:)/.test(identifier)) return null;
  return identifier;
}

function friendlyRouteSource(source: string | null | undefined) {
  return source === "event_anchor" ? "事件锚点与史料顺序"
    : source === "legacy_movement_claims" ? "史料中的移动关系"
      : null;
}

export function RouteDetails({ payload, routeSource }: { payload: HistoricalRoutePresentationPayload; routeSource?: string | null }) {
  const segments = routeSegments(payload);
  const failed = failedRouteSegments(payload);
  const failedFragments = failedRouteFragments(payload);
  const summary = payload.presentation_summary;
  const limitations = [
    ...(summary?.limitations ?? []),
    ...(payload.road_network?.limitations ?? []),
    ...(payload.location_warnings ?? []),
  ].filter((item, index, all) => item && all.indexOf(item) === index);
  const panels = payload.knowledge_panels ?? [];
  const markers = markerFeatures(payload);
  const historicalMarkers = markers.filter(feature => feature.properties.layer_type !== "reconstructed_crossing");
  const crossings = markers.filter(feature => feature.properties.layer_type === "reconstructed_crossing");

  return <aside className="right-panels" aria-label="Historical route details">
    <section className="knowledge-panel authority-card">
      <p className="panel-kicker">历史依据</p><h2>历史路点</h2>
      <p>这些点及其大致顺序来自后端接受的史料约束；地图线路不等于史料直接记载的精确行军轨迹。</p>
      {friendlyRouteSource(routeSource) && <p className="route-source"><strong>路点来源：</strong>{friendlyRouteSource(routeSource)}</p>}
      <ol className="waypoint-list">{historicalMarkers.map((feature, index) => {
        const metadata = waypointPopupMetadata(payload, feature);
        const panel = panels.find((item) => item.waypoint_id === metadata.waypointId);
        return <li key={`${metadata.waypointId}-${index}`}><strong>{metadata.name}</strong><span>第 {index + 1} 站 · {metadata.evidenceCount} 条证据引用</span>{panel?.source_references?.length ? <small>{panel.source_references.join("；")}</small> : null}</li>;
      })}</ol>
    </section>
    <section className="knowledge-panel reconstruction-card">
      <p className="panel-kicker">算法重建</p><h2>候选路线</h2>
      <p>{summary?.route_interpretation ?? "算法仅在历史路点之间重建地理候选路径，不创造历史事实。"}</p>
      <ul className="segment-list">{segments.map((segment, index) => {
        const from = visibleEndpoint(payload, segment.from);
        const to = visibleEndpoint(payload, segment.to);
        return <li key={`${segment.id}-${index}`} className={`segment-${segment.kind}`}><strong>{segmentLabel(segment.kind)}</strong><span>{from && to ? `${from} → ${to}` : "历史路点之间"}</span>{segment.distanceKm !== undefined ? <span>距离 {formatMetric(segment.distanceKm, " km")}</span> : null}{segment.cost !== undefined ? <span>重建成本 {formatMetric(segment.cost)}</span> : null}{segment.terrainSource ? <small>地形数据：{segment.terrainSource}</small> : null}{segment.failureStatus ? <small className="gap-warning">原因：{segment.failureStatus}</small> : null}</li>;
      })}</ul>
      {crossings.length > 0 && <ul className="segment-list">{crossings.map((feature, index) => <li key={`crossing-${index}`}><strong>{waypointPopupMetadata(payload, feature).name}</strong><span>算法推定穿越点 · 非历史路点</span></li>)}</ul>}
      {!segments.length && <p className="route-empty">历史路点已确定，但当前没有可显示的候选几何。</p>}
      {failed.length > 0 && <p className="gap-warning" role="status">{failed.length} 个区段未能可靠重建；地图不会用直线补齐。</p>}
      {failedFragments.length > 0 && <p className="gap-warning" role="status">{failedFragments.length} 个证据片段未能安全重建；已成功片段仍单独显示，片段之间不连线。</p>}
    </section>
    {limitations.length > 0 && <section className="knowledge-panel limitations-card"><p className="panel-kicker">限制</p><h2>如何理解这条路线</h2><ul>{limitations.map((item) => <li key={item}>{item}</li>)}</ul></section>}
  </aside>;
}
