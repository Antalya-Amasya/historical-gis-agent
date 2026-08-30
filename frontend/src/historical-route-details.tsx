import { failedRouteSegments, markerFeatures, routeSegments, type HistoricalRoutePresentationPayload, waypointPopupMetadata } from "./phase10-contract";

function formatMetric(value: number | undefined, suffix = "") {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}`
    : "未提供";
}

export function segmentLabel(kind: ReturnType<typeof routeSegments>[number]["kind"]) {
  return kind === "roman_road" ? "古罗马道路优先重建"
    : kind === "terrain" ? "地形 A* 算法重建"
      : kind === "connector" ? "道路接入连接"
        : "未能重建的区段";
}

function friendlyRouteSource(source: string | null | undefined) {
  return source === "event_anchor" ? "事件锚点与史料顺序"
    : source === "legacy_movement_claims" ? "史料中的移动关系"
      : null;
}

export function RouteDetails({ payload, routeSource }: { payload: HistoricalRoutePresentationPayload; routeSource?: string | null }) {
  const segments = routeSegments(payload);
  const failed = failedRouteSegments(payload);
  const summary = payload.presentation_summary;
  const limitations = [
    ...(summary?.limitations ?? []),
    ...(payload.road_network?.limitations ?? []),
    ...(payload.location_warnings ?? []),
  ].filter((item, index, all) => item && all.indexOf(item) === index);
  const panels = payload.knowledge_panels ?? [];

  return <aside className="right-panels" aria-label="Historical route details">
    <section className="knowledge-panel authority-card">
      <p className="panel-kicker">历史依据</p><h2>历史路点</h2>
      <p>这些点及其大致顺序来自后端接受的史料约束；地图线路不等于史料直接记载的精确行军轨迹。</p>
      {friendlyRouteSource(routeSource) && <p className="route-source"><strong>路点来源：</strong>{friendlyRouteSource(routeSource)}</p>}
      <ol className="waypoint-list">{markerFeatures(payload).map((feature, index) => {
        const metadata = waypointPopupMetadata(payload, feature);
        const panel = panels.find((item) => item.waypoint_id === metadata.waypointId);
        return <li key={metadata.waypointId || index}><strong>{metadata.name}</strong><span>第 {index + 1} 站 · {metadata.evidenceCount} 条证据引用</span>{panel?.source_references?.length ? <small>{panel.source_references.join("；")}</small> : null}</li>;
      })}</ol>
    </section>
    <section className="knowledge-panel reconstruction-card">
      <p className="panel-kicker">算法重建</p><h2>候选路线</h2>
      <p>{summary?.route_interpretation ?? "算法仅在历史路点之间重建地理候选路径，不创造历史事实。"}</p>
      <ul className="segment-list">{segments.map((segment) => <li key={segment.id} className={`segment-${segment.kind}`}><strong>{segmentLabel(segment.kind)}</strong><span>{segment.from && segment.to ? `${segment.from} → ${segment.to}` : "历史路点之间"}</span>{segment.distanceKm !== undefined ? <span>距离 {formatMetric(segment.distanceKm, " km")}</span> : null}{segment.cost !== undefined ? <span>重建成本 {formatMetric(segment.cost)}</span> : null}{segment.terrainSource ? <small>地形数据：{segment.terrainSource}</small> : null}{segment.failureStatus ? <small className="gap-warning">原因：{segment.failureStatus}</small> : null}</li>)}</ul>
      {!segments.length && <p className="route-empty">历史路点已确定，但当前没有可显示的候选几何。</p>}
      {failed.length > 0 && <p className="gap-warning" role="status">{failed.length} 个区段未能可靠重建；地图不会用直线补齐。</p>}
    </section>
    {limitations.length > 0 && <section className="knowledge-panel limitations-card"><p className="panel-kicker">限制</p><h2>如何理解这条路线</h2><ul>{limitations.map((item) => <li key={item}>{item}</li>)}</ul></section>}
  </aside>;
}
