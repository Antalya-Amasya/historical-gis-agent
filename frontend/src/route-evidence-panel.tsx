import type { AgentHistoricalEvidence } from "./phase10-contract";

export function RouteEvidencePanel({
  evidence,
  resolvedPlaces,
  waypoints,
  limitations,
}: {
  evidence: AgentHistoricalEvidence[];
  resolvedPlaces: string[];
  waypoints: string[];
  limitations: string[];
}) {
  if (!evidence.length && !resolvedPlaces.length && !waypoints.length && !limitations.length) {
    return null;
  }

  return (
    <aside className="right-panels route-evidence-panel" aria-label="Historical evidence summary">
      {waypoints.length > 0 && (
        <section className="knowledge-panel authority-card">
          <p className="panel-kicker">历史路点</p>
          <h2>已识别地点顺序</h2>
          <ol className="waypoint-list">
            {waypoints.map((name) => (
              <li key={name}>
                <strong>{name}</strong>
              </li>
            ))}
          </ol>
        </section>
      )}
      {resolvedPlaces.length > 0 && (
        <section className="knowledge-panel authority-card">
          <p className="panel-kicker">地理解析</p>
          <h2>已解析地点</h2>
          <ul className="segment-list">
            {resolvedPlaces.map((name) => (
              <li key={name}>
                <strong>{name}</strong>
              </li>
            ))}
          </ul>
        </section>
      )}
      {evidence.length > 0 && (
        <section className="knowledge-panel authority-card">
          <p className="panel-kicker">史料依据</p>
          <h2>回答引用的史料</h2>
          <ul className="segment-list">
            {[...new Map(evidence.filter((item) => item?.id).map((item) => [item.id, item])).values()].map((item) => (
              <li key={item.id}>
                <strong>{item.author} · {item.work} {item.locator}</strong>
                <span>{item.excerpt ?? item.text ?? "Evidence excerpt unavailable."}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
      {limitations.length > 0 && (
        <section className="knowledge-panel limitations-card">
          <p className="panel-kicker">限制</p>
          <h2>结果说明</h2>
          <ul>
            {limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      )}
    </aside>
  );
}
