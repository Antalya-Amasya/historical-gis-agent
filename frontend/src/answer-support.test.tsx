import React from "react";
import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { AgentHistoricalEvidence } from "./phase10-contract";
import { fetchAgentHistoricalRoutePresentation } from "./phase10-contract";
import { RouteEvidencePanel } from "./route-evidence-panel";

const source = (id: string) => ({ id, author: "Source", work: "Work", locator: "1", excerpt: `Passage ${id}` });
const workspace = Array.from({ length: 20 }, (_, i) => source(`e${i}`));
const presentation = { route: { route_id: "r", route_name: "Route", confidence: 0.7 },
  geojson: { type: "FeatureCollection", features: [] } };

async function response(status: "NO_ROUTE" | "FULL_ROUTE" | "PARTIAL", support?: ReturnType<typeof source>[]) {
  return fetchAgentHistoricalRoutePresentation("query", "session", async () => ({ ok: true, status: 200,
    json: async () => ({ reply: "Entry was prevented; no completed route should be reconstructed.", route_result_status: status,
      state: { historical_evidence: workspace, supporting_evidence: support,
        historical_route_presentation: status === "NO_ROUTE" ? null : presentation } }) }));
}
function markup(evidence: AgentHistoricalEvidence[]) {
  return renderToStaticMarkup(<RouteEvidencePanel evidence={evidence} resolvedPlaces={[]} waypoints={[]} limitations={[]} />);
}

describe("final answer support presentation", () => {
  it.each(["NO_ROUTE", "FULL_ROUTE", "PARTIAL"] as const)("%s uses support only, preserving retrieval separately", async (status) => {
    const result = await response(status, [workspace[0]]);
    expect(result.routeResultStatus).toBe(status);
    expect(result.evidence).toHaveLength(20);
    expect(result.supportingEvidence).toEqual([workspace[0]]);
    expect(result.reply).toContain("Entry was prevented");
    const html = markup(result.supportingEvidence);
    expect(html).toContain("史料依据");
    expect(html).toContain("Passage e0");
    expect(html).not.toContain("Passage e1");
    expect(html).not.toContain("相关证据");
  });
  it("missing and zero support never fall back to retrieval", async () => {
    for (const support of [undefined, []]) {
      const result = await response("NO_ROUTE", support);
      expect(result.supportingEvidence).toEqual([]);
      expect(markup(result.supportingEvidence)).toBe("");
    }
  });
  it("multiple sources preserve citation order and deduplicate IDs", async () => {
    const result = await response("PARTIAL", [workspace[1], workspace[0], workspace[1]]);
    expect(result.supportingEvidence.map((e) => e.id)).toEqual(["e1", "e0"]);
    const html = markup([workspace[1], workspace[0], workspace[1]]);
    expect(html.split("Passage e1")).toHaveLength(2);
    expect(html.indexOf("Passage e1")).toBeLessThan(html.indexOf("Passage e0"));
  });
  it("sparse evidence text does not crash support rendering", () => {
    expect(markup([{ id: "sparse", author: "", work: "", locator: "", excerpt: undefined }])).toContain("Evidence excerpt unavailable.");
  });
});
