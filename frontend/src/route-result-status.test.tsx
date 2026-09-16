import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AnswerResult } from "./answer-result";
import { RouteEvidencePanel } from "./route-evidence-panel";
import { fetchAgentHistoricalRoutePresentation } from "./phase10-contract";
import { routeResultStatusLabel, shouldRenderRouteMap } from "./route-result-status";

const terrainPresentation = {
  route: { route_id: "terrain-route", route_name: "Evidence-constrained route", confidence: 0.72 },
  geojson: {
    type: "FeatureCollection" as const,
    features: [{
      type: "Feature" as const,
      geometry: { type: "LineString" as const, coordinates: [[4, 43], [7, 45]] as [number, number][] },
      properties: {},
    }],
  },
};

function mockResponse(status: string | null, state: Record<string, unknown> = {}) {
  return {
    ok: true,
    status: 200,
    json: async () => ({
      reply: "Agent reply",
      route_result_status: status,
      state: {
        historical_route_presentation: status === "FULL_ROUTE" || status === "PARTIAL" ? terrainPresentation : null,
        historical_evidence: [{ id: "e1", author: "Caesar", work: "Civil War", locator: "XXIII", excerpt: "Libo sailed from Oricum." }],
        historical_route: status === "NO_ROUTE" ? null : {
          ordered_points: [
            { historical_place: { canonical_name: "Orikon" }, evidence_refs: ["e1"] },
            { historical_place: { canonical_name: "Brundisium" }, evidence_refs: ["e1"] },
          ],
        },
        resolved_places: [{ canonical_name: "Orikon" }],
        ...state,
      },
    }),
  };
}

describe("route_result_status contract integration", () => {
  it("maps FULL_ROUTE from the top-level API field", async () => {
    const result = await fetchAgentHistoricalRoutePresentation("route", "s1", async () => mockResponse("FULL_ROUTE"));
    expect(result.routeResultStatus).toBe("FULL_ROUTE");
    expect(result.payload).not.toBeNull();
    expect(shouldRenderRouteMap(result.routeResultStatus, Boolean(result.payload))).toBe(true);
  });

  it("maps PARTIAL without treating it as a fatal error", async () => {
    const result = await fetchAgentHistoricalRoutePresentation("route", "s1", async () => mockResponse("PARTIAL"));
    expect(result.routeResultStatus).toBe("PARTIAL");
    expect(result.evidence).toHaveLength(1);
    expect(shouldRenderRouteMap(result.routeResultStatus, Boolean(result.payload))).toBe(true);
  });

  it("maps NO_ROUTE and keeps evidence without presentation geometry", async () => {
    const result = await fetchAgentHistoricalRoutePresentation("route", "s1", async () => mockResponse("NO_ROUTE"));
    expect(result.routeResultStatus).toBe("NO_ROUTE");
    expect(result.payload).toBeNull();
    expect(result.evidence).toHaveLength(1);
    expect(shouldRenderRouteMap(result.routeResultStatus, Boolean(result.payload))).toBe(false);
  });

  it("maps ERROR distinctly from NO_ROUTE", async () => {
    const result = await fetchAgentHistoricalRoutePresentation("route", "s1", async () => mockResponse("ERROR"));
    expect(result.routeResultStatus).toBe("ERROR");
    expect(result.payload).toBeNull();
  });

  it("preserves null status for non-route responses", async () => {
    const result = await fetchAgentHistoricalRoutePresentation("ordinary", "s1", async () => ({
      ok: true,
      status: 200,
      json: async () => ({ reply: "Normal answer", route_result_status: null, state: {} }),
    }));
    expect(result.routeResultStatus).toBeNull();
    expect(result.payload).toBeNull();
  });
});

describe("route result UI labels", () => {
  it("renders distinct user-facing labels for each status", () => {
    expect(routeResultStatusLabel("FULL_ROUTE")).toContain("路线已生成");
    expect(routeResultStatusLabel("PARTIAL")).toContain("部分");
    expect(routeResultStatusLabel("NO_ROUTE")).toContain("不足以生成可靠路线");
    expect(routeResultStatusLabel("ERROR")).toContain("查询处理失败");
  });

  it("shows NO_ROUTE as a neutral notice rather than an error banner", () => {
    const markup = renderToStaticMarkup(<AnswerResult reply="Insufficient evidence." routeResultStatus="NO_ROUTE" />);
    expect(markup).toContain("route-notice--no-route");
    expect(markup).not.toContain("route-notice--error");
  });

  it("shows ERROR with error styling", () => {
    const markup = renderToStaticMarkup(<AnswerResult reply="Provider unavailable." routeResultStatus="ERROR" />);
    expect(markup).toContain("route-notice--error");
  });

  it("keeps evidence visible for NO_ROUTE", () => {
    const markup = renderToStaticMarkup(
      <RouteEvidencePanel
        evidence={[{ id: "e1", author: "Caesar", work: "Civil War", locator: "XXIII", excerpt: "Libo sailed from Oricum." }]}
        resolvedPlaces={["Orikon"]}
        waypoints={[]}
        limitations={[]}
      />,
    );
    expect(markup).toContain("Libo sailed from Oricum.");
    expect(markup).toContain("Orikon");
  });
});
