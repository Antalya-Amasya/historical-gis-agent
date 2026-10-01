import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { RouteDetails } from "./historical-route-details";
import { RouteEvidencePanel } from "./route-evidence-panel";
import { AnswerResult } from "./answer-result";
import { drawableRouteSegments, fetchAgentHistoricalRoutePresentation, loadHistoricalRoutePresentation, markerFeatures, routeSegments, toLeafletLineCoordinates, waypointPopupMetadata, type GeoJsonFeature, type HistoricalRoutePresentationPayload, type AgentHistoricalEvidence } from "./phase10-contract";

function fixture(vertices = 1000, roles = ["roman_road", "access_connector", "terrain_candidate"], waypoints = 2): HistoricalRoutePresentationPayload {
  const features: GeoJsonFeature[] = Array.from({ length: waypoints }, (_, i) => ({ type: "Feature", geometry: { type: "Point", coordinates: [10 + i / 10, 40 + i / 10] }, properties: { layer_type: "historical_anchor", waypoint_id: `anchor-${i}`, name: `Anchor ${i}`, evidence_refs: [`e-${i}`], coordinate_role: "representative_point" } }));
  roles.forEach((role, i) => features.push({ type: "Feature", geometry: role === "failed_gap" ? null : { type: "LineString", coordinates: Array.from({ length: i === 0 ? vertices : 4 }, (_, j) => [10 + i / 10 + j / vertices / 10, 40 + i / 10 + j / vertices / 10]) }, properties: { layer_type: role === "direct_water_edge" ? role : role === "simulated_coastal_access" ? role : "roman_road_segment", segment_role: role, leg_index: i + 1, segment_distance_m: 1000 } }));
  return { route: { route_id: "synthetic", confidence: 0.7 }, geojson: { type: "FeatureCollection", features }, road_network: { source: "Synthetic GIS", route_status: "PARTIAL", aggregate: { successful_leg_count: roles.length, failed_leg_count: 0, total_network_distance_m: 1000, total_access_connector_distance_m: 10, road_type_counts: {}, segment_status_counts: {}, chronology_counts: {} }, limitations: [], legs: roles.map((_, i) => ({ leg_index: i + 1, source_anchor_id: "anchor-0", destination_anchor_id: "anchor-1", candidate: { network_distance_m: 1000 } })) } };
}
function evidence(count: number): AgentHistoricalEvidence[] { return Array.from({ length: count }, (_, i) => ({ id: `e-${i}`, author: "Author", work: `Work ${i % 5}`, locator: `${i}`, ...(i % 2 ? { text: "Passage" } : {}) })); }
const mixed = ["roman_road", "access_connector", "simulated_coastal_access", "direct_water_edge", "simulated_coastal_access", "terrain_candidate"];

describe("Generic frontend adversarial contract", () => {
  it("handles twelve road/terrain/connectors without endpoint inflation", () => {
    const p = fixture(1000, Array.from({ length: 12 }, (_, i) => ["roman_road", "terrain_candidate", "access_connector"][i % 3]));
    expect(drawableRouteSegments(p)).toHaveLength(12); expect(markerFeatures(p)).toHaveLength(2);
    expect(renderToStaticMarkup(<RouteDetails payload={p} />)).toContain("Anchor 1");
  });
  it.each(["FULL_ROUTE", "PARTIAL"] as const)("handles mixed modes under %s", status => {
    const p = fixture(1000, mixed); expect(routeSegments(p).map(x => x.kind)).toEqual(["roman_road", "connector", "simulated_coastal_access", "direct_water_edge", "simulated_coastal_access", "terrain"]);
    expect(renderToStaticMarkup(<><AnswerResult reply="Simulation only." routeResultStatus={status} /><RouteDetails payload={p} /></>)).toContain("海上重建路线");
  });
  it.each([50, 100, 250])("renders %i evidence items with optional text", count => {
    expect(renderToStaticMarkup(<RouteEvidencePanel evidence={evidence(count)} resolvedPlaces={[]} waypoints={[]} limitations={[]} />).match(/<li/g)).toHaveLength(count);
  });
  it.each([1000, 10000, 25000])("converts %i vertices with bounded completion", vertices => {
    const start = performance.now(); const p = fixture(vertices); expect(toLeafletLineCoordinates(drawableRouteSegments(p)[0].feature)).toHaveLength(vertices);
    renderToStaticMarkup(<RouteDetails payload={p} />); console.log(`geometry ${vertices}: ${(performance.now()-start).toFixed(2)} ms`);
  });
  it("preserves eight historical markers and representative-coordinate labels", () => {
    const p = fixture(1000, mixed, 8); expect(markerFeatures(p)).toHaveLength(8);
    expect(waypointPopupMetadata(p, markerFeatures(p)[0]).description).toContain("不是考古精确点");
  });
  it("never draws a failed gap even if a malformed feature contains geometry", () => {
    const p=fixture(1000,["roman_road","failed_gap","terrain_candidate"]);
    p.geojson.features[3].geometry={type:"LineString",coordinates:[[10,40],[11,41]]};
    expect(drawableRouteSegments(p)).toHaveLength(2); expect(renderToStaticMarkup(<RouteDetails payload={p} />)).toContain("地图不会用直线补齐");
  });
  it("handles omitted optional presentation metadata", async () => {
    const p=fixture();
    const r=await fetchAgentHistoricalRoutePresentation("generic","session",async()=>({ok:true,status:200,json:async()=>({reply:"",route_result_status:"PARTIAL",state:{historical_route_presentation:p}})}));
    expect(r.evidence).toEqual([]); expect(r.routeDiagnosticSummary).toBeNull(); expect(()=>renderToStaticMarkup(<RouteDetails payload={p} />)).not.toThrow();
  });
  it("handles empty/one-point geometry and unknown roles", () => {
    const p=fixture(1,["future_role"]); expect(routeSegments(p)[0].kind).toBe("connector");
    expect(toLeafletLineCoordinates(drawableRouteSegments(p)[0].feature)).toHaveLength(1);
    p.geojson.features[2].geometry={type:"LineString",coordinates:[]}; expect(toLeafletLineCoordinates(p.geojson.features[2])).toEqual([]);
    delete p.geojson.features[0].properties.name; expect(waypointPopupMetadata(p,p.geojson.features[0]).name).toBe("Unnamed waypoint");
  });
  it("rejects invalid serialized coordinates before the map effect", async () => {
    const p=fixture(4); const malformed=JSON.parse(JSON.stringify(p)); malformed.geojson.features[2].geometry.coordinates[0]=[null,40];
    await expect(fetchAgentHistoricalRoutePresentation("generic","session",async()=>({ok:true,status:200,json:async()=>({reply:"",route_result_status:"FULL_ROUTE",state:{historical_route_presentation:malformed}})}))).rejects.toThrow(/coordinate/i);
  });
  it.each([[181, 40], [10, 91], [null, 40]])("rejects invalid marker coordinates %j", async (longitude, latitude) => {
    const malformed=JSON.parse(JSON.stringify(fixture(4)));
    malformed.geojson.features[0].geometry.coordinates=[longitude,latitude];
    const request=async()=>({ok:true,status:200,json:async()=>({reply:"",route_result_status:"FULL_ROUTE",state:{historical_route_presentation:malformed}})});
    let rejected=false;try{await fetchAgentHistoricalRoutePresentation("generic","session",request);}catch(error){rejected=error instanceof Error && /coordinate/i.test(error.message);}
    expect(rejected).toBe(true);
  });
  it("does not label a simulation crossing as a historical stop", () => {
    const p=fixture();p.geojson.features.push({type:"Feature",geometry:{type:"Point",coordinates:[10.05,40.05]},properties:{layer_type:"reconstructed_crossing",waypoint_id:"simulation-node",name:"Simulated crossing"}});
    const html=renderToStaticMarkup(<RouteDetails payload={p}/>); const historical=html.split('reconstruction-card')[0];
    expect(historical).not.toContain("Simulated crossing");expect(html).toContain("Simulated crossing");expect(html).toContain("非历史路点");
  });
  it("renders duplicate cited-support identities only once", () => {
    const e=evidence(50).map(x=>({...x,id:"duplicate"})); expect(renderToStaticMarkup(<RouteEvidencePanel evidence={e} resolvedPlaces={[]} waypoints={[]} limitations={[]} />).match(/<li/g)).toHaveLength(1);
  });
  it("runs fifty deterministic valid shape combinations", async () => {
    for(let i=0;i<50;i++){
      const p=fixture(10+i*10,i%2?mixed:["roman_road","failed_gap","terrain_candidate"],2+i%8);
      const status=(["FULL_ROUTE","PARTIAL","NO_ROUTE"] as const)[i%3];
      const r=await fetchAgentHistoricalRoutePresentation("generic","session",async()=>({ok:true,status:200,json:async()=>({reply:"Synthetic",route_result_status:status,state:{historical_route_presentation:status==="NO_ROUTE"?null:p,historical_evidence:evidence(i*5),historical_route_diagnostics:{anchor_count:0,projection_diagnostics:["FUTURE_CODE:dummy"]}}})}));
      expect(r.routeResultStatus).toBe(status); expect(()=>renderToStaticMarkup(<RouteEvidencePanel evidence={r.evidence} resolvedPlaces={[]} waypoints={[]} limitations={[]} />)).not.toThrow();
      if(r.payload){loadHistoricalRoutePresentation(r.payload);for(const s of drawableRouteSegments(r.payload))toLeafletLineCoordinates(s.feature);renderToStaticMarkup(<RouteDetails payload={r.payload}/>);}
    }
  });
  it("propagates rejected/aborted requests for the UI error handler", async()=>{
    await expect(fetchAgentHistoricalRoutePresentation("generic","session",async()=>({ok:false,status:503,json:async()=>({})}))).rejects.toThrow();
    await expect(fetchAgentHistoricalRoutePresentation("generic","session",async()=>{throw new DOMException("Aborted","AbortError");})).rejects.toThrow("Aborted");
  });
});
