export type RouteResultStatus = "FULL_ROUTE" | "PARTIAL" | "NO_ROUTE" | "ERROR";
import type { RouteDiagnosticSummary } from "./phase10-contract";

export function routeDiagnosticSummaryLabel(summary: RouteDiagnosticSummary | null | undefined): string | null {
  switch (summary) {
    case "NON_EXACT_ANCHOR": return "部分地点已识别，但现有坐标精度不足以构建可靠路线。";
    case "PLACE_RESOLUTION": return "部分地点未能充分解析或缺少可用坐标，暂无法构建可靠路线。";
    case "MIXED": return "部分地点坐标精度不足，另有地点解析或坐标信息不足；这些限制可能涉及不同的史料事件。";
    default: return null;
  }
}

export function routeResultStatusLabel(status: RouteResultStatus | null | undefined): string | null {
  switch (status) {
    case "FULL_ROUTE":
      return "路线已生成";
    case "PARTIAL":
      return "部分路线 / 部分证据可用";
    case "NO_ROUTE":
      return "当前证据不足以生成可靠路线";
    case "ERROR":
      return "查询处理失败";
    default:
      return null;
  }
}

export function routeResultNoticeClass(status: RouteResultStatus | null | undefined): string | null {
  switch (status) {
    case "FULL_ROUTE":
      return "route-notice route-notice--full";
    case "PARTIAL":
      return "route-notice route-notice--partial";
    case "NO_ROUTE":
      return "route-notice route-notice--no-route";
    case "ERROR":
      return "route-notice route-notice--error";
    default:
      return null;
  }
}

export function shouldRenderRouteMap(
  status: RouteResultStatus | null | undefined,
  hasPresentation: boolean,
): boolean {
  return hasPresentation && (status === "FULL_ROUTE" || status === "PARTIAL");
}
