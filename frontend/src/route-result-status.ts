export type RouteResultStatus = "FULL_ROUTE" | "PARTIAL" | "NO_ROUTE" | "ERROR";

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
