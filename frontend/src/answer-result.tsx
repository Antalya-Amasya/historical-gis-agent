import { AnswerMarkdown } from "./answer-markdown";
import { routeDiagnosticSummaryLabel, routeResultNoticeClass, routeResultStatusLabel, type RouteResultStatus } from "./route-result-status";
import type { RouteDiagnosticSummary } from "./phase10-contract";

export function AnswerResult({
  reply,
  routeResultStatus,
  routeDiagnosticSummary,
}: {
  reply: string;
  routeResultStatus?: RouteResultStatus | null;
  routeDiagnosticSummary?: RouteDiagnosticSummary | null;
}) {
  const label = routeResultStatusLabel(routeResultStatus);
  const noticeClass = routeResultNoticeClass(routeResultStatus);
  const detail = routeResultStatus === "NO_ROUTE" ? routeDiagnosticSummaryLabel(routeDiagnosticSummary) : null;
  return (
    <section className="presentation-summary answer-markdown">
      <h2>Agent answer</h2>
      {label && noticeClass ? <p className={noticeClass} role="status">{label}</p> : null}
      {detail ? <p className="route-diagnostic-summary">{detail}</p> : null}
      <AnswerMarkdown answer={reply} />
    </section>
  );
}
