import { AnswerMarkdown } from "./answer-markdown";
import { routeResultNoticeClass, routeResultStatusLabel, type RouteResultStatus } from "./route-result-status";

export function AnswerResult({
  reply,
  routeResultStatus,
}: {
  reply: string;
  routeResultStatus?: RouteResultStatus | null;
}) {
  const label = routeResultStatusLabel(routeResultStatus);
  const noticeClass = routeResultNoticeClass(routeResultStatus);
  return (
    <section className="presentation-summary answer-markdown">
      <h2>Agent answer</h2>
      {label && noticeClass ? <p className={noticeClass} role="status">{label}</p> : null}
      <AnswerMarkdown answer={reply} />
    </section>
  );
}
