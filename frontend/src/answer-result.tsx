import { AnswerMarkdown } from "./answer-markdown";

type RouteStatus = "COMPLETE" | "PARTIAL" | "UNAVAILABLE" | undefined;

export function AnswerResult({ reply, routeStatus }: { reply: string; routeStatus: RouteStatus }) {
  return <section className="presentation-summary answer-markdown">
    <h2>Agent answer</h2>
    <AnswerMarkdown answer={reply} />
    {routeStatus === "UNAVAILABLE" && <p className="route-status">A requested route could not be safely reconstructed from the available evidence and geographic data.</p>}
  </section>;
}
