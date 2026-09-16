import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AnswerResult } from "./answer-result";

describe("production answer presentation", () => {
  it("renders Markdown structurally without accepting raw HTML", () => {
    const markup = renderToStaticMarkup(<AnswerResult reply={"## Test heading\n\n**important**\n\n- item 1\n- item 2\n\n<script>alert('xss')</script>"} routeResultStatus={null} />);
    expect(markup).toContain("<h2>Test heading</h2>");
    expect(markup).toContain("<strong>important</strong>");
    expect(markup).toContain("<li>item 1</li>");
    expect(markup).not.toContain("<script>");
  });

  it("keeps an answer-only response free of a route status notice", () => {
    const markup = renderToStaticMarkup(<AnswerResult reply="A historical answer." routeResultStatus={null} />);
    expect(markup).toContain("A historical answer.");
    expect(markup).not.toContain("route-notice");
  });

  it("shows a partial route notice without fabricating geometry", () => {
    const markup = renderToStaticMarkup(<AnswerResult reply="Route evidence is limited." routeResultStatus="PARTIAL" />);
    expect(markup).toContain("route-notice--partial");
    expect(markup).not.toContain("LineString");
  });
});
