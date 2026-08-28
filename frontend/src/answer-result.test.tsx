import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AnswerResult } from "./answer-result";

describe("production answer presentation", () => {
  it("renders Markdown structurally without accepting raw HTML", () => {
    const markup = renderToStaticMarkup(<AnswerResult reply={"## Test heading\n\n**important**\n\n- item 1\n- item 2\n\n<script>alert('xss')</script>"} routeStatus={undefined} />);
    expect(markup).toContain("<h2>Test heading</h2>");
    expect(markup).toContain("<strong>important</strong>");
    expect(markup).toContain("<li>item 1</li>");
    expect(markup).not.toContain("<script>");
  });

  it("keeps an answer-only response free of a GIS failure notice", () => {
    const markup = renderToStaticMarkup(<AnswerResult reply="A historical answer." routeStatus={undefined} />);
    expect(markup).toContain("A historical answer.");
    expect(markup).not.toContain("could not be safely reconstructed");
  });

  it("shows an explicit unavailable state without fabricating geometry", () => {
    const markup = renderToStaticMarkup(<AnswerResult reply="Route evidence is limited." routeStatus="UNAVAILABLE" />);
    expect(markup).toContain("could not be safely reconstructed");
    expect(markup).not.toContain("LineString");
  });
});
