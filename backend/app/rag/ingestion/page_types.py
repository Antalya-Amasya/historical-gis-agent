NON_BODY = {"front_matter", "advertisement", "copyright", "toc", "gutenberg_footer"}


def classify_page(text: str, page_number: int, book: str | None) -> str:
    value = text.lower()
    if "project gutenberg" in value or "gutenberg-tm" in value:
        return "gutenberg_footer"
    if page_number < 15 and ("contents" in value or "table of contents" in value):
        return "toc"
    if page_number < 15 and ("copyright" in value or "all rights reserved" in value):
        return "copyright"
    if page_number < 20 and ("dictionary" in value or "published by" in value or "advertisement" in value):
        return "advertisement"
    if page_number < 20:
        return "front_matter"
    return "body" if book != "unknown" else "unknown"


def should_include_in_index(page_type: str) -> bool:
    return page_type not in NON_BODY
