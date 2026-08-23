import re

from .models import PageDocument

BOOK_PATTERN = re.compile(r"(?im)^\s*book\s+([ivxlcdm]+|\d+)\b")
CHAPTER_PATTERN = re.compile(r"(?im)^\s*(?:chapter|chap\.)\s+([ivxlcdm]+|\d+)\b")
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10", "xx": "20", "xxi": "21"}


def normalize_label(value: str) -> str:
    return ROMAN.get(value.lower(), value)


def annotate_structure(pages: list[PageDocument]) -> list[dict[str, object]]:
    """Carry the last detected book/chapter forward as reliable metadata."""
    book: str | None = None
    chapter: str | None = None
    annotated: list[dict[str, object]] = []
    for page in pages:
        book_match = BOOK_PATTERN.search(page.text)
        chapter_match = CHAPTER_PATTERN.search(page.text)
        if book_match:
            book = normalize_label(book_match.group(1))
            chapter = None
        if chapter_match:
            chapter = normalize_label(chapter_match.group(1))
        annotated.append({"page": page, "book": book, "chapter": chapter, "section": None})
    return annotated
