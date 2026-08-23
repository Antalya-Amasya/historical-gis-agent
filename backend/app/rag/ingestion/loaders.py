from pathlib import Path

import pymupdf

from .models import PageDocument


def inspect_text_quality(text: str) -> tuple[str, bool, float]:
    """Classify the text layer without performing OCR."""
    if len(text.strip()) < 40:
        return "empty", True, 1.0
    abnormal = sum(
        not (character.isprintable() or character.isspace()) or character == "�"
        for character in text
    )
    ratio = abnormal / len(text)
    if ratio > 0.03:
        return "poor", True, ratio
    return "good", False, ratio


def load_pdf(path: Path) -> list[PageDocument]:
    """Extract each PDF page independently and preserve provenance."""
    pages: list[PageDocument] = []
    document = pymupdf.open(path)
    try:
        for index, page in enumerate(document):
            text = page.get_text("text")
            quality, requires_ocr, abnormal_ratio = inspect_text_quality(text)
            pages.append(
                PageDocument(
                    source_file=path.as_posix(),
                    page_number=index + 1,
                    text=text,
                    extraction_quality=quality,
                    requires_ocr=requires_ocr,
                    abnormal_character_ratio=abnormal_ratio,
                )
            )
    finally:
        document.close()
    return pages
