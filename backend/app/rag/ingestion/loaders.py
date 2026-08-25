from pathlib import Path
from re import compile as re_compile
from xml.etree import ElementTree
from zipfile import ZipFile

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


_BOOK_HEADING = re_compile(r"^BOOK\s+([IVXLCDM]+|\d+)\b", flags=0)
_CHAPTER_HEADING = re_compile(r"^([IVXLCDM]+|\d+)\.\W*")
_ROMAN = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", "VII": "7", "VIII": "8"}

def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()

def _epub_spine(archive: ZipFile) -> list[str]:
    container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
    rootfile = next(element.attrib["full-path"] for element in container.iter() if _local_name(element.tag) == "rootfile")
    package = ElementTree.fromstring(archive.read(rootfile))
    prefix = Path(rootfile).parent
    manifest = {item.attrib["id"]: item.attrib["href"] for item in package.iter() if _local_name(item.tag) == "item"}
    return [(prefix / manifest[itemref.attrib["idref"]]).as_posix() for itemref in package.iter() if _local_name(itemref.tag) == "itemref" and itemref.attrib["idref"] in manifest]

def _normalise_epub_text(element) -> str:
    return " ".join("".join(element.itertext()).split())

def load_epub(path: Path) -> list[PageDocument]:
    """Read the explicit Book I–VII XHTML structure of an offline EPUB; no network or inference."""
    sections: list[PageDocument] = []
    current_book: str | None = None
    current_chapter: str | None = None
    current_text: list[str] = []
    enabled = False

    def flush() -> None:
        nonlocal current_text
        if current_book and current_chapter and current_text:
            text = f"BOOK {current_book}\nCHAPTER {current_chapter}\n" + "\n".join(current_text)
            quality, requires_ocr, abnormal_ratio = inspect_text_quality(text)
            sections.append(PageDocument(path.as_posix(), len(sections) + 1, text, quality, requires_ocr, abnormal_ratio))
        current_text = []

    with ZipFile(path) as archive:
        for item in _epub_spine(archive):
            if not item.lower().endswith((".xhtml", ".html")):
                continue
            try:
                root = ElementTree.fromstring(archive.read(item))
            except ElementTree.ParseError as exc:
                raise ValueError(f"invalid EPUB XHTML: {item}") from exc
            for element in root.iter():
                tag = _local_name(element.tag)
                if tag not in {"h3", "h4", "h5", "p"}:
                    continue
                text = _normalise_epub_text(element)
                if not text:
                    continue
                if tag == "h4" and text.upper() == "THE WAR IN GAUL":
                    enabled = True
                    continue
                if tag in {"h3", "h5"}:
                    book = _BOOK_HEADING.match(text.upper())
                    if book:
                        flush()
                        current_book = _ROMAN.get(book.group(1), book.group(1))
                        current_chapter = None
                        enabled = enabled and int(current_book) <= 7
                    continue
                if not enabled or current_book is None:
                    continue
                chapter = _CHAPTER_HEADING.match(text)
                if chapter:
                    flush()
                    current_chapter = _ROMAN.get(chapter.group(1).upper(), chapter.group(1))
                if current_chapter:
                    current_text.append(text)
    flush()
    return sections
