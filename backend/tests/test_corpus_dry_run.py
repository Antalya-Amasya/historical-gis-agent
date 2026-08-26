import json
from pathlib import Path
from zipfile import ZipFile

from backend.app.rag.ingestion.corpus_dry_run import document_chunks, dry_run
from backend.app.rag.ingestion.corpus_registry import CorpusDocument
from backend.app.rag.ingestion.document_sections import DocumentSection
from backend.app.rag.ingestion.generic_epub import classify_label, load_epub_sections, roman_to_int


def _epub(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container><rootfiles><rootfile full-path='OPS/content.opf'/></rootfiles></container>")
        archive.writestr("OPS/content.opf", "<package><manifest><item id='a' href='a.xhtml'/></manifest><spine><itemref idref='a'/></spine></package>")
        archive.writestr("OPS/a.xhtml", "<html><body><h1>Book I</h1><h2>Chapter 1</h2><p>Some text about a historical event.</p></body></html>")


def test_document_chunk_ids_are_deterministic() -> None:
    document = CorpusDocument(document_id="work", author="A", work="W", language="en", source_type="primary_source", filename="w.epub")
    section = DocumentSection("work", "A", "W", None, "I", "1", None, "Chapter 1", "text " * 100, "w.epub", 0, "a.xhtml")
    assert [item.id for item in document_chunks([section], document, size=30)] == [item.id for item in document_chunks([section], document, size=30)]


def test_dry_run_never_embeds_or_writes_chroma(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"; incoming.mkdir(); _epub(incoming / "work.epub")
    registry = tmp_path / "corpus.json"
    registry.write_text(json.dumps({"registry_version": 1, "documents": [{"document_id": "work", "author": "A", "work": "W", "volume": None, "language": "en", "source_type": "primary_source", "filename": "work.epub", "period_start_bce": None, "period_end_bce": None, "source_url": None, "license": "public_domain", "enabled": True, "parser_hints": {}}]}), encoding="utf-8")
    report = dry_run(incoming, registry, tmp_path / "report.json")
    assert report["embedding_called"] is False and report["chroma_written"] is False
    assert report["documents"][0]["detected_book_count"] == 1


def test_navigation_fragment_alignment_and_roman_normalization(tmp_path: Path) -> None:
    epub = tmp_path / "nav.epub"
    with ZipFile(epub, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container><rootfiles><rootfile full-path='OPS/content.opf'/></rootfiles></container>")
        archive.writestr("OPS/content.opf", "<package><manifest><item id='nav' href='nav.xhtml' properties='nav'/><item id='a' href='a.xhtml'/></manifest><spine><itemref idref='a'/></spine></package>")
        archive.writestr("OPS/nav.xhtml", "<html xmlns:epub='http://www.idpf.org/2007/ops'><body><nav epub:type='toc'><ol><li><a href='a.xhtml#book'>Book XXI</a><ol><li><a href='a.xhtml#chapter'>Chapter IV</a></li></ol></li></ol></nav></body></html>")
        archive.writestr("OPS/a.xhtml", "<html><body><h1 id='book'>Book XXI</h1><p>Book text.</p><h2 id='chapter'>Chapter IV</h2><p>Chapter text.</p></body></html>")
    document = CorpusDocument(document_id="work", author="A", work="W", language="en", source_type="primary_source", filename="nav.epub")
    sections, _ = load_epub_sections(epub, document)
    assert [(section.book, section.chapter) for section in sections] == [("21", None), ("21", "4")]
    assert sections[-1].navigation_path == ("Book XXI", "Chapter IV")
    assert roman_to_int("XLIX") == 49 and roman_to_int("IIV") is None
    assert classify_label("(BOOK 36, editor.)") == ("book", "36")


def test_chunk_id_does_not_depend_on_source_path() -> None:
    document = CorpusDocument(document_id="work", author="A", work="W", language="en", source_type="primary_source", filename="w.epub")
    first = DocumentSection("work", "A", "W", None, "1", None, None, "Book I", "text " * 20, "one/w.epub", 0, "a.xhtml", 4)
    moved = DocumentSection("work", "A", "W", None, "1", None, None, "Book I", "text " * 20, "another/w.epub", 0, "a.xhtml", 4)
    assert [chunk.id for chunk in document_chunks([first], document, size=30)] == [chunk.id for chunk in document_chunks([moved], document, size=30)]


def test_ncx_navigation_is_used_when_epub3_nav_is_absent(tmp_path: Path) -> None:
    epub = tmp_path / "ncx.epub"
    with ZipFile(epub, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container><rootfiles><rootfile full-path='OPS/content.opf'/></rootfiles></container>")
        archive.writestr("OPS/content.opf", "<package><manifest><item id='n' href='toc.ncx' media-type='application/x-dtbncx+xml'/><item id='a' href='a.xhtml'/></manifest><spine toc='n'><itemref idref='a'/></spine></package>")
        archive.writestr("OPS/toc.ncx", "<ncx><navMap><navPoint><navLabel><text>Book IX</text></navLabel><content src='a.xhtml#b'/><navPoint><navLabel><text>Section 5</text></navLabel><content src='a.xhtml#s'/></navPoint></navPoint></navMap></ncx>")
        archive.writestr("OPS/a.xhtml", "<html><body><h1 id='b'>Book IX</h1><p>One.</p><h2 id='s'>Section 5</h2><p>Two.</p></body></html>")
    document = CorpusDocument(document_id="work", author="A", work="W", language="en", source_type="primary_source", filename="ncx.epub")
    sections, _ = load_epub_sections(epub, document)
    assert sections[-1].navigation_source == "ncx"
    assert sections[-1].book == "9" and sections[-1].section == "5"
