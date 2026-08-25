import json
from pathlib import Path
from zipfile import ZipFile

from backend.app.rag.ingestion.loaders import load_epub
from backend.app.rag.ingestion.pipeline import ingest_epub


def write_epub(path: Path) -> None:
    container = """<?xml version='1.0'?><container><rootfiles><rootfile full-path='OEBPS/content.opf'/></rootfiles></container>"""
    opf = """<?xml version='1.0'?><package><manifest><item id='text' href='text.xhtml' media-type='application/xhtml+xml'/></manifest><spine><itemref idref='text'/></spine></package>"""
    xhtml = """<html><body><h4>THE WAR IN GAUL</h4><h5>BOOK I</h5><p>I. First Book text.</p><p>II. More first Book text.</p><h3>BOOK II</h3><p>I. Second Book text.</p><h3>BOOK VII</h3><p>I. Seventh Book text.</p><h3>BOOK VIII</h3><p>I. Excluded continuation.</p></body></html>"""
    with ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/text.xhtml", xhtml)


def manifest(path: Path) -> Path:
    path.write_text(json.dumps({"corpus_id": "caesar_gallic_war", "author": "Julius Caesar", "work": "De Bello Gallico", "language": "en", "source": "Local test source", "books": ["1", "2", "7"]}), encoding="utf-8")
    return path


def test_epub_reader_preserves_explicit_book_and_chapter_structure(tmp_path: Path) -> None:
    epub = tmp_path / "caesar.epub"
    write_epub(epub)

    sections = load_epub(epub)

    assert [(section.text.splitlines()[0], section.text.splitlines()[1]) for section in sections] == [("BOOK 1", "CHAPTER 1"), ("BOOK 1", "CHAPTER 2"), ("BOOK 2", "CHAPTER 1"), ("BOOK 7", "CHAPTER 1")]
    assert all("Excluded continuation" not in section.text for section in sections)


def test_epub_chunks_have_caesar_manifest_and_provenance_metadata(tmp_path: Path) -> None:
    epub = tmp_path / "caesar.epub"
    write_epub(epub)
    _, log, chunks = ingest_epub(epub, manifest(tmp_path / "corpus.json"))

    assert log["book_detection_result"] == ["1", "2", "7"]
    assert {chunk.metadata["author"] for chunk in chunks} == {"Julius Caesar"}
    assert {chunk.metadata["work"] for chunk in chunks} == {"De Bello Gallico"}
    assert {chunk.metadata["source"] for chunk in chunks} == {"Local test source"}
    assert {chunk.metadata["provenance"] for chunk in chunks} == {"manifest_declared_epub"}
    assert {chunk.metadata["corpus_id"] for chunk in chunks} == {"caesar_gallic_war"}
    assert all(chunk.metadata["book"] in {"1", "2", "7"} and chunk.metadata["chapter"] != "unknown" for chunk in chunks)


def test_epub_corpus_metadata_is_isolated_from_existing_primary_source_names(tmp_path: Path) -> None:
    epub = tmp_path / "caesar.epub"
    write_epub(epub)
    _, _, chunks = ingest_epub(epub, manifest(tmp_path / "corpus.json"))

    assert all(chunk.metadata["corpus_id"] == "caesar_gallic_war" for chunk in chunks)
    assert all(chunk.metadata["author"] not in {"Polybius", "Livy"} for chunk in chunks)
