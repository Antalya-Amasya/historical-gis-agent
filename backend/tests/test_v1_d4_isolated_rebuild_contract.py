import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _rebuild_script():
    path = ROOT / "scripts" / "v1_d4_isolated_caesar_rebuild.py"
    spec = importlib.util.spec_from_file_location("v1_d4_isolated_caesar_rebuild", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


from backend.app.rag.ingestion.corpus_cli import build_parser, production_config
from backend.app.rag.ingestion.corpus_dry_run import (
    CHUNKER_VERSION,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_OVERLAP_CHARS,
    PARSER_VERSION,
    document_chunks,
)
from backend.app.rag.ingestion.corpus_registry import CorpusDocument
from backend.app.rag.ingestion.document_sections import DocumentSection


def test_v1_d4_uses_production_parser_and_window_contract() -> None:
    assert PARSER_VERSION == "generic-epub-navigation-v2"
    assert CHUNKER_VERSION == "section-window-v1"
    assert DEFAULT_CHUNK_CHARS == 4000
    assert DEFAULT_OVERLAP_CHARS == 600


def test_v1_d4_cli_can_select_isolated_collection_without_all_documents(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--resume",
            "--document-id",
            "caesar_gallic_civil_wars",
            "--collection",
            "v1_d4_caesar_10657_smoke",
            "--registry",
            str(tmp_path / "registry.json"),
            "--incoming",
            str(tmp_path / "incoming"),
        ]
    )
    config = production_config(args, state_dir=tmp_path, default_collection="roman_republic_primary_sources_v2")
    assert config.collection_name == "v1_d4_caesar_10657_smoke"
    assert config.document_ids == ("caesar_gallic_civil_wars",)


def test_v1_d4_chunk_ids_are_stable_for_identical_bytes() -> None:
    document = CorpusDocument(
        document_id="caesar_gallic_civil_wars",
        author="Julius Caesar",
        work="Gallic War + Civil War",
        language="en",
        source_type="primary_source",
        filename="pg10657.epub.noimages",
        source_url="https://www.gutenberg.org/ebooks/10657",
        license="Public domain in the USA.",
    )
    section = DocumentSection(
        "caesar_gallic_civil_wars",
        "Julius Caesar",
        "Gallic War + Civil War",
        None,
        "I",
        "1",
        None,
        "Chapter 1",
        "All Gaul is divided into three parts. " * 80,
        "pg10657.epub.noimages",
        0,
        "a.xhtml",
    )
    first = document_chunks([section], document)
    second = document_chunks([section], document)
    assert [item.id for item in first] == [item.id for item in second]
    assert [item.text for item in first] == [item.text for item in second]
    assert all(item.metadata["parser_version"] == PARSER_VERSION for item in first)
    assert all(item.metadata["source_url"] == "https://www.gutenberg.org/ebooks/10657" for item in first)


def test_v1_d4_defaults_remain_caesar_10657() -> None:
    rebuild = _rebuild_script()
    DEFAULT_COLLECTION = rebuild.DEFAULT_COLLECTION
    DEFAULT_DOCUMENT_ID = rebuild.DEFAULT_DOCUMENT_ID
    DEFAULT_EDITION_PAGE = rebuild.DEFAULT_EDITION_PAGE
    catalog_noimages_href = rebuild.catalog_noimages_href
    expected_opf_identifier = rebuild.expected_opf_identifier
    gutenberg_ebook_id = rebuild.gutenberg_ebook_id
    opf_matches_edition = rebuild.opf_matches_edition
    source_document = rebuild.source_document

    assert DEFAULT_DOCUMENT_ID == "caesar_gallic_civil_wars"
    assert DEFAULT_EDITION_PAGE.endswith("/10657")
    assert DEFAULT_COLLECTION == "v1_d4_caesar_10657_smoke"
    document = source_document(
        "pg10657.epub.noimages",
        document_id=DEFAULT_DOCUMENT_ID,
        author="Julius Caesar",
        work="Gallic War + Civil War",
        source_url=DEFAULT_EDITION_PAGE,
    )
    assert document.document_id == "caesar_gallic_civil_wars"
    assert document.source_url == "https://www.gutenberg.org/ebooks/10657"
    assert gutenberg_ebook_id(DEFAULT_EDITION_PAGE) == "10657"
    assert catalog_noimages_href("10657") == "/ebooks/10657.epub.noimages"
    assert expected_opf_identifier("10657") == "http://www.gutenberg.org/10657"
    assert opf_matches_edition({"identifier": ["http://www.gutenberg.org/10657"]}, "10657")
    assert not opf_matches_edition({"identifier": ["http://www.gutenberg.org/10657"]}, "6386")


def test_v1_d5_suetonius_6386_is_independently_selectable() -> None:
    rebuild = _rebuild_script()
    catalog_noimages_href = rebuild.catalog_noimages_href
    catalog_noimages_url = rebuild.catalog_noimages_url
    expected_opf_identifier = rebuild.expected_opf_identifier
    gutenberg_ebook_id = rebuild.gutenberg_ebook_id
    opf_matches_edition = rebuild.opf_matches_edition
    source_document = rebuild.source_document

    edition = "https://www.gutenberg.org/ebooks/6386"
    assert gutenberg_ebook_id(edition) == "6386"
    assert catalog_noimages_href("6386") == "/ebooks/6386.epub.noimages"
    assert expected_opf_identifier("6386") == "http://www.gutenberg.org/6386"
    html = '<a href="/ebooks/6386.epub.noimages">EPUB (no images)</a>'
    assert catalog_noimages_url(html, edition) == "https://www.gutenberg.org/ebooks/6386.epub.noimages"
    document = source_document(
        "pg6386.epub.noimages",
        document_id="suetonius_julius_caesar",
        author="Suetonius",
        work="Julius Caesar",
        source_url=edition,
    )
    caesar = source_document(
        "pg10657.epub.noimages",
        document_id="caesar_gallic_civil_wars",
        author="Julius Caesar",
        work="Gallic War + Civil War",
        source_url="https://www.gutenberg.org/ebooks/10657",
    )
    assert document.document_id != caesar.document_id
    assert document.source_url != caesar.source_url
    assert opf_matches_edition({"identifier": ["http://www.gutenberg.org/6386"]}, "6386")
    assert not opf_matches_edition({"identifier": ["http://www.gutenberg.org/6386"]}, "10657")


def test_v1_d5_rejects_wrong_catalog_href_and_isolation_port() -> None:
    catalog_noimages_url = _rebuild_script().catalog_noimages_url
    source = (ROOT / "scripts" / "v1_d4_isolated_caesar_rebuild.py").read_text(encoding="utf-8")
    assert "PersistentClient" in source
    assert "HttpClient" not in source
    assert "never connects to Chroma on port 8002" in source

    edition = "https://www.gutenberg.org/ebooks/6386"
    try:
        catalog_noimages_url("<a href='/ebooks/10657.epub.noimages'>wrong</a>", edition)
    except SystemExit as exc:
        assert "6386" in str(exc)
    else:
        raise AssertionError("wrong edition href must be rejected")


def test_v1_d5_rejects_wrong_input_hash(tmp_path) -> None:
    main = _rebuild_script().main

    epub = tmp_path / "pg6386.epub.noimages"
    epub.write_bytes(b"not-an-epub")
    try:
        main(
            [
                "--epub",
                str(epub),
                "--work-dir",
                str(tmp_path / "work"),
                "--expected-sha256",
                "00" * 32,
                "--edition-page",
                "https://www.gutenberg.org/ebooks/6386",
                "--document-id",
                "suetonius_julius_caesar",
            ]
        )
    except SystemExit as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("wrong input hash must be rejected")
