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
