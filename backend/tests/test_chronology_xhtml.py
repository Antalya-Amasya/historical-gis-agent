from zipfile import ZipFile

from backend.app.chronology.registry import extract_epub_document_blocks, records
from backend.app.routes.temporal import EvidenceTemporalResolver


def test_namespaced_gutenberg_style_xhtml_preserves_blocks_and_resolves_date():
    raw = b'''<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body>
    <div><h2>Chapter I</h2><div><p>The office was vacant in <span>130 B.C.</span>.</p></div></div>
    <blockquote>Later testimony.</blockquote><script>ignored()</script><style>.ignored{}</style>
    </body></html>'''

    blocks = extract_epub_document_blocks(raw, "OEBPS/chapter.xhtml")

    assert [(block["type"], block["text"]) for block in blocks] == [
        ("heading", "Chapter I"),
        ("paragraph", "The office was vacant in 130 B.C.."),
        ("blockquote", "Later testimony."),
    ]
    readings, codes = EvidenceTemporalResolver().resolve(blocks[1]["text"], "fixture")
    assert codes == ["TEMPORAL_RESOLVED"]
    assert readings[0].raw_expression == "130 B.C."
    assert readings[0].normalized_start == "-130"


def test_div_fallback_does_not_duplicate_nested_paragraph():
    raw = b'''<html xmlns="http://www.w3.org/1999/xhtml"><body>
    <div>Outer text<p>Only paragraph <span>31 B.C.</span>.</p></div>
    </body></html>'''

    blocks = extract_epub_document_blocks(raw, "chapter.xhtml")

    assert [(block["type"], block["text"]) for block in blocks] == [
        ("paragraph", "Only paragraph 31 B.C.."),
    ]


def test_gutenberg_style_paragraph_becomes_one_provenanced_chronology_record(tmp_path):
    epub = tmp_path / "fixture.epub"
    with ZipFile(epub, "w") as archive:
        archive.writestr("META-INF/container.xml", '''<container><rootfiles>
            <rootfile full-path="OEBPS/content.opf"/></rootfiles></container>''')
        archive.writestr("OEBPS/content.opf", '''<package xmlns="http://www.idpf.org/2007/opf">
          <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Fixture</dc:title><dc:creator>Tester</dc:creator></metadata>
          <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
          <spine><itemref idref="chapter"/></spine></package>''')
        archive.writestr("OEBPS/chapter.xhtml", '''<html xmlns="http://www.w3.org/1999/xhtml"><body>
          <div><h2>Chronology</h2><p>The office was vacant in <span>130 B.C.</span>.</p></div></body></html>''')

    _, rows, _, _ = records(epub)

    assert len(rows) == 1
    assert rows[0]["normalized_start"] == "-130"
    assert rows[0]["source_locator"] == {
        "epub_document": "OEBPS/chapter.xhtml", "section_index": 1,
        "heading": "Chronology", "paragraph_index": 0,
    }
