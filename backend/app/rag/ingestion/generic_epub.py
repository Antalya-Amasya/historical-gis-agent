"""Generic EPUB navigation/spine parsing; it never branches on author or work."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree
from zipfile import ZipFile

from .corpus_registry import CorpusDocument
from .document_sections import DocumentSection, EpubNavigationEntry

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_TEXT = _HEADINGS | {"p", "div", "li", "blockquote"}
_LEVELS = ("volume", "book", "chapter", "section")
_LABEL = re.compile(r"(?:^|[\[(]\s*)(?:the\s+)?(volume|vol\.?|book|liber|chapter|chap\.?|caput|section|part)\s+([ivxlcdm]+|\d+)\b", re.I)


def _name(tag: str) -> str: return tag.rsplit("}", 1)[-1].lower()
def _text(element) -> str: return " ".join("".join(element.itertext()).split())


def roman_to_int(value: str) -> int | None:
    token = value.upper()
    if not re.fullmatch(r"[IVXLCDM]+", token or ""): return None
    values = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    total = sum(-values[c] if i + 1 < len(token) and values[c] < values[token[i + 1]] else values[c] for i,c in enumerate(token))
    parts = ((1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")); rest=total; result=""
    for n,g in parts: count,rest=divmod(rest,n); result += g * count
    return total if result == token else None


def classify_label(label: str) -> tuple[str, str] | None:
    match = _LABEL.search(label)
    if not match: return None
    kind, value = match.groups(); kind = {"vol":"volume","vol.":"volume","liber":"book","chap":"chapter","chap.":"chapter","caput":"chapter","part":"section"}.get(kind.lower(),kind.lower())
    return kind, str(roman_to_int(value) or value)


@dataclass(frozen=True)
class EpubStructure:
    spine: list[str]
    navigation: list[EpubNavigationEntry]
    navigation_source: str | None


def _href(base: PurePosixPath, href: str) -> tuple[str, str | None]:
    parsed=urlsplit(href); item=(base / unquote(parsed.path)).as_posix() if parsed.path else ""
    return str(PurePosixPath(item)), unquote(parsed.fragment) or None


def _package(archive: ZipFile):
    container=ElementTree.fromstring(archive.read("META-INF/container.xml")); opf=next(x.attrib["full-path"] for x in container.iter() if _name(x.tag)=="rootfile")
    package=ElementTree.fromstring(archive.read(opf)); manifest={x.attrib["id"]:dict(x.attrib) for x in package.iter() if _name(x.tag)=="item"}; base=PurePosixPath(opf).parent
    spine=[(base / manifest[x.attrib["idref"]]["href"]).as_posix() for x in package.iter() if _name(x.tag)=="itemref" and x.attrib.get("idref") in manifest]
    return base, manifest, spine


def _entries(root, base, spine, source):
    entries=[]; order=0
    def visit(node, depth, parents):
        nonlocal order
        for li in [x for x in list(node) if _name(x.tag)=="li"]:
            anchor=next((x for x in list(li) if _name(x.tag)=="a" and x.attrib.get("href")),None); label=_text(anchor) if anchor is not None else _text(li); path=parents
            if anchor is not None and label:
                normalized,fragment=_href(base,anchor.attrib["href"]); entries.append(EpubNavigationEntry(label,anchor.attrib["href"],normalized,fragment,depth,parents,source,spine.index(normalized) if normalized in spine else None,order)); order+=1; path=parents+(label,)
            for child in list(li):
                if _name(child.tag)=="ol": visit(child,depth+1,path)
    if source == "ncx":
        def ncx(node,depth,parents):
            nonlocal order
            for point in [x for x in list(node) if _name(x.tag)=="navpoint"]:
                label=next((_text(x) for x in point.iter() if _name(x.tag)=="text"),""); link=next((x.attrib.get("src") for x in point.iter() if _name(x.tag)=="content"),None); path=parents
                if label and link:
                    normalized,fragment=_href(base,link); entries.append(EpubNavigationEntry(label,link,normalized,fragment,depth,parents,source,spine.index(normalized) if normalized in spine else None,order)); order+=1; path=parents+(label,)
                ncx(point,depth+1,path)
        ncx(next((x for x in root.iter() if _name(x.tag)=="navmap"),root),1,())
    else:
        toc=next((x for x in root.iter() if _name(x.tag)=="nav" and "toc" in " ".join(x.attrib.values()).lower()),root); ol=next((x for x in toc.iter() if _name(x.tag)=="ol"),None)
        if ol is not None: visit(ol,1,())
    return entries


def epub_structure(path: Path) -> EpubStructure:
    with ZipFile(path) as archive:
        base,manifest,spine=_package(archive); nav=next((x for x in manifest.values() if "nav" in x.get("properties","").split()),None)
        if nav:
            parsed=_entries(ElementTree.fromstring(archive.read((base/nav["href"]).as_posix())),base,spine,"epub3_nav")
            # A navigation document containing only a notice is not a usable table of contents.
            # The generic cardinality check permits the EPUB2 NCX fallback without work-specific rules.
            if len(parsed) > 1: return EpubStructure(spine,parsed,"epub3_nav")
        ncx=next((x for x in manifest.values() if "ncx" in x.get("media-type","").lower() or x.get("href","").lower().endswith(".ncx")),None)
        return EpubStructure(spine,_entries(ElementTree.fromstring(archive.read((base/ncx["href"]).as_posix())),base,spine,"ncx") if ncx else [],"ncx" if ncx else None)


def epub_spine(path: Path) -> list[str]: return epub_structure(path).spine


def _labels(label, current):
    result=dict(current); found=classify_label(label)
    if found:
        kind,value=found; result[kind]=value
        for lower in _LEVELS[_LEVELS.index(kind)+1:]: result[lower]=None
    return result


def load_epub_sections(path: Path, document: CorpusDocument) -> tuple[list[DocumentSection], int]:
    structure=epub_structure(path); sections=[]; by_target={(x.normalized_href,x.fragment):x for x in structure.navigation}; by_item={x.normalized_href:x for x in structure.navigation if x.fragment is None}; labels={x:None for x in _LEVELS}
    with ZipFile(path) as archive:
        for spine_index,spine_item in enumerate(structure.spine):
            if not spine_item.lower().endswith((".xhtml",".html",".htm")): continue
            try: root=ElementTree.fromstring(archive.read(spine_item))
            except (KeyError,ElementTree.ParseError) as exc: raise ValueError(f"EPUB spine item cannot be parsed: {spine_item}") from exc
            active=by_item.get(spine_item); parts=[]; heading=None; heading_level=None
            def apply(entry):
                nonlocal active,labels
                if entry is not None:
                    active=entry; inherited={x:None for x in _LEVELS}
                    for label in entry.navigation_path: inherited=_labels(label,inherited)
                    labels=inherited
            def flush():
                nonlocal parts
                content="\n".join(parts).strip()
                if content:
                    sections.append(DocumentSection(document.document_id,document.author,document.work,labels["volume"] or document.volume,labels["book"],labels["chapter"],labels["section"],heading,content,path.name,spine_index,spine_item,len(sections),heading_level,active.navigation_path if active else (),active.source_type if active else None,active.normalized_href if active else spine_item,active.fragment if active else None,0,len(content)))
                parts=[]
            apply(active)
            for element in root.iter():
                identifier=element.attrib.get("id") or element.attrib.get("name"); target=by_target.get((spine_item,identifier)) if identifier else None; tag=_name(element.tag)
                if target and parts: flush()
                apply(target)
                if tag not in _TEXT: continue
                value=_text(element)
                if not value: continue
                if tag in _HEADINGS:
                    if parts: flush()
                    heading=value; heading_level=int(tag[1]); labels=_labels(value,labels)
                else: parts.append(value)
            flush()
    return sections,len(structure.spine)
