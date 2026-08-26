"""Generic EPUB structural records. Labels are observed, never inferred as history."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpubNavigationEntry:
    label: str
    href: str
    normalized_href: str
    fragment: str | None
    depth: int
    parent_path: tuple[str, ...]
    source_type: str
    spine_index: int | None
    order: int

    @property
    def navigation_path(self) -> tuple[str, ...]:
        return self.parent_path + (self.label,)


@dataclass(frozen=True)
class DocumentSection:
    document_id: str
    author: str
    work: str
    volume: str | None
    book: str | None
    chapter: str | None
    section: str | None
    heading: str | None
    text: str
    source_file: str
    spine_index: int
    spine_item: str
    section_index: int = 0
    heading_level: int | None = None
    navigation_path: tuple[str, ...] = ()
    navigation_source: str | None = None
    href: str | None = None
    fragment: str | None = None
    start_offset: int = 0
    end_offset: int = 0
