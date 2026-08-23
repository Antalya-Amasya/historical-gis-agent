from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageDocument:
    source_file: str
    page_number: int
    text: str
    extraction_quality: str
    requires_ocr: bool
    abnormal_character_ratio: float


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)
