import hashlib

from .cleaners import clean_text
from .models import TextChunk


def fixed_window(text: str, metadata: dict[str, object], size: int = 4000, overlap: int = 600, strategy: str = "fixed_window") -> list[TextChunk]:
    chunks: list[TextChunk] = []
    step = max(1, size - min(overlap, size // 4))
    for start in range(0, len(text), step):
        part = text[start : start + size].strip()
        if part:
            chunk_id = hashlib.sha256(f"{metadata['source_file']}:{start}:{part}".encode()).hexdigest()[:24]
            chunks.append(TextChunk(chunk_id, part, metadata | {"chunk_strategy": strategy}))
    return chunks


def structure_aware(records: list[dict[str, object]], base_metadata: dict[str, object], size: int = 4000) -> list[TextChunk]:
    chunks = []
    for record in records:
        page = record["page"]
        metadata = base_metadata | {"book": record["book"] or "unknown", "chapter": record["chapter"] or "unknown", "section": "unknown", "page_start": page.page_number, "page_end": page.page_number, "extraction_quality": page.extraction_quality}
        chunks.extend(fixed_window(clean_text(page.text), metadata, size, strategy="structure_aware"))
    return chunks
