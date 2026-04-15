"""Chunking helpers for parsed paper sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ra.parsing.pdf_parser import Section as ParsedSection


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    section_ordinal: int
    ordinal: int
    text: str
    token_count: int


class SimpleSectionChunker:
    """Split parsed sections into small text chunks."""

    def __init__(self, max_characters: int = 1200, overlap_characters: int = 120) -> None:
        self.max_characters = max(1, max_characters)
        self.overlap_characters = max(0, min(overlap_characters, self.max_characters - 1))

    def chunk_sections(self, sections: Sequence[ParsedSection]) -> list[ChunkDraft]:
        chunks: list[ChunkDraft] = []
        ordinal = 1
        for section_index, section in enumerate(sections, start=1):
            content = (section.content or "").strip()
            if not content:
                continue
            paragraphs = [part.strip() for part in content.split("\n\n") if part.strip()] or [content]
            for paragraph in paragraphs:
                if not paragraph:
                    continue
                start = 0
                while start < len(paragraph):
                    end = min(len(paragraph), start + self.max_characters)
                    text = paragraph[start:end].strip()
                    if not text:
                        break
                    chunks.append(
                        ChunkDraft(
                            section_ordinal=section_index,
                            ordinal=ordinal,
                            text=text,
                            token_count=len(text.split()),
                        )
                    )
                    ordinal += 1
                    if end >= len(paragraph):
                        break
                    start = max(end - self.overlap_characters, start + 1)
        return chunks
