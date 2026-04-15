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

    def __init__(self, max_words: int = 120) -> None:
        self.max_words = max(1, max_words)

    def chunk_sections(self, sections: Sequence[ParsedSection]) -> list[ChunkDraft]:
        chunks: list[ChunkDraft] = []
        ordinal = 1
        for section_index, section in enumerate(sections, start=1):
            content = (section.content or "").strip()
            if not content:
                continue
            paragraphs = [part.strip() for part in content.split("\n\n") if part.strip()] or [content]
            for paragraph in paragraphs:
                words = paragraph.split()
                if not words:
                    continue
                for start in range(0, len(words), self.max_words):
                    window = words[start : start + self.max_words]
                    text = " ".join(window).strip()
                    if not text:
                        continue
                    chunks.append(
                        ChunkDraft(
                            section_ordinal=section_index,
                            ordinal=ordinal,
                            text=text,
                            token_count=len(window),
                        )
                    )
                    ordinal += 1
        return chunks
