"""Persistence helpers for parsed sections and chunks."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from paperbase.db.models import Chunk, PaperFile, Section
from paperbase.parsing.chunker import ChunkDraft
from ra.parsing.pdf_parser import Section as ParsedSection


class ParsedPaperStore:
    """Store parse results into canonical Paperbase tables."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def replace_parse_output(
        self,
        *,
        paper_id: str,
        sections: Sequence[ParsedSection],
        chunks: Sequence[ChunkDraft],
        paper_file_id: str,
    ) -> tuple[int, int]:
        self.session.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
        self.session.execute(delete(Section).where(Section.paper_id == paper_id))

        persisted_sections: list[Section] = []
        for index, section in enumerate(sections, start=1):
            next_page_start = sections[index].page_start if index < len(sections) else None
            page_end = next_page_start if next_page_start is not None else None
            persisted = Section(
                paper_id=paper_id,
                title=section.title,
                ordinal=index,
                page_start=section.page_start,
                page_end=page_end,
                text=section.content,
            )
            self.session.add(persisted)
            persisted_sections.append(persisted)

        self.session.flush()
        section_by_ordinal = {section.ordinal: section for section in persisted_sections}

        for chunk in chunks:
            section = section_by_ordinal.get(chunk.section_ordinal)
            self.session.add(
                Chunk(
                    paper_id=paper_id,
                    section_id=section.id if section is not None else None,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    embedding_status="pending",
                )
            )

        file_record = self.session.execute(
            select(PaperFile).where(PaperFile.id == paper_file_id)
        ).scalar_one()
        file_record.parser_status = "parsed"
        self.session.commit()
        return len(persisted_sections), len(chunks)
