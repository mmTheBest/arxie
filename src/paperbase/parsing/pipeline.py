"""Parser pipeline for Paperbase PDFs."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import PaperFile
from paperbase.db.repositories import PaperFileRepository
from paperbase.parsing.chunker import SimpleSectionChunker
from paperbase.parsing.store import ParsedPaperStore
from paperbase.storage import StorageResolver
from ra.parsing.pdf_parser import PDFParser


@dataclass(frozen=True, slots=True)
class PaperParseResult:
    paper_id: str
    paper_file_id: str
    section_count: int
    chunk_count: int
class PaperParsePipeline:
    """Load a stored PDF, parse it, chunk it, and persist the results."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        parser: PDFParser | None = None,
        chunker: SimpleSectionChunker | None = None,
        storage_resolver: StorageResolver | None = None,
        max_chunk_characters: int = 1200,
        chunk_overlap_characters: int = 120,
    ) -> None:
        self.session_factory = session_factory
        self.parser = parser or PDFParser()
        self.storage_resolver = storage_resolver or StorageResolver()
        self.chunker = chunker or SimpleSectionChunker(
            max_characters=max_chunk_characters,
            overlap_characters=chunk_overlap_characters,
        )

    def parse_paper(self, paper_id: str, paper_file_id: str | None = None) -> PaperParseResult:
        with self.session_factory() as session:
            file_record = self._get_pdf_file(session, paper_id, paper_file_id)

        pdf_path = self.storage_resolver.resolve(file_record.storage_uri)
        document = self.parser.parse(pdf_path)
        sections = self.parser.extract_sections(document)
        chunks = self.chunker.chunk_sections(sections)

        with self.session_factory() as session:
            section_count, chunk_count = ParsedPaperStore(session).replace_parse_output(
                paper_id=paper_id,
                sections=sections,
                chunks=chunks,
                paper_file_id=file_record.id,
            )

        return PaperParseResult(
            paper_id=paper_id,
            paper_file_id=file_record.id,
            section_count=section_count,
            chunk_count=chunk_count,
        )

    def _get_pdf_file(self, session: Session, paper_id: str, paper_file_id: str | None) -> PaperFile:
        if paper_file_id is not None:
            statement = select(PaperFile).where(
                PaperFile.id == paper_file_id,
                PaperFile.paper_id == paper_id,
            )
            file_record = session.execute(statement).scalar_one_or_none()
            if file_record is None:
                raise ValueError(f"No PDF file found for paper_id={paper_id} and paper_file_id={paper_file_id}")
            return file_record

        file_records = PaperFileRepository(session).list_for_paper(paper_id=paper_id, file_kind="pdf")
        if not file_records:
            raise ValueError(f"No PDF file registered for paper_id={paper_id}")
        return file_records[0]
