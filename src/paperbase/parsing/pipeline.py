"""Parser pipeline for Paperbase PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import PaperFile
from paperbase.db.repositories import PaperFileRepository
from paperbase.parsing.chunker import SimpleSectionChunker
from paperbase.parsing.store import ParsedPaperStore
from ra.parsing.pdf_parser import PDFParser


@dataclass(frozen=True, slots=True)
class PaperParseResult:
    paper_id: str
    paper_file_id: str
    section_count: int
    chunk_count: int


def _path_from_storage_uri(storage_uri: str) -> Path:
    parsed = urlparse(storage_uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(storage_uri)


class PaperParsePipeline:
    """Load a stored PDF, parse it, chunk it, and persist the results."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        parser: PDFParser | None = None,
        chunker: SimpleSectionChunker | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.parser = parser or PDFParser()
        self.chunker = chunker or SimpleSectionChunker()

    def parse_paper(self, paper_id: str) -> PaperParseResult:
        with self.session_factory() as session:
            file_record = self._get_primary_pdf_file(session, paper_id)

        pdf_path = _path_from_storage_uri(file_record.storage_uri)
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

    def _get_primary_pdf_file(self, session: Session, paper_id: str) -> PaperFile:
        file_records = PaperFileRepository(session).list_for_paper(paper_id=paper_id, file_kind="pdf")
        if not file_records:
            raise ValueError(f"No PDF file registered for paper_id={paper_id}")
        return file_records[0]
