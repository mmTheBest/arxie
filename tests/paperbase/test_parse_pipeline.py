from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Chunk, PaperFile, Section
from paperbase.db.repositories import PaperFileRepository, PaperRepository
from paperbase.db.session import make_session_factory
from paperbase.parsing.pipeline import PaperParsePipeline
from ra.parsing.pdf_parser import ParsedDocument, Section as ParsedSection


class FakePDFParser:
    def parse(self, pdf_path: Path) -> ParsedDocument:  # noqa: ARG002
        return ParsedDocument(
            text=(
                "Abstract\nStructured extraction helps.\n\n"
                "Methods\nWe chunk the document into evidence-backed segments."
            ),
            pages=[
                "Abstract\nStructured extraction helps.",
                "Methods\nWe chunk the document into evidence-backed segments.",
            ],
            metadata={"title": "Parsed paper"},
        )

    def extract_sections(self, doc: ParsedDocument) -> list[ParsedSection]:
        return [
            ParsedSection(title="Abstract", content="Structured extraction helps.", page_start=0),
            ParsedSection(
                title="Methods",
                content="We chunk the document into evidence-backed segments.",
                page_start=1,
            ),
        ]


def test_parse_pipeline_persists_sections_and_chunks_for_a_stored_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub pdf\n")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(pdf_path),
            canonical_title="Sample Paper",
        )
        PaperFileRepository(session).upsert(
            paper_id=paper.id,
            storage_uri=pdf_path.resolve().as_uri(),
            file_kind="pdf",
            mime_type="application/pdf",
            parser_status="pending",
        )
        paper_id = paper.id

    pipeline = PaperParsePipeline(session_factory=session_factory, parser=FakePDFParser())
    result = pipeline.parse_paper(paper_id)

    assert result.paper_id == paper_id
    assert result.section_count == 2
    assert result.chunk_count >= 2

    with session_factory() as session:
        sections = session.execute(select(Section).order_by(Section.ordinal.asc())).scalars().all()
        chunks = session.execute(select(Chunk).order_by(Chunk.ordinal.asc())).scalars().all()
        file_record = session.execute(select(PaperFile)).scalar_one()

    assert [section.title for section in sections] == ["Abstract", "Methods"]
    assert len(chunks) >= 2
    assert all(chunk.paper_id == paper_id for chunk in chunks)
    assert file_record.parser_status == "parsed"
