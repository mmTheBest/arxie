from __future__ import annotations

from pathlib import Path

import httpx
from sqlalchemy import select

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Chunk, Paper, PaperFile, Section
from paperbase.db.session import make_session_factory
from paperbase.ingest.local_library import import_local_pdf_directory
from paperbase.parsing.pipeline import PaperParsePipeline
from paperbase.storage import StorageResolver
from ra.parsing.pdf_parser import ParsedDocument, Section as ParsedSection


def _write_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n%stub pdf\n")


class FakePDFParser:
    def parse(self, pdf_path: Path) -> ParsedDocument:
        assert pdf_path.exists()
        return ParsedDocument(
            text="Abstract text.\n\nMethods text with enough content to chunk twice.",
            pages=["Abstract text.\nMethods text with enough content to chunk twice."],
            metadata={"title": pdf_path.stem},
        )

    def extract_sections(self, doc: ParsedDocument) -> list[ParsedSection]:
        return [
            ParsedSection(title="Abstract", content="Abstract text.", page_start=0),
            ParsedSection(
                title="Methods",
                content="Methods text with enough content to chunk twice for testing.",
                page_start=0,
            ),
        ]


def test_parse_pipeline_persists_sections_chunks_and_parser_status(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "SamplePapers"
    corpus_dir.mkdir()
    _write_pdf(corpus_dir / "Alpha Paper.pdf")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    import_local_pdf_directory(source_dir=corpus_dir, session_factory=session_factory)

    with session_factory() as session:
        paper = session.execute(select(Paper)).scalar_one()
        file_record = session.execute(select(PaperFile)).scalar_one()

    pipeline = PaperParsePipeline(
        session_factory=session_factory,
        parser=FakePDFParser(),
        max_chunk_characters=24,
        chunk_overlap_characters=0,
    )

    result = pipeline.parse_paper(paper_id=paper.id, paper_file_id=file_record.id)

    assert result.paper_id == paper.id
    assert result.section_count == 2
    assert result.chunk_count >= 3

    with session_factory() as session:
        sections = session.execute(select(Section).order_by(Section.ordinal.asc())).scalars().all()
        chunks = session.execute(select(Chunk).order_by(Chunk.ordinal.asc())).scalars().all()
        stored_file = session.execute(select(PaperFile)).scalar_one()

    assert [section.title for section in sections] == ["Abstract", "Methods"]
    assert all(chunk.paper_id == paper.id for chunk in chunks)
    assert stored_file.parser_status == "parsed"


def test_parse_pipeline_can_download_remote_pdf_before_parsing(tmp_path: Path) -> None:
    pdf_bytes = b"%PDF-1.4\n%downloaded pdf\n"
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = session.execute(select(Paper)).scalar_one_or_none()
        if paper is not None:
            raise AssertionError("expected empty database")

    with session_factory() as session:
        remote_paper = Paper(
            provider="crossref",
            external_id="10.1000/example",
            canonical_title="Remote PDF Paper",
        )
        session.add(remote_paper)
        session.flush()
        session.add(
            PaperFile(
                paper_id=remote_paper.id,
                storage_uri="https://example.org/papers/remote.pdf",
                file_kind="pdf",
                mime_type="application/pdf",
                parser_status="pending",
            )
        )
        session.commit()
        paper_id = remote_paper.id

    class RecordingPDFParser(FakePDFParser):
        def parse(self, pdf_path: Path) -> ParsedDocument:
            assert pdf_path.exists()
            assert pdf_path.read_bytes() == pdf_bytes
            return super().parse(pdf_path)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=pdf_bytes))
    client = httpx.Client(transport=transport)
    resolver = StorageResolver(client=client, cache_dir=tmp_path / "downloads")

    pipeline = PaperParsePipeline(
        session_factory=session_factory,
        parser=RecordingPDFParser(),
        storage_resolver=resolver,
    )

    result = pipeline.parse_paper(paper_id=paper_id)

    assert result.paper_id == paper_id
    downloaded_files = list((tmp_path / "downloads").glob("*.pdf"))
    assert downloaded_files


def test_parse_pipeline_uses_text_only_parser_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_dir = tmp_path / "SamplePapers"
    corpus_dir.mkdir()
    _write_pdf(corpus_dir / "Alpha Paper.pdf")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    import_local_pdf_directory(source_dir=corpus_dir, session_factory=session_factory)

    with session_factory() as session:
        paper = session.execute(select(Paper)).scalar_one()

    constructed_with: list[bool] = []

    class RecordingParser(FakePDFParser):
        def __init__(self, *, include_supplemental_tables: bool = True) -> None:
            constructed_with.append(include_supplemental_tables)

    monkeypatch.setattr("paperbase.parsing.pipeline.PDFParser", RecordingParser)

    PaperParsePipeline(session_factory=session_factory).parse_paper(paper_id=paper.id)

    assert constructed_with == [False]


def test_parse_pipeline_removes_nul_bytes_before_persisting_text(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "SamplePapers"
    corpus_dir.mkdir()
    _write_pdf(corpus_dir / "Nul Paper.pdf")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    import_local_pdf_directory(source_dir=corpus_dir, session_factory=session_factory)

    with session_factory() as session:
        paper = session.execute(select(Paper)).scalar_one()

    class NulPDFParser(FakePDFParser):
        def parse(self, pdf_path: Path) -> ParsedDocument:
            return ParsedDocument(
                text="Abstract\x00 text.",
                pages=["Abstract\x00 text."],
                metadata={},
            )

        def extract_sections(self, doc: ParsedDocument) -> list[ParsedSection]:
            return [
                ParsedSection(
                    title="Abstract\x00",
                    content="Text with\x00 a database-hostile control character.",
                    page_start=0,
                )
            ]

    PaperParsePipeline(
        session_factory=session_factory,
        parser=NulPDFParser(),
        max_chunk_characters=120,
        chunk_overlap_characters=0,
    ).parse_paper(paper_id=paper.id)

    with session_factory() as session:
        section = session.execute(select(Section)).scalar_one()
        chunk = session.execute(select(Chunk)).scalar_one()

    assert "\x00" not in section.title
    assert "\x00" not in section.text
    assert "\x00" not in chunk.text
