from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import paperbase.db.models as db_models
from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Chunk, ExtractionRun, Figure, GlossaryTerm, PaperFile, Section
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from paperbase.parsing.pipeline import PaperParsePipeline
from ra.parsing.pdf_parser import ParsedDocument, Section as ParsedSection


def _load_worker_class():
    spec = find_spec("services.paperbase_worker.runtime")
    assert spec is not None, "services.paperbase_worker.runtime must exist."

    background_job_model = getattr(db_models, "BackgroundJob", None)
    assert background_job_model is not None, "BackgroundJob model must be defined."

    module = import_module("services.paperbase_worker.runtime")
    worker_cls = getattr(module, "PaperbaseWorker", None)
    assert worker_cls is not None, "PaperbaseWorker must be defined."
    return background_job_model, worker_cls


class FakeSearchBackend:
    def __init__(self) -> None:
        self.index_calls: list[str] = []
        self.bulk_calls: list[tuple[str, int]] = []

    def ensure_index(self, index_name: str, template: dict[str, object]) -> None:  # noqa: ARG002
        self.index_calls.append(index_name)

    def bulk_index(self, index_name: str, documents: list[dict[str, object]]) -> None:
        self.bulk_calls.append((index_name, len(documents)))

    def search(self, index_name: str, query: dict[str, object], size: int) -> list[dict[str, object]]:  # noqa: ARG002
        return []


class FakePDFParser:
    def parse(self, pdf_path: Path) -> ParsedDocument:  # noqa: ARG002
        return ParsedDocument(
            text="Methods\nscLong improves AUROC on scRegNetBench.",
            pages=["Methods\nscLong improves AUROC on scRegNetBench."],
            metadata={},
        )

    def extract_sections(self, doc: ParsedDocument) -> list[ParsedSection]:
        return [
            ParsedSection(
                title="Methods",
                content="scLong improves AUROC on scRegNetBench.",
                page_start=0,
            )
        ]


def test_worker_executes_search_reindex_job(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    backend = FakeSearchBackend()

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
            abstract="Long-range gene context modeling.",
            publication_year=2026,
            venue="Nature",
        )
        session.add_all(
            [
                PaperFile(
                    paper_id=paper.id,
                    storage_uri="file:///tmp/scLong.pdf",
                    file_kind="pdf",
                    mime_type="application/pdf",
                    parser_status="parsed",
                ),
                Section(
                    paper_id=paper.id,
                    title="Methods",
                    ordinal=1,
                    text="We evaluate AUROC on scRegNetBench.",
                ),
                Chunk(
                    paper_id=paper.id,
                    section_id=None,
                    ordinal=1,
                    text="We evaluate AUROC on scRegNetBench.",
                ),
                Figure(
                    paper_id=paper.id,
                    page_number=2,
                    figure_label="Figure 1",
                    caption="Model architecture.",
                ),
                ExtractionRun(
                    paper_id=paper.id,
                    model_name="fake-extractor",
                    prompt_version="paperbase-v1",
                    schema_version="schema-v1",
                    status="completed",
                ),
            ]
        )
        job = background_job_model(job_type="search_reindex", payload_json={})
        session.add(job)
        session.commit()
        job_id = job.id

    worker = worker_cls(session_factory=session_factory, search_backend=backend)
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id
    assert backend.index_calls == [
        "paperbase-papers",
        "paperbase-chunks",
        "paperbase-figures",
        "paperbase-tables",
    ]
    assert backend.bulk_calls == [
        ("paperbase-papers", 1),
        ("paperbase-chunks", 1),
        ("paperbase-figures", 1),
        ("paperbase-tables", 0),
    ]

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json == {"indexed": {"papers": 1, "chunks": 1, "figures": 1, "tables": 0}}
    assert stored_job.error_message is None


def test_worker_executes_collection_extract_job(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    from paperbase.extract.contracts import GlossaryTermExtraction, StructuredExtractionBundle
    from paperbase.schemas.extraction import EvidenceSpanPayload

    class FakeExtractionClient:
        model_name = "fake-extractor"

        def extract(
            self,
            *,
            paper_text: str,
            schema_payload: dict[str, object],
        ) -> StructuredExtractionBundle:
            assert "scLong" in paper_text
            assert schema_payload["domain"] == "scRegNet"
            return StructuredExtractionBundle(
                glossary_terms=[
                    GlossaryTermExtraction(
                        term="AUROC",
                        definition="Area under the receiver operating characteristic curve.",
                        evidence_spans=[
                            EvidenceSpanPayload(
                                target_type="glossary_term",
                                quote_text="AUROC is the evaluation metric used by scLong.",
                                page_number=1,
                            )
                        ],
                    )
                ]
            )

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub pdf\n")
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(pdf_path),
            canonical_title="scLong",
        )
        session.add(
            PaperFile(
                paper_id=paper.id,
                storage_uri=pdf_path.resolve().as_uri(),
                file_kind="pdf",
                mime_type="application/pdf",
                parser_status="pending",
            )
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Local field-specific corpus.",
        )
        CollectionRepository(session).add_paper(
            collection_id=collection.id,
            paper_id=paper.id,
        )
        job = background_job_model(
            job_type="collection_extract",
            payload_json={
                "collection_id": collection.id,
                "schema_payload": {"domain": "scRegNet"},
                "prompt_version": "paperbase-v1",
                "schema_version": "schema-v1",
                "extraction_profile_id": None,
                "limit": None,
            },
        )
        session.add(job)
        session.commit()
        job_id = job.id
        collection_id = collection.id

    PaperParsePipeline(session_factory=session_factory, parser=FakePDFParser()).parse_paper(paper.id)

    worker = worker_cls(
        session_factory=session_factory,
        extraction_client_factory=lambda: FakeExtractionClient(),
    )
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        glossary_terms = session.query(GlossaryTerm).all()

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json == {
        "collection_id": collection_id,
        "extracted_paper_count": 1,
        "skipped_paper_ids": [],
    }
    assert [term.term for term in glossary_terms] == ["AUROC"]


def test_worker_marks_job_failed_when_dependencies_are_missing(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        job = background_job_model(job_type="search_reindex", payload_json={})
        session.add(job)
        session.commit()
        job_id = job.id

    worker = worker_cls(session_factory=session_factory)
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)

    assert stored_job is not None
    assert stored_job.status == "failed"
    assert stored_job.error_message is not None
    assert "search backend" in stored_job.error_message.lower()


def test_worker_executes_local_library_ingest_job(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    source_dir = tmp_path / "library"
    source_dir.mkdir()
    (source_dir / "paper-one.pdf").write_bytes(b"%PDF-1.4\n%stub pdf\n")
    (source_dir / "paper-two.pdf").write_bytes(b"%PDF-1.4\n%stub pdf\n")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        job = background_job_model(
            job_type="local_library_ingest",
            payload_json={
                "source_dir": str(source_dir),
                "owner_id": "local-user",
                "collection_title": "Sample Library",
                "collection_description": "Imported papers.",
            },
        )
        session.add(job)
        session.commit()
        job_id = job.id

    worker = worker_cls(session_factory=session_factory)
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        collections = CollectionRepository(session).list_collections()

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json == {
        "collection_title": "Sample Library",
        "total_pdf_files": 2,
        "imported_papers": 2,
        "reused_papers": 0,
    }
    assert len(collections) == 1
    assert collections[0].title == "Sample Library"


def test_worker_executes_collection_parse_job(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub pdf\n")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(pdf_path),
            canonical_title="scLong",
        )
        session.add(
            PaperFile(
                paper_id=paper.id,
                storage_uri=pdf_path.resolve().as_uri(),
                file_kind="pdf",
                mime_type="application/pdf",
                parser_status="pending",
            )
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Curated field-specific corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)
        job = background_job_model(
            job_type="collection_parse",
            payload_json={
                "collection_id": collection.id,
                "limit": None,
            },
        )
        session.add(job)
        session.commit()
        job_id = job.id
        collection_id = collection.id

    worker = worker_cls(
        session_factory=session_factory,
        parser_factory=lambda: FakePDFParser(),
    )
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        sections = session.query(Section).all()
        chunks = session.query(Chunk).all()
        file_record = session.query(PaperFile).one()

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json == {
        "collection_id": collection_id,
        "parsed_paper_count": 1,
        "skipped_paper_ids": [],
        "section_count": 1,
        "chunk_count": 1,
    }
    assert len(sections) == 1
    assert len(chunks) == 1
    assert file_record.parser_status == "parsed"
