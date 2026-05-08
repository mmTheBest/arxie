from __future__ import annotations

from datetime import UTC, datetime, timedelta
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import fitz
import paperbase.db.models as db_models
from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import (
    Chunk,
    Dataset,
    ExtractionRun,
    Figure,
    GlossaryTerm,
    Method,
    PaperFile,
    ResearchDesignElement,
    ResearchArtifact,
    ResearchMessage,
    ResearchThread,
    ResultRow,
    Section,
    TableArtifact,
)
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from paperbase.ingest.models import CanonicalPaperSeed
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


class CountingPDFParser(FakePDFParser):
    def __init__(self) -> None:
        self.parse_calls = 0

    def parse(self, pdf_path: Path) -> ParsedDocument:
        self.parse_calls += 1
        return super().parse(pdf_path)


class FailingPDFParser(FakePDFParser):
    def parse(self, pdf_path: Path) -> ParsedDocument:
        raise RuntimeError("bad pdf text")


class FakeProviderResolver:
    def __init__(self, seeds: dict[tuple[str, str], CanonicalPaperSeed]) -> None:
        self.seeds = seeds

    def fetch_identifier(self, *, kind: str, value: str) -> CanonicalPaperSeed:
        return self.seeds[(kind, value)]


def _write_caption_pdf(pdf_path: Path, *lines: str) -> None:
    document = fitz.open()
    page = document.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 16
    document.save(pdf_path)
    document.close()


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


def test_worker_recovers_stale_running_jobs_before_claiming_new_work(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    backend = FakeSearchBackend()

    with session_factory() as session:
        stale_job = background_job_model(
            job_type="search_reindex",
            status="running",
            payload_json={},
            started_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=1),
            attempt_count=1,
        )
        pending_job = background_job_model(
            job_type="search_reindex",
            status="pending",
            payload_json={},
        )
        session.add_all([stale_job, pending_job])
        session.commit()
        stale_job_id = stale_job.id
        pending_job_id = pending_job.id

    worker = worker_cls(
        session_factory=session_factory,
        search_backend=backend,
        stale_running_seconds=60,
    )
    processed_job_id = worker.process_next_job()

    assert processed_job_id == stale_job_id

    with session_factory() as session:
        stored_stale_job = session.get(background_job_model, stale_job_id)
        stored_pending_job = session.get(background_job_model, pending_job_id)

    assert stored_stale_job is not None
    assert stored_stale_job.status == "completed"
    assert stored_stale_job.attempt_count == 2
    assert stored_pending_job is not None
    assert stored_pending_job.status == "pending"


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


def test_worker_executes_research_agent_run_job(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="/tmp/benchmark.pdf",
            canonical_title="Benchmark Paper",
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Local field-specific corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)
        section = Section(
            paper_id=paper.id,
            title="Methods",
            ordinal=1,
            text="We compare against baseline models using AUROC and ablation studies.",
        )
        dataset = Dataset(
            paper_id=paper.id,
            normalized_name="scregnetbench",
            display_name="scRegNetBench",
        )
        method = Method(
            paper_id=paper.id,
            normalized_name="graph_model",
            display_name="Graph model",
        )
        metric = Method(
            paper_id=paper.id,
            normalized_name="baseline",
            display_name="Baseline method",
        )
        session.add_all([section, dataset, method, metric])
        session.flush()
        session.add(
            ResultRow(
                paper_id=paper.id,
                dataset_id=dataset.id,
                method_id=method.id,
                metric_id=metric.id,
                value_numeric=0.91,
                notes="Ablation validates graph prior.",
            )
        )
        session.add(
            ResearchDesignElement(
                paper_id=paper.id,
                element_type="ablation",
                title="Graph-prior ablation",
                description="Remove the graph prior to isolate its effect.",
            )
        )
        thread = ResearchThread(
            owner_id="local-user",
            title="Experiment design",
            collection_id=collection.id,
            selected_paper_ids_json=[paper.id],
        )
        session.add(thread)
        session.flush()
        user_message = ResearchMessage(
            thread_id=thread.id,
            role="user",
            content="Design an experiment with baselines and ablations.",
        )
        artifact = ResearchArtifact(
            thread_id=thread.id,
            collection_id=collection.id,
            artifact_type="experiment_plan",
            title="Experiment plan",
            status="pending",
            input_payload_json={"message": user_message.content},
        )
        session.add_all([user_message, artifact])
        session.flush()
        job = background_job_model(
            job_type="research_agent_run",
            payload_json={
                "thread_id": thread.id,
                "message_id": user_message.id,
                "artifact_id": artifact.id,
                "collection_id": collection.id,
                "user_message": user_message.content,
                "artifact_type": "experiment_plan",
                "selected_paper_ids": [paper.id],
            },
        )
        session.add(job)
        session.commit()
        job_id = job.id
        artifact_id = artifact.id

    worker = worker_cls(session_factory=session_factory)
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id
    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        stored_artifact = session.get(ResearchArtifact, artifact_id)
        assistant_messages = session.query(ResearchMessage).filter_by(role="assistant").all()

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_artifact is not None
    assert stored_artifact.status == "completed"
    assert stored_artifact.output_payload_json["artifact_type"] == "experiment_plan"
    assert "baselines" in stored_artifact.output_payload_json
    assert "general_methodology" in stored_artifact.output_payload_json
    assert stored_artifact.evidence_payload_json["papers"][0]["title"] == "Benchmark Paper"
    assert stored_artifact.evidence_payload_json["papers"][0]["research_design_elements"][0]["element_type"] == "ablation"
    assert len(assistant_messages) == 1


def test_worker_collection_extract_honors_selected_paper_ids(tmp_path: Path) -> None:
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

    first_pdf_path = tmp_path / "first.pdf"
    second_pdf_path = tmp_path / "second.pdf"
    first_pdf_path.write_bytes(b"%PDF-1.4\n%first pdf\n")
    second_pdf_path.write_bytes(b"%PDF-1.4\n%second pdf\n")
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        first_paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(first_pdf_path),
            canonical_title="First Paper",
        )
        second_paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(second_pdf_path),
            canonical_title="Second Paper",
        )
        session.add_all(
            [
                PaperFile(
                    paper_id=first_paper.id,
                    storage_uri=first_pdf_path.resolve().as_uri(),
                    file_kind="pdf",
                    mime_type="application/pdf",
                    parser_status="pending",
                ),
                PaperFile(
                    paper_id=second_paper.id,
                    storage_uri=second_pdf_path.resolve().as_uri(),
                    file_kind="pdf",
                    mime_type="application/pdf",
                    parser_status="pending",
                ),
            ]
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Local field-specific corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=first_paper.id, position=1)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=second_paper.id, position=2)
        job = background_job_model(
            job_type="collection_extract",
            payload_json={
                "collection_id": collection.id,
                "schema_payload": {"domain": "scRegNet"},
                "prompt_version": "paperbase-v1",
                "schema_version": "schema-v1",
                "extraction_profile_id": None,
                "limit": None,
                "paper_ids": [second_paper.id],
            },
        )
        session.add(job)
        session.commit()
        job_id = job.id
        first_paper_id = first_paper.id
        second_paper_id = second_paper.id
        collection_id = collection.id

    PaperParsePipeline(session_factory=session_factory, parser=FakePDFParser()).parse_paper(first_paper_id)
    PaperParsePipeline(session_factory=session_factory, parser=FakePDFParser()).parse_paper(second_paper_id)

    worker = worker_cls(
        session_factory=session_factory,
        extraction_client_factory=lambda: FakeExtractionClient(),
    )
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        extracted_paper_ids = {
            run.paper_id for run in session.query(ExtractionRun).filter(ExtractionRun.status == "completed").all()
        }

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json == {
        "collection_id": collection_id,
        "extracted_paper_count": 1,
        "skipped_paper_ids": [],
    }
    assert first_paper_id not in extracted_paper_ids
    assert second_paper_id in extracted_paper_ids


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
        "collection_id": collections[0].id,
        "collection_title": "Sample Library",
        "total_pdf_files": 2,
        "imported_papers": 2,
        "reused_papers": 0,
    }
    assert len(collections) == 1
    assert collections[0].title == "Sample Library"


def test_worker_executes_provider_identifier_ingest_job(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        job = background_job_model(
            job_type="provider_identifier_ingest",
            payload_json={
                "owner_id": "local-user",
                "collection_title": "Provider Import",
                "collection_description": "Imported from identifiers.",
                "identifiers": [
                    {"kind": "arxiv", "value": "2503.01682v1"},
                    {"kind": "doi", "value": "10.1038/example"},
                ],
            },
        )
        session.add(job)
        session.commit()
        job_id = job.id

    worker = worker_cls(
        session_factory=session_factory,
        provider_resolver=FakeProviderResolver(
            {
                ("arxiv", "2503.01682v1"): CanonicalPaperSeed(
                    provider="arxiv",
                    external_id="2503.01682v1",
                    canonical_title="scLong",
                    abstract="Long-range gene context.",
                    publication_year=2025,
                    venue="arXiv",
                    doi="10.1038/example",
                    arxiv_id="2503.01682v1",
                    pdf_url="https://arxiv.org/pdf/2503.01682v1.pdf",
                    authors=["Alice Smith"],
                    source_payload={"source": "arxiv"},
                ),
                ("doi", "10.1038/example"): CanonicalPaperSeed(
                    provider="crossref",
                    external_id="10.1038/example",
                    canonical_title="scLong journal record",
                    abstract="Journal abstract.",
                    publication_year=2025,
                    venue="Nature",
                    doi="10.1038/example",
                    pdf_url="https://example.org/scLong.pdf",
                    authors=["Alice Smith", "Bob Jones"],
                    source_payload={"source": "crossref"},
                ),
            }
        ),
    )
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        collections = CollectionRepository(session).list_collections()
        papers = session.query(db_models.Paper).all()
        sources = session.query(db_models.PaperSource).all()

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert len(collections) == 1
    assert len(papers) == 1
    assert len(sources) == 2
    assert stored_job.result_json == {
        "collection_id": collections[0].id,
        "collection_title": "Provider Import",
        "requested_count": 2,
        "imported_papers": 1,
        "reused_papers": 1,
        "paper_ids": [papers[0].id, papers[0].id],
        "skipped_identifiers": [],
    }


def test_worker_executes_paper_metadata_refresh_job(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="crossref",
            external_id="10.1038/example",
            canonical_title="Original Title",
            abstract="Original abstract.",
            publication_year=2024,
            venue="Nature",
            doi="10.1038/example",
            authors=["Alice Smith"],
        )
        session.add(
            db_models.PaperSource(
                paper_id=paper.id,
                provider="crossref",
                provider_record_id="10.1038/example",
                is_primary=True,
                source_payload={"source": "crossref"},
            )
        )
        job = background_job_model(
            job_type="paper_metadata_refresh",
            payload_json={"paper_ids": [paper.id]},
        )
        session.add(job)
        session.commit()
        job_id = job.id
        paper_id = paper.id

    worker = worker_cls(
        session_factory=session_factory,
        provider_resolver=FakeProviderResolver(
            {
                ("doi", "10.1038/example"): CanonicalPaperSeed(
                    provider="crossref",
                    external_id="10.1038/example",
                    canonical_title="Refreshed Title",
                    abstract="Refreshed abstract.",
                    publication_year=2025,
                    venue="Nature",
                    doi="10.1038/example",
                    pdf_url="https://example.org/refreshed.pdf",
                    authors=["Alice Smith", "Bob Jones"],
                    source_payload={"source": "crossref"},
                    raw_metadata={"publisher": "Nature"},
                )
            }
        ),
    )
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        refreshed = session.get(db_models.Paper, paper_id)
        files = session.query(db_models.PaperFile).all()

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json == {
        "requested_count": 1,
        "refreshed_papers": 1,
        "skipped_paper_ids": [],
    }
    assert refreshed is not None
    assert refreshed.canonical_title == "Refreshed Title"
    assert refreshed.publication_year == 2025
    assert len(files) == 1
    assert files[0].storage_uri == "https://example.org/refreshed.pdf"


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
        "figure_count": 0,
        "table_count": 0,
    }
    assert len(sections) == 1
    assert len(chunks) == 1
    assert file_record.parser_status == "parsed"


def test_worker_collection_parse_skips_already_parsed_papers_by_default(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    parsed_pdf_path = tmp_path / "already-parsed.pdf"
    pending_pdf_path = tmp_path / "pending.pdf"
    parsed_pdf_path.write_bytes(b"%PDF-1.4\n%parsed pdf\n")
    pending_pdf_path.write_bytes(b"%PDF-1.4\n%pending pdf\n")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        parsed_paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(parsed_pdf_path),
            canonical_title="Already Parsed",
        )
        pending_paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(pending_pdf_path),
            canonical_title="Pending Parse",
        )
        session.add_all(
            [
                PaperFile(
                    paper_id=parsed_paper.id,
                    storage_uri=parsed_pdf_path.resolve().as_uri(),
                    file_kind="pdf",
                    mime_type="application/pdf",
                    parser_status="parsed",
                ),
                PaperFile(
                    paper_id=pending_paper.id,
                    storage_uri=pending_pdf_path.resolve().as_uri(),
                    file_kind="pdf",
                    mime_type="application/pdf",
                    parser_status="pending",
                ),
                Section(
                    paper_id=parsed_paper.id,
                    title="Methods",
                    ordinal=1,
                    page_start=0,
                    page_end=None,
                    text="Existing parsed text.",
                ),
            ]
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Curated field-specific corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=parsed_paper.id, position=1)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=pending_paper.id, position=2)
        job = background_job_model(
            job_type="collection_parse",
            payload_json={"collection_id": collection.id, "limit": None},
        )
        session.add(job)
        session.commit()
        job_id = job.id
        collection_id = collection.id

    parser = CountingPDFParser()
    worker = worker_cls(session_factory=session_factory, parser_factory=lambda: parser)
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id
    assert parser.parse_calls == 1

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        sections = session.query(Section).order_by(Section.paper_id.asc(), Section.ordinal.asc()).all()
        file_statuses = {
            file_record.paper_id: file_record.parser_status
            for file_record in session.query(PaperFile).all()
        }

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json == {
        "collection_id": collection_id,
        "parsed_paper_count": 1,
        "skipped_paper_ids": [],
        "section_count": 1,
        "chunk_count": 1,
        "figure_count": 0,
        "table_count": 0,
    }
    assert len(sections) == 2
    assert set(file_statuses.values()) == {"parsed"}


def test_worker_collection_parse_honors_selected_paper_ids(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    first_pdf_path = tmp_path / "first.pdf"
    second_pdf_path = tmp_path / "second.pdf"
    first_pdf_path.write_bytes(b"%PDF-1.4\n%first pdf\n")
    second_pdf_path.write_bytes(b"%PDF-1.4\n%second pdf\n")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        first_paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(first_pdf_path),
            canonical_title="First Pending Paper",
        )
        second_paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(second_pdf_path),
            canonical_title="Second Pending Paper",
        )
        session.add_all(
            [
                PaperFile(
                    paper_id=first_paper.id,
                    storage_uri=first_pdf_path.resolve().as_uri(),
                    file_kind="pdf",
                    mime_type="application/pdf",
                    parser_status="pending",
                ),
                PaperFile(
                    paper_id=second_paper.id,
                    storage_uri=second_pdf_path.resolve().as_uri(),
                    file_kind="pdf",
                    mime_type="application/pdf",
                    parser_status="pending",
                ),
            ]
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="Curated field-specific corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=first_paper.id, position=1)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=second_paper.id, position=2)
        job = background_job_model(
            job_type="collection_parse",
            payload_json={
                "collection_id": collection.id,
                "limit": None,
                "paper_ids": [second_paper.id],
            },
        )
        session.add(job)
        session.commit()
        job_id = job.id
        first_paper_id = first_paper.id
        second_paper_id = second_paper.id
        collection_id = collection.id

    parser = CountingPDFParser()
    worker = worker_cls(session_factory=session_factory, parser_factory=lambda: parser)
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id
    assert parser.parse_calls == 1

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        parsed_paper_ids = {section.paper_id for section in session.query(Section).all()}

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json["collection_id"] == collection_id
    assert stored_job.result_json["parsed_paper_count"] == 1
    assert first_paper_id not in parsed_paper_ids
    assert second_paper_id in parsed_paper_ids


def test_worker_collection_parse_logs_skipped_paper_errors(tmp_path: Path, caplog) -> None:
    background_job_model, worker_cls = _load_worker_class()

    pdf_path = tmp_path / "bad.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%bad pdf\n")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id=str(pdf_path),
            canonical_title="Bad PDF",
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
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id, position=1)
        job = background_job_model(
            job_type="collection_parse",
            payload_json={"collection_id": collection.id, "limit": None},
        )
        session.add(job)
        session.commit()
        paper_id = paper.id
        job_id = job.id

    worker = worker_cls(session_factory=session_factory, parser_factory=FailingPDFParser)

    with caplog.at_level("WARNING", logger="paperbase.parsing.runner"):
        processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id
    assert "Skipping paper after parse failure" in caplog.text
    assert paper_id in caplog.text
    assert "bad pdf text" in caplog.text


def test_worker_collection_parse_populates_figure_and_table_artifacts_when_pdf_has_captions(tmp_path: Path) -> None:
    background_job_model, worker_cls = _load_worker_class()

    pdf_path = tmp_path / "captioned.pdf"
    _write_caption_pdf(
        pdf_path,
        "Methods",
        "Figure 1. Model architecture for gene regulatory inference.",
        "Table 1. Benchmark comparison across scRegNet methods.",
    )

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
            payload_json={"collection_id": collection.id, "limit": None},
        )
        session.add(job)
        session.commit()
        job_id = job.id
        collection_id = collection.id

    worker = worker_cls(session_factory=session_factory)
    processed_job_id = worker.process_next_job()

    assert processed_job_id == job_id

    with session_factory() as session:
        stored_job = session.get(background_job_model, job_id)
        figures = session.query(Figure).all()
        tables = session.query(TableArtifact).all()

    assert stored_job is not None
    assert stored_job.status == "completed"
    assert stored_job.result_json == {
        "collection_id": collection_id,
        "parsed_paper_count": 1,
        "skipped_paper_ids": [],
        "section_count": 1,
        "chunk_count": 1,
        "figure_count": 1,
        "table_count": 1,
    }
    assert len(figures) == 1
    assert figures[0].figure_label == "Figure 1"
    assert len(tables) == 1
    assert tables[0].table_label == "Table 1"
