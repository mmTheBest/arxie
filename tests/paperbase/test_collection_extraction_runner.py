from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Collection, ExtractionRun, GlossaryTerm
from paperbase.db.repositories import CollectionRepository, PaperFileRepository, PaperRepository
from paperbase.db.session import make_session_factory
from paperbase.parsing.pipeline import PaperParsePipeline
from ra.parsing.pdf_parser import ParsedDocument, Section as ParsedSection


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


def test_collection_extraction_runner_executes_for_collection_members(tmp_path: Path) -> None:
    from paperbase.extract.contracts import GlossaryTermExtraction, StructuredExtractionBundle
    from paperbase.extract.runner import CollectionExtractionRunner
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
            evidence = EvidenceSpanPayload(
                target_type="glossary_term",
                quote_text="AUROC is the evaluation metric used by scLong.",
                page_number=1,
            )
            return StructuredExtractionBundle(
                glossary_terms=[
                    GlossaryTermExtraction(
                        term="AUROC",
                        definition="Area under the receiver operating characteristic curve.",
                        evidence_spans=[evidence],
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
        PaperFileRepository(session).upsert(
            paper_id=paper.id,
            storage_uri=pdf_path.resolve().as_uri(),
            file_kind="pdf",
            mime_type="application/pdf",
            parser_status="pending",
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
        collection_id = collection.id

    PaperParsePipeline(session_factory=session_factory, parser=FakePDFParser()).parse_paper(paper.id)
    summary = CollectionExtractionRunner(
        session_factory=session_factory,
        client=FakeExtractionClient(),
    ).extract_collection(
        collection_id=collection_id,
        schema_payload={"domain": "scRegNet"},
        prompt_version="paperbase-v1",
        schema_version="schema-v1",
    )

    assert summary.collection_id == collection_id
    assert summary.extracted_paper_count == 1
    assert len(summary.extraction_run_ids) == 1

    with session_factory() as session:
        stored_collection = session.execute(
            select(Collection).where(Collection.id == collection_id)
        ).scalar_one()
        glossary_terms = session.execute(select(GlossaryTerm)).scalars().all()
        extraction_runs = session.execute(select(ExtractionRun)).scalars().all()

    assert stored_collection.title == "SamplePapers"
    assert [term.term for term in glossary_terms] == ["AUROC"]
    assert [run.status for run in extraction_runs] == ["completed"]
