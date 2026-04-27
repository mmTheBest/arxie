from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import (
    Dataset,
    EngineeringTrick,
    EvidenceSpan,
    ExtractionRun,
    Finding,
    GlossaryTerm,
    Limitation,
    Method,
    Metric,
    ResultRow,
)
from paperbase.db.repositories import PaperFileRepository, PaperRepository
from paperbase.db.session import make_session_factory
from paperbase.parsing.pipeline import PaperParsePipeline
from ra.parsing.pdf_parser import ParsedDocument, Section as ParsedSection


class FakePDFParser:
    def parse(self, pdf_path: Path) -> ParsedDocument:  # noqa: ARG002
        return ParsedDocument(
            text=(
                "Abstract\nscLong improves AUROC.\n\n"
                "Methods\nWe evaluate on scRegNetBench using AUROC."
            ),
            pages=[
                "Abstract\nscLong improves AUROC.",
                "Methods\nWe evaluate on scRegNetBench using AUROC.",
            ],
            metadata={},
        )

    def extract_sections(self, doc: ParsedDocument) -> list[ParsedSection]:
        return [
            ParsedSection(title="Abstract", content="scLong improves AUROC.", page_start=0),
            ParsedSection(
                title="Methods",
                content="We evaluate on scRegNetBench using AUROC.",
                page_start=1,
            ),
        ]


def test_extraction_pipeline_persists_structured_entities_and_evidence(tmp_path: Path) -> None:
    from paperbase.extract.contracts import GlossaryTermExtraction, StructuredExtractionBundle
    from paperbase.extract.pipeline import PaperExtractionPipeline
    from paperbase.schemas.extraction import (
        DatasetExtraction,
        EngineeringTrickExtraction,
        EvidenceSpanPayload,
        FindingExtraction,
        LimitationExtraction,
        MethodExtraction,
        MetricExtraction,
        ResultExtraction,
    )

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
                target_type="result_row",
                quote_text="scLong improves AUROC to 0.91 on scRegNetBench.",
                page_number=2,
            )
            return StructuredExtractionBundle(
                datasets=[DatasetExtraction(display_name="scRegNetBench", evidence_spans=[evidence])],
                methods=[MethodExtraction(display_name="scLong", evidence_spans=[evidence])],
                metrics=[MetricExtraction(display_name="AUROC", evidence_spans=[evidence])],
                results=[
                    ResultExtraction(
                        dataset_name="scRegNetBench",
                        method_name="scLong",
                        metric_name="AUROC",
                        value_numeric=0.91,
                        evidence_spans=[evidence],
                    )
                ],
                findings=[
                    FindingExtraction(
                        statement="scLong improves AUROC on scRegNetBench.",
                        evidence_spans=[evidence],
                    )
                ],
                limitations=[
                    LimitationExtraction(
                        statement="The benchmark only covers a small number of curated cell states.",
                        evidence_spans=[evidence],
                    )
                ],
                glossary_terms=[
                    GlossaryTermExtraction(
                        term="AUROC",
                        definition="Area under the receiver operating characteristic curve.",
                        evidence_spans=[evidence],
                    )
                ],
                engineering_tricks=[
                    EngineeringTrickExtraction(
                        title="Context pretraining",
                        description="Pretrain with long-range gene context windows.",
                        evidence_spans=[evidence],
                    )
                ],
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
        paper_id = paper.id

    PaperParsePipeline(session_factory=session_factory, parser=FakePDFParser()).parse_paper(paper_id)
    result = PaperExtractionPipeline(
        session_factory=session_factory,
        client=FakeExtractionClient(),
    ).extract_paper(
        paper_id=paper_id,
        schema_payload={"domain": "scRegNet"},
        prompt_version="paperbase-v1",
        schema_version="schema-v1",
    )

    assert result.paper_id == paper_id
    assert result.run_status == "completed"
    assert result.result_count == 1

    with session_factory() as session:
        assert session.execute(select(ExtractionRun)).scalar_one().model_name == "fake-extractor"
        assert session.execute(select(Dataset)).scalar_one().display_name == "scRegNetBench"
        assert session.execute(select(Method)).scalar_one().display_name == "scLong"
        assert session.execute(select(Metric)).scalar_one().display_name == "AUROC"
        assert session.execute(select(ResultRow)).scalar_one().value_numeric == 0.91
        assert session.execute(select(Finding)).scalar_one().statement.startswith("scLong improves")
        assert session.execute(select(Limitation)).scalar_one().statement.startswith("The benchmark only covers")
        assert session.execute(select(GlossaryTerm)).scalar_one().term == "AUROC"
        assert session.execute(select(EngineeringTrick)).scalar_one().title == "Context pretraining"
        evidence_spans = session.execute(select(EvidenceSpan)).scalars().all()

    assert len(evidence_spans) >= 5
