from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Dataset, Figure, Method, Metric, ResultRow
from paperbase.db.repositories import PaperFileRepository, PaperRepository
from paperbase.db.session import make_session_factory
from paperbase.parsing.pipeline import PaperParsePipeline
from ra.parsing.pdf_parser import ParsedDocument, Section as ParsedSection
from services.paperbase_api.app import create_app


class FakePDFParser:
    def parse(self, pdf_path: Path) -> ParsedDocument:  # noqa: ARG002
        return ParsedDocument(
            text="Abstract\nscLong improves AUROC.\n\nMethods\nWe evaluate on scRegNetBench.",
            pages=[
                "Abstract\nscLong improves AUROC.",
                "Methods\nWe evaluate on scRegNetBench.",
            ],
            metadata={},
        )

    def extract_sections(self, doc: ParsedDocument) -> list[ParsedSection]:
        return [
            ParsedSection(title="Abstract", content="scLong improves AUROC.", page_start=0),
            ParsedSection(title="Methods", content="We evaluate on scRegNetBench.", page_start=1),
        ]


def test_paperbase_api_exposes_search_fetch_fulltext_figures_and_compare(tmp_path: Path) -> None:
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
            abstract="Long-range gene context modeling.",
            publication_year=2026,
            venue="Nature",
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

    with session_factory() as session:
        dataset = Dataset(paper_id=paper_id, normalized_name="scregnetbench", display_name="scRegNetBench")
        method = Method(paper_id=paper_id, normalized_name="sclong", display_name="scLong")
        metric = Metric(paper_id=paper_id, normalized_name="auroc", display_name="AUROC")
        session.add_all(
            [
                dataset,
                method,
                metric,
                Figure(paper_id=paper_id, page_number=2, figure_label="Figure 1", caption="Model architecture."),
            ]
        )
        session.flush()
        session.add(
            ResultRow(
                paper_id=paper_id,
                dataset_id=dataset.id,
                method_id=method.id,
                metric_id=metric.id,
                value_numeric=0.91,
            )
        )
        session.commit()

    client = TestClient(create_app(session_factory=session_factory))

    search_response = client.get("/api/v1/search/papers", params={"q": "scLong"})
    paper_response = client.get(f"/api/v1/papers/{paper_id}")
    fulltext_response = client.get(f"/api/v1/papers/{paper_id}/fulltext")
    figures_response = client.get(f"/api/v1/papers/{paper_id}/figures")
    compare_response = client.post(
        "/api/v1/compare/results",
        json={"dataset": "scRegNetBench", "metric": "AUROC"},
    )

    assert search_response.status_code == 200
    assert search_response.json()["data"][0]["id"] == paper_id

    assert paper_response.status_code == 200
    assert paper_response.json()["data"]["title"] == "scLong"

    assert fulltext_response.status_code == 200
    assert fulltext_response.json()["data"]["sections"][0]["title"] == "Abstract"

    assert figures_response.status_code == 200
    assert figures_response.json()["data"][0]["figure_label"] == "Figure 1"

    assert compare_response.status_code == 200
    assert compare_response.json()["data"][0]["metric"] == "AUROC"
    assert compare_response.json()["data"][0]["value_numeric"] == 0.91
