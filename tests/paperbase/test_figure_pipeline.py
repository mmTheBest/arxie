from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Figure
from paperbase.db.repositories import PaperFileRepository, PaperRepository
from paperbase.db.session import make_session_factory
from paperbase.figures.models import FigureCandidate
from paperbase.figures.pipeline import FigureExtractionPipeline


def test_figure_pipeline_persists_placeholder_figure_metadata(tmp_path: Path) -> None:
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
            parser_status="parsed",
        )
        paper_id = paper.id

    pipeline = FigureExtractionPipeline(
        session_factory=session_factory,
        extractor=lambda pdf_path: [  # noqa: ARG005
            FigureCandidate(
                page_number=2,
                figure_label="Figure 1",
                caption="Model architecture for gene regulatory inference.",
            )
        ],
    )
    result = pipeline.extract_and_store(paper_id)

    assert result.paper_id == paper_id
    assert result.figure_count == 1

    with session_factory() as session:
        figures = session.execute(select(Figure)).scalars().all()

    assert len(figures) == 1
    assert figures[0].figure_label == "Figure 1"
    assert figures[0].caption == "Model architecture for gene regulatory inference."
