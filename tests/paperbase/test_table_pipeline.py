from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from paperbase.db.bootstrap import initialize_database
from paperbase.db.repositories import PaperFileRepository, PaperRepository
from paperbase.db.session import make_session_factory


def test_table_pipeline_persists_placeholder_table_metadata(tmp_path: Path) -> None:
    from paperbase.db.models import TableArtifact
    from paperbase.tables.models import TableCandidate
    from paperbase.tables.pipeline import TableExtractionPipeline

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

    pipeline = TableExtractionPipeline(
        session_factory=session_factory,
        extractor=lambda pdf_path: [  # noqa: ARG005
            TableCandidate(
                page_number=3,
                table_label="Table 1",
                caption="Benchmark comparison across scRegNet methods.",
                structured_payload_json={"columns": ["Method", "AUROC"], "rows": 4},
            )
        ],
    )
    result = pipeline.extract_and_store(paper_id)

    assert result.paper_id == paper_id
    assert result.table_count == 1

    with session_factory() as session:
        tables = session.execute(select(TableArtifact)).scalars().all()

    assert len(tables) == 1
    assert tables[0].table_label == "Table 1"
    assert tables[0].caption == "Benchmark comparison across scRegNet methods."
    assert tables[0].structured_payload_json == {"columns": ["Method", "AUROC"], "rows": 4}
