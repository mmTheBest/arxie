from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Collection, CollectionPaper, Paper, PaperFile
from paperbase.db.session import make_session_factory
from paperbase.ingest.local_library import import_local_pdf_directory


def _write_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n%stub pdf\n")


def test_import_local_pdf_directory_creates_collection_papers_and_files(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "SamplePapers"
    corpus_dir.mkdir()
    _write_pdf(corpus_dir / "Alpha Paper.pdf")
    _write_pdf(corpus_dir / "Beta Paper.pdf")
    (corpus_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    result = import_local_pdf_directory(
        source_dir=corpus_dir,
        session_factory=session_factory,
        owner_id="local-user",
    )

    assert result.collection_title == "SamplePapers"
    assert result.total_pdf_files == 2
    assert result.imported_papers == 2
    assert result.reused_papers == 0

    with session_factory() as session:
        papers = session.execute(select(Paper).order_by(Paper.canonical_title.asc())).scalars().all()
        files = session.execute(select(PaperFile).order_by(PaperFile.storage_uri.asc())).scalars().all()
        collections = session.execute(select(Collection)).scalars().all()
        memberships = session.execute(select(CollectionPaper)).scalars().all()

    assert [paper.canonical_title for paper in papers] == ["Alpha Paper", "Beta Paper"]
    assert len(files) == 2
    assert len(collections) == 1
    assert len(memberships) == 2
    assert all(file_record.file_kind == "pdf" for file_record in files)


def test_import_local_pdf_directory_is_idempotent_for_same_folder(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "SamplePapers"
    corpus_dir.mkdir()
    _write_pdf(corpus_dir / "Alpha Paper.pdf")
    _write_pdf(corpus_dir / "Beta Paper.pdf")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    first = import_local_pdf_directory(source_dir=corpus_dir, session_factory=session_factory)
    second = import_local_pdf_directory(source_dir=corpus_dir, session_factory=session_factory)

    assert first.imported_papers == 2
    assert second.imported_papers == 0
    assert second.reused_papers == 2
    assert second.collection_id == first.collection_id

    with session_factory() as session:
        paper_count = session.execute(select(Paper)).scalars().all()
        file_count = session.execute(select(PaperFile)).scalars().all()
        membership_count = session.execute(select(CollectionPaper)).scalars().all()

    assert len(paper_count) == 2
    assert len(file_count) == 2
    assert len(membership_count) == 2
