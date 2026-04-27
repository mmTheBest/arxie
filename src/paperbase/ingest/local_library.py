"""Import local PDF directories into Paperbase."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.repositories import CollectionRepository, PaperFileRepository, PaperRepository


@dataclass(frozen=True, slots=True)
class LocalLibraryImportResult:
    collection_id: str
    collection_title: str
    total_pdf_files: int
    imported_papers: int
    reused_papers: int


def _canonical_title_for_path(pdf_path: Path) -> str:
    return pdf_path.stem.replace("_", " ").strip()


def _content_hash_for_path(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def _paper_object_key(*, paper_id: str, content_hash: str, suffix: str) -> str:
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"papers/{paper_id}/source-{content_hash}{normalized_suffix}"


def import_local_pdf_directory(
    *,
    source_dir: Path,
    session_factory: sessionmaker[Session],
    owner_id: str = "local-user",
    collection_title: str | None = None,
    collection_description: str | None = None,
    object_store: object | None = None,
) -> LocalLibraryImportResult:
    """Import all PDFs from a local directory into a curated Paperbase collection."""

    resolved_source_dir = source_dir.expanduser().resolve()
    if not resolved_source_dir.exists():
        raise FileNotFoundError(str(resolved_source_dir))
    if not resolved_source_dir.is_dir():
        raise NotADirectoryError(str(resolved_source_dir))

    pdf_paths = sorted(resolved_source_dir.rglob("*.pdf"), key=lambda path: path.name.lower())
    title = collection_title or resolved_source_dir.name
    description = collection_description or f"Imported from local directory {resolved_source_dir}"

    imported_papers = 0
    reused_papers = 0

    with session_factory() as session:
        paper_repository = PaperRepository(session)
        file_repository = PaperFileRepository(session)
        collection_repository = CollectionRepository(session)

        collection = collection_repository.create_or_get(
            owner_id=owner_id,
            title=title,
            description=description,
            tags=["local-library"],
        )

        for position, pdf_path in enumerate(pdf_paths, start=1):
            external_id = str(pdf_path.resolve())
            content_hash = _content_hash_for_path(pdf_path)
            existing_paper = paper_repository.get_by_provider_id("local_filesystem", external_id)
            paper = paper_repository.upsert(
                provider="local_filesystem",
                external_id=external_id,
                canonical_title=_canonical_title_for_path(pdf_path),
                tags=["local-library"],
                raw_metadata={
                    "file_name": pdf_path.name,
                    "source_dir": str(resolved_source_dir),
                },
            )
            if existing_paper is None:
                imported_papers += 1
            else:
                reused_papers += 1

            storage_uri = pdf_path.resolve().as_uri()
            if object_store is not None:
                storage_uri = object_store.put_file(
                    key=_paper_object_key(
                        paper_id=paper.id,
                        content_hash=content_hash,
                        suffix=pdf_path.suffix or ".pdf",
                    ),
                    source_path=pdf_path,
                    content_type="application/pdf",
                )
            file_repository.upsert(
                paper_id=paper.id,
                storage_uri=storage_uri,
                file_kind="pdf",
                content_hash=content_hash,
                mime_type="application/pdf",
                parser_status="pending",
            )
            collection_repository.add_paper(
                collection_id=collection.id,
                paper_id=paper.id,
                position=position,
            )

        return LocalLibraryImportResult(
            collection_id=collection.id,
            collection_title=collection.title,
            total_pdf_files=len(pdf_paths),
            imported_papers=imported_papers,
            reused_papers=reused_papers,
        )
