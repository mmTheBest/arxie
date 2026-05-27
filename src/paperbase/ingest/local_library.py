"""Import local PDF directories into Paperbase."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from paperbase.config import load_paperbase_config
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


def _is_under_any_root(path: Path, allowed_roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in allowed_roots)


def _resolved_hosted_allowed_roots(allowed_roots: Sequence[str | Path]) -> tuple[Path, ...]:
    resolved_roots = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
    if not resolved_roots:
        raise PermissionError("Local PDF import is disabled because no allowed roots are configured.")
    return resolved_roots


def _validate_hosted_source_dir(source_dir: Path, allowed_roots: Sequence[Path]) -> None:
    if not _is_under_any_root(source_dir, allowed_roots):
        raise PermissionError("Local PDF import source_dir is outside configured allowed roots.")


def _contains_symlink(path: Path, *, root: Path) -> bool:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        return path.is_symlink()

    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _validated_hosted_pdf_paths(
    *,
    source_dir: Path,
    pdf_paths: Sequence[Path],
    allowed_roots: Sequence[Path],
) -> list[Path]:
    validated_paths: list[Path] = []
    for pdf_path in pdf_paths:
        resolved_pdf_path = pdf_path.resolve()
        if not _is_under_any_root(resolved_pdf_path, allowed_roots):
            raise PermissionError("Local PDF import contains a PDF outside configured allowed roots.")
        if _contains_symlink(pdf_path, root=source_dir):
            raise PermissionError("Local PDF import contains a symlinked PDF path.")
        validated_paths.append(pdf_path)
    return validated_paths


def _discover_pdf_paths(source_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: str(path.relative_to(source_dir)).casefold(),
    )


def import_local_pdf_directory(
    *,
    source_dir: Path,
    session_factory: sessionmaker[Session],
    owner_id: str = "local-user",
    collection_title: str | None = None,
    collection_description: str | None = None,
    object_store: object | None = None,
    hosted_mode: bool | None = None,
    local_path_import_allowed_roots: Sequence[str | Path] | None = None,
) -> LocalLibraryImportResult:
    """Import all PDFs from a local directory into a curated Paperbase collection."""

    config = (
        load_paperbase_config()
        if hosted_mode is None or local_path_import_allowed_roots is None
        else None
    )
    resolved_hosted_mode = config.hosted_mode if hosted_mode is None and config is not None else bool(hosted_mode)
    resolved_allowed_roots: Sequence[str | Path] = (
        local_path_import_allowed_roots
        if local_path_import_allowed_roots is not None
        else (
            *config.local_path_import_allowed_roots,
            config.upload_staging_dir,
        )
    )
    resolved_source_dir = source_dir.expanduser().resolve()
    if not resolved_source_dir.exists():
        raise FileNotFoundError(str(resolved_source_dir))
    if not resolved_source_dir.is_dir():
        raise NotADirectoryError(str(resolved_source_dir))

    if resolved_hosted_mode:
        resolved_allowed_roots = _resolved_hosted_allowed_roots(resolved_allowed_roots)
        _validate_hosted_source_dir(resolved_source_dir, resolved_allowed_roots)

    pdf_paths = _discover_pdf_paths(resolved_source_dir)
    if resolved_hosted_mode:
        pdf_paths = _validated_hosted_pdf_paths(
            source_dir=resolved_source_dir,
            pdf_paths=pdf_paths,
            allowed_roots=resolved_allowed_roots,
        )
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
