"""Worker runtime for queued Paperbase background jobs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.repositories import BackgroundJobRepository
from paperbase.extract.runner import CollectionExtractionRunner
from paperbase.ingest.local_library import import_local_pdf_directory
from paperbase.parsing.runner import CollectionParseRunner
from paperbase.search.runtime import PaperbaseSearchReindexer


class PaperbaseWorker:
    """Claim and execute queued Paperbase jobs from the canonical database."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        search_backend: object | None = None,
        extraction_client_factory: Callable[[], object] | None = None,
        parser_factory: Callable[[], object] | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.search_backend = search_backend
        self.extraction_client_factory = extraction_client_factory
        self.parser_factory = parser_factory

    def process_next_job(self) -> str | None:
        with self.session_factory() as session:
            repository = BackgroundJobRepository(session)
            job = repository.claim_next()
            if job is None:
                return None
            job_id = job.id
            job_type = job.job_type
            payload = dict(job.payload_json or {})

        try:
            result_json = self._execute_job(job_type=job_type, payload=payload)
        except Exception as exc:  # noqa: BLE001
            with self.session_factory() as session:
                repository = BackgroundJobRepository(session)
                repository.mark_failed(job_id, error_message=str(exc))
            return job_id

        with self.session_factory() as session:
            repository = BackgroundJobRepository(session)
            repository.mark_completed(job_id, result_json=result_json)
        return job_id

    def _execute_job(self, *, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if job_type == "search_reindex":
            return self._execute_search_reindex()
        if job_type == "collection_extract":
            return self._execute_collection_extract(payload)
        if job_type == "local_library_ingest":
            return self._execute_local_library_ingest(payload)
        if job_type == "collection_parse":
            return self._execute_collection_parse(payload)
        raise RuntimeError(f"Unsupported background job type: {job_type}")

    def _execute_search_reindex(self) -> dict[str, Any]:
        if self.search_backend is None:
            raise RuntimeError("Search backend is not configured for worker execution.")

        reindexer = PaperbaseSearchReindexer(
            session_factory=self.session_factory,
            backend=self.search_backend,
        )
        indexed = reindexer.reindex_all()
        return {"indexed": indexed}

    def _execute_collection_extract(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.extraction_client_factory is None:
            raise RuntimeError("Extraction client factory is not configured for worker execution.")

        runner = CollectionExtractionRunner(
            session_factory=self.session_factory,
            client=self.extraction_client_factory(),
        )
        summary = runner.extract_collection(
            collection_id=str(payload["collection_id"]),
            schema_payload=dict(payload.get("schema_payload") or {}),
            prompt_version=str(payload["prompt_version"]),
            schema_version=str(payload["schema_version"]),
            extraction_profile_id=payload.get("extraction_profile_id"),
            limit=payload.get("limit"),
        )
        return {
            "collection_id": summary.collection_id,
            "extracted_paper_count": summary.extracted_paper_count,
            "skipped_paper_ids": list(summary.skipped_paper_ids),
        }

    def _execute_local_library_ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = import_local_pdf_directory(
            source_dir=Path(str(payload["source_dir"])),
            session_factory=self.session_factory,
            owner_id=str(payload.get("owner_id") or "local-user"),
            collection_title=payload.get("collection_title"),
            collection_description=payload.get("collection_description"),
        )
        return {
            "collection_title": result.collection_title,
            "total_pdf_files": result.total_pdf_files,
            "imported_papers": result.imported_papers,
            "reused_papers": result.reused_papers,
        }

    def _execute_collection_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary = CollectionParseRunner(
            session_factory=self.session_factory,
            parser_factory=self.parser_factory,
        ).parse_collection(
            collection_id=str(payload["collection_id"]),
            limit=payload.get("limit"),
        )
        return {
            "collection_id": summary.collection_id,
            "parsed_paper_count": summary.parsed_paper_count,
            "skipped_paper_ids": list(summary.skipped_paper_ids),
            "section_count": summary.section_count,
            "chunk_count": summary.chunk_count,
        }
