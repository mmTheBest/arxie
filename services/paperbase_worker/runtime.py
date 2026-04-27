"""Worker runtime for queued Paperbase background jobs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.repositories import BackgroundJobRepository
from paperbase.extract.runner import CollectionExtractionRunner
from paperbase.ingest.local_library import import_local_pdf_directory
from paperbase.ingest.provider_identifiers import (
    IdentifierInput,
    ingest_provider_identifiers,
    refresh_paper_metadata,
)
from paperbase.search.embeddings import EmbeddingProvider
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
        provider_resolver: object | None = None,
        job_consumer: object | None = None,
        job_dispatcher: object | None = None,
        retry_limit: int = 3,
        embedding_provider: EmbeddingProvider | None = None,
        object_store: object | None = None,
        download_cache_dir: str | None = None,
        download_cache_ttl_seconds: int = 86400,
    ) -> None:
        self.session_factory = session_factory
        self.search_backend = search_backend
        self.extraction_client_factory = extraction_client_factory
        self.parser_factory = parser_factory
        self.provider_resolver = provider_resolver
        self.job_consumer = job_consumer
        self.job_dispatcher = job_dispatcher
        self.retry_limit = retry_limit
        self.embedding_provider = embedding_provider
        self.object_store = object_store
        self.download_cache_dir = download_cache_dir
        self.download_cache_ttl_seconds = download_cache_ttl_seconds

    def process_next_job(self) -> str | None:
        with self.session_factory() as session:
            repository = BackgroundJobRepository(session)
            job = repository.claim_next()
            if job is None:
                return None
        return self._execute_claimed_job(job)

    def process_next_dispatched_job(self, timeout_seconds: float | None = None) -> str | None:
        if self.job_consumer is None:
            return self.process_next_job()
        job_id = self.job_consumer.receive(timeout_seconds)
        if job_id is None:
            return None
        with self.session_factory() as session:
            repository = BackgroundJobRepository(session)
            job = repository.claim_by_id(job_id)
            if job is None:
                return None
        return self._execute_claimed_job(job)

    def _execute_claimed_job(self, job) -> str:  # noqa: ANN001
        job_id = job.id
        job_type = job.job_type
        payload = dict(job.payload_json or {})

        try:
            result_json = self._execute_job(job_type=job_type, payload=payload)
        except Exception as exc:  # noqa: BLE001
            self._handle_failed_job(job_id=job_id, attempt_count=job.attempt_count, error_message=str(exc))
            return job_id

        with self.session_factory() as session:
            repository = BackgroundJobRepository(session)
            repository.mark_completed(job_id, result_json=result_json)
        return job_id

    def _handle_failed_job(self, *, job_id: str, attempt_count: int, error_message: str) -> None:
        should_retry = self.job_dispatcher is not None and attempt_count < self.retry_limit
        with self.session_factory() as session:
            repository = BackgroundJobRepository(session)
            if should_retry:
                repository.mark_pending(job_id, error_message=error_message)
            else:
                repository.mark_failed(job_id, error_message=error_message)
        if should_retry:
            self.job_dispatcher.dispatch(job_id)

    def _execute_job(self, *, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if job_type == "search_reindex":
            return self._execute_search_reindex()
        if job_type == "collection_extract":
            return self._execute_collection_extract(payload)
        if job_type == "local_library_ingest":
            return self._execute_local_library_ingest(payload)
        if job_type == "collection_parse":
            return self._execute_collection_parse(payload)
        if job_type == "provider_identifier_ingest":
            return self._execute_provider_identifier_ingest(payload)
        if job_type == "paper_metadata_refresh":
            return self._execute_paper_metadata_refresh(payload)
        raise RuntimeError(f"Unsupported background job type: {job_type}")

    def _execute_search_reindex(self) -> dict[str, Any]:
        if self.search_backend is None:
            raise RuntimeError("Search backend is not configured for worker execution.")

        reindexer = PaperbaseSearchReindexer(
            session_factory=self.session_factory,
            backend=self.search_backend,
            embedding_provider=self.embedding_provider,
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
            object_store=self.object_store,
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
            object_store=self.object_store,
            cache_dir=self.download_cache_dir,
            cache_ttl_seconds=self.download_cache_ttl_seconds,
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
            "figure_count": summary.figure_count,
            "table_count": summary.table_count,
        }

    def _execute_provider_identifier_ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        identifiers = [
            IdentifierInput(kind=str(item["kind"]), value=str(item["value"]))
            for item in list(payload.get("identifiers") or [])
        ]
        result = ingest_provider_identifiers(
            identifiers=identifiers,
            session_factory=self.session_factory,
            resolver=self.provider_resolver,
            owner_id=str(payload.get("owner_id") or "local-user"),
            collection_id=payload.get("collection_id"),
            collection_title=payload.get("collection_title"),
            collection_description=payload.get("collection_description"),
            object_store=self.object_store,
        )
        return {
            "collection_id": result.collection_id,
            "collection_title": result.collection_title,
            "requested_count": result.requested_count,
            "imported_papers": result.imported_papers,
            "reused_papers": result.reused_papers,
            "paper_ids": list(result.paper_ids),
            "skipped_identifiers": list(result.skipped_identifiers),
        }

    def _execute_paper_metadata_refresh(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = refresh_paper_metadata(
            paper_ids=[str(paper_id) for paper_id in list(payload.get("paper_ids") or [])],
            session_factory=self.session_factory,
            resolver=self.provider_resolver,
            object_store=self.object_store,
        )
        return {
            "requested_count": result.requested_count,
            "refreshed_papers": result.refreshed_papers,
            "skipped_paper_ids": list(result.skipped_paper_ids),
        }
