"""Deterministic local research-agent runner backed by Paperbase evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.repositories import ResearchRepository
from paperbase.research.agent_runtime import PaperbaseResearchAgentRuntime


@dataclass(frozen=True, slots=True)
class ResearchAgentRunResult:
    collection_id: str
    thread_id: str
    artifact_id: str
    artifact_type: str
    evidence_paper_count: int
    run_id: str | None = None


class PaperbaseResearchAgentRunner:
    """Synthesize research artifacts from a prepared local paper collection."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        model_name: str = "local-research-agent",
        prompt_version: str = "research-agent-v1",
        model_client: object | None = None,
        search_backend: object | None = None,
        embedding_provider: object | None = None,
        project_id: str | None = None,
        backend_retrieval_enabled: bool = False,
    ) -> None:
        self.session_factory = session_factory
        self.model_name = model_name
        self.prompt_version = prompt_version
        self.model_client = model_client
        self.search_backend = search_backend
        self.embedding_provider = embedding_provider
        self.project_id = project_id
        self.backend_retrieval_enabled = backend_retrieval_enabled

    def run(self, payload: dict[str, Any]) -> ResearchAgentRunResult:
        with self.session_factory() as session:
            runtime_payload = {
                **payload,
                "prompt_version": payload.get("prompt_version") or self.prompt_version,
            }
            runtime_result = PaperbaseResearchAgentRuntime(
                model_client=self.model_client,
                search_backend=self.search_backend,
                embedding_provider=self.embedding_provider,
                project_id=self.project_id,
                backend_retrieval_enabled=self.backend_retrieval_enabled,
            ).execute(session, runtime_payload)

        return ResearchAgentRunResult(
            collection_id=str(payload["collection_id"]),
            thread_id=str(payload["thread_id"]),
            artifact_id=str(payload["artifact_id"]),
            artifact_type=str(runtime_result["artifact_type"]),
            evidence_paper_count=int(runtime_result["evidence_paper_count"]),
            run_id=str(runtime_result["run_id"]),
        )

    def mark_artifact_failed(self, *, artifact_id: str, error_message: str) -> None:
        with self.session_factory() as session:
            repository = ResearchRepository(session)
            if repository.get_artifact(artifact_id) is None:
                return
            repository.update_artifact(
                artifact_id,
                status="failed",
                error_message=error_message,
            )
