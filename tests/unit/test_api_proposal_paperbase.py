from __future__ import annotations

from fastapi.testclient import TestClient

from ra.api import create_app
from ra.retrieval.unified import Paper, WorkspaceContext


class _CollectionRetrieverStub:
    def __init__(self) -> None:
        self.collection_calls: list[tuple[str, str | None, int]] = []
        self.workspace_calls: list[str] = []

    async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
        return []

    async def get_paper(self, identifier: str):  # noqa: ARG002
        return None

    async def get_collection_papers(
        self,
        collection_id: str,
        *,
        query: str | None = None,
        limit: int = 50,
    ) -> list[Paper]:
        self.collection_calls.append((collection_id, query, limit))
        return [
            Paper(
                id="pb-1",
                title="GRNFormer for single-cell perturbation prediction",
                abstract="GRNFormer improves perturbation prediction on scRegNetBench.",
                authors=["Alice Example"],
                year=2026,
                venue="Nature",
                citation_count=3,
                pdf_url=None,
                doi=None,
                arxiv_id=None,
                source="both",
            )
        ]

    async def get_workspace_context(self, workspace_id: str) -> WorkspaceContext | None:
        self.workspace_calls.append(workspace_id)
        return WorkspaceContext(
            workspace_id=workspace_id,
            title="scRegNet workspace",
            collection_id="collection-2",
            saved_query="scRegNet perturbation models",
            focus_note="Use persisted benchmark evidence.",
            active_filters={},
            pinned_paper_ids=["pb-1"],
        )

    async def close(self) -> None:
        return None


def test_query_proposal_evidence_can_load_papers_from_paperbase_collection() -> None:
    retriever = _CollectionRetrieverStub()
    client = TestClient(create_app(retriever_factory=lambda: retriever))

    response = client.post(
        "/api/proposal/evidence/query",
        json={
            "claim": "Perturbation transformers improve single-cell prediction quality.",
            "paperbase_collection_id": "collection-1",
            "pinned_paper_ids": ["pb-1"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["bucket_counts"]["supporting"] >= 1
    assert retriever.collection_calls == [
        ("collection-1", "Perturbation transformers improve single-cell prediction quality.", 50)
    ]


def test_query_proposal_evidence_can_use_workspace_context_defaults() -> None:
    retriever = _CollectionRetrieverStub()
    client = TestClient(create_app(retriever_factory=lambda: retriever))

    response = client.post(
        "/api/proposal/evidence/query",
        json={
            "claim": "Perturbation transformers improve single-cell prediction quality.",
            "workspace_id": "workspace-1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["bucket_counts"]["supporting"] >= 1
    assert payload["supporting"][0]["pinned"] is True
    assert retriever.workspace_calls == ["workspace-1"]
    assert retriever.collection_calls == [
        ("collection-2", "Perturbation transformers improve single-cell prediction quality.", 50)
    ]
