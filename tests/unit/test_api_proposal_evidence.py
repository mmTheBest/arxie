from __future__ import annotations

from fastapi.testclient import TestClient

from ra.api import create_app


class _StubRetriever:
    async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
        return []

    async def get_paper(self, identifier: str):  # noqa: ARG002
        return None

    async def close(self) -> None:
        return None


def _mk_app():
    return create_app(retriever_factory=lambda: _StubRetriever())


def test_query_proposal_evidence_endpoint_returns_bucketed_result_with_provenance_links() -> None:
    client = TestClient(_mk_app())

    response = client.post(
        "/api/proposal/evidence/query",
        json={
            "claim": "Transformers improve machine translation quality across benchmarks.",
            "pinned_paper_ids": ["p1", "p2"],
            "papers": [
                {
                    "paper_id": "p1",
                    "title": "Positive evidence",
                    "abstract": "Transformers improve machine translation quality.",
                    "doi": "10.1000/xyz123",
                },
                {
                    "paper_id": "p2",
                    "title": "Adjacent evidence",
                    "abstract": "Transformer deployment constraints and runtime trade-offs.",
                    "pdf_url": "https://example.org/evidence.pdf",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["bucket_counts"]["supporting"] >= 1
    all_items = payload["supporting"] + payload["contradicting"] + payload["adjacent"]
    by_id = {item["paper_id"]: item for item in all_items}
    assert by_id["p1"]["provenance_link"] == "https://doi.org/10.1000/xyz123"
    assert by_id["p2"]["provenance_link"] == "https://example.org/evidence.pdf"
