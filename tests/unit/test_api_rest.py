from __future__ import annotations

from typing import Callable

import pytest
from fastapi.testclient import TestClient

from ra.api import create_app
from ra.retrieval.unified import Paper


class _StubRetriever:
    def __init__(self, *, search_result=None, paper_result=None, error: Exception | None = None) -> None:
        self._search_result = search_result if search_result is not None else []
        self._paper_result = paper_result
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        return None

    async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
        if self._error is not None:
            raise self._error
        return self._search_result

    async def get_paper(self, identifier: str):  # noqa: ARG002
        if self._error is not None:
            raise self._error
        return self._paper_result


class _StubAgent:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def arun(self, query: str) -> str:  # noqa: ARG002
        return self._answer


def _mk_app(
    *,
    retriever_factory: Callable[[], _StubRetriever] | None = None,
    agent_factory: Callable[[], _StubAgent] | None = None,
):
    return create_app(retriever_factory=retriever_factory, agent_factory=agent_factory)


def test_health_endpoint_returns_ok():
    client = TestClient(_mk_app())

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_endpoint_returns_results():
    paper = Paper(
        id="p1",
        title="Attention Is All You Need",
        abstract="Transformers",
        authors=["Ashish Vaswani"],
        year=2017,
        venue="NeurIPS",
        citation_count=100,
        pdf_url="https://example.com/p1.pdf",
        doi="10.5555/3295222.3295349",
        arxiv_id="1706.03762",
        source="both",
    )

    app = _mk_app(retriever_factory=lambda: _StubRetriever(search_result=[paper]))
    client = TestClient(app)

    resp = client.post(
        "/search",
        json={"query": "transformer model", "limit": 5, "source": "both"},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["query"] == "transformer model"
    assert payload["count"] == 1
    assert payload["results"][0]["id"] == "p1"
    assert payload["results"][0]["title"] == "Attention Is All You Need"


def test_search_endpoint_maps_value_error_to_bad_request():
    app = _mk_app(retriever_factory=lambda: _StubRetriever(error=ValueError("invalid query")))
    client = TestClient(app)

    resp = client.post(
        "/search",
        json={"query": "transformer model", "limit": 5, "source": "both"},
    )

    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"


def test_retrieve_endpoint_returns_paper_not_found():
    app = _mk_app(retriever_factory=lambda: _StubRetriever(paper_result=None))
    client = TestClient(app)

    resp = client.post("/retrieve", json={"identifier": "10.1000/xyz123"})

    assert resp.status_code == 404
    payload = resp.json()
    assert payload["error"] == "paper_not_found"


def test_answer_endpoint_returns_agent_response():
    app = _mk_app(agent_factory=lambda: _StubAgent("## Answer\nok\n\n## References\nNone."))
    client = TestClient(app)

    resp = client.post("/answer", json={"query": "What is retrieval augmented generation?"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["query"].startswith("What is retrieval")
    assert "## Answer" in payload["answer"]


def test_answer_endpoint_maps_agent_factory_errors_to_service_unavailable():
    def _failing_agent_factory() -> _StubAgent:
        raise ValueError("OPENAI_API_KEY missing")

    app = _mk_app(agent_factory=_failing_agent_factory)
    client = TestClient(app)

    resp = client.post("/answer", json={"query": "test"})

    assert resp.status_code == 503
    payload = resp.json()
    assert payload["error"] == "agent_unavailable"


def test_validation_errors_use_structured_payload():
    client = TestClient(_mk_app())

    resp = client.post("/search", json={"limit": 3, "source": "both"})

    assert resp.status_code == 422
    payload = resp.json()
    assert payload["error"] == "validation_error"
    assert isinstance(payload["details"], list)
    assert payload["details"]
