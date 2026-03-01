from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from ra.api import create_app
from ra.retrieval.unified import Paper


class _StubRetriever:
    def __init__(
        self,
        *,
        search_result=None,
        paper_result=None,
        error: Exception | None = None,
    ) -> None:
        self._search_result = search_result if search_result is not None else []
        self._paper_result = paper_result
        self._error = error
        self.close_calls = 0

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

    async def search_batch(
        self,
        requests: list[tuple[str, int, tuple[str, ...]]],  # noqa: ARG002
        *,
        max_concurrency: int = 4,  # noqa: ARG002
    ):
        if self._error is not None:
            raise self._error
        return [list(self._search_result) for _ in requests]

    async def get_papers_batch(
        self,
        identifiers: list[str],  # noqa: ARG002
        *,
        max_concurrency: int = 8,  # noqa: ARG002
    ):
        if self._error is not None:
            raise self._error
        return [self._paper_result for _ in identifiers]

    async def close(self) -> None:
        self.close_calls += 1


class _StubAgent:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def arun(self, query: str) -> str:  # noqa: ARG002
        return self._answer


class _StubLitReviewAgent:
    def __init__(self, review: str) -> None:
        self._review = review

    async def arun(self, topic: str) -> str:  # noqa: ARG002
        return self._review


def _mk_app(
    *,
    retriever_factory: Callable[[], _StubRetriever] | None = None,
    agent_factory: Callable[[], _StubAgent] | None = None,
    lit_review_agent_factory: Callable[[], _StubLitReviewAgent] | None = None,
):
    return create_app(
        retriever_factory=retriever_factory,
        agent_factory=agent_factory,
        lit_review_agent_factory=lit_review_agent_factory,
    )


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


def test_lit_review_endpoint_returns_structured_review():
    observed: dict[str, object] = {}
    expected_review = (
        "## Introduction\nIntro.\n\n"
        "## Thematic Groups\n- Theme A\n\n"
        "## Key Findings\n- Finding\n\n"
        "## Research Gaps\n- Gap\n\n"
        "## Future Directions\n- Next steps"
    )

    class _TrackingLitReviewAgent(_StubLitReviewAgent):
        async def arun(self, topic: str) -> str:
            observed["topic"] = topic
            return await super().arun(topic)

    def _lit_review_factory() -> _TrackingLitReviewAgent:
        return _TrackingLitReviewAgent(expected_review)

    app = _mk_app(lit_review_agent_factory=_lit_review_factory)
    client = TestClient(app)

    resp = client.post("/api/lit-review", json={"topic": "graph neural networks"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["topic"] == "graph neural networks"
    assert payload["review"] == expected_review
    assert observed["topic"] == "graph neural networks"


def test_lit_review_endpoint_maps_factory_errors_to_service_unavailable():
    def _failing_lit_review_factory() -> _StubLitReviewAgent:
        raise ValueError("OPENAI_API_KEY missing")

    app = _mk_app(lit_review_agent_factory=_failing_lit_review_factory)
    client = TestClient(app)

    resp = client.post("/api/lit-review", json={"topic": "test"})

    assert resp.status_code == 503
    payload = resp.json()
    assert payload["error"] == "agent_unavailable"


def test_query_endpoint_passes_deep_flag_to_agent_factory():
    observed: dict[str, object] = {}

    class _TrackingAgent(_StubAgent):
        async def arun(self, query: str) -> str:
            observed["query"] = query
            return await super().arun(query)

    def _agent_factory(*, deep_search: bool = False) -> _TrackingAgent:
        observed["deep_search"] = deep_search
        return _TrackingAgent("## Answer\nok\n\n## References\nNone.")

    app = _mk_app(agent_factory=_agent_factory)
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={"query": "What is retrieval augmented generation?", "deep": True},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["query"].startswith("What is retrieval")
    assert payload["answer"] == "## Answer\nok\n\n## References\nNone."
    assert observed["deep_search"] is True
    assert observed["query"] == "What is retrieval augmented generation?"


def test_query_endpoint_defaults_deep_to_false():
    observed: list[bool] = []

    def _agent_factory(*, deep_search: bool = False) -> _StubAgent:
        observed.append(deep_search)
        return _StubAgent("## Answer\nok\n\n## References\nNone.")

    app = _mk_app(agent_factory=_agent_factory)
    client = TestClient(app)

    resp = client.post("/query", json={"query": "test"})

    assert resp.status_code == 200
    assert observed == [False]


def test_validation_errors_use_structured_payload():
    client = TestClient(_mk_app())

    resp = client.post("/search", json={"limit": 3, "source": "both"})

    assert resp.status_code == 422
    payload = resp.json()
    assert payload["error"] == "validation_error"
    assert isinstance(payload["details"], list)
    assert payload["details"]


def test_openapi_metadata_and_tags_are_documented():
    client = TestClient(_mk_app())

    resp = client.get("/openapi.json")

    assert resp.status_code == 200
    schema = resp.json()

    assert schema["info"]["contact"]["name"] == "Academic Research Assistant Maintainers"
    assert schema["info"]["license"]["name"] == "MIT"

    tags = {tag["name"]: tag for tag in schema["tags"]}
    assert "pipeline" in tags
    assert "research workflow endpoints" in tags["pipeline"]["description"].lower()
    assert "system" in tags
    assert "service health" in tags["system"]["description"].lower()


def test_openapi_search_endpoint_documents_examples_and_errors():
    client = TestClient(_mk_app())

    resp = client.get("/openapi.json")

    assert resp.status_code == 200
    schema = resp.json()
    operation = schema["paths"]["/search"]["post"]

    assert operation["summary"] == "Search Academic Papers"
    assert "semantic scholar" in operation["description"].lower()

    request_examples = operation["requestBody"]["content"]["application/json"]["examples"]
    assert "cross_source_discovery" in request_examples
    assert request_examples["cross_source_discovery"]["value"]["source"] == "both"

    bad_request = operation["responses"]["400"]
    assert bad_request["description"] == "Invalid search request."
    assert bad_request["content"]["application/json"]["example"]["error"] == "invalid_input"


def test_openapi_component_schemas_include_field_descriptions():
    client = TestClient(_mk_app())

    resp = client.get("/openapi.json")

    assert resp.status_code == 200
    schema = resp.json()
    search_request_schema = schema["components"]["schemas"]["SearchRequest"]
    query_schema = search_request_schema["properties"]["query"]

    assert query_schema["description"] == "Natural-language query for paper discovery."
    assert query_schema["examples"][0] == "transformer architecture for long-context summarization"


def test_search_batch_endpoint_returns_ordered_results():
    p1 = Paper(
        id="q1-paper",
        title="Batch Paper A",
        abstract="A",
        authors=["Alice"],
        year=2023,
        venue="ConfA",
        citation_count=11,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="semantic_scholar",
    )
    p2 = Paper(
        id="q2-paper",
        title="Batch Paper B",
        abstract="B",
        authors=["Bob"],
        year=2024,
        venue="ConfB",
        citation_count=7,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="arxiv",
    )

    class _BatchRetriever(_StubRetriever):
        async def search_batch(
            self,
            requests: list[tuple[str, int, tuple[str, ...]]],  # noqa: ARG002
            *,
            max_concurrency: int = 4,  # noqa: ARG002
        ):
            return [[p1], [p2]]

    client = TestClient(_mk_app(retriever_factory=lambda: _BatchRetriever()))
    resp = client.post(
        "/search/batch",
        json={
            "requests": [
                {"query": "query one", "limit": 3, "source": "semantic_scholar"},
                {"query": "query two", "limit": 3, "source": "arxiv"},
            ],
            "max_concurrency": 2,
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 2
    assert payload["results"][0]["query"] == "query one"
    assert payload["results"][0]["results"][0]["id"] == "q1-paper"
    assert payload["results"][1]["query"] == "query two"
    assert payload["results"][1]["results"][0]["id"] == "q2-paper"


def test_retrieve_batch_endpoint_returns_missing_papers_as_null():
    paper = Paper(
        id="p-1",
        title="Retrieved Paper",
        abstract=None,
        authors=["Alice"],
        year=2020,
        venue=None,
        citation_count=3,
        pdf_url=None,
        doi="10.1234/paper",
        arxiv_id=None,
        source="semantic_scholar",
    )

    class _BatchRetriever(_StubRetriever):
        async def get_papers_batch(
            self,
            identifiers: list[str],  # noqa: ARG002
            *,
            max_concurrency: int = 8,  # noqa: ARG002
        ):
            return [paper, None]

    client = TestClient(_mk_app(retriever_factory=lambda: _BatchRetriever()))
    resp = client.post(
        "/retrieve/batch",
        json={
            "requests": [
                {"identifier": "10.1234/paper"},
                {"identifier": "missing-id"},
            ],
            "max_concurrency": 4,
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] == 2
    assert payload["found"] == 1
    assert payload["results"][0]["paper"]["id"] == "p-1"
    assert payload["results"][1]["paper"] is None


def test_app_reuses_single_retriever_instance_across_requests():
    instances: list[_StubRetriever] = []

    def _factory() -> _StubRetriever:
        retriever = _StubRetriever(search_result=[], paper_result=None)
        instances.append(retriever)
        return retriever

    with TestClient(_mk_app(retriever_factory=_factory)) as client:
        _ = client.post("/search", json={"query": "test", "limit": 5, "source": "both"})
        _ = client.post("/retrieve", json={"identifier": "10.1000/xyz123"})

    assert len(instances) == 1
    assert instances[0].close_calls == 1
