import pytest
import httpx

from ra.retrieval.semantic_scholar import Author, Paper, SemanticScholarClient


def test_author_from_api_defaults():
    a = Author.from_api({})
    assert a.author_id == ""
    assert a.name == "Unknown"


def test_paper_from_api_parses_external_ids_and_open_access_pdf():
    payload = {
        "paperId": "abc123",
        "title": "A Study",
        "abstract": None,
        "year": 2024,
        "authors": [{"authorId": "a1", "name": "Alice"}],
        "venue": "TestConf",
        "citationCount": 7,
        "isOpenAccess": True,
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        "externalIds": {"DOI": "10.5555/XYZ", "ArXiv": "2101.00001"},
    }

    p = Paper.from_api(payload)
    assert p.paper_id == "abc123"
    assert p.title == "A Study"
    assert p.abstract is None
    assert p.year == 2024
    assert [a.name for a in p.authors] == ["Alice"]
    assert p.venue == "TestConf"
    assert p.citation_count == 7
    assert p.is_open_access is True
    assert p.pdf_url == "https://example.com/paper.pdf"
    assert p.doi == "10.5555/XYZ"
    assert p.arxiv_id == "2101.00001"
    assert p.external_ids == {"DOI": "10.5555/XYZ", "ArXiv": "2101.00001"}


class _CountingRateLimiter:
    def __init__(self) -> None:
        self.calls = 0

    async def acquire(self) -> None:
        self.calls += 1


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"data": []}


class _FakeHttpClient:
    async def request(self, method: str, endpoint: str, params=None):  # noqa: ANN001, ARG002
        return _FakeResponse()


class _EndpointCaptureHttpClient:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload or {}
        self.last_endpoint: str | None = None

    async def request(self, method: str, endpoint: str, params=None):  # noqa: ANN001, ARG002
        self.last_endpoint = endpoint
        return _FakeResponseWithPayload(self.payload)


class _StatusThenSuccessHttpClient:
    def __init__(self, status_codes: list[int], payload: dict[str, object] | None = None) -> None:
        self.status_codes = status_codes
        self.payload = payload or {"data": []}
        self.calls = 0

    async def request(self, method: str, endpoint: str, params=None):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.status_codes:
            status = self.status_codes.pop(0)
            request = httpx.Request(method, f"https://api.semanticscholar.org{endpoint}")
            response = httpx.Response(status_code=status, request=request)
            raise httpx.HTTPStatusError("transient failure", request=request, response=response)
        return _FakeResponseWithPayload(self.payload)


class _RequestErrorThenSuccessHttpClient:
    def __init__(self, failures: int, payload: dict[str, object] | None = None) -> None:
        self.failures = failures
        self.payload = payload or {"data": []}
        self.calls = 0

    async def request(self, method: str, endpoint: str, params=None):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls <= self.failures:
            request = httpx.Request(method, f"https://api.semanticscholar.org{endpoint}")
            raise httpx.RequestError("network failure", request=request)
        return _FakeResponseWithPayload(self.payload)


class _FakeResponseWithPayload:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


@pytest.mark.asyncio
async def test_request_uses_rate_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = _CountingRateLimiter()
    client = SemanticScholarClient(rate_limiter=limiter, max_retries=1)

    async def _fake_get_client():
        return _FakeHttpClient()

    monkeypatch.setattr(client, "_get_client", _fake_get_client)

    await client._request("GET", "/paper/search", params={"query": "transformer"})
    assert limiter.calls == 1


@pytest.mark.asyncio
async def test_request_retries_retryable_http_status_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _CountingRateLimiter()
    client = SemanticScholarClient(rate_limiter=limiter, max_retries=3)
    fake_http = _StatusThenSuccessHttpClient(
        status_codes=[503, 500],
        payload={"data": [{"paperId": "ok"}]},
    )
    sleep_calls: list[int] = []

    async def _fake_get_client():
        return fake_http

    async def _fake_sleep(delay: int) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(client, "_get_client", _fake_get_client)
    monkeypatch.setattr("ra.retrieval.semantic_scholar.asyncio.sleep", _fake_sleep)

    payload = await client._request("GET", "/paper/search", params={"query": "transformer"})

    assert payload == {"data": [{"paperId": "ok"}]}
    assert fake_http.calls == 3
    assert sleep_calls == [1, 2]


@pytest.mark.asyncio
async def test_request_retries_request_errors_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _CountingRateLimiter()
    client = SemanticScholarClient(rate_limiter=limiter, max_retries=4)
    fake_http = _RequestErrorThenSuccessHttpClient(
        failures=3,
        payload={"data": [{"paperId": "ok"}]},
    )
    sleep_calls: list[int] = []

    async def _fake_get_client():
        return fake_http

    async def _fake_sleep(delay: int) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(client, "_get_client", _fake_get_client)
    monkeypatch.setattr("ra.retrieval.semantic_scholar.asyncio.sleep", _fake_sleep)

    payload = await client._request("GET", "/paper/search", params={"query": "transformer"})

    assert payload == {"data": [{"paperId": "ok"}]}
    assert fake_http.calls == 4
    assert sleep_calls == [1, 2, 4]


@pytest.mark.asyncio
async def test_get_paper_url_encodes_identifier_path_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SemanticScholarClient(max_retries=1)
    fake_http = _EndpointCaptureHttpClient(payload={"paperId": "ok"})

    async def _fake_get_client():
        return fake_http

    monkeypatch.setattr(client, "_get_client", _fake_get_client)

    _ = await client.get_paper("DOI:10.1000/xyz")
    assert fake_http.last_endpoint == "/paper/DOI:10.1000%2Fxyz"


@pytest.mark.asyncio
async def test_search_rejects_control_characters_in_query() -> None:
    client = SemanticScholarClient(max_retries=1)
    with pytest.raises(ValueError, match="query"):
        await client.search("transformer\x00models", limit=5)


@pytest.mark.asyncio
async def test_get_client_configures_connection_pool_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            captured_kwargs.update(kwargs)
            self.is_closed = False

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr("ra.retrieval.semantic_scholar.httpx.AsyncClient", _FakeAsyncClient)

    client = SemanticScholarClient(
        max_connections=77,
        max_keepalive_connections=13,
        keepalive_expiry=55.0,
    )
    _ = await client._get_client()

    limits = captured_kwargs["limits"]
    assert isinstance(limits, httpx.Limits)
    assert limits.max_connections == 77
    assert limits.max_keepalive_connections == 13
    assert limits.keepalive_expiry == 55.0


@pytest.mark.asyncio
async def test_search_batch_returns_ordered_results() -> None:
    client = SemanticScholarClient(max_retries=1)

    async def _fake_search(query: str, limit: int = 10, **kwargs):  # noqa: ANN001, ARG001
        return [
            Paper(
                paper_id=f"id-{query}",
                title=f"title-{query}",
                abstract=None,
                year=None,
                authors=[],
                venue=None,
                citation_count=0,
                is_open_access=False,
                pdf_url=None,
                doi=None,
                arxiv_id=None,
                external_ids={},
            )
        ]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(client, "search", _fake_search)
    try:
        results = await client.search_batch(["q1", "q2"], max_concurrency=2)
    finally:
        monkeypatch.undo()

    assert [batch[0].paper_id for batch in results] == ["id-q1", "id-q2"]


@pytest.mark.asyncio
async def test_get_papers_batch_returns_ordered_results() -> None:
    client = SemanticScholarClient(max_retries=1)

    async def _fake_get_paper(paper_id: str):  # noqa: ANN001
        return Paper(
            paper_id=paper_id,
            title=f"title-{paper_id}",
            abstract=None,
            year=None,
            authors=[],
            venue=None,
            citation_count=0,
            is_open_access=False,
            pdf_url=None,
            doi=None,
            arxiv_id=None,
            external_ids={},
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(client, "get_paper", _fake_get_paper)
    try:
        results = await client.get_papers_batch(["a", "b"], max_concurrency=2)
    finally:
        monkeypatch.undo()

    assert [paper.paper_id if paper else None for paper in results] == ["a", "b"]
