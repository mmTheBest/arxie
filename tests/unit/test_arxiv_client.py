import pytest
import httpx

from ra.retrieval.arxiv import ArxivClient


ATOM_XML_WITH_DUPES = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\" xmlns:arxiv=\"http://arxiv.org/schemas/atom\">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v1</id>
    <updated>2021-01-02T00:00:00Z</updated>
    <published>2021-01-01T00:00:00Z</published>
    <title>  A   Title\n</title>
    <summary>Abstract\n with   spaces</summary>
    <author><name>Alice A.</name></author>
    <author><name>Bob B.</name></author>
    <category term=\"cs.AI\" />
    <link rel=\"related\" type=\"application/pdf\" title=\"pdf\" href=\"http://arxiv.org/pdf/2101.00001v1\" />
    <arxiv:doi> 10.1000/XYZ </arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2101.00001v1</id>
    <updated>2021-01-02T00:00:00Z</updated>
    <published>2021-01-01T00:00:00Z</published>
    <title>A Title</title>
    <summary>Duplicate entry should be deduped</summary>
    <author><name>Alice A.</name></author>
    <category term=\"cs.AI\" />
  </entry>
</feed>
"""


def test_arxiv_parse_feed_normalizes_and_dedupes_by_arxiv_id():
    client = ArxivClient()
    papers = client._parse_feed(ATOM_XML_WITH_DUPES)

    assert len(papers) == 1
    p = papers[0]

    assert p.arxiv_id == "2101.00001v1"
    assert p.title == "A Title"
    assert p.abstract == "Abstract with spaces"
    assert p.authors == ["Alice A.", "Bob B."]
    assert p.categories == ["cs.AI"]
    assert p.doi == "10.1000/XYZ"
    # Prefer explicit PDF link when present
    assert p.pdf_url == "http://arxiv.org/pdf/2101.00001v1"


def test_arxiv_parse_entry_falls_back_to_canonical_pdf_url_when_missing_link():
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\" xmlns:arxiv=\"http://arxiv.org/schemas/atom\">
  <entry>
    <id>http://arxiv.org/abs/9999.12345</id>
    <title>Test</title>
    <summary>Test</summary>
  </entry>
</feed>
"""
    client = ArxivClient()
    papers = client._parse_feed(xml)
    assert len(papers) == 1
    assert papers[0].pdf_url == "https://arxiv.org/pdf/9999.12345.pdf"


class _CountingRateLimiter:
    def __init__(self) -> None:
        self.calls = 0

    async def acquire(self) -> None:
        self.calls += 1


@pytest.mark.asyncio
async def test_arxiv_respect_rate_limit_uses_rate_limiter() -> None:
    limiter = _CountingRateLimiter()
    client = ArxivClient(rate_limiter=limiter)

    await client._respect_rate_limit()
    assert limiter.calls == 1


class _RequestErrorThenSuccessQueryClient:
    def __init__(self, failures: int, payload: str) -> None:
        self.failures = failures
        self.payload = payload
        self.calls = 0

    async def get(self, endpoint: str, params=None):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls <= self.failures:
            request = httpx.Request("GET", f"https://export.arxiv.org{endpoint}")
            raise httpx.RequestError("network failure", request=request)
        return _TextResponse(self.payload)


class _TextResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _RequestErrorThenSuccessStreamClient:
    def __init__(self, failures: int, payload: bytes) -> None:
        self.failures = failures
        self.payload = payload
        self.calls = 0

    def stream(self, method: str, url: str):  # noqa: ANN001
        self.calls += 1
        should_fail = self.calls <= self.failures
        return _MaybeFailStreamContext(should_fail=should_fail, method=method, url=url, payload=self.payload)


class _MaybeFailStreamContext:
    def __init__(self, should_fail: bool, method: str, url: str, payload: bytes) -> None:
        self.should_fail = should_fail
        self.method = method
        self.url = url
        self.payload = payload

    async def __aenter__(self):
        if self.should_fail:
            request = httpx.Request(self.method, self.url)
            raise httpx.RequestError("stream failure", request=request)
        return self

    async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        yield self.payload


@pytest.mark.asyncio
async def test_arxiv_request_retries_request_errors_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _CountingRateLimiter()
    client = ArxivClient(rate_limiter=limiter, max_retries=4)
    fake_http = _RequestErrorThenSuccessQueryClient(
        failures=3,
        payload=ATOM_XML_WITH_DUPES,
    )
    sleep_calls: list[int] = []

    async def _fake_get_client():
        return fake_http

    async def _fake_sleep(delay: int) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(client, "_get_client", _fake_get_client)
    monkeypatch.setattr("ra.retrieval.arxiv.asyncio.sleep", _fake_sleep)

    xml = await client._request({"search_query": "all:transformer"})
    assert "2101.00001v1" in xml
    assert fake_http.calls == 4
    assert sleep_calls == [1, 2, 4]


@pytest.mark.asyncio
async def test_download_pdf_retries_request_errors_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    limiter = _CountingRateLimiter()
    client = ArxivClient(rate_limiter=limiter, max_retries=4)
    fake_http = _RequestErrorThenSuccessStreamClient(
        failures=3,
        payload=b"%PDF-1.7 data",
    )
    sleep_calls: list[int] = []

    async def _fake_get_client():
        return fake_http

    async def _fake_sleep(delay: int) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(client, "_get_client", _fake_get_client)
    monkeypatch.setattr("ra.retrieval.arxiv.asyncio.sleep", _fake_sleep)

    out = await client.download_pdf("2101.00001", tmp_path / "paper.pdf")

    assert out.exists()
    assert out.read_bytes() == b"%PDF-1.7 data"
    assert fake_http.calls == 4
    assert sleep_calls == [1, 2, 4]
