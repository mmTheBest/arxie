import pytest

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


@pytest.mark.asyncio
async def test_request_uses_rate_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = _CountingRateLimiter()
    client = SemanticScholarClient(rate_limiter=limiter, max_retries=1)

    async def _fake_get_client():
        return _FakeHttpClient()

    monkeypatch.setattr(client, "_get_client", _fake_get_client)

    await client._request("GET", "/paper/search", params={"query": "transformer"})
    assert limiter.calls == 1
