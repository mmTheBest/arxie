import time
import socket
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from ra.retrieval.arxiv import ArxivClient
from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.retrieval.unified import UnifiedRetriever


pytestmark = pytest.mark.integration
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True, scope="module")
def _skip_if_external_apis_unreachable() -> None:
    hosts = ("api.semanticscholar.org", "export.arxiv.org")
    unreachable: list[str] = []

    for host in hosts:
        try:
            socket.gethostbyname(host)
        except OSError as exc:
            unreachable.append(f"{host} ({exc})")

    if unreachable:
        joined = ", ".join(unreachable)
        pytest.skip(f"External retrieval APIs unreachable: {joined}")


class APIMetrics:
    """Tiny metrics collector for integration tests.

    Emits summary metrics through the shared logging pipeline.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    async def timed(self, name: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        start = time.perf_counter()
        try:
            return await fn()
        finally:
            self.calls.append((name, time.perf_counter() - start))

    def report(self) -> None:
        total = sum(dt for _, dt in self.calls)
        logger.info(
            "retrieval smoke metrics summary",
            extra={
                "event": "retrieval_smoke.metrics.summary",
                "calls": len(self.calls),
                "total_seconds": round(total, 3),
            },
        )
        by_name: dict[str, list[float]] = {}
        for name, dt in self.calls:
            by_name.setdefault(name, []).append(dt)
        for name, dts in sorted(by_name.items()):
            n = len(dts)
            avg = sum(dts) / n
            mx = max(dts)
            logger.info(
                "retrieval smoke metrics by operation",
                extra={
                    "event": "retrieval_smoke.metrics.operation",
                    "operation": name,
                    "count": n,
                    "avg_seconds": round(avg, 3),
                    "max_seconds": round(mx, 3),
                },
            )


@pytest.fixture
def api_metrics() -> APIMetrics:
    m = APIMetrics()
    yield m
    m.report()


@pytest.mark.asyncio
async def test_semantic_scholar_search(api_metrics: APIMetrics) -> None:
    client = SemanticScholarClient()
    try:
        results = await api_metrics.timed(
            "semantic_scholar.search",
            lambda: client.search("transformer attention mechanism", limit=5),
        )
        assert len(results) >= 1
        first = results[0]
        assert first.title and first.title.strip()
        assert first.authors and first.authors[0].name.strip()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_arxiv_search(api_metrics: APIMetrics) -> None:
    client = ArxivClient()
    try:
        results = await api_metrics.timed(
            "arxiv.search",
            lambda: client.search("large language models", limit=5),
        )
        assert len(results) >= 1
        assert results[0].title and results[0].title.strip()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_unified_search(api_metrics: APIMetrics) -> None:
    async with UnifiedRetriever() as r:
        results = await api_metrics.timed(
            "unified.search",
            lambda: r.search("retrieval augmented generation", limit=10),
        )

    assert len(results) >= 1

    # Smoke-check dedup: unified.search() should never return duplicate IDs.
    ids = [p.id for p in results]
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_get_paper_by_arxiv_id(api_metrics: APIMetrics) -> None:
    async with UnifiedRetriever() as r:
        paper = await api_metrics.timed(
            "unified.get_paper(arxiv)",
            lambda: r.get_paper("1706.03762"),
        )

    assert paper is not None
    assert "attention" in (paper.title or "").lower()


@pytest.mark.asyncio
async def test_get_paper_by_doi(api_metrics: APIMetrics) -> None:
    # arXiv's DOI for Attention Is All You Need is minted by arXiv: 10.48550/arXiv.1706.03762
    async with UnifiedRetriever() as r:
        paper = await api_metrics.timed(
            "unified.get_paper(doi)",
            lambda: r.get_paper("10.48550/arXiv.1706.03762"),
        )

    assert paper is not None
    assert "attention" in (paper.title or "").lower()
