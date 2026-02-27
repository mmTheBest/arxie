from __future__ import annotations

from dataclasses import dataclass

import pytest

from ra.retrieval.arxiv import ArxivPaper
from ra.retrieval.semantic_scholar import Author, Paper as S2Paper
from ra.retrieval.unified import UnifiedRetriever


@dataclass
class DummySemanticScholarClient:
    results: list[S2Paper]

    async def search(self, query: str, limit: int = 10):  # noqa: ARG002
        return self.results

    async def close(self) -> None:
        return None


@dataclass
class DummyArxivClient:
    results: list[ArxivPaper]

    async def search(self, query: str, limit: int = 10, category: str | None = None):  # noqa: ARG002
        return self.results

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_unified_retriever_dedup_by_doi_and_merges_sources():
    s2 = S2Paper(
        paper_id="s2-1",
        title="Paper Title",
        abstract="s2 abstract",
        year=2021,
        authors=[Author(author_id="a1", name="Alice")],
        venue="Venue",
        citation_count=42,
        is_open_access=True,
        pdf_url=None,
        doi="10.1000/xyz",
        arxiv_id=None,
        external_ids={"DOI": "10.1000/xyz"},
    )
    ax = ArxivPaper(
        arxiv_id="2101.00001",
        title="Paper Title (arxiv)",
        abstract="ax abstract",
        authors=["Alice"],
        published="2021-01-01T00:00:00Z",
        updated=None,
        categories=["cs.AI"],
        pdf_url="https://arxiv.org/pdf/2101.00001.pdf",
        doi="doi:10.1000/XYZ",
    )

    r = UnifiedRetriever(
        semantic_scholar=DummySemanticScholarClient([s2]),
        arxiv=DummyArxivClient([ax]),
    )

    results = await r.search("q", limit=10)
    assert len(results) == 1

    p = results[0]
    assert p.source == "both"
    # Prefer Semantic Scholar as primary metadata
    assert p.title == "Paper Title"
    assert p.abstract == "s2 abstract"
    assert p.citation_count == 42
    # Fill missing PDF URL from arXiv
    assert p.pdf_url == "https://arxiv.org/pdf/2101.00001.pdf"


@pytest.mark.asyncio
async def test_unified_retriever_dedup_by_arxiv_id():
    s2 = S2Paper(
        paper_id="s2-1",
        title="Some Title",
        abstract=None,
        year=2021,
        authors=[Author(author_id="a1", name="Alice")],
        venue=None,
        citation_count=1,
        is_open_access=False,
        pdf_url=None,
        doi=None,
        arxiv_id="2101.00001",
        external_ids={"ArXiv": "2101.00001"},
    )
    ax = ArxivPaper(
        arxiv_id="2101.00001",
        title="Some Title",
        abstract="ax abstract",
        authors=["Alice"],
        published="2021-01-01T00:00:00Z",
        updated=None,
        categories=[],
        pdf_url="https://arxiv.org/pdf/2101.00001.pdf",
        doi=None,
    )

    r = UnifiedRetriever(
        semantic_scholar=DummySemanticScholarClient([s2]),
        arxiv=DummyArxivClient([ax]),
    )

    results = await r.search("q", limit=10)
    assert len(results) == 1
    assert results[0].source == "both"


@pytest.mark.asyncio
async def test_unified_retriever_dedup_by_fallback_title_year_first_author():
    p1 = S2Paper(
        paper_id="s2-1",
        title="Deep   Learning for NLP",
        abstract=None,
        year=2020,
        authors=[Author(author_id="a1", name="Alice"), Author(author_id="a2", name="Bob")],
        venue=None,
        citation_count=5,
        is_open_access=False,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        external_ids={},
    )
    p2 = S2Paper(
        paper_id="s2-2",
        title=" deep learning  for  nlp ",
        abstract="newer abstract",
        year=2020,
        authors=[Author(author_id="aX", name="ALICE")],
        venue=None,
        citation_count=10,
        is_open_access=False,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        external_ids={},
    )

    r = UnifiedRetriever(
        semantic_scholar=DummySemanticScholarClient([p1, p2]),
        arxiv=DummyArxivClient([]),
    )

    results = await r.search("q", limit=10)
    assert len(results) == 1
    merged = results[0]
    # Citation count uses max
    assert merged.citation_count == 10
    # Abstract filled from secondary (primary had None)
    assert merged.abstract == "newer abstract"
