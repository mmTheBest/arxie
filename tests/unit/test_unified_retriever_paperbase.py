from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from ra.parsing.pdf_parser import Section
from ra.retrieval.semantic_scholar import Author, Paper as S2Paper
from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.tools.retrieval_tools import make_retrieval_tools


@dataclass
class DummyPaperbaseGateway:
    search_results: list[Paper] = field(default_factory=list)
    paper: Paper | None = None
    sections: list[Section] = field(default_factory=list)
    structured_data: dict[str, object] | None = None
    search_calls: list[tuple[str, int]] = field(default_factory=list)
    get_paper_calls: list[str] = field(default_factory=list)
    get_sections_calls: list[str] = field(default_factory=list)
    get_structured_data_calls: list[str] = field(default_factory=list)

    async def search(self, query: str, limit: int = 10) -> list[Paper]:
        self.search_calls.append((query, limit))
        return self.search_results[:limit]

    async def get_paper(self, identifier: str) -> Paper | None:
        self.get_paper_calls.append(identifier)
        return self.paper

    async def get_stored_sections(self, identifier: str) -> list[Section]:
        self.get_sections_calls.append(identifier)
        return list(self.sections)

    async def get_paper_structured_data(self, identifier: str) -> dict[str, object] | None:
        self.get_structured_data_calls.append(identifier)
        return dict(self.structured_data or {}) if self.structured_data is not None else None

    async def close(self) -> None:
        return None


@dataclass
class DummySemanticScholarClient:
    results: list[S2Paper] = field(default_factory=list)
    paper: S2Paper | None = None
    search_calls: list[tuple[str, int]] = field(default_factory=list)
    get_paper_calls: list[str] = field(default_factory=list)

    async def search(self, query: str, limit: int = 10):
        self.search_calls.append((query, limit))
        return self.results[:limit]

    async def get_paper(self, identifier: str):
        self.get_paper_calls.append(identifier)
        return self.paper

    async def close(self) -> None:
        return None


@dataclass
class DummyArxivClient:
    search_calls: list[tuple[str, int]] = field(default_factory=list)

    async def search(self, query: str, limit: int = 10, category: str | None = None):  # noqa: ARG002
        self.search_calls.append((query, limit))
        return []

    async def get_paper(self, arxiv_id: str):  # noqa: ARG002
        return None

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_unified_retriever_search_uses_paperbase_first_without_external_fallback_when_enough():
    gateway = DummyPaperbaseGateway(
        search_results=[
            Paper(
                id="pb-1",
                title="scLong",
                abstract="local corpus result",
                authors=[],
                year=2026,
                venue="Nature",
                citation_count=None,
                pdf_url=None,
                doi=None,
                arxiv_id=None,
                source="both",
            )
        ]
    )
    s2 = DummySemanticScholarClient(
        results=[
            S2Paper(
                paper_id="s2-1",
                title="fallback",
                abstract=None,
                year=2026,
                authors=[Author(author_id="a1", name="Alice")],
                venue=None,
                citation_count=10,
                is_open_access=False,
                pdf_url=None,
                doi=None,
                arxiv_id=None,
                external_ids={},
            )
        ]
    )
    arxiv = DummyArxivClient()
    retriever = UnifiedRetriever(
        semantic_scholar=s2,
        arxiv=arxiv,
        paperbase_gateway=gateway,
    )

    results = await retriever.search("scLong", limit=1)

    assert [paper.id for paper in results] == ["pb-1"]
    assert gateway.search_calls == [("scLong", 1)]
    assert s2.search_calls == []
    assert arxiv.search_calls == []


@pytest.mark.asyncio
async def test_unified_retriever_search_falls_back_to_external_when_paperbase_insufficient():
    gateway = DummyPaperbaseGateway(search_results=[])
    s2 = DummySemanticScholarClient(
        results=[
            S2Paper(
                paper_id="s2-1",
                title="external result",
                abstract=None,
                year=2025,
                authors=[Author(author_id="a1", name="Alice")],
                venue=None,
                citation_count=10,
                is_open_access=False,
                pdf_url=None,
                doi=None,
                arxiv_id=None,
                external_ids={},
            )
        ]
    )
    retriever = UnifiedRetriever(
        semantic_scholar=s2,
        arxiv=DummyArxivClient(),
        paperbase_gateway=gateway,
    )

    results = await retriever.search("external", limit=1)

    assert [paper.id for paper in results] == ["s2-1"]
    assert gateway.search_calls == [("external", 1)]
    assert s2.search_calls


@pytest.mark.asyncio
async def test_unified_retriever_get_paper_uses_paperbase_first():
    gateway = DummyPaperbaseGateway(
        paper=Paper(
            id="pb-1",
            title="scLong",
            abstract="local corpus result",
            authors=[],
            year=2026,
            venue="Nature",
            citation_count=None,
            pdf_url=None,
            doi="10.1000/scLong",
            arxiv_id=None,
            source="both",
        )
    )
    s2 = DummySemanticScholarClient()
    retriever = UnifiedRetriever(
        semantic_scholar=s2,
        arxiv=DummyArxivClient(),
        paperbase_gateway=gateway,
    )

    paper = await retriever.get_paper("pb-1")

    assert paper is not None
    assert paper.id == "pb-1"
    assert gateway.get_paper_calls == ["pb-1"]
    assert s2.get_paper_calls == []


@pytest.mark.asyncio
async def test_read_paper_fulltext_uses_stored_paperbase_sections_before_pdf_download():
    paper = Paper(
        id="pb-1",
        title="scLong",
        abstract="local corpus result",
        authors=[],
        year=2026,
        venue="Nature",
        citation_count=None,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="both",
    )
    gateway = DummyPaperbaseGateway(
        paper=paper,
        sections=[
            Section(title="Abstract", content="Abstract section", page_start=0),
            Section(title="Methods", content="Methods section", page_start=1),
        ],
    )
    retriever = UnifiedRetriever(
        semantic_scholar=DummySemanticScholarClient(),
        arxiv=DummyArxivClient(),
        paperbase_gateway=gateway,
    )
    tools = make_retrieval_tools(retriever=retriever)
    tool = next(t for t in tools if t.name == "read_paper_fulltext")

    raw = await tool.ainvoke({"paper_id": "pb-1"})
    payload = json.loads(raw)

    assert payload["paper_id"] == "pb-1"
    assert payload["abstract"] == "Abstract section"
    assert payload["methods"] == "Methods section"
    assert gateway.get_sections_calls == ["pb-1"]


@pytest.mark.asyncio
async def test_get_paper_structured_data_tool_uses_paperbase_gateway():
    paper = Paper(
        id="pb-1",
        title="scLong",
        abstract="local corpus result",
        authors=[],
        year=2026,
        venue="Nature",
        citation_count=None,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="both",
    )
    gateway = DummyPaperbaseGateway(
        paper=paper,
        structured_data={
            "paper_id": "pb-1",
            "methods": [{"display_name": "scLong"}],
            "limitations": [{"statement": "Evaluation is limited to curated datasets."}],
            "result_rows": [{"metric": "AUROC", "value_numeric": 0.91}],
        },
    )
    retriever = UnifiedRetriever(
        semantic_scholar=DummySemanticScholarClient(),
        arxiv=DummyArxivClient(),
        paperbase_gateway=gateway,
    )
    tools = make_retrieval_tools(retriever=retriever)
    tool = next(t for t in tools if t.name == "get_paper_structured_data")

    raw = await tool.ainvoke({"paper_id": "pb-1"})
    payload = json.loads(raw)

    assert payload["paper_id"] == "pb-1"
    assert payload["methods"][0]["display_name"] == "scLong"
    assert payload["limitations"][0]["statement"] == "Evaluation is limited to curated datasets."
    assert gateway.get_structured_data_calls == ["pb-1"]


@pytest.mark.asyncio
async def test_unified_retriever_get_full_text_uses_stored_paperbase_sections():
    paper = Paper(
        id="pb-1",
        title="scLong",
        abstract="local corpus result",
        authors=[],
        year=2026,
        venue="Nature",
        citation_count=None,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="both",
    )
    gateway = DummyPaperbaseGateway(
        sections=[
            Section(title="Abstract", content="Abstract section", page_start=0),
            Section(title="Methods", content="Methods section", page_start=1),
        ],
    )
    retriever = UnifiedRetriever(
        semantic_scholar=DummySemanticScholarClient(),
        arxiv=DummyArxivClient(),
        paperbase_gateway=gateway,
    )

    full_text = await retriever.get_full_text(paper)

    assert full_text == "Abstract\n\nAbstract section\n\nMethods\n\nMethods section"
    assert gateway.get_sections_calls == ["pb-1"]
