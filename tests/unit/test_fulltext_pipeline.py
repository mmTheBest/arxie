from __future__ import annotations

import json
from pathlib import Path

import pytest

from ra.parsing.pdf_parser import ParsedDocument
from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.tools.retrieval_tools import make_retrieval_tools


@pytest.mark.asyncio
async def test_unified_get_full_text_downloads_parses_and_cleans_up(monkeypatch: pytest.MonkeyPatch):
    parsed_path: Path | None = None

    class FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001,ARG002
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
            return None

        async def get(self, url: str):  # noqa: ARG002
            return FakeResponse(b"%PDF-1.7 dummy")

    class FakePDFParser:
        def parse(self, pdf_path: Path) -> ParsedDocument:
            nonlocal parsed_path
            parsed_path = pdf_path
            assert pdf_path.exists()
            return ParsedDocument(text="EXTRACTED TEXT", pages=["p1"], metadata={})

    monkeypatch.setattr("ra.retrieval.unified.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("ra.retrieval.unified.PDFParser", lambda: FakePDFParser())

    r = UnifiedRetriever(semantic_scholar=None, arxiv=None)
    paper = Paper(
        id="p1",
        title="t",
        abstract=None,
        authors=[],
        year=None,
        venue=None,
        citation_count=None,
        pdf_url="https://example.com/paper.pdf",
        doi=None,
        arxiv_id=None,
        source="arxiv",
    )

    text = await r.get_full_text(paper)
    assert text == "EXTRACTED TEXT"

    assert parsed_path is not None
    # ensure temp file was deleted after parsing
    assert not parsed_path.exists()


@pytest.mark.asyncio
async def test_unified_get_full_text_no_pdf_url_returns_empty(monkeypatch: pytest.MonkeyPatch):
    class BoomAsyncClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001,ARG002
            raise AssertionError("HTTP client should not be constructed when pdf_url is missing")

    monkeypatch.setattr("ra.retrieval.unified.httpx.AsyncClient", BoomAsyncClient)

    r = UnifiedRetriever(semantic_scholar=None, arxiv=None)
    paper = Paper(
        id="p1",
        title="t",
        abstract=None,
        authors=[],
        year=None,
        venue=None,
        citation_count=None,
        pdf_url=None,
        doi=None,
        arxiv_id=None,
        source="arxiv",
    )

    assert await r.get_full_text(paper) == ""


@pytest.mark.asyncio
async def test_tool_get_paper_full_text_calls_retriever(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    class DummyRetriever:
        async def get_paper(self, identifier: str):
            calls.append(f"get_paper:{identifier}")
            return Paper(
                id="p1",
                title="t",
                abstract=None,
                authors=[],
                year=None,
                venue=None,
                citation_count=None,
                pdf_url="https://example.com/paper.pdf",
                doi=None,
                arxiv_id=None,
                source="arxiv",
            )

        async def get_full_text(self, paper: Paper) -> str:  # noqa: ARG002
            calls.append("get_full_text")
            return "FULLTEXT"

    tools = make_retrieval_tools(retriever=DummyRetriever())
    tool = next(t for t in tools if t.name == "get_paper_full_text")

    out = await tool.ainvoke({"identifier": "10.1234/xyz"})
    assert out == "FULLTEXT"
    assert calls == ["get_paper:10.1234/xyz", "get_full_text"]


@pytest.mark.asyncio
async def test_tool_search_papers_handles_internal_errors_gracefully():
    class DummyRetriever:
        async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
            raise RuntimeError("search backend down")

    class DummySemanticScholar:
        async def get_citations(self, paper_id: str, limit: int):  # noqa: ARG002
            return []

    tools = make_retrieval_tools(
        retriever=DummyRetriever(),
        semantic_scholar=DummySemanticScholar(),
    )
    tool = next(t for t in tools if t.name == "search_papers")

    raw = await tool.ainvoke({"query": "rag", "limit": 5, "source": "both"})
    payload = json.loads(raw)

    assert payload["error"] == "tool_execution_failed"
    assert payload["tool"] == "search_papers"


@pytest.mark.asyncio
async def test_tool_get_paper_citations_handles_internal_errors_gracefully():
    class DummyRetriever:
        async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
            return []

    class DummySemanticScholar:
        async def get_citations(self, paper_id: str, limit: int):  # noqa: ARG002
            raise RuntimeError("citations backend down")

    tools = make_retrieval_tools(
        retriever=DummyRetriever(),
        semantic_scholar=DummySemanticScholar(),
    )
    tool = next(t for t in tools if t.name == "get_paper_citations")

    raw = await tool.ainvoke({"paper_id": "s2id", "limit": 10})
    payload = json.loads(raw)

    assert payload["error"] == "tool_execution_failed"
    assert payload["tool"] == "get_paper_citations"


@pytest.mark.asyncio
async def test_tool_get_paper_full_text_handles_internal_errors_gracefully():
    class DummyRetriever:
        async def get_paper(self, identifier: str):  # noqa: ARG002
            raise RuntimeError("get_paper backend down")

    class DummySemanticScholar:
        async def get_citations(self, paper_id: str, limit: int):  # noqa: ARG002
            return []

    tools = make_retrieval_tools(
        retriever=DummyRetriever(),
        semantic_scholar=DummySemanticScholar(),
    )
    tool = next(t for t in tools if t.name == "get_paper_full_text")

    out = await tool.ainvoke({"identifier": "10.1234/xyz"})
    assert out == ""
