from __future__ import annotations

import json

import httpx
import pytest

from ra.parsing.pdf_parser import ParsedDocument, Section
from ra.retrieval.unified import Paper
from ra.tools.retrieval_tools import ReadPaperFullTextArgs, make_retrieval_tools


@pytest.mark.asyncio
async def test_read_paper_fulltext_tool_exists_and_has_expected_schema():
    class DummyRetriever:
        async def get_paper(self, identifier: str):  # noqa: ARG002
            return None

    class DummySemanticScholar:
        async def get_citations(self, paper_id: str, limit: int):  # noqa: ARG002
            return []

    tools = make_retrieval_tools(
        retriever=DummyRetriever(),
        semantic_scholar=DummySemanticScholar(),
    )
    tool = next(t for t in tools if t.name == "read_paper_fulltext")

    assert tool.args_schema is ReadPaperFullTextArgs
    schema = tool.args_schema.model_json_schema()
    assert schema["required"] == ["paper_id"]
    assert "paper_id" in schema["properties"]


@pytest.mark.asyncio
async def test_read_paper_fulltext_downloads_pdf_and_returns_structured_sections(
    monkeypatch: pytest.MonkeyPatch,
):
    pdf_bytes = b"%PDF-1.7 dummy pdf"
    download_urls: list[str] = []
    client_init_kwargs: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001
            client_init_kwargs.append(dict(kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
            return None

        async def get(self, url: str):
            download_urls.append(url)
            return FakeResponse(pdf_bytes)

    class FakeRetriever:
        async def get_paper(self, identifier: str) -> Paper:
            return Paper(
                id=identifier,
                title="Attention Is All You Need",
                abstract=None,
                authors=["A"],
                year=2017,
                venue="NeurIPS",
                citation_count=0,
                pdf_url="https://example.org/paper.pdf",
                doi=None,
                arxiv_id=None,
                source="semantic_scholar",
            )

    class FakePDFParser:
        def parse_from_bytes(self, data: bytes) -> ParsedDocument:
            assert data == pdf_bytes
            return ParsedDocument(text="full text", pages=["page 1"], metadata={})

        def extract_sections(self, doc: ParsedDocument) -> list[Section]:
            assert doc.text == "full text"
            return [
                Section(title="Abstract", content="Abstract section", page_start=0),
                Section(title="Methodology", content="Methods section", page_start=1),
                Section(title="Results", content="Results section", page_start=2),
                Section(title="Discussion", content="Discussion section", page_start=3),
                Section(title="Conclusions", content="Conclusion section", page_start=4),
            ]

    monkeypatch.setattr("ra.tools.retrieval_tools.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("ra.tools.retrieval_tools.PDFParser", lambda: FakePDFParser())

    tools = make_retrieval_tools(retriever=FakeRetriever())
    tool = next(t for t in tools if t.name == "read_paper_fulltext")

    raw = await tool.ainvoke({"paper_id": "paper-123"})
    payload = json.loads(raw)

    assert payload["paper_id"] == "paper-123"
    assert payload["title"] == "Attention Is All You Need"
    assert payload["abstract"] == "Abstract section"
    assert payload["methods"] == "Methods section"
    assert payload["results"] == "Results section"
    assert payload["discussion"] == "Discussion section"
    assert payload["conclusion"] == "Conclusion section"
    assert download_urls == ["https://example.org/paper.pdf"]
    assert client_init_kwargs
    assert "timeout" in client_init_kwargs[0]


@pytest.mark.asyncio
async def test_read_paper_fulltext_returns_pdf_unavailable_error():
    class FakeRetriever:
        async def get_paper(self, identifier: str) -> Paper:
            return Paper(
                id=identifier,
                title="No PDF Paper",
                abstract=None,
                authors=[],
                year=None,
                venue=None,
                citation_count=None,
                pdf_url=None,
                doi=None,
                arxiv_id=None,
                source="semantic_scholar",
            )

    tools = make_retrieval_tools(retriever=FakeRetriever())
    tool = next(t for t in tools if t.name == "read_paper_fulltext")

    raw = await tool.ainvoke({"paper_id": "p-no-pdf"})
    payload = json.loads(raw)

    assert payload["tool"] == "read_paper_fulltext"
    assert payload["error"] == "pdf_unavailable"


@pytest.mark.asyncio
async def test_read_paper_fulltext_returns_download_failure_error(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeRetriever:
        async def get_paper(self, identifier: str) -> Paper:
            return Paper(
                id=identifier,
                title="Download Fail Paper",
                abstract=None,
                authors=[],
                year=None,
                venue=None,
                citation_count=None,
                pdf_url="https://example.org/unreachable.pdf",
                doi=None,
                arxiv_id=None,
                source="semantic_scholar",
            )

    class FailingAsyncClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN001,ARG002
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
            return None

        async def get(self, url: str):
            request = httpx.Request("GET", url)
            raise httpx.RequestError("network down", request=request)

    monkeypatch.setattr("ra.tools.retrieval_tools.httpx.AsyncClient", FailingAsyncClient)

    tools = make_retrieval_tools(retriever=FakeRetriever())
    tool = next(t for t in tools if t.name == "read_paper_fulltext")

    raw = await tool.ainvoke({"paper_id": "p-download-fail"})
    payload = json.loads(raw)

    assert payload["tool"] == "read_paper_fulltext"
    assert payload["error"] == "download_failed"


@pytest.mark.asyncio
async def test_read_paper_fulltext_returns_parse_failure_error(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeRetriever:
        async def get_paper(self, identifier: str) -> Paper:
            return Paper(
                id=identifier,
                title="Parse Fail Paper",
                abstract=None,
                authors=[],
                year=None,
                venue=None,
                citation_count=None,
                pdf_url="https://example.org/paper.pdf",
                doi=None,
                arxiv_id=None,
                source="semantic_scholar",
            )

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

    class FailingPDFParser:
        def parse_from_bytes(self, data: bytes) -> ParsedDocument:  # noqa: ARG002
            raise RuntimeError("parser exploded")

    monkeypatch.setattr("ra.tools.retrieval_tools.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("ra.tools.retrieval_tools.PDFParser", lambda: FailingPDFParser())

    tools = make_retrieval_tools(retriever=FakeRetriever())
    tool = next(t for t in tools if t.name == "read_paper_fulltext")

    raw = await tool.ainvoke({"paper_id": "p-parse-fail"})
    payload = json.loads(raw)

    assert payload["tool"] == "read_paper_fulltext"
    assert payload["error"] == "parse_failed"
