"""LangChain tool wrappers for paper retrieval.

These tools wrap the project's retrieval clients so that a LangChain agent can:
- search for papers
- fetch paper metadata by identifier
- fetch citations for a paper

For now, tools return JSON-serializable dicts (via `json.dumps`) to keep the
interface stable for downstream UI / API layers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Awaitable
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from langchain_core.tools import StructuredTool

from ra.parsing import PDFParser, Section
from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.utils.security import sanitize_identifier, sanitize_user_text

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class _BackgroundEventLoop:
    """Run sync tool calls on a single dedicated asyncio loop."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _bootstrap_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop and self._loop.is_running():
                return self._loop

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._bootstrap_loop,
                name="ra-retrieval-tools-loop",
                daemon=True,
            )
            self._thread.start()
            started = self._ready.wait(timeout=5)
            if not started or self._loop is None:
                raise RuntimeError("Failed to initialize background event loop for tools.")
            return self._loop

    def run(self, coro: Awaitable[_T]) -> _T:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()


_SYNC_LOOP = _BackgroundEventLoop()


def _run_coroutine_sync(coro: Awaitable[_T]) -> _T:
    """Run an async coroutine from sync tool invocations."""
    return _SYNC_LOOP.run(coro)


def _paper_to_dict(p: Paper) -> dict:
    return {
        "id": p.id,
        "title": p.title,
        "abstract": p.abstract,
        "authors": p.authors,
        "year": p.year,
        "venue": p.venue,
        "citation_count": p.citation_count,
        "pdf_url": p.pdf_url,
        "doi": p.doi,
        "arxiv_id": p.arxiv_id,
        "source": p.source,
        "citation": p.to_citation(),
    }


def _tool_error_payload(tool: str, error: Exception) -> str:
    return json.dumps(
        {
            "tool": tool,
            "error": "tool_execution_failed",
            "message": str(error),
        },
        ensure_ascii=False,
    )


class SearchPapersArgs(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Search query.")
    limit: int = Field(10, ge=1, le=50, description="Max number of results.")
    source: Literal["semantic_scholar", "arxiv", "both"] = Field(
        "both",
        description="Which source(s) to search. 'both' searches S2 + arXiv.",
    )

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        return sanitize_user_text(value, field_name="query", max_length=1000)


class GetPaperDetailsArgs(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    identifier: str = Field(
        ...,
        description="Paper identifier: Semantic Scholar paperId, DOI (optionally prefixed with DOI:), or arXiv id.",
    )

    @field_validator("identifier")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        return sanitize_identifier(value, field_name="identifier", max_length=256)


class GetPaperFullTextArgs(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    identifier: str = Field(
        ...,
        description=(
            "Paper identifier: Semantic Scholar paperId, DOI (optionally prefixed with DOI:), or arXiv id."
        ),
    )

    @field_validator("identifier")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        return sanitize_identifier(value, field_name="identifier", max_length=256)


class ReadPaperFullTextArgs(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    paper_id: str = Field(
        ...,
        description=(
            "Paper identifier: Semantic Scholar paperId, DOI (optionally prefixed with DOI:), or arXiv id."
        ),
    )

    @field_validator("paper_id")
    @classmethod
    def _validate_paper_id(cls, value: str) -> str:
        return sanitize_identifier(value, field_name="paper_id", max_length=256)


class GetPaperCitationsArgs(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    paper_id: str = Field(
        ...,
        description="Semantic Scholar paperId (preferred) or another identifier accepted by Semantic Scholar.",
    )
    limit: int = Field(20, ge=1, le=100, description="Max number of citations.")

    @field_validator("paper_id")
    @classmethod
    def _validate_paper_id(cls, value: str) -> str:
        return sanitize_identifier(value, field_name="paper_id", max_length=256)


def _tool_named_error_payload(tool: str, error: str, message: str, **extra: object) -> str:
    payload: dict[str, object] = {
        "tool": tool,
        "error": error,
        "message": message,
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _find_section_content(
    sections: list[Section],
    *,
    heading_aliases: tuple[str, ...],
) -> str | None:
    aliases = tuple(alias.lower() for alias in heading_aliases)
    for section in sections:
        title = (section.title or "").strip().lower()
        if any(alias in title for alias in aliases):
            content = (section.content or "").strip()
            if content:
                return content
    return None


def make_retrieval_tools(
    retriever: UnifiedRetriever | None = None,
    semantic_scholar: SemanticScholarClient | None = None,
) -> list[StructuredTool]:
    """Create the LangChain tools used by the research agent."""

    retriever = retriever or UnifiedRetriever()
    semantic_scholar = semantic_scholar or SemanticScholarClient()

    async def search_papers(query: str, limit: int = 10, source: str = "both") -> str:
        """Search for papers across one or more sources."""
        try:
            sources = ("semantic_scholar", "arxiv") if source == "both" else (source,)
            papers = await retriever.search(query=query, limit=limit, sources=sources)
            payload = {
                "query": query,
                "count": len(papers),
                "results": [_paper_to_dict(p) for p in papers],
            }
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            logger.exception("Tool execution failed: search_papers")
            return _tool_error_payload("search_papers", exc)

    def search_papers_sync(query: str, limit: int = 10, source: str = "both") -> str:
        return _run_coroutine_sync(search_papers(query=query, limit=limit, source=source))

    async def get_paper_details(identifier: str) -> str:
        """Fetch paper metadata for a single paper by identifier."""
        try:
            paper = await retriever.get_paper(identifier)
            payload = {"identifier": identifier, "paper": _paper_to_dict(paper) if paper else None}
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            logger.exception("Tool execution failed: get_paper_details")
            return _tool_error_payload("get_paper_details", exc)

    def get_paper_details_sync(identifier: str) -> str:
        return _run_coroutine_sync(get_paper_details(identifier=identifier))

    async def get_paper_full_text(identifier: str) -> str:
        """Fetch a paper, download its PDF (if available), and return extracted text."""
        try:
            paper = await retriever.get_paper(identifier)
            if not paper:
                return ""
            return await retriever.get_full_text(paper)
        except Exception:
            logger.exception("Tool execution failed: get_paper_full_text")
            return ""

    def get_paper_full_text_sync(identifier: str) -> str:
        return _run_coroutine_sync(get_paper_full_text(identifier=identifier))

    async def read_paper_fulltext(paper_id: str) -> str:
        """Download and parse a paper PDF into structured core sections."""
        tool_name = "read_paper_fulltext"
        try:
            paper = await retriever.get_paper(paper_id)
        except Exception as exc:
            logger.exception("Tool execution failed: read_paper_fulltext")
            return _tool_error_payload(tool_name, exc)

        if not paper:
            return _tool_named_error_payload(
                tool_name,
                "paper_not_found",
                "No paper found for the provided identifier.",
                paper_id=paper_id,
            )

        pdf_url = (paper.pdf_url or "").strip()
        if not pdf_url:
            return _tool_named_error_payload(
                tool_name,
                "pdf_unavailable",
                "No PDF URL is available for this paper.",
                paper_id=paper.id,
            )

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()
        except httpx.TimeoutException:
            return _tool_named_error_payload(
                tool_name,
                "download_timeout",
                "Timed out while downloading the PDF.",
                paper_id=paper.id,
                pdf_url=pdf_url,
            )
        except httpx.HTTPError as exc:
            logger.warning("PDF download failed for %s: %s", paper.id, exc)
            return _tool_named_error_payload(
                tool_name,
                "download_failed",
                "Failed to download the PDF.",
                paper_id=paper.id,
                pdf_url=pdf_url,
            )

        try:
            parser = PDFParser()
            parsed = parser.parse_from_bytes(response.content)
            sections = parser.extract_sections(parsed)
        except Exception as exc:
            logger.warning("PDF parse failed for %s: %s", paper.id, exc)
            return _tool_named_error_payload(
                tool_name,
                "parse_failed",
                "Failed to parse the PDF.",
                paper_id=paper.id,
                pdf_url=pdf_url,
            )

        payload = {
            "paper_id": paper.id,
            "title": (paper.title or "").strip() or None,
            "abstract": _find_section_content(
                sections,
                heading_aliases=("abstract",),
            )
            or (paper.abstract.strip() if isinstance(paper.abstract, str) and paper.abstract.strip() else None),
            "methods": _find_section_content(
                sections,
                heading_aliases=(
                    "method",
                    "methods",
                    "methodology",
                    "materials and methods",
                    "experiment",
                ),
            ),
            "results": _find_section_content(
                sections,
                heading_aliases=("results",),
            ),
            "discussion": _find_section_content(
                sections,
                heading_aliases=("discussion",),
            ),
            "conclusion": _find_section_content(
                sections,
                heading_aliases=("conclusion", "conclusions", "future work"),
            ),
        }
        return json.dumps(payload, ensure_ascii=False)

    def read_paper_fulltext_sync(paper_id: str) -> str:
        return _run_coroutine_sync(read_paper_fulltext(paper_id=paper_id))

    async def get_paper_citations(paper_id: str, limit: int = 20) -> str:
        """Fetch papers that cite the given paper (Semantic Scholar)."""
        try:
            citing = await semantic_scholar.get_citations(paper_id=paper_id, limit=limit)
            payload = {
                "paper_id": paper_id,
                "count": len(citing),
                "results": [p.to_dict() for p in citing],
            }
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            logger.exception("Tool execution failed: get_paper_citations")
            return _tool_error_payload("get_paper_citations", exc)

    def get_paper_citations_sync(paper_id: str, limit: int = 20) -> str:
        return _run_coroutine_sync(get_paper_citations(paper_id=paper_id, limit=limit))

    return [
        StructuredTool(
            name="search_papers",
            description=(
                "Search for relevant academic papers. Use this first to discover candidate sources. "
                "Returns JSON with a list of normalized paper metadata and citation strings."
            ),
            args_schema=SearchPapersArgs,
            func=search_papers_sync,
            coroutine=search_papers,
        ),
        StructuredTool(
            name="get_paper_details",
            description=(
                "Get detailed metadata for a specific paper by identifier (DOI, arXiv id, or Semantic Scholar paperId). "
                "Returns JSON with normalized metadata and a formatted citation string when available."
            ),
            args_schema=GetPaperDetailsArgs,
            func=get_paper_details_sync,
            coroutine=get_paper_details,
        ),
        StructuredTool(
            name="get_paper",
            description=(
                "Alias for get_paper_details. Get detailed metadata for a specific paper by identifier (DOI, arXiv id, or Semantic Scholar paperId). "
                "Returns JSON with normalized metadata and a formatted citation string when available."
            ),
            args_schema=GetPaperDetailsArgs,
            func=get_paper_details_sync,
            coroutine=get_paper_details,
        ),

        StructuredTool(
            name="get_paper_full_text",
            description=(
                "Download a paper's PDF (when available) and extract its full text. "
                "Input can be a Semantic Scholar paperId, a DOI, or an arXiv id. "
                "Returns plain text (empty string if unavailable or extraction fails)."
            ),
            args_schema=GetPaperFullTextArgs,
            func=get_paper_full_text_sync,
            coroutine=get_paper_full_text,
        ),
        StructuredTool(
            name="read_paper_fulltext",
            description=(
                "Read and structure full text for a paper from its PDF. "
                "Use this when the user asks for specific methodology, results, discussion details, or conclusions. "
                "Returns JSON with title, abstract, methods, results, discussion, and conclusion sections."
            ),
            args_schema=ReadPaperFullTextArgs,
            func=read_paper_fulltext_sync,
            coroutine=read_paper_fulltext,
        ),

        StructuredTool(
            name="get_paper_citations",
            description=(
                "Get papers that cite a given paper (Semantic Scholar). Useful for forward citation chasing. "
                "Returns JSON with citing papers."
            ),
            args_schema=GetPaperCitationsArgs,
            func=get_paper_citations_sync,
            coroutine=get_paper_citations,
        ),
    ]
