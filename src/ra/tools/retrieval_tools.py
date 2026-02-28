"""LangChain tool wrappers for paper retrieval.

These tools wrap the project's retrieval clients so that a LangChain agent can:
- search for papers
- fetch paper metadata by identifier
- fetch citations for a paper

For now, tools return JSON-serializable dicts (via `json.dumps`) to keep the
interface stable for downstream UI / API layers.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from langchain_core.tools import StructuredTool

from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.retrieval.unified import Paper, UnifiedRetriever
from ra.utils.security import sanitize_identifier, sanitize_user_text

logger = logging.getLogger(__name__)


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

    async def get_paper_details(identifier: str) -> str:
        """Fetch paper metadata for a single paper by identifier."""
        try:
            paper = await retriever.get_paper(identifier)
            payload = {"identifier": identifier, "paper": _paper_to_dict(paper) if paper else None}
            return json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            logger.exception("Tool execution failed: get_paper_details")
            return _tool_error_payload("get_paper_details", exc)

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

    return [
        StructuredTool(
            name="search_papers",
            description=(
                "Search for relevant academic papers. Use this first to discover candidate sources. "
                "Returns JSON with a list of normalized paper metadata and citation strings."
            ),
            args_schema=SearchPapersArgs,
            coroutine=search_papers,
        ),
        StructuredTool(
            name="get_paper_details",
            description=(
                "Get detailed metadata for a specific paper by identifier (DOI, arXiv id, or Semantic Scholar paperId). "
                "Returns JSON with normalized metadata and a formatted citation string when available."
            ),
            args_schema=GetPaperDetailsArgs,
            coroutine=get_paper_details,
        ),
        StructuredTool(
            name="get_paper",
            description=(
                "Alias for get_paper_details. Get detailed metadata for a specific paper by identifier (DOI, arXiv id, or Semantic Scholar paperId). "
                "Returns JSON with normalized metadata and a formatted citation string when available."
            ),
            args_schema=GetPaperDetailsArgs,
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
            coroutine=get_paper_full_text,
        ),

        StructuredTool(
            name="get_paper_citations",
            description=(
                "Get papers that cite a given paper (Semantic Scholar). Useful for forward citation chasing. "
                "Returns JSON with citing papers."
            ),
            args_schema=GetPaperCitationsArgs,
            coroutine=get_paper_citations,
        ),
    ]
