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
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.retrieval.unified import Paper, UnifiedRetriever


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


class SearchPapersArgs(BaseModel):
    query: str = Field(..., description="Search query.")
    limit: int = Field(10, ge=1, le=50, description="Max number of results.")
    source: Literal["semantic_scholar", "arxiv", "both"] = Field(
        "both",
        description="Which source(s) to search. 'both' searches S2 + arXiv.",
    )


class GetPaperDetailsArgs(BaseModel):
    identifier: str = Field(
        ...,
        description="Paper identifier: Semantic Scholar paperId, DOI (optionally prefixed with DOI:), or arXiv id.",
    )


class GetPaperFullTextArgs(BaseModel):
    identifier: str = Field(
        ...,
        description=(
            "Paper identifier: Semantic Scholar paperId, DOI (optionally prefixed with DOI:), or arXiv id."
        ),
    )


class GetPaperCitationsArgs(BaseModel):
    paper_id: str = Field(
        ...,
        description="Semantic Scholar paperId (preferred) or another identifier accepted by Semantic Scholar.",
    )
    limit: int = Field(20, ge=1, le=100, description="Max number of citations.")


def make_retrieval_tools(
    retriever: UnifiedRetriever | None = None,
    semantic_scholar: SemanticScholarClient | None = None,
) -> list[StructuredTool]:
    """Create the LangChain tools used by the research agent."""

    retriever = retriever or UnifiedRetriever()
    semantic_scholar = semantic_scholar or SemanticScholarClient()

    async def search_papers(query: str, limit: int = 10, source: str = "both") -> str:
        """Search for papers across one or more sources."""
        sources = ("semantic_scholar", "arxiv") if source == "both" else (source,)
        papers = await retriever.search(query=query, limit=limit, sources=sources)
        payload = {
            "query": query,
            "count": len(papers),
            "results": [_paper_to_dict(p) for p in papers],
        }
        return json.dumps(payload, ensure_ascii=False)

    async def get_paper_details(identifier: str) -> str:
        """Fetch paper metadata for a single paper by identifier."""
        paper = await retriever.get_paper(identifier)
        payload = {"identifier": identifier, "paper": _paper_to_dict(paper) if paper else None}
        return json.dumps(payload, ensure_ascii=False)

    async def get_paper_full_text(identifier: str) -> str:
        """Fetch a paper, download its PDF (if available), and return extracted text."""
        paper = await retriever.get_paper(identifier)
        if not paper:
            return ""
        return await retriever.get_full_text(paper)

    async def get_paper_citations(paper_id: str, limit: int = 20) -> str:
        """Fetch papers that cite the given paper (Semantic Scholar)."""
        citing = await semantic_scholar.get_citations(paper_id=paper_id, limit=limit)
        payload = {
            "paper_id": paper_id,
            "count": len(citing),
            "results": [p.to_dict() for p in citing],
        }
        return json.dumps(payload, ensure_ascii=False)

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
