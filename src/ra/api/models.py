"""Request/response models for the REST API layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ra import __version__
from ra.retrieval.unified import Paper

SearchSource = Literal["semantic_scholar", "arxiv", "both"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(10, ge=1, le=100)
    source: SearchSource = Field("both")


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    identifier: str = Field(..., min_length=1, max_length=256)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=4000)


class PaperResponse(BaseModel):
    id: str
    title: str
    abstract: str | None
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int | None
    pdf_url: str | None
    doi: str | None
    arxiv_id: str | None
    source: SearchSource
    citation: str

    @classmethod
    def from_paper(cls, paper: Paper) -> "PaperResponse":
        return cls(
            id=paper.id,
            title=paper.title,
            abstract=paper.abstract,
            authors=paper.authors,
            year=paper.year,
            venue=paper.venue,
            citation_count=paper.citation_count,
            pdf_url=paper.pdf_url,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            source=paper.source,
            citation=paper.to_citation(),
        )


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[PaperResponse]


class RetrieveResponse(BaseModel):
    identifier: str
    paper: PaperResponse


class AnswerResponse(BaseModel):
    query: str
    answer: str


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "academic-research-assistant"
    version: str = __version__


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: list[dict[str, Any]] | None = None
