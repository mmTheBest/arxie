"""Request/response models for the REST API layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ra import __version__
from ra.retrieval.unified import Paper

SearchSource = Literal["semantic_scholar", "arxiv", "both"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "query": "transformer architecture for long-context summarization",
                "limit": 5,
                "source": "both",
            }
        },
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language query for paper discovery.",
        examples=["transformer architecture for long-context summarization"],
    )
    limit: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of papers to return (1-100).",
        examples=[10],
    )
    source: SearchSource = Field(
        "both",
        description="Search source selector: Semantic Scholar, arXiv, or both.",
        examples=["both"],
    )


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={"example": {"identifier": "10.5555/3295222.3295349"}},
    )

    identifier: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Paper identifier (DOI, arXiv ID, or provider-specific paper ID).",
        examples=["10.5555/3295222.3295349"],
    )


class AnswerRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "query": "What are the main limitations of retrieval-augmented generation?"
            }
        },
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Research question to answer using the retrieval and reasoning pipeline.",
        examples=["What are the main limitations of retrieval-augmented generation?"],
    )


class SearchBatchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    requests: list[SearchRequest] = Field(
        ...,
        min_length=1,
        max_length=25,
        description="Batch of search requests to execute asynchronously.",
    )
    max_concurrency: int = Field(
        4,
        ge=1,
        le=32,
        description="Maximum number of in-flight searches processed in parallel.",
        examples=[4],
    )


class RetrieveBatchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    requests: list[RetrieveRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Batch of paper identifier lookups to execute asynchronously.",
    )
    max_concurrency: int = Field(
        8,
        ge=1,
        le=32,
        description="Maximum number of in-flight retrieve calls processed in parallel.",
        examples=[8],
    )


class PaperResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "p1",
                "title": "Attention Is All You Need",
                "abstract": "Transformers are sequence transduction models...",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "year": 2017,
                "venue": "NeurIPS",
                "citation_count": 100000,
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                "doi": "10.5555/3295222.3295349",
                "arxiv_id": "1706.03762",
                "source": "both",
                "citation": "Vaswani, A., ... (2017). Attention Is All You Need.",
            }
        }
    )

    id: str = Field(..., description="Provider-specific identifier for the paper.")
    title: str = Field(..., description="Paper title.")
    abstract: str | None = Field(None, description="Paper abstract when available.")
    authors: list[str] = Field(..., description="Ordered list of author names.")
    year: int | None = Field(None, description="Publication year.")
    venue: str | None = Field(None, description="Publication venue, conference, or journal.")
    citation_count: int | None = Field(None, description="Citation count from metadata provider.")
    pdf_url: str | None = Field(None, description="Direct URL to a PDF when available.")
    doi: str | None = Field(None, description="Digital Object Identifier (DOI), if present.")
    arxiv_id: str | None = Field(None, description="arXiv identifier, if present.")
    source: SearchSource = Field(..., description="Originating source for this paper metadata.")
    citation: str = Field(..., description="Formatted citation string.")

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
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "transformer architecture for long-context summarization",
                "count": 1,
                "results": [
                    {
                        "id": "p1",
                        "title": "Attention Is All You Need",
                        "abstract": "Transformers are sequence transduction models...",
                        "authors": ["Ashish Vaswani", "Noam Shazeer"],
                        "year": 2017,
                        "venue": "NeurIPS",
                        "citation_count": 100000,
                        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                        "doi": "10.5555/3295222.3295349",
                        "arxiv_id": "1706.03762",
                        "source": "both",
                        "citation": "Vaswani, A., ... (2017). Attention Is All You Need.",
                    }
                ],
            }
        }
    )

    query: str = Field(..., description="Original search query.")
    count: int = Field(..., description="Total number of papers returned.")
    results: list[PaperResponse] = Field(..., description="Search result papers.")


class SearchBatchItemResponse(BaseModel):
    query: str = Field(..., description="Original query for this batch item.")
    count: int = Field(..., description="Number of papers returned for this query.")
    results: list[PaperResponse] = Field(..., description="Search results for this query.")


class SearchBatchResponse(BaseModel):
    count: int = Field(..., description="Total number of queries in the batch.")
    results: list[SearchBatchItemResponse] = Field(
        ...,
        description="Ordered results for each query in the request batch.",
    )


class RetrieveResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "identifier": "10.5555/3295222.3295349",
                "paper": PaperResponse.model_config["json_schema_extra"]["example"],
            }
        }
    )

    identifier: str = Field(..., description="Identifier requested by the client.")
    paper: PaperResponse = Field(..., description="Resolved paper metadata.")


class RetrieveBatchItemResponse(BaseModel):
    identifier: str = Field(..., description="Identifier requested by the client.")
    paper: PaperResponse | None = Field(
        None,
        description="Resolved paper metadata or null when not found.",
    )


class RetrieveBatchResponse(BaseModel):
    count: int = Field(..., description="Total number of identifiers in the batch.")
    found: int = Field(..., description="Number of identifiers that resolved to a paper.")
    results: list[RetrieveBatchItemResponse] = Field(
        ...,
        description="Ordered retrieval results for each identifier.",
    )


class AnswerResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the main limitations of retrieval-augmented generation?",
                "answer": "## Answer\nRAG systems can struggle with stale indexes...",
            }
        }
    )

    query: str = Field(..., description="Original research question.")
    answer: str = Field(..., description="Structured Markdown answer from the research agent.")


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "service": "academic-research-assistant",
                "version": __version__,
            }
        }
    )

    status: str = Field("ok", description="Health status for the API service.")
    service: str = Field("academic-research-assistant", description="Service identifier.")
    version: str = Field(__version__, description="Current application version.")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "invalid_input",
                "message": "query must not be empty",
                "details": [{"loc": ["body", "query"], "msg": "Field required"}],
            }
        }
    )

    error: str = Field(..., description="Machine-readable error key.")
    message: str = Field(..., description="Human-readable error message.")
    details: list[dict[str, Any]] | None = Field(
        None,
        description="Optional structured details for validation and downstream debugging.",
    )
