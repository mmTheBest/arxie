"""Request and response models for the Paperbase API service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "paperbase-api"
    version: str = "0.1.0"


class PaperSummaryResponse(BaseModel):
    id: str
    title: str
    abstract: str | None = None
    publication_year: int | None = None
    venue: str | None = None
    provider: str
    external_id: str
    doi: str | None = None
    arxiv_id: str | None = None


class SearchPapersResponse(BaseModel):
    data: list[PaperSummaryResponse]


class SectionResponse(BaseModel):
    id: str
    title: str
    ordinal: int
    page_start: int | None = None
    page_end: int | None = None
    text: str


class PaperDetailResponse(PaperSummaryResponse):
    pass


class SinglePaperResponse(BaseModel):
    data: PaperDetailResponse


class FulltextResponseData(BaseModel):
    paper_id: str
    title: str
    sections: list[SectionResponse]


class FulltextResponse(BaseModel):
    data: FulltextResponseData


class FigureResponse(BaseModel):
    id: str
    page_number: int | None = None
    figure_label: str | None = None
    caption: str | None = None
    storage_uri: str | None = None


class FiguresResponse(BaseModel):
    data: list[FigureResponse]


class CompareResultsRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    dataset: str = Field(..., min_length=1, max_length=255)
    metric: str = Field(..., min_length=1, max_length=255)
    collection_id: str | None = Field(None, min_length=1, max_length=36)


class CompareResultItemResponse(BaseModel):
    paper_id: str
    paper_title: str
    dataset: str
    method: str | None = None
    metric: str
    value_numeric: float | None = None
    value_text: str | None = None
    comparator_text: str | None = None
    notes: str | None = None


class CompareResultsResponse(BaseModel):
    data: list[CompareResultItemResponse]
