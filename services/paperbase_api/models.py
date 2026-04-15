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


class CollectionCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=5000)
    owner_id: str = Field(default="local-user", min_length=1, max_length=128)
    scope_type: str = Field(default="private", min_length=1, max_length=64)
    tags: list[str] = Field(default_factory=list)
    extraction_profile_id: str | None = Field(None, min_length=1, max_length=36)


class CollectionSummaryResponse(BaseModel):
    id: str
    owner_id: str
    scope_type: str
    title: str
    description: str | None = None
    extraction_profile_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class SingleCollectionResponse(BaseModel):
    data: CollectionSummaryResponse


class CollectionsResponse(BaseModel):
    data: list[CollectionSummaryResponse]


class CollectionPaperCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    paper_id: str = Field(..., min_length=1, max_length=36)
    position: int | None = Field(None, ge=1)
    membership_note: str | None = Field(None, max_length=5000)


class CollectionPaperMembershipResponse(BaseModel):
    id: str
    collection_id: str
    paper_id: str
    position: int | None = None
    membership_note: str | None = None
    paper: PaperSummaryResponse


class CollectionPapersResponse(BaseModel):
    data: list[CollectionPaperMembershipResponse]


class SingleCollectionPaperResponse(BaseModel):
    data: CollectionPaperMembershipResponse


class AnnotationCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    author_id: str = Field(default="local-user", min_length=1, max_length=128)
    collection_id: str | None = Field(None, min_length=1, max_length=36)
    target_type: str = Field(..., min_length=1, max_length=64)
    target_id: str = Field(..., min_length=1, max_length=36)
    body: str = Field(..., min_length=1, max_length=20000)
    tags: list[str] = Field(default_factory=list)
    status: str | None = Field(None, min_length=1, max_length=64)


class AnnotationResponse(BaseModel):
    id: str
    author_id: str
    collection_id: str | None = None
    target_type: str
    target_id: str
    body: str
    tags: list[str] = Field(default_factory=list)
    status: str | None = None


class SingleAnnotationResponse(BaseModel):
    data: AnnotationResponse


class AnnotationsResponse(BaseModel):
    data: list[AnnotationResponse]


class ExtractionProfileCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    owner_id: str = Field(default="local-user", min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=5000)
    scope_type: str = Field(default="private", min_length=1, max_length=64)
    schema_payload: dict[str, Any] = Field(default_factory=dict)
    active: bool = True


class ExtractionProfileResponse(BaseModel):
    id: str
    owner_id: str
    name: str
    description: str | None = None
    scope_type: str
    schema_payload: dict[str, Any] = Field(default_factory=dict)
    active: bool


class SingleExtractionProfileResponse(BaseModel):
    data: ExtractionProfileResponse


class ExtractionProfilesResponse(BaseModel):
    data: list[ExtractionProfileResponse]


class RunCollectionExtractionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    extraction_profile_id: str | None = Field(None, min_length=1, max_length=36)
    schema_payload: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str = Field(..., min_length=1, max_length=64)
    schema_version: str = Field(..., min_length=1, max_length=64)
    limit: int | None = Field(None, ge=1)


class CollectionExtractionSummaryResponse(BaseModel):
    collection_id: str
    extracted_paper_count: int
    extraction_run_ids: list[str]
    skipped_paper_ids: list[str]


class SingleCollectionExtractionSummaryResponse(BaseModel):
    data: CollectionExtractionSummaryResponse
