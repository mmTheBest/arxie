"""Request and response models for the Paperbase API service."""

from __future__ import annotations

from typing import Any, Literal

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
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class SearchPapersResponse(BaseModel):
    data: list[PaperSummaryResponse]


class SearchChunkHitResponse(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    section_title: str | None = None
    text: str


class SearchChunksResponse(BaseModel):
    data: list[SearchChunkHitResponse]


class SearchArtifactHitResponse(BaseModel):
    artifact_type: str
    artifact_id: str
    paper_id: str
    paper_title: str
    page_number: int | None = None
    label: str | None = None
    caption: str | None = None
    structured_payload: dict[str, Any] = Field(default_factory=dict)


class SearchArtifactsResponse(BaseModel):
    data: list[SearchArtifactHitResponse]


class SearchStatusResponseData(BaseModel):
    backend_configured: bool
    backend_type: str | None = None


class SearchStatusResponse(BaseModel):
    data: SearchStatusResponseData


class BackgroundJobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class SingleBackgroundJobResponse(BaseModel):
    data: BackgroundJobResponse


class BackgroundJobsResponse(BaseModel):
    data: list[BackgroundJobResponse]


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


class TableResponse(BaseModel):
    id: str
    page_number: int | None = None
    table_label: str | None = None
    caption: str | None = None
    storage_uri: str | None = None
    structured_payload: dict[str, Any] = Field(default_factory=dict)


class TablesResponse(BaseModel):
    data: list[TableResponse]


class StructuredNamedArtifactResponse(BaseModel):
    id: str
    display_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultRowArtifactResponse(BaseModel):
    id: str
    dataset_id: str | None = None
    dataset: str | None = None
    method_id: str | None = None
    method: str | None = None
    metric_id: str | None = None
    metric: str | None = None
    value_numeric: float | None = None
    value_text: str | None = None
    comparator_text: str | None = None
    notes: str | None = None


class GlossaryTermArtifactResponse(BaseModel):
    id: str
    term: str
    definition: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FindingArtifactResponse(BaseModel):
    id: str
    statement: str
    polarity: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineeringTrickArtifactResponse(BaseModel):
    id: str
    title: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionRunArtifactResponse(BaseModel):
    id: str
    model_name: str
    prompt_version: str
    schema_version: str
    status: str


class EvidenceSpanArtifactResponse(BaseModel):
    id: str
    extraction_run_id: str | None = None
    target_type: str
    target_id: str | None = None
    page_number: int | None = None
    quote_text: str | None = None
    section_id: str | None = None
    chunk_id: str | None = None


class PaperStructuredDataResponseData(BaseModel):
    paper_id: str
    figures: list[FigureResponse]
    tables: list[TableResponse]
    datasets: list[StructuredNamedArtifactResponse]
    methods: list[StructuredNamedArtifactResponse]
    metrics: list[StructuredNamedArtifactResponse]
    result_rows: list[ResultRowArtifactResponse]
    glossary_terms: list[GlossaryTermArtifactResponse]
    findings: list[FindingArtifactResponse]
    engineering_tricks: list[EngineeringTrickArtifactResponse]
    extraction_runs: list[ExtractionRunArtifactResponse]
    evidence_spans: list[EvidenceSpanArtifactResponse]


class PaperStructuredDataResponse(BaseModel):
    data: PaperStructuredDataResponseData


class CompareResultsRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    dataset: str = Field(..., min_length=1, max_length=255)
    metric: str = Field(..., min_length=1, max_length=255)
    collection_id: str | None = Field(None, min_length=1, max_length=36)
    include_evidence: bool = False


class CompareEvidenceSpanResponse(BaseModel):
    id: str
    page_number: int | None = None
    quote_text: str | None = None


class CompareResultItemResponse(BaseModel):
    result_row_id: str
    paper_id: str
    paper_title: str
    dataset: str
    method: str | None = None
    metric: str
    value_numeric: float | None = None
    value_text: str | None = None
    comparator_text: str | None = None
    notes: str | None = None
    evidence_spans: list[CompareEvidenceSpanResponse] = Field(default_factory=list)


class CompareResultsResponse(BaseModel):
    data: list[CompareResultItemResponse]


class CompareMethodsRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    collection_id: str | None = Field(None, min_length=1, max_length=36)
    dataset: str | None = Field(None, min_length=1, max_length=255)
    metric: str | None = Field(None, min_length=1, max_length=255)
    limit: int = Field(default=20, ge=1, le=100)


class CompareMethodBestResultResponse(BaseModel):
    paper_id: str
    paper_title: str
    dataset: str | None = None
    metric: str | None = None
    value_numeric: float | None = None
    value_text: str | None = None


class CompareMethodItemResponse(BaseModel):
    method: str
    paper_count: int
    result_count: int
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    best_result: CompareMethodBestResultResponse | None = None


class CompareMethodsResponse(BaseModel):
    data: list[CompareMethodItemResponse]


class CompareEngineeringTricksRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    collection_id: str | None = Field(None, min_length=1, max_length=36)
    method: str | None = Field(None, min_length=1, max_length=255)
    limit: int = Field(default=20, ge=1, le=100)


class ComparePaperReferenceResponse(BaseModel):
    paper_id: str
    paper_title: str


class CompareEngineeringTrickItemResponse(BaseModel):
    title: str
    description: str
    paper_count: int
    papers: list[ComparePaperReferenceResponse] = Field(default_factory=list)


class CompareEngineeringTricksResponse(BaseModel):
    data: list[CompareEngineeringTrickItemResponse]


class CompareFigureTableRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    collection_id: str | None = Field(None, min_length=1, max_length=36)
    method: str | None = Field(None, min_length=1, max_length=255)
    limit: int = Field(default=20, ge=1, le=100)


class CompareFigureItemResponse(BaseModel):
    id: str
    paper_id: str
    paper_title: str
    page_number: int | None = None
    figure_label: str | None = None
    caption: str | None = None
    storage_uri: str | None = None


class CompareFiguresResponse(BaseModel):
    data: list[CompareFigureItemResponse]


class CompareTableItemResponse(BaseModel):
    id: str
    paper_id: str
    paper_title: str
    page_number: int | None = None
    table_label: str | None = None
    caption: str | None = None
    storage_uri: str | None = None
    structured_payload: dict[str, Any] = Field(default_factory=dict)


class CompareTablesResponse(BaseModel):
    data: list[CompareTableItemResponse]


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


class WorkspaceCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=5000)
    owner_id: str = Field(default="local-user", min_length=1, max_length=128)
    collection_id: str | None = Field(None, min_length=1, max_length=36)
    saved_query: str | None = Field(None, max_length=1000)
    focus_note: str | None = Field(None, max_length=10000)
    active_filters: dict[str, Any] = Field(default_factory=dict)
    pinned_paper_ids: list[str] = Field(default_factory=list)


class WorkspaceUpdateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=5000)
    collection_id: str | None = Field(None, min_length=1, max_length=36)
    saved_query: str | None = Field(None, max_length=1000)
    focus_note: str | None = Field(None, max_length=10000)
    active_filters: dict[str, Any] | None = None
    pinned_paper_ids: list[str] | None = None


class WorkspaceSummaryResponse(BaseModel):
    id: str
    owner_id: str
    title: str
    description: str | None = None
    collection_id: str | None = None
    saved_query: str | None = None
    focus_note: str | None = None
    active_filters: dict[str, Any] = Field(default_factory=dict)
    pinned_paper_ids: list[str] = Field(default_factory=list)


class WorkspaceDetailResponse(WorkspaceSummaryResponse):
    collection: CollectionSummaryResponse | None = None
    pinned_papers: list[PaperSummaryResponse] = Field(default_factory=list)


class SingleWorkspaceResponse(BaseModel):
    data: WorkspaceDetailResponse


class WorkspacesResponse(BaseModel):
    data: list[WorkspaceSummaryResponse]


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


class CollectionSummaryNamedArtifactResponse(BaseModel):
    id: str
    display_name: str


class CollectionSummaryGlossaryTermResponse(BaseModel):
    id: str
    term: str
    definition: str


class CollectionSummaryEngineeringTrickResponse(BaseModel):
    id: str
    title: str
    description: str


class CollectionSummaryFigureResponse(BaseModel):
    id: str
    page_number: int | None = None
    figure_label: str | None = None
    caption: str | None = None


class CollectionSummaryTableResponse(BaseModel):
    id: str
    page_number: int | None = None
    table_label: str | None = None
    caption: str | None = None


class CollectionSummaryResultRowResponse(BaseModel):
    id: str
    paper_id: str
    paper_title: str
    dataset: str | None = None
    method: str | None = None
    metric: str | None = None
    value_numeric: float | None = None
    value_text: str | None = None


class CollectionStructuredSummaryResponseData(BaseModel):
    collection_id: str
    paper_count: int
    extracted_paper_count: int
    datasets: list[CollectionSummaryNamedArtifactResponse]
    methods: list[CollectionSummaryNamedArtifactResponse]
    metrics: list[CollectionSummaryNamedArtifactResponse]
    figures: list[CollectionSummaryFigureResponse]
    tables: list[CollectionSummaryTableResponse]
    glossary_terms: list[CollectionSummaryGlossaryTermResponse]
    engineering_tricks: list[CollectionSummaryEngineeringTrickResponse]
    top_result_rows: list[CollectionSummaryResultRowResponse]


class CollectionStructuredSummaryResponse(BaseModel):
    data: CollectionStructuredSummaryResponseData


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
    preset_name: str | None = Field(None, min_length=1, max_length=128)
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


class ExtractionProfilePresetResponse(BaseModel):
    name: str
    title: str
    domain: str
    description: str
    schema_payload: dict[str, Any] = Field(default_factory=dict)


class ExtractionProfilePresetsResponse(BaseModel):
    data: list[ExtractionProfilePresetResponse]


class RunCollectionExtractionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    extraction_profile_id: str | None = Field(None, min_length=1, max_length=36)
    schema_payload: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str = Field(..., min_length=1, max_length=64)
    schema_version: str = Field(..., min_length=1, max_length=64)
    limit: int | None = Field(None, ge=1)


class RunCollectionParseRequest(BaseModel):
    limit: int | None = Field(None, ge=1)


class LocalLibraryIngestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    source_dir: str = Field(..., min_length=1, max_length=4096)
    owner_id: str = Field(default="local-user", min_length=1, max_length=128)
    collection_title: str | None = Field(None, min_length=1, max_length=255)
    collection_description: str | None = Field(None, max_length=5000)


class ProviderIdentifierItemRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    kind: Literal["doi", "arxiv", "openalex"]
    value: str = Field(..., min_length=1, max_length=512)


class ProviderIdentifierIngestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    owner_id: str = Field(default="local-user", min_length=1, max_length=128)
    collection_id: str | None = Field(None, min_length=1, max_length=36)
    collection_title: str | None = Field(None, min_length=1, max_length=255)
    collection_description: str | None = Field(None, max_length=5000)
    identifiers: list[ProviderIdentifierItemRequest] = Field(default_factory=list, min_length=1)


class PaperMetadataRefreshRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    paper_ids: list[str] = Field(default_factory=list, min_length=1, max_length=200)


class CollectionExtractionSummaryResponse(BaseModel):
    collection_id: str
    extracted_paper_count: int
    extraction_run_ids: list[str]
    skipped_paper_ids: list[str]


class SingleCollectionExtractionSummaryResponse(BaseModel):
    data: CollectionExtractionSummaryResponse
