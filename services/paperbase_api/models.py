"""Request and response models for the Arxie API service."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from paperbase.version import get_version


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "arxie-api"
    version: str = Field(default_factory=get_version)


class DependencyStatusResponse(BaseModel):
    name: str
    ok: bool
    detail: str
    required: bool = True


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    service: str = "arxie-api"
    version: str = Field(default_factory=get_version)
    dependencies: list[DependencyStatusResponse] = Field(default_factory=list)


class ModelProviderStatusResponse(BaseModel):
    provider: str
    model_name: str | None = None
    configured: bool
    usable: bool
    missing_setup: list[str] = Field(default_factory=list)
    setup_hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    command: str | None = None
    command_available: bool | None = None
    allow_agentic_cli: bool = False
    login_status: str = "not_applicable"
    login_status_checked: bool = False
    login_status_command: str | None = None


class WorkerModelProviderStatusResponse(BaseModel):
    provider: str | None = None
    matches_api_provider: bool | None = None
    source: str
    setup_hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorkerHeartbeatStatusResponse(BaseModel):
    heartbeat_status: Literal["online", "stale", "unavailable", "unknown_project"]
    expected_runtime_scope: Literal["default", "project", "unknown_project"]
    active_worker_count: int = 0
    stale_worker_count: int = 0
    latest_seen_seconds_ago: int | None = None
    heartbeat_stale_after_seconds: int
    setup_hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProjectDataPathStatusResponse(BaseModel):
    runtime_data_scope: Literal["default", "project", "unknown_project"]
    project_id_present: bool
    project_found: bool | None = None
    registry_available: bool
    registry_file_exists: bool
    registered_project_count: int = 0
    database_backend: str
    hosted_mode: bool
    host_path_import_policy: Literal[
        "local_unrestricted",
        "hosted_allowlisted",
        "hosted_disabled",
    ]
    allowed_root_count: int = 0
    setup_hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RuntimeStatusResponseData(BaseModel):
    service: str = "arxie-api"
    version: str = Field(default_factory=get_version)
    model_provider: ModelProviderStatusResponse
    worker_model_provider: WorkerModelProviderStatusResponse
    worker_heartbeat: WorkerHeartbeatStatusResponse
    project_data_paths: ProjectDataPathStatusResponse


class RuntimeStatusResponse(BaseModel):
    data: RuntimeStatusResponseData


class RuntimeModelProviderSmokeResponseData(BaseModel):
    status: Literal["success", "not_configured", "failed"]
    grade: Literal[
        "passed",
        "missing_setup",
        "provider_disabled",
        "client_factory_missing",
        "client_factory_failed",
        "provider_call_failed",
    ]
    grade_label: str
    provider: str
    model_name: str | None = None
    message: str
    next_actions: list[str] = Field(default_factory=list)
    missing_setup: list[str] = Field(default_factory=list)
    setup_hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RuntimeModelProviderSmokeResponse(BaseModel):
    data: RuntimeModelProviderSmokeResponseData


class ProjectOpenRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    root_path: str = Field(..., min_length=1, max_length=4096)
    title: str | None = Field(None, min_length=1, max_length=255)


class ProjectResponse(BaseModel):
    id: str
    title: str
    root_path: str
    database_path: str
    last_opened_at: str


class SingleProjectResponse(BaseModel):
    data: ProjectResponse


class ProjectsResponse(BaseModel):
    data: list[ProjectResponse]


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


class LimitationArtifactResponse(BaseModel):
    id: str
    statement: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EngineeringTrickArtifactResponse(BaseModel):
    id: str
    title: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchDesignElementArtifactResponse(BaseModel):
    id: str
    element_type: str
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
    limitations: list[LimitationArtifactResponse]
    engineering_tricks: list[EngineeringTrickArtifactResponse]
    research_design_elements: list[ResearchDesignElementArtifactResponse]
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
    paper_count: int = 0
    parsed_paper_count: int = 0
    extracted_paper_count: int = 0
    latest_job_status: str | None = None
    latest_parse_job_status: str | None = None
    latest_extraction_job_status: str | None = None
    failed_job_count: int = 0


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


class StudyBriefItemRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=255)
    text: str = Field(..., min_length=1, max_length=5000)


StudyBriefSourceId = Annotated[str, Field(min_length=1, max_length=36)]


class StudyBriefPayload(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    aim: str = Field(default="", max_length=10000)
    hypothesis: str = Field(default="", max_length=10000)
    constraints: list[StudyBriefItemRequest] = Field(default_factory=list, max_length=50)
    confirmed_decisions: list[StudyBriefItemRequest] = Field(default_factory=list, max_length=50)
    open_risks: list[StudyBriefItemRequest] = Field(default_factory=list, max_length=50)
    linked_source_ids: list[StudyBriefSourceId] = Field(default_factory=list, max_length=200)


class StudyBriefUpdateRequest(BaseModel):
    expected_version: int = Field(..., ge=0)
    brief: StudyBriefPayload


class StudyBriefProposalChange(BaseModel):
    field: Literal[
        "aim",
        "hypothesis",
        "constraints",
        "confirmed_decisions",
        "open_risks",
        "linked_source_ids",
    ]
    before: Any
    after: Any


class StudyBriefProposalResponse(BaseModel):
    workspace_id: str
    artifact_id: str
    artifact_type: str
    artifact_title: str
    current_version: int
    proposed_brief: StudyBriefPayload
    changes: list[StudyBriefProposalChange] = Field(default_factory=list)


class SingleStudyBriefProposalResponse(BaseModel):
    data: StudyBriefProposalResponse


class StudyBriefProposalAcceptRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    artifact_id: str = Field(..., min_length=1, max_length=36)
    expected_version: int = Field(..., ge=0)
    brief: StudyBriefPayload


class StudyBriefResponse(BaseModel):
    id: str | None = None
    workspace_id: str
    brief: StudyBriefPayload
    version: int
    updated_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class SingleStudyBriefResponse(BaseModel):
    data: StudyBriefResponse


StudySourceType = Literal["text", "code_path", "draft_path", "results_path"]


class StudySourceCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    source_type: StudySourceType
    title: str = Field(..., min_length=1, max_length=255)
    path: str | None = Field(None, min_length=1, max_length=2000)
    content: str | None = Field(None, min_length=1)


class ArtifactFileEntryResponse(BaseModel):
    name: str
    relative_path: str
    path: str
    entry_type: Literal["directory", "file"]
    source_type: StudySourceType | None = None
    size_bytes: int | None = None
    source_mtime_ns: int | None = None
    selectable: bool = False


class ArtifactFolderListingResponse(BaseModel):
    root_path: str
    relative_path: str
    entries: list[ArtifactFileEntryResponse] = Field(default_factory=list)
    truncated: bool = False
    ignored_count: int = 0


class SingleArtifactFolderListingResponse(BaseModel):
    data: ArtifactFolderListingResponse


class StudySourceResponse(BaseModel):
    id: str
    workspace_id: str
    source_type: str
    title: str
    path: str | None = None
    content: str | None = None
    summary: str | None = None
    read_status: str
    error_message: str | None = None
    source_size_bytes: int | None = None
    source_mtime_ns: int | None = None
    is_stale: bool = False


class SingleStudySourceResponse(BaseModel):
    data: StudySourceResponse


class StudySourcesResponse(BaseModel):
    data: list[StudySourceResponse]


ResearchArtifactType = Literal[
    "field_patterns",
    "hypotheses",
    "literature_review",
    "comparison",
    "experiment_plan",
    "critique",
    "experiment_backlog",
    "benchmark_plan",
    "revision_plan",
    "assumption_map",
]
ResearchPaperLabelValue = Literal[
    "exemplar",
    "baseline",
    "preliminary",
    "similar_method",
    "strong_design",
    "weak_design",
    "ignore",
    "neutral",
]


class ResearchThreadCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(..., min_length=1, max_length=255)
    collection_id: str = Field(..., min_length=1, max_length=36)
    owner_id: str = Field(default="local-user", min_length=1, max_length=128)
    workspace_id: str | None = Field(None, min_length=1, max_length=36)
    selected_paper_ids: list[str] = Field(default_factory=list, max_length=200)


class ResearchThreadResponse(BaseModel):
    id: str
    owner_id: str
    title: str
    collection_id: str
    workspace_id: str | None = None
    selected_paper_ids: list[str] = Field(default_factory=list)
    status: str


class ResearchMessageCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(..., min_length=1, max_length=20000)
    artifact_type: ResearchArtifactType | None = None
    suggestion_id: str | None = Field(None, min_length=1, max_length=128)
    source_ids: list[str] = Field(default_factory=list, max_length=50)


class ResearchMessageResponse(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    artifact_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchArtifactPatchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str | None = Field(None, min_length=1, max_length=255)
    is_saved: bool | None = None
    saved_format: Literal["markdown", "csv", "script", "note"] | None = None
    saved_title: str | None = Field(None, min_length=1, max_length=255)


class ResearchArtifactResponse(BaseModel):
    id: str
    collection_id: str
    thread_id: str | None = None
    artifact_type: str
    title: str
    status: str
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    evidence_payload: dict[str, Any] = Field(default_factory=dict)
    model_name: str | None = None
    prompt_version: str | None = None
    error_message: str | None = None
    is_saved: bool = False
    saved_format: str | None = None
    saved_title: str | None = None


class ResearchThreadDetailResponseData(ResearchThreadResponse):
    messages: list[ResearchMessageResponse] = Field(default_factory=list)
    artifacts: list[ResearchArtifactResponse] = Field(default_factory=list)


class SingleResearchThreadResponse(BaseModel):
    data: ResearchThreadResponse


class ResearchThreadDetailResponse(BaseModel):
    data: ResearchThreadDetailResponseData


class ResearchThreadsResponse(BaseModel):
    data: list[ResearchThreadResponse]


class SingleResearchArtifactResponse(BaseModel):
    data: ResearchArtifactResponse


class ResearchArtifactsResponse(BaseModel):
    data: list[ResearchArtifactResponse]


class ResearchAgentStepResponse(BaseModel):
    id: str
    run_id: str
    attempt_number: int = 1
    ordinal: int
    step_type: str
    label: str
    status: str
    input_json: dict[str, Any] = Field(default_factory=dict)
    output_json: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class StudyContextPackResponse(BaseModel):
    id: str
    run_id: str
    attempt_number: int = 1
    collection_id: str
    workspace_id: str | None = None
    task_type: str
    cache_key: str | None = None
    context_summary: dict[str, Any] = Field(default_factory=dict)
    intelligence_layers_summary: dict[str, Any] = Field(default_factory=dict)
    retrieval_summary: dict[str, Any] = Field(default_factory=dict)
    context_materialization_summary: dict[str, Any] = Field(default_factory=dict)
    selection_diagnostics: dict[str, Any] = Field(default_factory=dict)
    selected_item_counts: dict[str, Any] = Field(default_factory=dict)
    readiness_warnings: list[str] = Field(default_factory=list)


class ResearchValidationReportResponse(BaseModel):
    id: str
    run_id: str
    attempt_number: int = 1
    artifact_id: str
    harness_status: str
    missing_evidence: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    readiness_blockers: list[str] = Field(default_factory=list)
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    validation_issue_counts: dict[str, Any] = Field(default_factory=dict)
    support_label_counts: dict[str, Any] = Field(default_factory=dict)
    recommendation_health: dict[str, Any] = Field(default_factory=dict)
    reference_integrity: dict[str, int] = Field(default_factory=dict)
    support_screening: dict[str, Any] = Field(default_factory=dict)
    task_evidence_roles: dict[str, list[str]] = Field(default_factory=dict)
    task_quality: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)


class ResearchAgentRunResponse(BaseModel):
    id: str
    thread_id: str | None = None
    artifact_id: str
    collection_id: str
    workspace_id: str | None = None
    skill_id: str
    artifact_type: str
    model_policy: str
    status: str
    input_json: dict[str, Any] = Field(default_factory=dict)
    model_name: str | None = None
    error_message: str | None = None
    steps: list[ResearchAgentStepResponse] = Field(default_factory=list)
    context_pack: StudyContextPackResponse | None = None
    validation_report: ResearchValidationReportResponse | None = None


class SingleResearchAgentRunResponse(BaseModel):
    data: ResearchAgentRunResponse


class ResearchMessageJobResponseData(BaseModel):
    message: ResearchMessageResponse
    artifact: ResearchArtifactResponse
    job: BackgroundJobResponse
    run_id: str | None = None


class ResearchMessageJobResponse(BaseModel):
    data: ResearchMessageJobResponseData


class ResearchSuggestionResponse(BaseModel):
    id: str
    label: str
    instruction: str
    skill_id: str
    artifact_type: str
    readiness: str


class ResearchSuggestionsResponse(BaseModel):
    data: list[ResearchSuggestionResponse]


class PaperResearchLabelPatchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_label: ResearchPaperLabelValue
    notes: str | None = Field(None, max_length=10000)


class PaperResearchLabelResponse(BaseModel):
    id: str
    collection_id: str
    paper_id: str
    user_label: str
    inferred_label: str | None = None
    inferred_signals: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class SinglePaperResearchLabelResponse(BaseModel):
    data: PaperResearchLabelResponse


class PaperResearchLabelsResponse(BaseModel):
    data: list[PaperResearchLabelResponse]


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
    is_parsed: bool = False
    is_extracted: bool = False
    parsed_section_count: int = 0
    completed_extraction_count: int = 0
    latest_parse_job_status: str | None = None
    latest_extraction_job_status: str | None = None
    latest_job_error: str | None = None


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


class CollectionSummaryResearchDesignElementResponse(BaseModel):
    id: str
    paper_id: str
    element_type: str
    title: str
    description: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionSummaryLimitationResponse(BaseModel):
    id: str
    statement: str


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


class PaperExtractionQualityResponse(BaseModel):
    paper_id: str
    paper_title: str
    structured_entity_count: int
    entity_counts: dict[str, int] = Field(default_factory=dict)
    evidence_span_count: int
    anchored_evidence_span_count: int
    unresolved_evidence_span_count: int
    evidence_span_anchor_diagnostics: dict[str, Any] = Field(default_factory=dict)
    latest_completed_extraction_run_id: str | None = None
    extraction_profile_id: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    freshness_status: str
    stale_reasons: list[str] = Field(default_factory=list)
    missing_structured_evidence: list[str] = Field(default_factory=list)


class CollectionExtractionQualityResponse(BaseModel):
    paper_count: int
    papers_with_completed_extraction_count: int
    fresh_paper_count: int
    stale_paper_count: int
    missing_extraction_paper_count: int
    total_structured_entity_count: int
    total_evidence_span_count: int
    anchored_evidence_span_count: int
    unresolved_evidence_span_count: int
    missing_structured_evidence: list[str] = Field(default_factory=list)
    papers: list[PaperExtractionQualityResponse] = Field(default_factory=list)


class CollectionExtractionRecoveryActionResponse(BaseModel):
    action_id: str
    action_type: Literal["parse", "extract", "review_evidence"]
    can_queue_job: bool
    priority: int
    label: str
    description: str
    paper_count: int
    paper_ids: list[str] = Field(default_factory=list)
    truncated: bool = False
    stale_reasons: list[str] = Field(default_factory=list)
    missing_structured_evidence: list[str] = Field(default_factory=list)
    unresolved_evidence_span_count: int = 0
    unresolved_evidence_span_samples: list[dict[str, Any]] = Field(default_factory=list)


class CollectionStructuredSummaryResponseData(BaseModel):
    collection_id: str
    paper_count: int
    parsed_paper_count: int
    extracted_paper_count: int
    latest_job_status: str | None = None
    latest_parse_job_status: str | None = None
    latest_extraction_job_status: str | None = None
    failed_job_count: int = 0
    datasets: list[CollectionSummaryNamedArtifactResponse]
    methods: list[CollectionSummaryNamedArtifactResponse]
    metrics: list[CollectionSummaryNamedArtifactResponse]
    figures: list[CollectionSummaryFigureResponse]
    tables: list[CollectionSummaryTableResponse]
    glossary_terms: list[CollectionSummaryGlossaryTermResponse]
    limitations: list[CollectionSummaryLimitationResponse]
    engineering_tricks: list[CollectionSummaryEngineeringTrickResponse]
    research_design_elements: list[CollectionSummaryResearchDesignElementResponse]
    top_result_rows: list[CollectionSummaryResultRowResponse]
    extraction_quality: CollectionExtractionQualityResponse
    extraction_recovery_actions: list[CollectionExtractionRecoveryActionResponse] = (
        Field(default_factory=list)
    )


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
    paper_ids: list[str] | None = Field(None, max_length=200)


class RunCollectionParseRequest(BaseModel):
    limit: int | None = Field(None, ge=1)
    paper_ids: list[str] | None = Field(None, max_length=200)


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
