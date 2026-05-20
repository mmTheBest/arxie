"""Pydantic models for study-agent API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ra.retrieval.unified import Paper
from ra.study import (
    EvidenceReference,
    StudyAgentRun,
    StudyBrief,
    StudyRecommendation,
    StudyRunStep,
    StudySource,
    StudySourceType,
    StudyTaskOutput,
    StudyTaskType,
)


class StudyCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    study_id: str = Field(..., min_length=1, max_length=128)
    title: str = Field(..., min_length=1, max_length=300)
    research_goal: str = Field(..., min_length=1, max_length=4000)
    collection_id: str | None = Field(None, max_length=256)
    domain: str | None = Field(None, max_length=256)
    current_method: str | None = Field(None, max_length=4000)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class StudyBriefUpdateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    expected_version: int = Field(..., ge=0)
    title: str | None = Field(None, min_length=1, max_length=300)
    research_goal: str | None = Field(None, min_length=1, max_length=4000)
    collection_id: str | None = Field(None, max_length=256)
    domain: str | None = Field(None, max_length=256)
    current_method: str | None = Field(None, max_length=4000)
    datasets: list[str] | None = None
    metrics: list[str] | None = None
    constraints: list[str] | None = None
    decisions: list[str] | None = None
    risks: list[str] | None = None
    open_questions: list[str] | None = None


class StudySourceCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    source_type: StudySourceType
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1, max_length=40000)
    summary: str | None = Field(None, max_length=2000)
    extracted_facts: list[str] = Field(default_factory=list)


class StudyPaperInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    paper_id: str = Field(..., min_length=1, max_length=256)
    title: str = Field(..., min_length=1, max_length=1000)
    abstract: str | None = Field(None, max_length=20000)
    year: int | None = None
    source: str = Field(default="semantic_scholar", max_length=64)

    def to_domain(self) -> Paper:
        source = str(self.source or "semantic_scholar").strip().lower()
        if source not in {"semantic_scholar", "arxiv", "both"}:
            source = "semantic_scholar"
        return Paper(
            id=self.paper_id,
            title=self.title,
            abstract=self.abstract,
            authors=[],
            year=self.year,
            venue=None,
            citation_count=None,
            pdf_url=None,
            doi=None,
            arxiv_id=None,
            source=source,
        )


class StudyRunCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    task_type: StudyTaskType
    query: str = Field(..., min_length=1, max_length=4000)
    papers: list[StudyPaperInput] = Field(default_factory=list)


class StudyBriefResponse(BaseModel):
    study_id: str
    title: str
    research_goal: str
    collection_id: str | None = None
    domain: str | None = None
    current_method: str | None = None
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    version: int
    created_at: str
    updated_at: str

    @classmethod
    def from_domain(cls, brief: StudyBrief) -> "StudyBriefResponse":
        return cls(**brief.to_dict())


class StudySourceResponse(BaseModel):
    source_id: str
    study_id: str
    source_type: StudySourceType
    title: str
    content: str
    summary: str
    extracted_facts: list[str] = Field(default_factory=list)
    version: int
    created_at: str
    updated_at: str

    @classmethod
    def from_domain(cls, source: StudySource) -> "StudySourceResponse":
        return cls(**source.to_dict())


class EvidenceReferenceResponse(BaseModel):
    reference_id: str
    source_type: str
    source_id: str
    source_title: str
    support_label: str
    summary: str
    locator: str | None = None
    confidence: float | None = None
    version: int | None = None

    @classmethod
    def from_domain(cls, ref: EvidenceReference) -> "EvidenceReferenceResponse":
        return cls(**ref.to_dict())


class StudyRecommendationResponse(BaseModel):
    category: str
    title: str
    rationale: str
    evidence_reference_ids: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, recommendation: StudyRecommendation) -> "StudyRecommendationResponse":
        return cls(**recommendation.to_dict())


class StudyTaskOutputResponse(BaseModel):
    summary: str
    recommendations: list[StudyRecommendationResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_context: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceReferenceResponse] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, output: StudyTaskOutput) -> "StudyTaskOutputResponse":
        return cls(
            summary=output.summary,
            recommendations=[
                StudyRecommendationResponse.from_domain(item)
                for item in output.recommendations
            ],
            warnings=list(output.warnings),
            missing_context=list(output.missing_context),
            evidence_refs=[EvidenceReferenceResponse.from_domain(item) for item in output.evidence_refs],
            next_actions=list(output.next_actions),
        )


class StudyRunStepResponse(BaseModel):
    step_type: str
    message: str
    status: str
    tool_name: str | None = None
    created_at: str
    details: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, step: StudyRunStep) -> "StudyRunStepResponse":
        return cls(**step.to_dict())


class StudyRunResponse(BaseModel):
    run_id: str
    study_id: str
    task_type: StudyTaskType
    query: str
    status: str
    context_pack_id: str
    steps: list[StudyRunStepResponse] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    output: StudyTaskOutputResponse | None = None
    created_at: str
    completed_at: str | None = None

    @classmethod
    def from_domain(cls, run: StudyAgentRun) -> "StudyRunResponse":
        return cls(
            run_id=run.run_id,
            study_id=run.study_id,
            task_type=run.task_type,
            query=run.query,
            status=run.status.value,
            context_pack_id=run.context_pack_id,
            steps=[StudyRunStepResponse.from_domain(step) for step in run.steps],
            warnings=list(run.warnings),
            output=StudyTaskOutputResponse.from_domain(run.output) if run.output else None,
            created_at=run.created_at,
            completed_at=run.completed_at,
        )


class StudyRunListResponse(BaseModel):
    study_id: str
    count: int
    results: list[StudyRunResponse]
