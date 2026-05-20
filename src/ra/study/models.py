"""Domain models for study-agent memory, context, and runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class StudyTaskType(str, Enum):
    """Research tasks supported by the deterministic study-agent runtime."""

    DESIGN_EXPERIMENTS = "design_experiments"
    FIND_BENCHMARKS = "find_benchmarks"
    REVIEW_DRAFT_CLAIMS = "review_draft_claims"


class StudySourceType(str, Enum):
    """User-attached source types for local study context."""

    DRAFT = "draft"
    NOTE = "note"
    CODE_SUMMARY = "code_summary"
    RESULT_SUMMARY = "result_summary"


class EvidenceSourceType(str, Enum):
    """Origin of evidence referenced by a recommendation."""

    PAPER = "paper"
    STUDY_BRIEF = "study_brief"
    USER_SOURCE = "user_source"


class EvidenceSupportLabel(str, Enum):
    """Relationship between a source and an agent output."""

    SUPPORTS = "supports"
    MIXED = "mixed"
    INFERRED = "inferred"
    USER_PROVIDED = "user_provided"
    MISSING = "missing"


class StudyRunStatus(str, Enum):
    """Lifecycle state for a study-agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class StudyToolCategory(str, Enum):
    """Tool side-effect class used by the study-agent runtime."""

    READ = "read"
    WRITE = "write"
    ADVANCED = "advanced"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with stable `Z` suffix."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class StudyBrief:
    """Durable memory for a user's active study."""

    study_id: str
    title: str
    research_goal: str
    collection_id: str | None = None
    domain: str | None = None
    current_method: str | None = None
    datasets: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    version: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudyBrief":
        return cls(
            study_id=_str(payload.get("study_id")),
            title=_str(payload.get("title")),
            research_goal=_str(payload.get("research_goal")),
            collection_id=_optional_str(payload.get("collection_id")),
            domain=_optional_str(payload.get("domain")),
            current_method=_optional_str(payload.get("current_method")),
            datasets=_tuple_str(payload.get("datasets")),
            metrics=_tuple_str(payload.get("metrics")),
            constraints=_tuple_str(payload.get("constraints")),
            decisions=_tuple_str(payload.get("decisions")),
            risks=_tuple_str(payload.get("risks")),
            open_questions=_tuple_str(payload.get("open_questions")),
            source_ids=_tuple_str(payload.get("source_ids")),
            version=int(payload.get("version", 0)),
            created_at=_str(payload.get("created_at") or utc_now_iso()),
            updated_at=_str(payload.get("updated_at") or utc_now_iso()),
        )


@dataclass(frozen=True, slots=True)
class StudySource:
    """User-provided context attached to a study."""

    source_id: str
    study_id: str
    source_type: StudySourceType
    title: str
    content: str
    summary: str
    extracted_facts: tuple[str, ...] = ()
    version: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudySource":
        return cls(
            source_id=_str(payload.get("source_id")),
            study_id=_str(payload.get("study_id")),
            source_type=StudySourceType(_str(payload.get("source_type") or StudySourceType.NOTE.value)),
            title=_str(payload.get("title")),
            content=_str(payload.get("content")),
            summary=_str(payload.get("summary")),
            extracted_facts=_tuple_str(payload.get("extracted_facts")),
            version=int(payload.get("version", 0)),
            created_at=_str(payload.get("created_at") or utc_now_iso()),
            updated_at=_str(payload.get("updated_at") or utc_now_iso()),
        )


@dataclass(frozen=True, slots=True)
class StudySourceContextRef:
    """Source summary selected for a context pack."""

    source_id: str
    source_title: str
    source_type: StudySourceType
    summary: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudySourceContextRef":
        return cls(
            source_id=_str(payload.get("source_id")),
            source_title=_str(payload.get("source_title")),
            source_type=StudySourceType(_str(payload.get("source_type") or StudySourceType.NOTE.value)),
            summary=_str(payload.get("summary")),
            version=int(payload.get("version", 0)),
        )


@dataclass(frozen=True, slots=True)
class StudyPaperContextRef:
    """Paper metadata selected for a context pack."""

    paper_id: str
    title: str
    abstract: str | None
    year: int | None = None
    relevance_score: float = 0.0
    source: str = "semantic_scholar"

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudyPaperContextRef":
        year = payload.get("year")
        return cls(
            paper_id=_str(payload.get("paper_id")),
            title=_str(payload.get("title")),
            abstract=_optional_str(payload.get("abstract")),
            year=int(year) if year is not None else None,
            relevance_score=float(payload.get("relevance_score", 0.0)),
            source=_str(payload.get("source") or "semantic_scholar"),
        )


@dataclass(frozen=True, slots=True)
class StudyContextPack:
    """Explicit bounded context selected for one study-agent run."""

    context_pack_id: str
    study_id: str
    task_type: StudyTaskType
    query: str
    brief_fields: dict[str, str] = field(default_factory=dict)
    source_refs: tuple[StudySourceContextRef, ...] = ()
    paper_refs: tuple[StudyPaperContextRef, ...] = ()
    selection_reasons: dict[str, str] = field(default_factory=dict)
    missing_context: tuple[str, ...] = ()
    token_budget: int = 8000
    source_versions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudyContextPack":
        return cls(
            context_pack_id=_str(payload.get("context_pack_id")),
            study_id=_str(payload.get("study_id")),
            task_type=StudyTaskType(_str(payload.get("task_type"))),
            query=_str(payload.get("query")),
            brief_fields={str(k): str(v) for k, v in dict(payload.get("brief_fields", {})).items()},
            source_refs=tuple(
                StudySourceContextRef.from_dict(item)
                for item in payload.get("source_refs", [])
                if isinstance(item, dict)
            ),
            paper_refs=tuple(
                StudyPaperContextRef.from_dict(item)
                for item in payload.get("paper_refs", [])
                if isinstance(item, dict)
            ),
            selection_reasons={
                str(k): str(v) for k, v in dict(payload.get("selection_reasons", {})).items()
            },
            missing_context=_tuple_str(payload.get("missing_context")),
            token_budget=int(payload.get("token_budget", 8000)),
            source_versions={
                str(k): int(v) for k, v in dict(payload.get("source_versions", {})).items()
            },
        )


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """Normalized reference attached to an agent recommendation."""

    reference_id: str
    source_type: EvidenceSourceType
    source_id: str
    source_title: str
    support_label: EvidenceSupportLabel
    summary: str
    locator: str | None = None
    confidence: float | None = None
    version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceReference":
        confidence = payload.get("confidence")
        version = payload.get("version")
        return cls(
            reference_id=_str(payload.get("reference_id")),
            source_type=EvidenceSourceType(_str(payload.get("source_type"))),
            source_id=_str(payload.get("source_id")),
            source_title=_str(payload.get("source_title")),
            support_label=EvidenceSupportLabel(_str(payload.get("support_label"))),
            summary=_str(payload.get("summary")),
            locator=_optional_str(payload.get("locator")),
            confidence=float(confidence) if confidence is not None else None,
            version=int(version) if version is not None else None,
        )


@dataclass(frozen=True, slots=True)
class StudyRecommendation:
    """One deterministic recommendation produced by a study-agent task."""

    category: str
    title: str
    rationale: str
    evidence_reference_ids: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudyRecommendation":
        return cls(
            category=_str(payload.get("category")),
            title=_str(payload.get("title")),
            rationale=_str(payload.get("rationale")),
            evidence_reference_ids=_tuple_str(payload.get("evidence_reference_ids")),
            next_actions=_tuple_str(payload.get("next_actions")),
        )


@dataclass(frozen=True, slots=True)
class StudyTaskOutput:
    """Structured output for a study-agent run."""

    summary: str
    recommendations: tuple[StudyRecommendation, ...] = ()
    warnings: tuple[str, ...] = ()
    missing_context: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceReference, ...] = ()
    next_actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudyTaskOutput":
        return cls(
            summary=_str(payload.get("summary")),
            recommendations=tuple(
                StudyRecommendation.from_dict(item)
                for item in payload.get("recommendations", [])
                if isinstance(item, dict)
            ),
            warnings=_tuple_str(payload.get("warnings")),
            missing_context=_tuple_str(payload.get("missing_context")),
            evidence_refs=tuple(
                EvidenceReference.from_dict(item)
                for item in payload.get("evidence_refs", [])
                if isinstance(item, dict)
            ),
            next_actions=_tuple_str(payload.get("next_actions")),
        )


@dataclass(frozen=True, slots=True)
class StudyRunStep:
    """Trace event recorded during a study-agent run."""

    step_type: str
    message: str
    status: str = "completed"
    tool_name: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudyRunStep":
        return cls(
            step_type=_str(payload.get("step_type")),
            message=_str(payload.get("message")),
            status=_str(payload.get("status") or "completed"),
            tool_name=_optional_str(payload.get("tool_name")),
            created_at=_str(payload.get("created_at") or utc_now_iso()),
            details=dict(payload.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class StudyAgentRun:
    """Traceable execution record for a study-agent task."""

    run_id: str
    study_id: str
    task_type: StudyTaskType
    query: str
    status: StudyRunStatus
    context_pack_id: str
    steps: tuple[StudyRunStep, ...]
    warnings: tuple[str, ...] = ()
    output: StudyTaskOutput | None = None
    created_at: str = field(default_factory=utc_now_iso)
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StudyAgentRun":
        output = payload.get("output")
        return cls(
            run_id=_str(payload.get("run_id")),
            study_id=_str(payload.get("study_id")),
            task_type=StudyTaskType(_str(payload.get("task_type"))),
            query=_str(payload.get("query")),
            status=StudyRunStatus(_str(payload.get("status"))),
            context_pack_id=_str(payload.get("context_pack_id")),
            steps=tuple(
                StudyRunStep.from_dict(item)
                for item in payload.get("steps", [])
                if isinstance(item, dict)
            ),
            warnings=_tuple_str(payload.get("warnings")),
            output=StudyTaskOutput.from_dict(output) if isinstance(output, dict) else None,
            created_at=_str(payload.get("created_at") or utc_now_iso()),
            completed_at=_optional_str(payload.get("completed_at")),
        )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple | list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


def _str(value: object) -> str:
    return str(value or "").strip()


def _optional_str(value: object) -> str | None:
    text = _str(value)
    return text or None


def _tuple_str(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, list | tuple | set):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple()
