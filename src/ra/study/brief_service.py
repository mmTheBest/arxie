"""Application service for study briefs, sources, and runs."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from ra.study.models import (
    StudyAgentRun,
    StudyBrief,
    StudySource,
    StudySourceType,
    utc_now_iso,
)
from ra.study.store import (
    JsonStudyStore,
    StudyNotFoundError,
    StudyRunNotFoundError,
)

_SUMMARY_CHARS = 180


class StudyBriefService:
    """Coordinates versioned study memory operations."""

    def __init__(self, *, store: JsonStudyStore | None = None) -> None:
        self._store = store or JsonStudyStore()

    @property
    def store(self) -> JsonStudyStore:
        return self._store

    def create_brief(
        self,
        *,
        study_id: str,
        title: str,
        research_goal: str,
        collection_id: str | None = None,
        domain: str | None = None,
        current_method: str | None = None,
        datasets: tuple[str, ...] | list[str] = (),
        metrics: tuple[str, ...] | list[str] = (),
        constraints: tuple[str, ...] | list[str] = (),
        decisions: tuple[str, ...] | list[str] = (),
        risks: tuple[str, ...] | list[str] = (),
        open_questions: tuple[str, ...] | list[str] = (),
    ) -> StudyBrief:
        brief = StudyBrief(
            study_id=_required(study_id, "study_id"),
            title=_required(title, "title"),
            research_goal=_required(research_goal, "research_goal"),
            collection_id=_optional(collection_id),
            domain=_optional(domain),
            current_method=_optional(current_method),
            datasets=_tuple(datasets),
            metrics=_tuple(metrics),
            constraints=_tuple(constraints),
            decisions=_tuple(decisions),
            risks=_tuple(risks),
            open_questions=_tuple(open_questions),
        )
        return self._store.create_brief(brief)

    def get_brief(self, study_id: str) -> StudyBrief:
        key = _required(study_id, "study_id")
        brief = self._store.get_brief(key)
        if brief is None:
            raise StudyNotFoundError(key)
        return brief

    def list_briefs(self) -> tuple[StudyBrief, ...]:
        return self._store.list_briefs()

    def update_brief(
        self,
        study_id: str,
        *,
        expected_version: int,
        title: str | None = None,
        research_goal: str | None = None,
        collection_id: str | None = None,
        domain: str | None = None,
        current_method: str | None = None,
        datasets: tuple[str, ...] | list[str] | None = None,
        metrics: tuple[str, ...] | list[str] | None = None,
        constraints: tuple[str, ...] | list[str] | None = None,
        decisions: tuple[str, ...] | list[str] | None = None,
        risks: tuple[str, ...] | list[str] | None = None,
        open_questions: tuple[str, ...] | list[str] | None = None,
    ) -> StudyBrief:
        current = self.get_brief(study_id)
        updated = replace(
            current,
            title=_required(title, "title") if title is not None else current.title,
            research_goal=(
                _required(research_goal, "research_goal")
                if research_goal is not None
                else current.research_goal
            ),
            collection_id=_optional(collection_id)
            if collection_id is not None
            else current.collection_id,
            domain=_optional(domain) if domain is not None else current.domain,
            current_method=_optional(current_method)
            if current_method is not None
            else current.current_method,
            datasets=_tuple(datasets) if datasets is not None else current.datasets,
            metrics=_tuple(metrics) if metrics is not None else current.metrics,
            constraints=_tuple(constraints)
            if constraints is not None
            else current.constraints,
            decisions=_tuple(decisions) if decisions is not None else current.decisions,
            risks=_tuple(risks) if risks is not None else current.risks,
            open_questions=_tuple(open_questions)
            if open_questions is not None
            else current.open_questions,
        )
        return self._store.save_brief(updated, expected_version=expected_version)

    def add_source(
        self,
        study_id: str,
        *,
        source_type: StudySourceType | str,
        title: str,
        content: str,
        summary: str | None = None,
        extracted_facts: tuple[str, ...] | list[str] = (),
    ) -> StudySource:
        brief = self.get_brief(study_id)
        source = StudySource(
            source_id=f"source-{uuid4().hex[:12]}",
            study_id=brief.study_id,
            source_type=(
                source_type
                if isinstance(source_type, StudySourceType)
                else StudySourceType(str(source_type))
            ),
            title=_required(title, "title"),
            content=_required(content, "content"),
            summary=_optional(summary) or _summarize(content),
            extracted_facts=_tuple(extracted_facts),
        )
        saved = self._store.add_source(source)
        source_ids = tuple(dict.fromkeys((*brief.source_ids, saved.source_id)))
        updated = replace(brief, source_ids=source_ids)
        self._store.save_brief(updated, expected_version=brief.version)
        return saved

    def list_sources(self, study_id: str) -> tuple[StudySource, ...]:
        return self._store.list_sources(_required(study_id, "study_id"))

    def save_run(self, run: StudyAgentRun) -> StudyAgentRun:
        return self._store.save_run(run)

    def get_run(self, study_id: str, run_id: str) -> StudyAgentRun:
        key = _required(study_id, "study_id")
        run_key = _required(run_id, "run_id")
        run = self._store.get_run(key, run_key)
        if run is None:
            raise StudyRunNotFoundError(key, run_key)
        return run

    def list_runs(self, study_id: str) -> tuple[StudyAgentRun, ...]:
        return self._store.list_runs(_required(study_id, "study_id"))

    def touch_brief(self, brief: StudyBrief) -> StudyBrief:
        return replace(brief, updated_at=utc_now_iso())


def _required(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def _optional(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _summarize(content: str) -> str:
    text = " ".join(str(content or "").split())
    if len(text) <= _SUMMARY_CHARS:
        return text
    return text[: _SUMMARY_CHARS - 1].rstrip() + "..."
