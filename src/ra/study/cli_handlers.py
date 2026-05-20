"""CLI handlers for local study-agent commands."""

from __future__ import annotations

from ra.study.brief_service import StudyBriefService
from ra.study.models import StudySourceType, StudyTaskType
from ra.study.runtime import StudyAgentRuntime
from ra.study.store import JsonStudyStore


def default_study_service() -> StudyBriefService:
    return StudyBriefService(store=JsonStudyStore())


def create_study(payload: object) -> dict[str, object]:
    service = default_study_service()
    brief = service.create_brief(
        study_id=getattr(payload, "study_id"),
        title=getattr(payload, "title"),
        research_goal=getattr(payload, "goal"),
    )
    return brief.to_dict()


def get_study_brief(payload: object) -> dict[str, object]:
    service = default_study_service()
    return service.get_brief(getattr(payload, "study_id")).to_dict()


def add_study_source(payload: object) -> dict[str, object]:
    service = default_study_service()
    source = service.add_source(
        getattr(payload, "study_id"),
        source_type=StudySourceType(getattr(payload, "source_type")),
        title=getattr(payload, "title"),
        content=getattr(payload, "text"),
    )
    return source.to_dict()


def run_study_task(payload: object) -> dict[str, object]:
    service = default_study_service()
    runtime = StudyAgentRuntime(service=service)
    run = runtime.run_task(
        study_id=getattr(payload, "study_id"),
        task_type=StudyTaskType(getattr(payload, "task")),
        query=getattr(payload, "query"),
    )
    return run.to_dict()
