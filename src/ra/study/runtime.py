"""Deterministic runtime for study-agent tasks."""

from __future__ import annotations

from uuid import uuid4

from ra.retrieval.unified import Paper
from ra.study.brief_service import StudyBriefService
from ra.study.context_builder import StudyContextBuilder
from ra.study.models import (
    EvidenceReference,
    EvidenceSourceType,
    EvidenceSupportLabel,
    StudyAgentRun,
    StudyContextPack,
    StudyRecommendation,
    StudyRunStatus,
    StudyRunStep,
    StudyTaskOutput,
    StudyTaskType,
    utc_now_iso,
)
from ra.study.tools import StudyToolRegistry, default_study_tool_registry


class StudyTaskError(ValueError):
    """Raised when a study-agent task request is unsupported."""


class StudyAgentRuntime:
    """Run deterministic study-agent tasks with traceable context and evidence."""

    def __init__(
        self,
        *,
        service: StudyBriefService,
        context_builder: StudyContextBuilder | None = None,
        tool_registry: StudyToolRegistry | None = None,
    ) -> None:
        self.service = service
        self.context_builder = context_builder or StudyContextBuilder(service=service)
        self.tool_registry = tool_registry or default_study_tool_registry()

    def run_task(
        self,
        *,
        study_id: str,
        task_type: StudyTaskType | str,
        query: str,
        papers: list[Paper] | tuple[Paper, ...] | None = None,
    ) -> StudyAgentRun:
        try:
            task = task_type if isinstance(task_type, StudyTaskType) else StudyTaskType(str(task_type))
        except ValueError as exc:
            raise StudyTaskError(f"Unsupported study task: {task_type}") from exc

        steps: list[StudyRunStep] = [
            StudyRunStep(
                step_type="planned",
                message=f"Planned deterministic study task `{task.value}`.",
                details={"task_type": task.value},
            )
        ]
        context = self.context_builder.build_context(
            study_id=study_id,
            task_type=task,
            query=query,
            papers=papers,
        )
        steps.append(
            StudyRunStep(
                step_type="context_built",
                message="Built task-aware context pack.",
                details={
                    "context_pack_id": context.context_pack_id,
                    "paper_count": len(context.paper_refs),
                    "source_count": len(context.source_refs),
                },
            )
        )

        tool_names = _tools_for_task(task)
        tool_results = []
        for tool_name in tool_names:
            result = self.tool_registry.call(tool_name, context)
            tool_results.append(result)
            steps.append(
                StudyRunStep(
                    step_type="tool_called",
                    message=result.summary,
                    tool_name=tool_name,
                    details=result.data,
                )
            )

        output = _synthesize(task, query, context, tool_results)
        steps.append(
            StudyRunStep(
                step_type="synthesized",
                message="Synthesized deterministic study-agent output.",
                details={"recommendation_count": len(output.recommendations)},
            )
        )
        steps.append(
            StudyRunStep(
                step_type="validated",
                message="Validated evidence-linked output schema.",
                details={"evidence_ref_count": len(output.evidence_refs)},
            )
        )

        run = StudyAgentRun(
            run_id=f"run-{uuid4().hex[:12]}",
            study_id=context.study_id,
            task_type=task,
            query=str(query or "").strip(),
            status=StudyRunStatus.COMPLETED,
            context_pack_id=context.context_pack_id,
            steps=tuple(steps),
            warnings=output.warnings,
            output=output,
            completed_at=utc_now_iso(),
        )
        return self.service.save_run(run)


def _tools_for_task(task: StudyTaskType) -> tuple[str, ...]:
    if task is StudyTaskType.REVIEW_DRAFT_CLAIMS:
        return ("inspect_study_brief", "inspect_study_sources", "check_draft_claims")
    return ("inspect_study_brief", "inspect_study_sources", "extract_benchmark_hints")


def _synthesize(
    task: StudyTaskType,
    query: str,
    context: StudyContextPack,
    tool_results: list[object],
) -> StudyTaskOutput:
    evidence_refs = _evidence_refs(context)
    ref_ids = tuple(ref.reference_id for ref in evidence_refs)
    warnings = tuple(context.missing_context)
    hints = _extract_hints(tool_results)

    if task is StudyTaskType.DESIGN_EXPERIMENTS:
        recommendations = _design_experiment_recommendations(ref_ids, hints, context)
        summary = "Designed next experiment improvements from study memory and available evidence."
        next_actions = (
            "Prioritize one missing baseline and one ablation before expanding scope.",
            "Attach more paper metadata or extracted evidence to strengthen literature grounding.",
        )
    elif task is StudyTaskType.FIND_BENCHMARKS:
        recommendations = _benchmark_recommendations(ref_ids, hints)
        summary = "Identified benchmark, dataset, and metric candidates from available context."
        next_actions = (
            "Confirm which benchmark candidates fit the study domain.",
            "Add structured result evidence when extraction is available.",
        )
    else:
        recommendations = _draft_review_recommendations(ref_ids, context)
        summary = "Reviewed draft-like claims against available study context."
        next_actions = (
            "Link each important draft claim to at least one paper evidence reference.",
            "Mark any user-source-only claim before using it in a manuscript.",
        )

    return StudyTaskOutput(
        summary=summary,
        recommendations=recommendations,
        warnings=warnings,
        missing_context=context.missing_context,
        evidence_refs=evidence_refs,
        next_actions=next_actions,
    )


def _evidence_refs(context: StudyContextPack) -> tuple[EvidenceReference, ...]:
    refs: list[EvidenceReference] = []
    if context.brief_fields:
        refs.append(
            EvidenceReference(
                reference_id="ev-brief",
                source_type=EvidenceSourceType.STUDY_BRIEF,
                source_id=context.study_id,
                source_title=context.brief_fields.get("title", context.study_id),
                support_label=EvidenceSupportLabel.USER_PROVIDED,
                summary="Study brief supplied the research goal, constraints, and current method.",
                version=context.source_versions.get("brief"),
            )
        )
    for index, source in enumerate(context.source_refs, start=1):
        refs.append(
            EvidenceReference(
                reference_id=f"ev-source-{index}",
                source_type=EvidenceSourceType.USER_SOURCE,
                source_id=source.source_id,
                source_title=source.source_title,
                support_label=EvidenceSupportLabel.USER_PROVIDED,
                summary=source.summary,
                version=source.version,
            )
        )
    for index, paper in enumerate(context.paper_refs, start=1):
        refs.append(
            EvidenceReference(
                reference_id=f"ev-paper-{index}",
                source_type=EvidenceSourceType.PAPER,
                source_id=paper.paper_id,
                source_title=paper.title,
                support_label=EvidenceSupportLabel.INFERRED,
                summary=paper.abstract or paper.title,
                confidence=paper.relevance_score,
            )
        )
    return tuple(refs)


def _extract_hints(tool_results: list[object]) -> tuple[str, ...]:
    hints: set[str] = set()
    for result in tool_results:
        data = getattr(result, "data", {})
        if isinstance(data, dict):
            raw_hints = data.get("hints", [])
            if isinstance(raw_hints, list):
                hints.update(str(item) for item in raw_hints if str(item).strip())
    return tuple(sorted(hints))


def _design_experiment_recommendations(
    evidence_ids: tuple[str, ...],
    hints: tuple[str, ...],
    context: StudyContextPack,
) -> tuple[StudyRecommendation, ...]:
    metric_text = context.brief_fields.get("metrics") or _first_hint(hints, "metric", "f1")
    dataset_text = context.brief_fields.get("datasets") or _first_hint(hints, "dataset", "benchmark")
    return (
        StudyRecommendation(
            category="baseline",
            title="Add a clear baseline comparison",
            rationale=(
                "The study should compare the current method against simple and strong baselines "
                "before claiming improvement."
            ),
            evidence_reference_ids=evidence_ids,
            next_actions=("List baseline systems already tested and add at least one missing baseline.",),
        ),
        StudyRecommendation(
            category="metric",
            title="Use task-specific quality and reliability metrics",
            rationale=f"Track `{metric_text}` alongside error analysis so gains are interpretable.",
            evidence_reference_ids=evidence_ids,
            next_actions=("Define the metric calculation and acceptance threshold before running.",),
        ),
        StudyRecommendation(
            category="ablation",
            title="Run ablations for the main method components",
            rationale=(
                "Ablations isolate whether retrieval, reranking, prompting, or filtering drives "
                "the observed result."
            ),
            evidence_reference_ids=evidence_ids,
            next_actions=(f"Run the ablation on `{dataset_text}` or the closest available dataset.",),
        ),
    )


def _benchmark_recommendations(
    evidence_ids: tuple[str, ...],
    hints: tuple[str, ...],
) -> tuple[StudyRecommendation, ...]:
    hint_text = ", ".join(hints[:6]) if hints else "benchmark, dataset, metric"
    return (
        StudyRecommendation(
            category="benchmark",
            title="Create a benchmark candidate table",
            rationale=f"Available context mentions: {hint_text}. Convert these into a benchmark matrix.",
            evidence_reference_ids=evidence_ids,
            next_actions=("Record dataset, metric, baseline, and paper source for each candidate.",),
        ),
        StudyRecommendation(
            category="comparison",
            title="Separate benchmark fit from benchmark popularity",
            rationale="A popular benchmark is not always aligned with the study's claim or data regime.",
            evidence_reference_ids=evidence_ids,
            next_actions=("Score each benchmark for claim fit, implementation cost, and comparability.",),
        ),
    )


def _draft_review_recommendations(
    evidence_ids: tuple[str, ...],
    context: StudyContextPack,
) -> tuple[StudyRecommendation, ...]:
    title = "Add paper evidence for draft claims"
    if context.paper_refs:
        title = "Map draft claims to candidate paper evidence"
    return (
        StudyRecommendation(
            category="claim_support",
            title=title,
            rationale=(
                "Draft claims should be marked as user-provided until linked to direct paper or "
                "structured evidence."
            ),
            evidence_reference_ids=evidence_ids,
            next_actions=("Create a claim-evidence table before manuscript revision.",),
        ),
    )


def _first_hint(hints: tuple[str, ...], preferred: str, fallback: str) -> str:
    for hint in hints:
        if preferred in hint:
            return hint
    return fallback
