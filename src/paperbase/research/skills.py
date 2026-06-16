"""Internal research-agent skills and quality harnesses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SKILL_VERSION = "2026-05-11"
_STUDY_BRIEF_PROPOSAL_ARTIFACT_TYPES = {
    "assumption_map",
    "benchmark_plan",
    "experiment_backlog",
    "experiment_plan",
    "hypotheses",
    "revision_plan",
}
_STUDY_BRIEF_LIST_LIMIT = 8
_STUDY_BRIEF_SOURCE_LIMIT = 20
_STUDY_BRIEF_TITLE_LIMIT = 255
_STUDY_BRIEF_TEXT_LIMIT = 1000
_STUDY_BRIEF_LONG_TEXT_LIMIT = 3000


class UnsupportedResearchSkillError(ValueError):
    """Raised when a research-agent payload names an unknown internal skill."""


@dataclass(frozen=True, slots=True)
class ResearchSkillContext:
    """Bounded context for one internal research skill run."""

    message: str
    artifact_type: str
    evidence_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResearchSkillResult:
    """Structured result produced by an internal research skill."""

    skill_id: str
    skill_version: str
    artifact_type: str
    output_payload: dict[str, Any]
    evidence_payload: dict[str, Any]
    harness_report: dict[str, Any]


SkillHandler = Callable[[ResearchSkillContext], dict[str, Any]]


class ResearchSkillRegistry:
    """Registry of deterministic internal research skills."""

    def __init__(self) -> None:
        self._handlers: dict[str, tuple[str, str, SkillHandler]] = {}

    def register(
        self,
        *,
        skill_id: str,
        artifact_type: str,
        description: str,
        handler: SkillHandler,
    ) -> None:
        self._handlers[skill_id] = (artifact_type, description, handler)

    def run(self, skill_id: str, context: ResearchSkillContext) -> ResearchSkillResult:
        if skill_id not in self._handlers:
            raise UnsupportedResearchSkillError(f"Unsupported research skill: {skill_id}")
        resolved_skill_id = skill_id
        artifact_type, _description, handler = self._handlers[resolved_skill_id]
        skill_context = ResearchSkillContext(
            message=context.message,
            artifact_type=artifact_type,
            evidence_payload=context.evidence_payload,
        )
        output_payload = handler(skill_context)
        harness_report = _build_harness_report(skill_context, output_payload=output_payload)
        output_payload = {
            **output_payload,
            "skill_id": resolved_skill_id,
            "skill_version": SKILL_VERSION,
            "harness_report": harness_report,
        }
        output_payload = with_study_brief_update(
            skill_context,
            output_payload,
            artifact_type=artifact_type,
            skill_id=resolved_skill_id,
        )
        evidence_payload = {
            **context.evidence_payload,
            "skill_id": resolved_skill_id,
            "skill_version": SKILL_VERSION,
            "harness_report": harness_report,
        }
        return ResearchSkillResult(
            skill_id=resolved_skill_id,
            skill_version=SKILL_VERSION,
            artifact_type=artifact_type,
            output_payload=output_payload,
            evidence_payload=evidence_payload,
            harness_report=harness_report,
        )


def default_research_skill_registry() -> ResearchSkillRegistry:
    registry = ResearchSkillRegistry()
    registry.register(
        skill_id="literature_review",
        artifact_type="literature_review",
        description="Synthesize collection papers into themes, gaps, and future directions.",
        handler=_literature_review,
    )
    registry.register(
        skill_id="quality_harness",
        artifact_type="critique",
        description="Evaluate evidence coverage, unsupported claims, and missing context.",
        handler=_quality_harness_output,
    )
    registry.register(
        skill_id="comparison",
        artifact_type="comparison",
        description="Compare methods, datasets, metrics, and result patterns in chat.",
        handler=_comparison_output,
    )
    registry.register(
        skill_id="benchmark_planning",
        artifact_type="benchmark_plan",
        description="Design benchmark, baseline, and metric plans grounded in the collection.",
        handler=_benchmark_planning,
    )
    registry.register(
        skill_id="experiment_planning",
        artifact_type="experiment_plan",
        description="Design experiments and ablations grounded in papers and user sources.",
        handler=_experiment_planning,
    )
    registry.register(
        skill_id="field_pattern_analysis",
        artifact_type="field_patterns",
        description="Extract recurring methods, datasets, metrics, and limitation patterns.",
        handler=_field_pattern_analysis,
    )
    registry.register(
        skill_id="experiment_backlog",
        artifact_type="experiment_backlog",
        description="Produce a prioritized backlog of experiments from collection evidence.",
        handler=_experiment_backlog,
    )
    registry.register(
        skill_id="assumption_mapping",
        artifact_type="assumption_map",
        description="Map assumptions worth challenging with paper-grounded tests.",
        handler=_assumption_mapping,
    )
    registry.register(
        skill_id="hypothesis_generation",
        artifact_type="hypotheses",
        description="Generate collection-grounded hypotheses and validation plans.",
        handler=_hypothesis_generation,
    )
    registry.register(
        skill_id="revision_planning",
        artifact_type="revision_plan",
        description="Plan paper or project revisions from collection evidence.",
        handler=_revision_planning,
    )
    return registry


def select_research_skill(
    message: str,
    *,
    suggestion_id: str | None = None,
    artifact_type: str | None = None,
) -> str:
    suggestion_map = {
        "literature_review_synthesis": "literature_review",
        "evidence_quality_check": "quality_harness",
        "benchmark_ablation_plan": "benchmark_planning",
        "revision_plan": "revision_planning",
    }
    if suggestion_id in suggestion_map:
        return suggestion_map[str(suggestion_id)]

    artifact_map = {
        "literature_review": "literature_review",
        "comparison": "comparison",
        "critique": "quality_harness",
        "benchmark_plan": "benchmark_planning",
        "revision_plan": "revision_planning",
        "experiment_plan": "experiment_planning",
        "experiment_backlog": "experiment_backlog",
        "hypotheses": "hypothesis_generation",
        "field_patterns": "field_pattern_analysis",
        "assumption_map": "assumption_mapping",
    }
    if artifact_type in artifact_map:
        return artifact_map[str(artifact_type)]

    normalized = message.casefold()
    if "hypothes" in normalized:
        return "hypothesis_generation"
    if any(term in normalized for term in ("literature review", "synthesize", "theme", "themes")):
        return "literature_review"
    if any(term in normalized for term in ("compare", "comparison", "contrast", "rank", "ranking")):
        return "comparison"
    if any(
        term in normalized
        for term in ("unsupported", "evidence coverage", "claim", "quality", "check")
    ):
        return "quality_harness"
    if any(term in normalized for term in ("benchmark", "baseline", "metric")):
        return "benchmark_planning"
    if any(term in normalized for term in ("revision", "improve", "draft")):
        return "revision_planning"
    return "experiment_planning"


def artifact_type_for_skill(skill_id: str, *, fallback: str = "experiment_plan") -> str:
    return {
        "literature_review": "literature_review",
        "comparison": "comparison",
        "quality_harness": "critique",
        "benchmark_planning": "benchmark_plan",
        "revision_planning": "revision_plan",
        "experiment_planning": "experiment_plan",
        "field_pattern_analysis": "field_patterns",
        "experiment_backlog": "experiment_backlog",
        "assumption_mapping": "assumption_map",
        "hypothesis_generation": "hypotheses",
    }.get(skill_id, fallback)


def with_study_brief_update(
    context: ResearchSkillContext,
    output_payload: dict[str, Any],
    *,
    artifact_type: str | None = None,
    skill_id: str | None = None,
) -> dict[str, Any]:
    """Attach a review-only Study Brief proposal to eligible skill outputs."""

    if "study_brief_update" in output_payload or "study_brief_updates" in output_payload:
        return output_payload

    resolved_artifact_type = str(
        artifact_type or output_payload.get("artifact_type") or context.artifact_type or ""
    )
    if resolved_artifact_type not in _STUDY_BRIEF_PROPOSAL_ARTIFACT_TYPES:
        return output_payload

    proposal = _study_brief_update_from_output(context, output_payload)
    if not _study_brief_update_has_content(proposal):
        return output_payload

    return {
        **output_payload,
        "study_brief_update": proposal,
        "study_brief_update_source": {
            "source": "research_skill_output",
            "artifact_type": resolved_artifact_type,
            "skill_id": str(skill_id or output_payload.get("skill_id") or "unknown"),
        },
    }


def _study_brief_update_from_output(
    context: ResearchSkillContext,
    output_payload: dict[str, Any],
) -> dict[str, Any]:
    proposal: dict[str, Any] = {}
    aim = _first_text(
        output_payload.get("aim"),
        output_payload.get("objective"),
        output_payload.get("summary"),
    )
    if aim:
        proposal["aim"] = _bounded_study_brief_text(
            aim,
            max_length=_STUDY_BRIEF_LONG_TEXT_LIMIT,
        )

    hypothesis = _hypothesis_from_output(output_payload)
    if hypothesis:
        proposal["hypothesis"] = _bounded_study_brief_text(
            hypothesis,
            max_length=_STUDY_BRIEF_LONG_TEXT_LIMIT,
        )

    constraints = _study_brief_constraints(output_payload)
    if constraints:
        proposal["constraints"] = constraints

    decisions = _study_brief_decisions(output_payload)
    if decisions:
        proposal["confirmed_decisions"] = decisions

    risks = _study_brief_risks(context, output_payload)
    if risks:
        proposal["open_risks"] = risks

    linked_source_ids = _linked_study_source_ids(context)
    if linked_source_ids:
        proposal["linked_source_ids"] = linked_source_ids
    return proposal


def _study_brief_update_has_content(proposal: dict[str, Any]) -> bool:
    return bool(
        proposal.get("aim")
        or proposal.get("hypothesis")
        or proposal.get("constraints")
        or proposal.get("confirmed_decisions")
        or proposal.get("open_risks")
    )


def _hypothesis_from_output(output_payload: dict[str, Any]) -> str:
    explicit = _first_text(output_payload.get("hypothesis"))
    if explicit:
        return explicit
    hypotheses = output_payload.get("hypotheses")
    if isinstance(hypotheses, list):
        for item in hypotheses:
            if isinstance(item, dict):
                claim = _first_text(
                    item.get("claim"),
                    item.get("hypothesis"),
                    item.get("title"),
                )
                if claim:
                    return claim
            elif isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def _study_brief_constraints(output_payload: dict[str, Any]) -> list[dict[str, str]]:
    constraints: list[dict[str, str]] = []
    constraints.extend(
        _grouped_study_brief_item("Evaluation datasets", output_payload.get("datasets"))
    )
    constraints.extend(
        _grouped_study_brief_item(
            "Evaluation metrics",
            output_payload.get("metrics_or_result_logic") or output_payload.get("metrics"),
        )
    )
    constraints.extend(
        _grouped_study_brief_item(
            "Candidate baselines",
            output_payload.get("candidate_baselines"),
        )
    )
    constraints.extend(_grouped_study_brief_item("Baselines", output_payload.get("baselines")))
    constraints.extend(_grouped_study_brief_item("Controls", output_payload.get("controls")))
    constraints.extend(
        _grouped_study_brief_item("Planned ablations", output_payload.get("ablations"))
    )
    return _dedupe_study_brief_items(constraints)[:_STUDY_BRIEF_LIST_LIMIT]


def _study_brief_decisions(output_payload: dict[str, Any]) -> list[dict[str, str]]:
    decisions: list[dict[str, str]] = []
    for field_name, title in (
        ("benchmark_recommendations", "Benchmark recommendation"),
        ("revision_priorities", "Revision priority"),
        ("backlog_items", "Experiment backlog item"),
        ("challenge_tests", "Assumption challenge test"),
        ("validation_plan", "Validation plan"),
    ):
        decisions.extend(_study_brief_items_from_values(title, output_payload.get(field_name)))

    if not decisions:
        decisions.extend(
            _study_brief_items_from_recommendations(output_payload.get("recommendations"))
        )
    return _dedupe_study_brief_items(decisions)[:_STUDY_BRIEF_LIST_LIMIT]


def _study_brief_risks(
    context: ResearchSkillContext,
    output_payload: dict[str, Any],
) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    risks.extend(_study_brief_items_from_values("Evidence risk", output_payload.get("limitations")))
    risks.extend(
        _study_brief_items_from_values(
            "Evidence risk",
            output_payload.get("paper_backed_risks"),
        )
    )
    risks.extend(
        _study_brief_items_from_values(
            "Source context risk",
            output_payload.get("source_context_risks"),
        )
    )
    risks.extend(
        _study_brief_items_from_values("Evidence gap", output_payload.get("evidence_gaps"))
    )
    risks.extend(
        _study_brief_items_from_values(
            "Readiness blocker",
            output_payload.get("readiness_blockers"),
        )
    )
    if not risks:
        risks.extend(
            _study_brief_items_from_values(
                "Evidence risk",
                _unique(
                    item for paper in _papers(context) for item in paper.get("limitations", [])
                ),
            )
        )
    return _dedupe_study_brief_items(risks)[:_STUDY_BRIEF_LIST_LIMIT]


def _study_brief_items_from_recommendations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, str]] = []
    for recommendation in value:
        if isinstance(recommendation, dict):
            title = _first_text(recommendation.get("title"), recommendation.get("name"))
            text = _first_text(
                recommendation.get("detail"),
                recommendation.get("text"),
                recommendation.get("summary"),
                recommendation.get("description"),
                title,
            )
            item = _study_brief_item(title or "Recommendation", text)
            if item is not None:
                items.append(item)
        else:
            item = _study_brief_item("Recommendation", recommendation)
            if item is not None:
                items.append(item)
    return items


def _study_brief_items_from_values(title: str, value: Any) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for text in _text_values(value):
        item = _study_brief_item(title, text)
        if item is not None:
            items.append(item)
    return items


def _grouped_study_brief_item(title: str, value: Any) -> list[dict[str, str]]:
    values = _text_values(value)
    if not values:
        return []
    item = _study_brief_item(title, ", ".join(values[:_STUDY_BRIEF_LIST_LIMIT]))
    return [item] if item is not None else []


def _study_brief_item(title: Any, text: Any) -> dict[str, str] | None:
    bounded_text = _bounded_study_brief_text(text, max_length=_STUDY_BRIEF_TEXT_LIMIT)
    if not bounded_text:
        return None
    bounded_title = _bounded_study_brief_text(
        title or "Item",
        max_length=_STUDY_BRIEF_TITLE_LIMIT,
    )
    return {"title": bounded_title or "Item", "text": bounded_text}


def _dedupe_study_brief_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        title = item.get("title", "").strip()
        text = item.get("text", "").strip()
        key = (title.casefold(), text.casefold())
        if not title or not text or key in seen:
            continue
        seen.add(key)
        deduped.append({"title": title, "text": text})
    return deduped


def _linked_study_source_ids(context: ResearchSkillContext) -> list[str]:
    source_ids = [
        str(source.get("source_id") or "").strip()[:36]
        for source in context.evidence_payload.get("sources", [])
        if isinstance(source, dict)
    ]
    return _unique(source_ids)[:_STUDY_BRIEF_SOURCE_LIMIT]


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []

    values: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                values.append(item.strip())
        elif isinstance(item, dict):
            text = _first_text(
                item.get("text"),
                item.get("detail"),
                item.get("summary"),
                item.get("description"),
                item.get("decision"),
                item.get("risk"),
                item.get("claim"),
                item.get("title"),
            )
            if text:
                values.append(text)
    return _unique(values)


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _bounded_study_brief_text(value: Any, *, max_length: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[:max_length]


def _literature_review(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(
        context, artifact_type="literature_review", title="Literature review synthesis"
    )
    methods = _unique(item for paper in papers for item in paper.get("methods", []))
    datasets = _unique(item for paper in papers for item in paper.get("datasets", []))
    metrics = _unique(item for paper in papers for item in paper.get("metrics", []))
    limitations = _unique(item for paper in papers for item in paper.get("limitations", []))
    section_terms = _unique(
        section.get("title")
        for paper in papers
        for section in paper.get("sections", [])
        if isinstance(section, dict)
    )
    themes = [
        {
            "title": "Method families and design assumptions",
            "summary": _join_or_default(
                methods, "The collection needs structured method extraction."
            ),
            "evidence_papers": _paper_titles(papers),
        },
        {
            "title": "Benchmark and measurement practice",
            "summary": _join_or_default(
                datasets + metrics, "Benchmarks and metrics are not extracted yet."
            ),
            "evidence_papers": _paper_titles(papers),
        },
        {
            "title": "Reported limitations and validity threats",
            "summary": _join_or_default(
                limitations, "Explicit limitations were not extracted yet."
            ),
            "evidence_papers": _paper_titles(papers),
        },
    ]
    return {
        **common,
        "search_methodology": {
            "source": "active Arxie collection",
            "screening": "Uses papers selected or present in the active collection.",
            "available_sections": section_terms[:8],
        },
        "themes": themes,
        "consensus": [
            "Use collection-recurring datasets and metrics as the comparison frame.",
            "Map method claims to explicit benchmark and ablation evidence.",
        ],
        "controversies": limitations[:5],
        "research_gaps": _research_gaps(limitations, context),
        "future_directions": [
            "Turn each gap into a falsifiable experiment with a baseline and metric.",
            "Prioritize studies that reuse collection benchmarks while challenging one assumption.",
            "Add claim-level evidence checks before drafting conclusions.",
        ],
    }


def _quality_harness_output(context: ResearchSkillContext) -> dict[str, Any]:
    harness = _build_harness_report(context, output_payload={})
    return {
        "artifact_type": "critique",
        "title": "Evidence quality harness",
        "request": context.message,
        "paper_count": len(_papers(context)),
        "evidence_references": (
            _evidence_references_with_selected_method_limitation_roles(context)
        ),
        "quality_checks": [
            {
                "name": "Paper evidence coverage",
                "status": "ready" if harness["evidence_coverage"] > 0 else "needs_attention",
                "detail": f"{harness['evidence_paper_count']} paper(s) available.",
            },
            {
                "name": "Missing context",
                "status": "ready" if not harness["missing_context"] else "needs_attention",
                "detail": "; ".join(harness["missing_context"]) or "No missing context detected.",
            },
        ],
    }


def _comparison_output(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(context, artifact_type="comparison", title="Comparison")
    rows: list[dict[str, Any]] = []
    method_names: list[str] = []
    dataset_names: list[str] = []
    metric_names: list[str] = []
    for paper in papers:
        title = str(paper.get("title") or "Untitled paper")
        methods = _unique(paper.get("methods", []))
        datasets = _unique(paper.get("datasets", []))
        metrics = _unique(paper.get("metrics", []))
        results = [result for result in paper.get("results", []) if isinstance(result, dict)]
        method_names.extend(methods)
        dataset_names.extend(datasets)
        metric_names.extend(metrics)
        if results:
            for result in results[:3]:
                rows.append(
                    {
                        "paper_title": title,
                        "method": str(
                            result.get("method") or (methods[0] if methods else "Unknown method")
                        ),
                        "dataset": str(result.get("dataset") or (datasets[0] if datasets else "")),
                        "metric": str(result.get("metric") or (metrics[0] if metrics else "")),
                        "value": result.get("value_numeric")
                        if result.get("value_numeric") is not None
                        else result.get("value_text"),
                        "notes": str(result.get("notes") or ""),
                    }
                )
        else:
            rows.append(
                {
                    "paper_title": title,
                    "method": methods[0] if methods else "Unknown method",
                    "dataset": datasets[0] if datasets else "",
                    "metric": metrics[0] if metrics else "",
                    "value": "",
                    "notes": "No result rows were extracted for this paper.",
                }
            )

    return {
        **common,
        "summary": "Comparison generated from selected papers."
        if papers
        else "No paper evidence was available for comparison.",
        "comparison_rows": rows[:12],
        "method_families": _unique(method_names)[:8],
        "datasets": _unique(dataset_names)[:8],
        "metrics": _unique(metric_names)[:8],
        "save_options": ["markdown", "csv"],
    }


def _benchmark_planning(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = {
        **_common_payload(context, artifact_type="benchmark_plan", title="Benchmark plan"),
        "evidence_references": _evidence_references_with_selected_benchmark_roles(context),
    }
    datasets = _unique(item for paper in papers for item in paper.get("datasets", []))
    metrics = _unique(item for paper in papers for item in paper.get("metrics", []))
    methods = _unique(item for paper in papers for item in paper.get("methods", []))
    structured_evidence_ready = bool(
        datasets or metrics or methods or any(paper.get("results") for paper in papers)
    )
    recommendation_texts = [
        "Reuse the most common dataset and metric from the collection as the primary comparison.",
        "Add one stress test that targets an extracted limitation or user-source concern.",
        "Report matched baselines under identical splits before adding new model variants.",
    ]
    return {
        **common,
        "structured_evidence_ready": structured_evidence_ready,
        "readiness_blockers": []
        if structured_evidence_ready
        else ["structured_extraction_missing"],
        "next_actions": []
        if structured_evidence_ready
        else ["Run extraction in Library for the selected papers."],
        "benchmark_recommendations": recommendation_texts,
        "recommendations": _support_labeled_recommendations(context, recommendation_texts),
        "datasets": datasets[:8],
        "metrics_or_result_logic": metrics[:8]
        or ["Define a metric for each claim before running."],
        "candidate_baselines": methods[:8]
        or ["Use the strongest method reported by exemplar papers."],
    }


def _experiment_planning(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = {
        **_common_payload(context, artifact_type="experiment_plan", title="Experiment plan"),
        "evidence_references": _evidence_references_with_selected_experiment_roles(context),
    }
    design_elements = [
        item
        for paper in papers
        for item in paper.get("research_design_elements", [])
        if isinstance(item, dict)
    ]
    return {
        **common,
        "objective": "Design a study whose claims can be compared against the selected collection.",
        "baselines": ["Use the strongest method reported by exemplar papers."],
        "ablations": [
            item["title"]
            for item in design_elements
            if item.get("element_type") == "ablation" and item.get("title")
        ][:3]
        or [
            "Remove one model assumption at a time and report the same metrics "
            "used by reference papers."
        ],
        "controls": [
            "Lock preprocessing and train/test splits before comparing methods.",
            "Report failure cases and sensitivity to key experimental variables.",
        ],
    }


def _field_pattern_analysis(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(context, artifact_type="field_patterns", title="Field patterns")
    methods = _unique(item for paper in papers for item in paper.get("methods", []))
    datasets = _unique(item for paper in papers for item in paper.get("datasets", []))
    metrics = _unique(item for paper in papers for item in paper.get("metrics", []))
    limitations = _unique(item for paper in papers for item in paper.get("limitations", []))
    evaluation_frame = _join_or_default(
        datasets + metrics,
        "datasets and metrics that still need extraction",
    )
    patterns = [
        f"Methods recur around {_join_or_default(methods, 'methods that still need extraction')}.",
        f"Evaluation is framed by {evaluation_frame}.",
        "Limitations cluster around "
        f"{_join_or_default(limitations, 'validity threats that still need extraction')}.",
    ]
    return {
        **common,
        "patterns": patterns,
        "method_patterns": methods[:8],
        "dataset_patterns": datasets[:8],
        "metric_patterns": metrics[:8],
        "limitation_patterns": limitations[:8],
    }


def _experiment_backlog(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(
        context, artifact_type="experiment_backlog", title="Experiment backlog"
    )
    design_elements = [
        item
        for paper in papers
        for item in paper.get("research_design_elements", [])
        if isinstance(item, dict) and item.get("title")
    ]
    backlog_items = [str(item.get("title")) for item in design_elements][:6] or [
        "Run matched baselines on the collection's most common dataset.",
        "Add one ablation that removes a core method assumption.",
        "Report sensitivity analysis for the strongest validity threat.",
    ]
    return {
        **common,
        "backlog_items": backlog_items,
        "prioritization_rule": (
            "Prioritize items with direct paper evidence and clear evaluation metrics."
        ),
        "next_actions": [
            "Assign each backlog item a dataset, metric, and expected failure mode.",
            "Link each planned experiment to at least one paper or study source.",
        ],
    }


def _assumption_mapping(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = {
        **_common_payload(context, artifact_type="assumption_map", title="Assumption map"),
        "evidence_references": (
            _evidence_references_with_selected_method_limitation_roles(context)
        ),
    }
    methods = _unique(item for paper in papers for item in paper.get("methods", []))
    limitations = _unique(item for paper in papers for item in paper.get("limitations", []))
    assumptions = [
        f"{method} remains reliable under the target data conditions." for method in methods[:3]
    ]
    assumptions.extend(
        f"The study can control for limitation: {limitation}." for limitation in limitations[:3]
    )
    return {
        **common,
        "assumptions_to_challenge": assumptions
        or [
            "The central method assumption is not yet explicit; extract methods "
            "or add study sources."
        ],
        "challenge_tests": [
            "Create one falsification test per assumption.",
            "Use collection metrics to define pass/fail criteria.",
        ],
        "evidence_gaps": _research_gaps(limitations, context),
    }


def _hypothesis_generation(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = {
        **_common_payload(
            context, artifact_type="hypotheses", title="Collection-grounded hypotheses"
        ),
        "evidence_references": _evidence_references_with_selected_hypothesis_roles(
            context
        ),
    }
    methods = _unique(item for paper in papers for item in paper.get("methods", []))
    datasets = _unique(item for paper in papers for item in paper.get("datasets", []))
    method = methods[0] if methods else "the target method"
    dataset = datasets[0] if datasets else "the benchmark corpus"
    return {
        **common,
        "hypotheses": [
            {
                "claim": (
                    f"{method} improves reliability on {dataset} when the key "
                    "design assumption is isolated."
                ),
                "rationale": "Collection evidence suggests a testable design assumption.",
                "test": "Run matched baselines and ablations under the same evaluation protocol.",
            }
        ],
        "validation_plan": [
            "Define a measurable claim for each hypothesis.",
            "Reuse collection datasets and metrics where possible to make "
            "comparisons interpretable.",
        ],
    }


def _revision_planning(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = {
        **_common_payload(context, artifact_type="revision_plan", title="Revision plan"),
        "evidence_references": (
            _evidence_references_with_selected_method_limitation_roles(context)
        ),
    }
    limitations = _unique(item for paper in papers for item in paper.get("limitations", []))
    source_summaries = [
        str(source.get("summary") or "")
        for source in context.evidence_payload.get("sources", [])
        if isinstance(source, dict) and source.get("summary")
    ]
    return {
        **common,
        "revision_priorities": [
            "State the main claim as a falsifiable comparison against prior work.",
            "Add controls or ablations for each extracted limitation that applies "
            "to the user's study.",
            "Align evaluation tables with datasets, metrics, and baselines that "
            "recur across the collection.",
        ],
        "paper_backed_risks": limitations[:6],
        "source_context_risks": source_summaries[:5],
    }


def _build_harness_report(
    context: ResearchSkillContext,
    *,
    output_payload: dict[str, Any],
) -> dict[str, Any]:
    papers = _papers(context)
    sources = [
        source for source in context.evidence_payload.get("sources", []) if isinstance(source, dict)
    ]
    missing_context: list[str] = []
    if not papers:
        missing_context.append("No paper evidence was available.")
    if output_payload.get("artifact_type") in {"literature_review", "benchmark_plan"}:
        extracted_signal = any(
            paper.get("methods")
            or paper.get("datasets")
            or paper.get("metrics")
            or paper.get("results")
            for paper in papers
        )
        if papers and not extracted_signal:
            missing_context.append("Structured extraction is missing for the selected papers.")
    evidence_coverage = (
        0.0 if not papers else min(1.0, len(papers) / max(1, len(_paper_titles(papers))))
    )
    return {
        "skill_version": SKILL_VERSION,
        "evidence_coverage": round(evidence_coverage, 3),
        "evidence_paper_count": len(papers),
        "source_context_count": len(sources),
        "missing_context": missing_context,
        "unsupported_claims": [] if papers else [context.message],
        "reproducibility_checks": [
            "Does the artifact name the papers or sources used?",
            "Are claims tied to benchmarks, metrics, methods, or limitations?",
        ],
    }


def _research_gaps(limitations: list[str], context: ResearchSkillContext) -> list[str]:
    source_gaps = [
        str(source.get("summary") or "")
        for source in context.evidence_payload.get("sources", [])
        if isinstance(source, dict) and source.get("summary")
    ]
    gaps = limitations[:4] + source_gaps[:3]
    return gaps or [
        "No explicit gap evidence was extracted yet; run extraction or add study sources."
    ]


def _common_payload(
    context: ResearchSkillContext, *, artifact_type: str, title: str
) -> dict[str, Any]:
    papers = _papers(context)
    evidence_references = _evidence_references(context)
    return {
        "artifact_type": artifact_type,
        "title": title,
        "request": context.message,
        "paper_count": len(papers),
        "evidence_basis": _paper_titles(papers),
        "evidence_references": evidence_references,
        "source_context": _source_context(context),
        "recommendations": _support_labeled_recommendations(
            context,
            [
                "Tie each recommendation to paper evidence, user context, or an "
                "explicit inference.",
                "Check that each planned claim has a dataset, metric, and failure mode.",
            ],
        ),
        "general_methodology": [
            "Separate claims, experimental variables, and evaluation criteria before "
            "writing new experiments.",
            "Use paper-level evidence as constraints, then identify where a new "
            "study can challenge one assumption.",
        ],
    }


def _source_context(context: ResearchSkillContext) -> list[dict[str, str]]:
    sources = [
        source for source in context.evidence_payload.get("sources", []) if isinstance(source, dict)
    ]
    return [
        {
            "title": str(source.get("title") or "Study source"),
            "source_type": str(source.get("source_type") or "text"),
            "summary": str(source.get("summary") or ""),
        }
        for source in sources
    ]


def _support_labeled_recommendations(
    context: ResearchSkillContext,
    recommendation_texts: list[str],
) -> list[dict[str, Any]]:
    evidence_references = _evidence_references(context)
    supporting_layers = _supporting_layers(context)
    evidence_references = _without_source_fact_memory_references(
        context,
        evidence_references,
    )
    supporting_layers = [layer for layer in supporting_layers if layer != "source_fact_memory"]
    support_status = "supported" if evidence_references else "speculative"
    return [
        {
            "title": text,
            "detail": text,
            "support_status": support_status,
            "supporting_layers": supporting_layers,
            "evidence_references": evidence_references,
        }
        for text in recommendation_texts
    ]


def _without_source_fact_memory_references(
    context: ResearchSkillContext,
    references: list[dict[str, str]],
) -> list[dict[str, str]]:
    source_fact_memory_ids = _source_fact_memory_ids(context)
    if not source_fact_memory_ids:
        return references
    return [
        reference
        for reference in references
        if reference.get("memory_record_id") not in source_fact_memory_ids
    ]


def _source_fact_memory_ids(context: ResearchSkillContext) -> set[str]:
    intelligence_layers = context.evidence_payload.get("intelligence_layers")
    if not isinstance(intelligence_layers, dict):
        return set()
    return {
        str(item["memory_record_id"])
        for item in intelligence_layers.get("source_fact_memory", [])
        if isinstance(item, dict) and item.get("memory_record_id")
    }


def _supporting_layers(context: ResearchSkillContext) -> list[str]:
    layers: list[str] = []
    evidence_payload = context.evidence_payload
    if _papers(context) or any(
        isinstance(source, dict) for source in evidence_payload.get("sources", [])
    ):
        layers.append("source_library")

    intelligence_layers = evidence_payload.get("intelligence_layers")
    if isinstance(intelligence_layers, dict):
        if any(isinstance(item, dict) for item in intelligence_layers.get("evidence_memory", [])):
            layers.append("evidence_memory")
        if any(isinstance(item, dict) for item in intelligence_layers.get("pattern_memory", [])):
            layers.append("pattern_memory")
        source_fact_memory = intelligence_layers.get("source_fact_memory", [])
        if any(isinstance(item, dict) for item in source_fact_memory):
            layers.append("source_fact_memory")
        field_graph = intelligence_layers.get("field_graph")
        if isinstance(field_graph, dict) and (
            any(isinstance(item, dict) for item in field_graph.get("nodes", []))
            or any(isinstance(item, dict) for item in field_graph.get("edges", []))
        ):
            layers.append("field_graph")
        if isinstance(intelligence_layers.get("study_brief"), dict):
            layers.append("study_brief")
    return _unique(layers)


def _papers(context: ResearchSkillContext) -> list[dict[str, Any]]:
    return [
        paper for paper in context.evidence_payload.get("papers", []) if isinstance(paper, dict)
    ]


def _paper_titles(papers: list[dict[str, Any]]) -> list[str]:
    return _unique(paper.get("title") for paper in papers)


def _evidence_references(context: ResearchSkillContext) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for paper in _papers(context):
        paper_id = paper.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            continue
        references.append(
            {
                "reference_type": "paper",
                "paper_id": paper_id,
                "label": str(paper.get("title") or "Untitled paper"),
            }
        )
    for source in context.evidence_payload.get("sources", []):
        if not isinstance(source, dict):
            continue
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            continue
        references.append(
            {
                "reference_type": "study_source",
                "source_id": source_id,
                "label": str(source.get("title") or "Study source"),
            }
        )
    references.extend(_intelligence_layer_references(context))
    return references


def _evidence_references_with_selected_method_limitation_roles(
    context: ResearchSkillContext,
) -> list[dict[str, str]]:
    references = [*_evidence_references(context), *_selected_method_limitation_references(context)]
    return _dedupe_evidence_references(references)


def _evidence_references_with_selected_hypothesis_roles(
    context: ResearchSkillContext,
) -> list[dict[str, str]]:
    references = [*_evidence_references(context), *_selected_hypothesis_role_references(context)]
    return _dedupe_evidence_references(references)


def _evidence_references_with_selected_experiment_roles(
    context: ResearchSkillContext,
) -> list[dict[str, str]]:
    references = [*_evidence_references(context), *_selected_experiment_role_references(context)]
    return _dedupe_evidence_references(references)


def _evidence_references_with_selected_benchmark_roles(
    context: ResearchSkillContext,
) -> list[dict[str, str]]:
    references = [*_evidence_references(context), *_selected_benchmark_role_references(context)]
    return _dedupe_evidence_references(references)


def _dedupe_evidence_references(
    references: list[dict[str, str]],
) -> list[dict[str, str]]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    deduped: list[dict[str, str]] = []
    for reference in references:
        key = tuple(sorted(reference.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(reference)
    return deduped


def _selected_experiment_role_references(
    context: ResearchSkillContext,
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for entity in context.evidence_payload.get("structured_entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("entity_type") or entity.get("type") or "")
        normalized_type = entity_type.casefold().strip()
        if normalized_type not in {"method", "metric"}:
            continue
        reference = _structured_entity_role_reference(entity, reference_type=normalized_type)
        if reference is not None:
            references.append(reference)

    for result in context.evidence_payload.get("result_evidence", []):
        if not isinstance(result, dict):
            continue
        reference = _result_role_reference(result)
        if reference is not None:
            references.append(reference)
    return references


def _selected_benchmark_role_references(
    context: ResearchSkillContext,
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for entity in context.evidence_payload.get("structured_entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("entity_type") or entity.get("type") or "")
        normalized_type = entity_type.casefold().strip()
        if normalized_type not in {"dataset", "method", "metric"}:
            continue
        reference = _structured_entity_role_reference(entity, reference_type=normalized_type)
        if reference is not None:
            references.append(reference)

    for result in context.evidence_payload.get("result_evidence", []):
        if not isinstance(result, dict):
            continue
        reference = _result_role_reference(result)
        if reference is not None:
            references.append(reference)
    return references


def _selected_hypothesis_role_references(
    context: ResearchSkillContext,
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for entity in context.evidence_payload.get("structured_entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("entity_type") or entity.get("type") or "")
        normalized_type = entity_type.casefold().strip()
        if normalized_type not in {"dataset", "method", "metric"}:
            continue
        reference = _structured_entity_role_reference(entity, reference_type=normalized_type)
        if reference is not None:
            references.append(reference)

    for result in context.evidence_payload.get("result_evidence", []):
        if not isinstance(result, dict):
            continue
        reference = _result_role_reference(result)
        if reference is not None:
            references.append(reference)

    for span in context.evidence_payload.get("evidence_spans", []):
        if not isinstance(span, dict) or not _span_mentions_hypothesis_validation(span):
            continue
        span_id_field = _first_available_key(
            span,
            ("evidence_span_id", "span_id", "evidence_id"),
        )
        if span_id_field is None:
            continue
        references.append(
            {
                "reference_type": "evidence_span",
                span_id_field: str(span[span_id_field]),
                "label": str(
                    span.get("quote_text")
                    or span.get("text")
                    or span.get("content")
                    or "Evidence span"
                ),
            }
        )

    for chunk in context.evidence_payload.get("chunks", []):
        if not isinstance(chunk, dict) or not _chunk_mentions_hypothesis_validation(chunk):
            continue
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        references.append(
            {
                "reference_type": "chunk",
                "chunk_id": chunk_id,
                "label": str(
                    chunk.get("text")
                    or chunk.get("content")
                    or chunk.get("section_text")
                    or "Chunk"
                ),
            }
        )
    return references


def _selected_method_limitation_references(
    context: ResearchSkillContext,
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    method_references: list[dict[str, str]] = []
    for entity in context.evidence_payload.get("structured_entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("entity_type") or entity.get("type") or "")
        if entity_type.casefold().strip() != "method":
            continue
        reference = _method_role_reference(entity)
        if reference is not None:
            method_references.append(reference)
    references.extend(method_references[:8])

    role_span_references: list[dict[str, str]] = []
    for span in context.evidence_payload.get("evidence_spans", []):
        if not isinstance(span, dict) or not _span_mentions_assumption_or_limitation(span):
            continue
        span_id_field = _first_available_key(
            span,
            ("evidence_span_id", "span_id", "evidence_id"),
        )
        if span_id_field is None:
            continue
        role_span_references.append(
            {
                "reference_type": "evidence_span",
                span_id_field: str(span[span_id_field]),
                "label": str(
                    span.get("quote_text")
                    or span.get("text")
                    or span.get("content")
                    or "Evidence span"
                ),
            }
        )
    references.extend(role_span_references[:8])

    role_chunk_references: list[dict[str, str]] = []
    for chunk in context.evidence_payload.get("chunks", []):
        if not isinstance(chunk, dict) or not _chunk_mentions_assumption_or_limitation(chunk):
            continue
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        role_chunk_references.append(
            {
                "reference_type": "chunk",
                "chunk_id": chunk_id,
                "label": str(
                    chunk.get("text")
                    or chunk.get("content")
                    or chunk.get("section_text")
                    or "Chunk"
                ),
            }
        )
    references.extend(role_chunk_references[:8])
    return references


def _structured_entity_role_reference(
    entity: dict[str, Any],
    *,
    reference_type: str,
) -> dict[str, str] | None:
    id_field = _first_available_key(
        entity,
        ("entity_id", f"{reference_type}_id"),
    )
    if id_field is None:
        return None
    return {
        "reference_type": reference_type,
        id_field: str(entity[id_field]),
        "label": str(
            entity.get("name")
            or entity.get("display_name")
            or entity.get("title")
            or entity[id_field]
        ),
    }


def _method_role_reference(
    entity: dict[str, Any],
) -> dict[str, str] | None:
    if isinstance(entity.get("entity_id"), str) and entity["entity_id"]:
        id_field = "entity_id"
    elif isinstance(entity.get("method_id"), str) and entity["method_id"]:
        id_field = "method_id"
    else:
        return None
    return {
        "reference_type": "method",
        id_field: str(entity[id_field]),
        "label": str(
            entity.get("name")
            or entity.get("display_name")
            or entity.get("title")
            or entity[id_field]
        ),
    }


def _result_role_reference(
    result: dict[str, Any],
) -> dict[str, str] | None:
    id_field = _first_available_key(result, ("result_row_id", "result_id"))
    if id_field is None:
        return None
    return {
        "reference_type": "result",
        id_field: str(result[id_field]),
        "label": str(
            result.get("value_text")
            or result.get("summary")
            or result.get("metric_name")
            or result.get("metric_id")
            or result[id_field]
        ),
    }


def _span_mentions_assumption_or_limitation(span: dict[str, Any]) -> bool:
    text = str(span.get("quote_text") or span.get("text") or span.get("content") or "")
    return _text_mentions_assumption_or_limitation(text)


def _chunk_mentions_assumption_or_limitation(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or item.get("content") or item.get("section_text") or "")
    return _text_mentions_assumption_or_limitation(text)


def _span_mentions_hypothesis_validation(span: dict[str, Any]) -> bool:
    text = str(span.get("quote_text") or span.get("text") or span.get("content") or "")
    return _text_mentions_hypothesis_validation(text)


def _chunk_mentions_hypothesis_validation(item: dict[str, Any]) -> bool:
    text = str(item.get("text") or item.get("content") or item.get("section_text") or "")
    return _text_mentions_hypothesis_validation(text)


def _text_mentions_assumption_or_limitation(text: str) -> bool:
    normalized_text = "".join(
        character if character.isalnum() else " " for character in text.casefold()
    )
    normalized_tokens = set(normalized_text.split())
    return bool(
        normalized_tokens
        & {
            "assume",
            "assumes",
            "assumption",
            "assumptions",
            "constraint",
            "constraints",
            "limitation",
            "limitations",
        }
    )


def _text_mentions_hypothesis_validation(text: str) -> bool:
    normalized_text = "".join(
        character if character.isalnum() else " " for character in text.casefold()
    )
    normalized_tokens = normalized_text.split()
    role_tokens = {
        "ablate",
        "ablated",
        "ablation",
        "ablations",
        "replicate",
        "replicated",
        "replication",
        "validate",
        "validated",
        "validation",
    }
    if set(normalized_tokens) & role_tokens:
        return True
    for index, token in enumerate(normalized_tokens):
        if token not in {"control", "controlled", "controls"}:
            continue
        window_start = max(0, index - 3)
        window_end = min(len(normalized_tokens), index + 4)
        nearby_tokens = (
            normalized_tokens[window_start:index]
            + normalized_tokens[index + 1 : window_end]
        )
        if set(nearby_tokens) & {
            "ablation",
            "baseline",
            "condition",
            "conditions",
            "donor",
            "experiment",
            "experimental",
            "external",
            "group",
            "held",
            "matched",
            "negative",
            "positive",
            "preprocessing",
            "randomized",
            "treatment",
            "validation",
        }:
            return True
    return False


def _first_available_key(
    item: dict[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        if isinstance(item.get(key), str) and item[key]:
            return key
    return None


def _intelligence_layer_references(context: ResearchSkillContext) -> list[dict[str, str]]:
    intelligence_layers = context.evidence_payload.get("intelligence_layers")
    if not isinstance(intelligence_layers, dict):
        return []
    references: list[dict[str, str]] = []
    for layer_name in ("evidence_memory", "pattern_memory", "source_fact_memory"):
        for item in intelligence_layers.get(layer_name, [])[:8]:
            if not isinstance(item, dict) or not item.get("memory_record_id"):
                continue
            references.append(
                {
                    "reference_type": "research_memory",
                    "memory_record_id": str(item["memory_record_id"]),
                    "label": str(item.get("title") or item.get("summary") or layer_name),
                }
            )
    field_graph = intelligence_layers.get("field_graph")
    if not isinstance(field_graph, dict):
        return references
    for node in field_graph.get("nodes", [])[:8]:
        if not isinstance(node, dict) or not node.get("graph_node_id"):
            continue
        references.append(
            {
                "reference_type": "field_graph_node",
                "graph_node_id": str(node["graph_node_id"]),
                "label": str(node.get("label") or node.get("node_type") or "Field graph node"),
            }
        )
    for edge in field_graph.get("edges", [])[:8]:
        if not isinstance(edge, dict) or not edge.get("graph_edge_id"):
            continue
        references.append(
            {
                "reference_type": "field_graph_edge",
                "graph_edge_id": str(edge["graph_edge_id"]),
                "label": str(edge.get("edge_type") or "Field graph edge"),
            }
        )
    return references


def _join_or_default(values: list[str], default: str) -> str:
    return ", ".join(values[:8]) if values else default


def _unique(values) -> list[str]:  # noqa: ANN001
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item or item.casefold() in seen:
            continue
        seen.add(item.casefold())
        cleaned.append(item)
    return cleaned
