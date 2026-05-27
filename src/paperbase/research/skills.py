"""Internal research-agent skills and quality harnesses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SKILL_VERSION = "2026-05-11"


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
    if any(term in normalized for term in ("unsupported", "evidence coverage", "claim", "quality", "check")):
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


def _literature_review(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(context, artifact_type="literature_review", title="Literature review synthesis")
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
            "summary": _join_or_default(methods, "The collection needs structured method extraction."),
            "evidence_papers": _paper_titles(papers),
        },
        {
            "title": "Benchmark and measurement practice",
            "summary": _join_or_default(datasets + metrics, "Benchmarks and metrics are not extracted yet."),
            "evidence_papers": _paper_titles(papers),
        },
        {
            "title": "Reported limitations and validity threats",
            "summary": _join_or_default(limitations, "Explicit limitations were not extracted yet."),
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
        "evidence_references": _evidence_references(context),
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
        results = [
            result for result in paper.get("results", [])
            if isinstance(result, dict)
        ]
        method_names.extend(methods)
        dataset_names.extend(datasets)
        metric_names.extend(metrics)
        if results:
            for result in results[:3]:
                rows.append(
                    {
                        "paper_title": title,
                        "method": str(result.get("method") or (methods[0] if methods else "Unknown method")),
                        "dataset": str(result.get("dataset") or (datasets[0] if datasets else "")),
                        "metric": str(result.get("metric") or (metrics[0] if metrics else "")),
                        "value": result.get("value_numeric") if result.get("value_numeric") is not None else result.get("value_text"),
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
        "summary": "Comparison generated from selected papers." if papers else "No paper evidence was available for comparison.",
        "comparison_rows": rows[:12],
        "method_families": _unique(method_names)[:8],
        "datasets": _unique(dataset_names)[:8],
        "metrics": _unique(metric_names)[:8],
        "save_options": ["markdown", "csv"],
    }


def _benchmark_planning(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(context, artifact_type="benchmark_plan", title="Benchmark plan")
    datasets = _unique(item for paper in papers for item in paper.get("datasets", []))
    metrics = _unique(item for paper in papers for item in paper.get("metrics", []))
    methods = _unique(item for paper in papers for item in paper.get("methods", []))
    structured_evidence_ready = bool(datasets or metrics or methods or any(paper.get("results") for paper in papers))
    return {
        **common,
        "structured_evidence_ready": structured_evidence_ready,
        "readiness_blockers": [] if structured_evidence_ready else ["structured_extraction_missing"],
        "next_actions": []
        if structured_evidence_ready
        else ["Run extraction in Library for the selected papers."],
        "benchmark_recommendations": [
            "Reuse the most common dataset and metric from the collection as the primary comparison.",
            "Add one stress test that targets an extracted limitation or user-source concern.",
            "Report matched baselines under identical splits before adding new model variants.",
        ],
        "datasets": datasets[:8],
        "metrics_or_result_logic": metrics[:8] or ["Define a metric for each claim before running."],
        "candidate_baselines": methods[:8] or ["Use the strongest method reported by exemplar papers."],
    }


def _experiment_planning(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(context, artifact_type="experiment_plan", title="Experiment plan")
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
        or ["Remove one model assumption at a time and report the same metrics used by reference papers."],
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
    patterns = [
        f"Methods recur around {_join_or_default(methods, 'methods that still need extraction')}.",
        f"Evaluation is framed by {_join_or_default(datasets + metrics, 'datasets and metrics that still need extraction')}.",
        f"Limitations cluster around {_join_or_default(limitations, 'validity threats that still need extraction')}.",
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
    common = _common_payload(context, artifact_type="experiment_backlog", title="Experiment backlog")
    design_elements = [
        item
        for paper in papers
        for item in paper.get("research_design_elements", [])
        if isinstance(item, dict) and item.get("title")
    ]
    backlog_items = [
        str(item.get("title"))
        for item in design_elements
    ][:6] or [
        "Run matched baselines on the collection's most common dataset.",
        "Add one ablation that removes a core method assumption.",
        "Report sensitivity analysis for the strongest validity threat.",
    ]
    return {
        **common,
        "backlog_items": backlog_items,
        "prioritization_rule": "Prioritize items with direct paper evidence and clear evaluation metrics.",
        "next_actions": [
            "Assign each backlog item a dataset, metric, and expected failure mode.",
            "Link each planned experiment to at least one paper or study source.",
        ],
    }


def _assumption_mapping(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(context, artifact_type="assumption_map", title="Assumption map")
    methods = _unique(item for paper in papers for item in paper.get("methods", []))
    limitations = _unique(item for paper in papers for item in paper.get("limitations", []))
    assumptions = [
        f"{method} remains reliable under the target data conditions."
        for method in methods[:3]
    ]
    assumptions.extend(
        f"The study can control for limitation: {limitation}."
        for limitation in limitations[:3]
    )
    return {
        **common,
        "assumptions_to_challenge": assumptions
        or ["The central method assumption is not yet explicit; extract methods or add study sources."],
        "challenge_tests": [
            "Create one falsification test per assumption.",
            "Use collection metrics to define pass/fail criteria.",
        ],
        "evidence_gaps": _research_gaps(limitations, context),
    }


def _hypothesis_generation(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(context, artifact_type="hypotheses", title="Collection-grounded hypotheses")
    methods = _unique(item for paper in papers for item in paper.get("methods", []))
    datasets = _unique(item for paper in papers for item in paper.get("datasets", []))
    method = methods[0] if methods else "the target method"
    dataset = datasets[0] if datasets else "the benchmark corpus"
    return {
        **common,
        "hypotheses": [
            {
                "claim": f"{method} improves reliability on {dataset} when the key design assumption is isolated.",
                "rationale": "Collection evidence suggests a testable design assumption.",
                "test": "Run matched baselines and ablations under the same evaluation protocol.",
            }
        ],
        "validation_plan": [
            "Define a measurable claim for each hypothesis.",
            "Reuse collection datasets and metrics where possible to make comparisons interpretable.",
        ],
    }


def _revision_planning(context: ResearchSkillContext) -> dict[str, Any]:
    papers = _papers(context)
    common = _common_payload(context, artifact_type="revision_plan", title="Revision plan")
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
            "Add controls or ablations for each extracted limitation that applies to the user's study.",
            "Align evaluation tables with datasets, metrics, and baselines that recur across the collection.",
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
        source
        for source in context.evidence_payload.get("sources", [])
        if isinstance(source, dict)
    ]
    missing_context: list[str] = []
    if not papers:
        missing_context.append("No paper evidence was available.")
    if output_payload.get("artifact_type") in {"literature_review", "benchmark_plan"}:
        extracted_signal = any(
            paper.get("methods") or paper.get("datasets") or paper.get("metrics") or paper.get("results")
            for paper in papers
        )
        if papers and not extracted_signal:
            missing_context.append("Structured extraction is missing for the selected papers.")
    evidence_coverage = 0.0 if not papers else min(1.0, len(papers) / max(1, len(_paper_titles(papers))))
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
    return gaps or ["No explicit gap evidence was extracted yet; run extraction or add study sources."]


def _common_payload(context: ResearchSkillContext, *, artifact_type: str, title: str) -> dict[str, Any]:
    papers = _papers(context)
    return {
        "artifact_type": artifact_type,
        "title": title,
        "request": context.message,
        "paper_count": len(papers),
        "evidence_basis": _paper_titles(papers),
        "evidence_references": _evidence_references(context),
        "source_context": _source_context(context),
        "general_methodology": [
            "Separate claims, experimental variables, and evaluation criteria before writing new experiments.",
            "Use paper-level evidence as constraints, then identify where a new study can challenge one assumption.",
        ],
    }


def _source_context(context: ResearchSkillContext) -> list[dict[str, str]]:
    sources = [
        source
        for source in context.evidence_payload.get("sources", [])
        if isinstance(source, dict)
    ]
    return [
        {
            "title": str(source.get("title") or "Study source"),
            "source_type": str(source.get("source_type") or "text"),
            "summary": str(source.get("summary") or ""),
        }
        for source in sources
    ]


def _papers(context: ResearchSkillContext) -> list[dict[str, Any]]:
    return [paper for paper in context.evidence_payload.get("papers", []) if isinstance(paper, dict)]


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
