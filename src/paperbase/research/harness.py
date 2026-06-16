"""Quality harnesses for Paperbase research-agent outputs."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from paperbase.research.task_quality import task_section_has_meaningful_content

SUPPORT_STATUS_VALUES = (
    "supported",
    "mixed",
    "inferred",
    "user_provided",
    "speculative",
)
SUPPORTING_LAYER_VALUES = (
    "evidence_memory",
    "pattern_memory",
    "source_fact_memory",
    "field_graph",
    "study_brief",
    "source_library",
)
EVIDENCE_BACKED_STATUSES = {"supported", "mixed", "user_provided"}
REFERENCE_BACKED_LAYERS = {
    "evidence_memory",
    "source_fact_memory",
    "field_graph",
    "study_brief",
    "source_library",
}
SUPPORTING_LAYER_REQUIRED_STATUSES = {"supported", "mixed", "inferred", "user_provided"}
RECOMMENDATION_FIELD_NAME = "recommendations"
RECOMMENDATION_FIELD_SUFFIX = "_recommendations"
REFERENCE_ID_FIELD_KEYS = {
    "paper_id": "paper_ids",
    "source_id": "source_ids",
    "study_source_id": "source_ids",
    "chunk_id": "chunk_ids",
    "evidence_span_id": "evidence_span_ids",
    "span_id": "evidence_span_ids",
    "evidence_id": "evidence_span_ids",
    "figure_id": "figure_ids",
    "table_id": "table_ids",
    "entity_id": "entity_ids",
    "dataset_id": "dataset_ids",
    "method_id": "method_ids",
    "metric_id": "metric_ids",
    "result_id": "result_row_ids",
    "result_row_id": "result_row_ids",
    "memory_record_id": "memory_record_ids",
    "graph_node_id": "graph_node_ids",
    "graph_edge_id": "graph_edge_ids",
    "study_brief_id": "study_brief_ids",
}
REFERENCE_TYPE_ALLOWED_ID_FIELDS = {
    "paper": {"paper_id"},
    "source": {"source_id", "study_source_id"},
    "study_source": {"source_id", "study_source_id"},
    "chunk": {"chunk_id"},
    "evidence_span": {"evidence_span_id", "span_id", "evidence_id"},
    "span": {"evidence_span_id", "span_id", "evidence_id"},
    "evidence": {"evidence_span_id", "span_id", "evidence_id"},
    "figure": {"figure_id"},
    "table": {"table_id"},
    "structured_entity": {"entity_id", "dataset_id", "method_id", "metric_id"},
    "dataset": {"entity_id", "dataset_id"},
    "method": {"entity_id", "method_id"},
    "metric": {"entity_id", "metric_id"},
    "result": {"result_id", "result_row_id"},
    "result_row": {"result_id", "result_row_id"},
    "research_memory": {"memory_record_id"},
    "memory": {"memory_record_id"},
    "evidence_memory": {"memory_record_id"},
    "pattern_memory": {"memory_record_id"},
    "source_fact_memory": {"memory_record_id"},
    "field_graph": {"graph_node_id", "graph_edge_id"},
    "field_graph_node": {"graph_node_id"},
    "field_graph_edge": {"graph_edge_id"},
    "graph_node": {"graph_node_id"},
    "graph_edge": {"graph_edge_id"},
    "study_brief": {"study_brief_id"},
    "brief": {"study_brief_id"},
}
REFERENCE_TYPE_ENTITY_INVENTORY_KEYS = {
    "dataset": "dataset_ids",
    "method": "method_ids",
    "metric": "metric_ids",
}
SPAN_CHUNK_SUPPORT_REFERENCE_FIELDS = (
    "chunk_id",
    "evidence_span_id",
    "span_id",
    "evidence_id",
)
STRUCTURED_SUPPORT_REFERENCE_FIELDS = (
    "entity_id",
    "dataset_id",
    "method_id",
    "metric_id",
    "result_id",
    "result_row_id",
)
SOURCE_FACT_SUPPORT_REFERENCE_FIELDS = ("memory_record_id",)
RECOMMENDATION_SUPPORT_TEXT_FIELDS = (
    "title",
    "detail",
    "claim",
    "summary",
    "rationale",
    "objective",
    "description",
    "text",
)
VALIDATION_ISSUE_COUNT_KEYS = ("category", "severity", "code")
VALIDATION_ISSUE_REMEDIATION_BY_CODE = {
    "readiness_warning": (
        "Review the readiness warning before trusting this artifact; rerun parse "
        "or extraction recovery if it names stale or missing evidence."
    ),
    "readiness_blocker": "Resolve the listed blocker and rerun the research instruction.",
    "no_paper_evidence": (
        "Select or ingest paper evidence before asking for a grounded artifact."
    ),
    "no_explicit_evidence_references": (
        "Add evidence references that point to selected papers, chunks, spans, "
        "sources, memory, or structured rows."
    ),
    "incompatible_evidence_reference_type": (
        "Fix the reference_type or target id so they describe the same selected "
        "evidence object."
    ),
    "invalid_evidence_reference_target": (
        "Replace the target id with an id present in the selected context or rerun "
        "with that evidence selected."
    ),
    "unverifiable_evidence_reference": (
        "Attach a concrete selected-context target id to the evidence reference."
    ),
    "semantic_entailment_mismatch": (
        "Downgrade or rewrite the claim so it does not exceed the cited evidence, "
        "or cite stronger support."
    ),
    "comparator_polarity_mismatch": (
        "Align the recommendation direction with the cited result or comparator, "
        "or cite a result that supports the stated direction."
    ),
    "negation_mismatch": (
        "Rewrite the claim to preserve the cited negation or cite positive evidence "
        "for the asserted term."
    ),
    "result_value_mismatch": (
        "Correct the stated numeric result or cite the result row that contains "
        "that value."
    ),
    "result_relation_mismatch": (
        "Use dataset, method, metric, and result-row references from the same "
        "selected result relation."
    ),
    "weak_lexical_support": (
        "Cite evidence whose visible text overlaps the recommendation's concrete "
        "method, dataset, metric, claim, or limitation."
    ),
    "support_text_unavailable": (
        "Rerun parsing or extraction, or choose a cited object with prompt-visible text."
    ),
    "missing_required_section": (
        "Fill the required task section before treating the artifact as complete."
    ),
    "missing_evidence_role_citation": (
        "Cite available task-role evidence for the missing groups, such as dataset, "
        "method, metric, result, validation, or limitation evidence."
    ),
    "missing_support_status": (
        "Set support_status to supported, mixed, inferred, user_provided, or speculative."
    ),
    "invalid_support_status": (
        "Set support_status to supported, mixed, inferred, user_provided, or speculative."
    ),
    "missing_supporting_layers": (
        "Add supporting_layers that describe whether the claim relies on source_library, "
        "evidence_memory, source_fact_memory, field_graph, study_brief, or pattern_memory."
    ),
    "invalid_supporting_layer": (
        "Replace unsupported layer names with registered supporting layer values."
    ),
    "recommendation_missing_evidence_references": (
        "Add recommendation-level evidence references for evidence-backed or "
        "user-provided claims."
    ),
    "unavailable_supporting_layer": (
        "Remove unavailable supporting layers or rerun after that layer is built into "
        "selected context."
    ),
    "source_fact_memory_marked_supported": (
        "Use user_provided or mixed for Study-source facts unless paper evidence also "
        "supports the claim."
    ),
    "speculative_claim_marked_supported": (
        "Mark speculative recommendations as speculative or cite direct support before "
        "marking them supported."
    ),
}
TASK_ARTIFACT_TYPES_BY_SKILL_ID = {
    "literature_review": "literature_review",
    "quality_harness": "critique",
    "comparison": "comparison",
    "benchmark_planning": "benchmark_plan",
    "experiment_planning": "experiment_plan",
    "field_pattern_analysis": "field_patterns",
    "experiment_backlog": "experiment_backlog",
    "assumption_mapping": "assumption_map",
    "hypothesis_generation": "hypotheses",
    "revision_planning": "revision_plan",
}
TASK_REQUIRED_SECTIONS_BY_ARTIFACT_TYPE = {
    "literature_review": ("themes", "research_gaps", "future_directions"),
    "critique": ("quality_checks",),
    "comparison": ("comparison_rows",),
    "benchmark_plan": (
        "benchmark_recommendations",
        "datasets",
        "metrics_or_result_logic",
        "candidate_baselines",
    ),
    "experiment_plan": ("objective", "baselines", "ablations", "controls"),
    "field_patterns": ("patterns",),
    "experiment_backlog": ("backlog_items",),
    "assumption_map": (
        "assumptions_to_challenge",
        "challenge_tests",
        "evidence_gaps",
    ),
    "hypotheses": ("hypotheses", "validation_plan"),
    "revision_plan": ("revision_priorities",),
}
TASK_EVIDENCE_ROLE_REQUIREMENTS_BY_ARTIFACT_TYPE = {
    "critique": {
        "assumption_or_limitation": ("assumption", "limitation"),
        "method": ("method",),
    },
    "experiment_plan": {
        "evaluation": ("result", "metric"),
        "method": ("method",),
    },
    "assumption_map": {
        "assumption_or_limitation": ("assumption", "limitation"),
        "method": ("method",),
    },
    "benchmark_plan": {
        "dataset": ("dataset",),
        "evaluation": ("result", "metric"),
        "method": ("method",),
    },
    "hypotheses": {
        "dataset": ("dataset",),
        "evaluation": ("result", "metric"),
        "method": ("method",),
        "validation": ("ablation", "control", "validation"),
    },
    "revision_plan": {
        "assumption_or_limitation": ("assumption", "limitation"),
        "method": ("method",),
    },
}
TEXT_EVIDENCE_ROLE_KEYWORDS = {
    "ablation": frozenset(("ablate", "ablated", "ablation", "ablations")),
    "assumption": frozenset(("assume", "assumes", "assumption", "assumptions")),
    "limitation": frozenset(("constraint", "constraints", "limitation", "limitations")),
    "validation": frozenset(
        ("replicate", "replicated", "replication", "validate", "validated", "validation")
    ),
}
CONTROL_ROLE_CONTEXT_TOKENS = frozenset(
    (
        "ablation",
        "baseline",
        "condition",
        "conditions",
        "donor",
        "experiment",
        "experimental",
        "group",
        "held",
        "held-out",
        "matched",
        "negative",
        "positive",
        "preprocessing",
        "treatment",
    )
)
MIN_SPAN_CHUNK_SUPPORT_TOKEN_OVERLAP = 2
MIN_STRUCTURED_EVIDENCE_SUPPORT_TOKEN_OVERLAP = 1
MIN_SOURCE_FACT_SUPPORT_TOKEN_OVERLAP = 2
NEGATION_SCOPE_TOKEN_WINDOW = 8
RESULT_VALUE_REFERENCE_FIELDS = ("result_id", "result_row_id")
RESULT_RELATION_REFERENCE_FIELDS = ("dataset_id", "method_id", "metric_id")
RESULT_VALUE_CLAIM_TOKEN_WINDOW = 2
RESULT_VALUE_GENERIC_ANCHOR_TOKENS = {"score", "value"}
NUMERIC_VALUE_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?"
NUMERIC_VALUE_RE = re.compile(
    rf"(?<![a-z0-9]){NUMERIC_VALUE_PATTERN}(?![a-z0-9])"
)
RESULT_COMPARATOR_NEGATIVE_RE = re.compile(
    r"\b(?:"
    r"underperform[a-z]*|"
    r"worse|"
    r"lower|"
    r"below|"
    r"inferior|"
    r"(?:do|does|did)\s+not\s+outperform|"
    r"fail(?:s|ed|ing)?\s+to\s+outperform|"
    r"fail(?:s|ed|ing)?\s+to\s+improv(?:e|es|ed|ing)|"
    r"no\s+improvement|"
    r"not\s+improv(?:e|es|ed|ing)"
    r")\b"
)
RESULT_COMPARATOR_POSITIVE_RE = re.compile(
    r"\b(?:"
    r"outperform[a-z]*|"
    r"beats?|"
    r"exceed[a-z]*|"
    r"surpass[a-z]*|"
    r"improv(?:e|es|ed|ing)|"
    r"higher|"
    r"better|"
    r"stronger|"
    r"superior"
    r")\b"
)
RESULT_COMPARATOR_DIRECTION_RE = re.compile(
    r"\b(?P<direction>higher|lower|smaller)[\s_-]+is[\s_-]+better\b"
)
ENTAILMENT_OVERCLAIM_CUE_TOKENS = frozenset(
    (
        "confirm",
        "confirmed",
        "confirms",
        "demonstrate",
        "demonstrated",
        "demonstrates",
        "establish",
        "established",
        "establishes",
        "prove",
        "proved",
        "proves",
        "proof",
        "validate",
        "validated",
        "validates",
    )
)
ENTAILMENT_CLAIM_FAMILY_TOKENS = {
    "robustness": frozenset(
        (
            "robust",
            "robustness",
            "sensitivity",
            "stress",
        )
    ),
    "generalization": frozenset(
        (
            "cross-domain",
            "cross-dataset",
            "distribution-shift",
            "external",
            "generalisation",
            "generalise",
            "generalised",
            "generalises",
            "generalization",
            "generalize",
            "generalized",
            "generalizes",
            "ood",
            "out-of-distribution",
            "transfer",
        )
    ),
    "deployment readiness": frozenset(
        (
            "deploy",
            "deployable",
            "deployed",
            "deployment",
            "deployment-ready",
            "deployment-readiness",
            "operational",
            "production",
            "production-ready",
            "readiness",
            "real-world",
        )
    ),
}
ENTAILMENT_SCALAR_RESULT_TOKENS = frozenset(
    (
        "accuracy",
        "auc",
        "auprc",
        "auroc",
        "benchmark",
        "f1",
        "metric",
        "metrics",
        "precision",
        "recall",
        "report",
        "reported",
        "reporting",
        "reports",
        "result",
        "results",
        "score",
        "scores",
    )
)
ENTAILMENT_CAUSAL_CLAIM_TOKENS = frozenset(
    (
        "cause",
        "caused",
        "causes",
        "causal",
        "causally",
        "lead",
        "leads",
        "led",
        "resulted",
    )
)
ENTAILMENT_CAUSAL_CLAIM_FOLLOWING_TOKENS = {
    "lead": frozenset(("to",)),
    "leads": frozenset(("to",)),
    "led": frozenset(("to",)),
    "resulted": frozenset(("in",)),
}
ENTAILMENT_CAUSAL_CLAIM_CAUTION_TOKENS = frozenset(
    (
        "avoid",
        "avoided",
        "avoids",
        "noncausal",
        "without",
    )
)
ENTAILMENT_CAUSAL_ADJECTIVE_TOKENS = frozenset(("causal", "causally"))
ENTAILMENT_CAUSAL_VERB_NEGATED_SUPPORT_SEQUENCES = (
    ("does", "not", "show"),
    ("does", "not", "prove"),
    ("does", "not", "support"),
    ("does", "not", "demonstrate"),
    ("does", "not", "establish"),
    ("did", "not", "show"),
    ("did", "not", "prove"),
    ("did", "not", "support"),
    ("did", "not", "demonstrate"),
    ("did", "not", "establish"),
    ("do", "not", "claim"),
    ("do", "not", "conclude"),
    ("do", "not", "infer"),
    ("cannot", "claim"),
    ("cannot", "conclude"),
    ("cannot", "infer"),
)
ENTAILMENT_CAUSAL_VERB_POST_DISCLAIMER_SEQUENCES = (
    ("is", "not", "supported"),
    ("is", "not", "warranted"),
    ("is", "unsupported"),
    ("is", "unwarranted"),
    ("are", "not", "supported"),
    ("are", "not", "warranted"),
    ("are", "unsupported"),
    ("are", "unwarranted"),
    ("was", "not", "supported"),
    ("was", "not", "warranted"),
    ("was", "unsupported"),
    ("was", "unwarranted"),
    ("were", "not", "supported"),
    ("were", "not", "warranted"),
    ("were", "unsupported"),
    ("were", "unwarranted"),
)
ENTAILMENT_CAUSAL_DISCLAIMER_GOVERNED_TOKEN_LIMIT = 6
ENTAILMENT_CAUSAL_DISCLAIMER_CONNECTOR_TOKENS = frozenset(
    ("and", "but", "however", "whereas", "while")
)
ENTAILMENT_CAUSAL_DISCLAIMER_TARGET_SWITCH_TOKENS = frozenset(
    (
        "caveat",
        "caveats",
        "endpoint",
        "endpoints",
        "interaction",
        "interactions",
        "limitation",
        "limitations",
        "mechanism",
        "mechanisms",
        "mediator",
        "mediators",
        "pathway",
        "pathways",
        "subgroup",
        "subgroups",
    )
)
ENTAILMENT_CAUSAL_POST_CAUTION_TOKENS = frozenset(
    ("not", "unsupported", "unwarranted")
)
ENTAILMENT_ASSOCIATIONAL_EVIDENCE_TOKENS = frozenset(
    (
        "associate",
        "associated",
        "association",
        "associations",
        "cohort",
        "correlate",
        "correlated",
        "correlation",
        "correlations",
        "link",
        "linked",
        "links",
        "observational",
        "relationship",
        "relationships",
    )
)
ENTAILMENT_CAUSAL_STUDY_SIGNAL_TOKENS = frozenset(
    (
        "experiment",
        "experimental",
        "intervention",
        "interventional",
        "mechanism",
        "mechanistic",
        "randomised",
        "randomized",
        "trial",
    )
)
ENTAILMENT_CAUSAL_STUDY_NONPOSITIVE_TOKENS = frozenset(
    (
        "could",
        "future",
        "need",
        "needed",
        "needs",
        "planned",
        "possible",
        "potential",
        "proposed",
        "required",
        "should",
        "may",
        "might",
        "hypothesised",
        "hypothesized",
        "hypothetical",
        "unclear",
        "unexamined",
        "unknown",
        "untested",
        "will",
        "would",
    )
)
NEGATION_CLAUSE_SPLIT_RE = re.compile(r"\b(?:but|however|whereas|while)\b|[.;]")
ANALYSIS_TOKEN_RE = re.compile(rf"{NUMERIC_VALUE_PATTERN}|[a-z][a-z0-9_+-]*")
SUPPORT_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "using",
    "with",
}
NEGATION_CUE_TOKENS = {
    "absent",
    "lack",
    "lacked",
    "lacking",
    "lacks",
    "no",
}
NEGATION_POSTPOSITIVE_CUE_TOKENS = {
    "absent",
    "lacking",
}
NEGATION_ACTION_TOKENS = {
    "cite",
    "cited",
    "cites",
    "evaluate",
    "evaluated",
    "evaluates",
    "include",
    "included",
    "includes",
    "perform",
    "performed",
    "performs",
    "report",
    "reported",
    "reports",
    "show",
    "showed",
    "shown",
    "shows",
    "test",
    "tested",
    "tests",
    "use",
    "used",
    "uses",
    "validate",
    "validated",
    "validates",
}
NEGATION_COMMA_CLAUSE_ACTION_RE = re.compile(
    r",\s+(?=(?:do\s+not\s+|not\s+)?(?:"
    + "|".join(sorted(NEGATION_ACTION_TOKENS))
    + r")\b)"
)
NEGATION_DENIAL_HEAD_TOKENS = {
    "evidence",
    "support",
}
NEGATION_DENIAL_LINK_TOKENS = {
    "for",
    "of",
}
NEGATION_COORDINATOR_TOKENS = {
    "and",
    "or",
}
NEGATION_EVIDENCE_HEAD_TOKENS = {
    "analysis",
    "analyses",
    "evidence",
    "evaluation",
    "evaluations",
    "experiment",
    "experiments",
    "finding",
    "findings",
    "result",
    "results",
    "study",
    "studies",
    "support",
    "test",
    "tests",
    "validation",
    "validations",
}
NEGATION_COMMA_LIST_RE = re.compile(
    r"\b(?:"
    + "|".join(sorted(re.escape(token) for token in NEGATION_CUE_TOKENS))
    + r")\s+(?P<items>[a-z0-9_+-]+(?:\s*,\s*(?:and\s+|or\s+)?[a-z0-9_+-]+)+)"
    + r"\s+(?:"
    + "|".join(sorted(re.escape(token) for token in NEGATION_EVIDENCE_HEAD_TOKENS))
    + r")\b"
)
NEGATION_SCOPE_BOUNDARY_TOKENS = {
    "against",
    "although",
    "but",
    "by",
    "for",
    "however",
    "in",
    "on",
    "though",
    "whereas",
    "while",
    "with",
}
NEGATION_TERM_STOPWORDS = SUPPORT_TOKEN_STOPWORDS | {
    "analysis",
    "analyses",
    "available",
    "been",
    "being",
    "but",
    "can",
    "cite",
    "cited",
    "cites",
    "could",
    "did",
    "do",
    "does",
    "done",
    "evidence",
    "finding",
    "findings",
    "had",
    "however",
    "not",
    "paper",
    "present",
    "result",
    "results",
    "should",
    "source",
    "support",
    "supported",
    "study",
    "studies",
    "was",
    "were",
    "whereas",
    "while",
    "would",
}


def validate_research_output(
    *,
    context: dict[str, Any],
    output_payload: dict[str, Any],
    readiness_warnings: list[str],
    forced_status: str | None = None,
    entailment_grader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    papers = [paper for paper in context.get("papers", []) if isinstance(paper, dict)]
    references = [
        reference
        for reference in output_payload.get("evidence_references", [])
        if isinstance(reference, dict)
    ]
    readiness_blockers = list(output_payload.get("readiness_blockers") or [])
    missing_evidence: list[str] = []
    unsupported_claims: list[str] = []
    recommendations = _recommendations(output_payload)
    available_layers = _available_layers(context)
    reference_diagnostics = _reference_diagnostics(
        context=context,
        output_payload=output_payload,
        recommendations=recommendations,
    )
    active_entailment_grader = _active_entailment_grader(entailment_grader)

    if not papers:
        missing_evidence.append("No paper evidence was available.")
    if papers and not references and forced_status != "blocked":
        missing_evidence.append("No explicit evidence references were provided.")
    if output_payload.get("summary") and not papers:
        unsupported_claims.append(str(output_payload["summary"]))

    recommendation_diagnostics = _recommendation_diagnostics(
        recommendations=recommendations,
        available_layers=available_layers,
    )
    support_diagnostics = _span_chunk_support_diagnostics(
        context=context,
        recommendations=recommendations,
        entailment_grader=active_entailment_grader,
        count_unknown_entailment_checks=entailment_grader is not None,
    )
    structured_support_diagnostics = _structured_evidence_support_diagnostics(
        context=context,
        recommendations=recommendations,
        entailment_grader=active_entailment_grader,
        count_unknown_entailment_checks=entailment_grader is not None,
    )
    source_fact_support_diagnostics = _source_fact_memory_support_diagnostics(
        context=context,
        recommendations=recommendations,
    )
    task_quality_diagnostics = _task_quality_diagnostics(output_payload=output_payload)
    task_evidence_role_diagnostics = _task_evidence_role_diagnostics(
        context=context,
        output_payload=output_payload,
        recommendations=recommendations,
    )
    missing_evidence.extend(recommendation_diagnostics["missing_evidence"])
    unsupported_claims.extend(recommendation_diagnostics["unsupported_claims"])
    missing_evidence.extend(reference_diagnostics["missing_evidence"])
    missing_evidence.extend(support_diagnostics["missing_evidence"])
    missing_evidence.extend(structured_support_diagnostics["missing_evidence"])
    missing_evidence.extend(source_fact_support_diagnostics["missing_evidence"])
    missing_evidence.extend(task_quality_diagnostics["missing_evidence"])
    missing_evidence.extend(task_evidence_role_diagnostics["missing_evidence"])
    entailment_report = _combined_entailment_report(
        support_diagnostics,
        structured_support_diagnostics,
    )

    if forced_status == "blocked" or readiness_blockers:
        harness_status = "blocked"
    elif missing_evidence or unsupported_claims:
        harness_status = "needs_attention"
    else:
        harness_status = "passed"

    validation_issues = _validation_issues(
        missing_evidence=missing_evidence,
        unsupported_claims=unsupported_claims,
        readiness_blockers=readiness_blockers,
        readiness_warnings=readiness_warnings,
    )

    return {
        "harness_status": harness_status,
        "missing_evidence": missing_evidence,
        "unsupported_claims": unsupported_claims,
        "readiness_blockers": readiness_blockers,
        "readiness_warnings": readiness_warnings,
        "validation_issues": validation_issues,
        "validation_issue_counts": _validation_issue_counts(validation_issues),
        "evidence_reference_count": len(references),
        "evidence_paper_count": len(papers),
        "source_context_count": len(
            [
                source
                for source in context.get("sources", [])
                if isinstance(source, dict)
            ]
        ),
        **reference_diagnostics["report"],
        **recommendation_diagnostics["report"],
        **support_diagnostics["report"],
        **structured_support_diagnostics["report"],
        **source_fact_support_diagnostics["report"],
        **task_quality_diagnostics["report"],
        **task_evidence_role_diagnostics["report"],
        **entailment_report,
    }


def _validation_issues(
    *,
    missing_evidence: list[str],
    unsupported_claims: list[str],
    readiness_blockers: list[Any],
    readiness_warnings: list[str],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for warning in readiness_warnings:
        issues.append(
            _validation_issue(
                category="readiness",
                severity="warning",
                code="readiness_warning",
                source="context",
                message=warning,
            )
        )
    for blocker in readiness_blockers:
        issues.append(
            _validation_issue(
                category="readiness",
                severity="blocker",
                code="readiness_blocker",
                source="output_payload",
                message=str(blocker),
            )
        )
    issues.extend(
        _missing_evidence_issue(message)
        for message in missing_evidence
        if isinstance(message, str) and message
    )
    issues.extend(
        _unsupported_claim_issue(message)
        for message in unsupported_claims
        if isinstance(message, str) and message
    )
    return issues


def _validation_issue(
    *,
    category: str,
    severity: str,
    code: str,
    source: str,
    message: str,
) -> dict[str, str]:
    return {
        "category": category,
        "severity": severity,
        "code": code,
        "source": source,
        "message": message,
        "remediation": _validation_issue_remediation(
            category=category,
            code=code,
        ),
    }


def _validation_issue_remediation(*, category: str, code: str) -> str:
    remediation = VALIDATION_ISSUE_REMEDIATION_BY_CODE.get(code)
    if remediation is not None:
        return remediation
    if category == "unsupported_claim":
        return "Revise the claim or add selected evidence that supports it."
    if category == "evidence":
        return "Add stronger selected evidence or revise the claim to match available evidence."
    return "Review the validation issue and rerun after revising the artifact or selected evidence."


def _validation_issue_counts(
    validation_issues: list[dict[str, str]],
) -> dict[str, Any]:
    counts: dict[str, Any] = {"total": len(validation_issues)}
    for key in VALIDATION_ISSUE_COUNT_KEYS:
        counter = Counter(
            str(issue[key])
            for issue in validation_issues
            if isinstance(issue.get(key), str) and issue.get(key)
        )
        counts[f"by_{key}"] = dict(sorted(counter.items()))
    return counts


def _missing_evidence_issue(message: str) -> dict[str, str]:
    code = "missing_evidence"
    category = "evidence"
    severity = "medium"
    if message == "No paper evidence was available.":
        code = "no_paper_evidence"
        severity = "high"
    elif message == "No explicit evidence references were provided.":
        code = "no_explicit_evidence_references"
    elif message.startswith("Evidence reference "):
        category = "reference"
        if "has incompatible reference_type" in message:
            code = "incompatible_evidence_reference_type"
        elif "targets unavailable selected context" in message:
            code = "invalid_evidence_reference_target"
        elif "has no selected-context target id" in message:
            code = "unverifiable_evidence_reference"
            severity = "low"
        else:
            code = "evidence_reference_issue"
    elif "semantic entailment mismatch" in message:
        category = "entailment"
        code = "semantic_entailment_mismatch"
        severity = "high"
    elif "comparator polarity mismatch" in message:
        category = "support_screen"
        code = "comparator_polarity_mismatch"
        severity = "high"
    elif "negation mismatch" in message:
        category = "support_screen"
        code = "negation_mismatch"
        severity = "high"
    elif "result-row value mismatch" in message:
        category = "support_screen"
        code = "result_value_mismatch"
        severity = "high"
    elif "result-row relation mismatch" in message:
        category = "support_screen"
        code = "result_relation_mismatch"
        severity = "high"
    elif "weak lexical support" in message:
        category = "support_screen"
        code = "weak_lexical_support"
    elif "has no prompt-visible text for support screening" in message:
        category = "support_screen"
        code = "support_text_unavailable"
    elif message.startswith("Task artifact is missing required section"):
        category = "task_quality"
        code = "missing_required_section"
        severity = "high"
    elif message.startswith("Task artifact is missing required evidence role"):
        category = "task_quality"
        code = "missing_evidence_role_citation"
    elif message.startswith("Recommendation is missing support_status"):
        category = "recommendation_contract"
        code = "missing_support_status"
    elif message.startswith("Recommendation has invalid support_status"):
        category = "recommendation_contract"
        code = "invalid_support_status"
    elif message.startswith("Recommendation is missing supporting_layers"):
        category = "recommendation_contract"
        code = "missing_supporting_layers"
    elif message.startswith("Recommendation cites invalid supporting layer"):
        category = "recommendation_contract"
        code = "invalid_supporting_layer"
    elif message.startswith("Recommendation is missing evidence references"):
        category = "recommendation_contract"
        code = "recommendation_missing_evidence_references"
    return _validation_issue(
        category=category,
        severity=severity,
        code=code,
        source="missing_evidence",
        message=message,
    )


def _unsupported_claim_issue(message: str) -> dict[str, str]:
    code = "unsupported_claim"
    category = "unsupported_claim"
    severity = "high"
    if message.startswith("Recommendation cites unavailable supporting layer"):
        category = "recommendation_contract"
        code = "unavailable_supporting_layer"
    elif "source_fact_memory" in message and "supported paper evidence" in message:
        category = "recommendation_contract"
        code = "source_fact_memory_marked_supported"
    elif message.startswith("Recommendation is speculative but marked supported"):
        category = "recommendation_contract"
        code = "speculative_claim_marked_supported"
    return _validation_issue(
        category=category,
        severity=severity,
        code=code,
        source="unsupported_claims",
        message=message,
    )


def _recommendations(output_payload: dict[str, Any]) -> list[dict[str, Any]]:
    top_level_recommendations = _recommendation_items(
        output_payload.get(RECOMMENDATION_FIELD_NAME),
        field_name=RECOMMENDATION_FIELD_NAME,
    )
    recommendations = list(top_level_recommendations)
    labeled_top_level_texts = {
        _recommendation_label(item).casefold()
        for item in top_level_recommendations
        if _recommendation_support_status(item)
        or _recommendation_supporting_layers(item)
        or _recommendation_references(item)
    }
    for field_name, raw_items in output_payload.items():
        if field_name == RECOMMENDATION_FIELD_NAME:
            continue
        if not _is_recommendation_field(field_name):
            continue
        for item in _recommendation_items(raw_items, field_name=field_name):
            if (
                item.get("_normalized_from_string") is True
                and _recommendation_label(item).casefold() in labeled_top_level_texts
            ):
                continue
            recommendations.append(item)
    return recommendations


def _is_recommendation_field(field_name: str) -> bool:
    return field_name == RECOMMENDATION_FIELD_NAME or field_name.endswith(
        RECOMMENDATION_FIELD_SUFFIX
    )


def _recommendation_items(raw_items: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    recommendations: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            recommendations.append({"recommendation_field": field_name, **item})
        elif isinstance(item, str) and item.strip():
            recommendations.append(
                {
                    "recommendation_field": field_name,
                    "title": item.strip(),
                    "_normalized_from_string": True,
                }
            )
    return recommendations


def _recommendation_support_status(recommendation: dict[str, Any]) -> str | None:
    status = recommendation.get("support_status")
    return status if isinstance(status, str) and status else None


def _recommendation_references(recommendation: dict[str, Any]) -> list[dict[str, Any]]:
    references = recommendation.get("evidence_references")
    if not isinstance(references, list):
        return []
    return [reference for reference in references if isinstance(reference, dict)]


def _reference_diagnostics(
    *,
    context: dict[str, Any],
    output_payload: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    inventory = _reference_inventory(context)
    valid = 0
    invalid = 0
    unverifiable = 0
    incompatible = 0
    missing_evidence: list[str] = []
    for location, reference in _output_references(
        output_payload=output_payload,
        recommendations=recommendations,
    ):
        status, reason, has_incompatible_type = _reference_status(
            reference,
            inventory=inventory,
        )
        if status == "valid":
            valid += 1
            continue
        if status == "invalid":
            invalid += 1
            if has_incompatible_type:
                incompatible += 1
        else:
            unverifiable += 1
        missing_evidence.append(
            f"Evidence reference {reason}: {location}; {_reference_label(reference)}"
        )
    return {
        "missing_evidence": missing_evidence,
        "report": {
            "valid_evidence_reference_count": valid,
            "invalid_evidence_reference_count": invalid,
            "unverifiable_evidence_reference_count": unverifiable,
            "incompatible_evidence_reference_type_count": incompatible,
        },
    }


def _output_references(
    *,
    output_payload: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    references: list[tuple[str, dict[str, Any]]] = [
        ("artifact evidence_references", reference)
        for reference in output_payload.get("evidence_references", [])
        if isinstance(reference, dict)
    ]
    for recommendation in recommendations:
        label = _recommendation_label(recommendation)
        field_name = str(recommendation.get("recommendation_field") or RECOMMENDATION_FIELD_NAME)
        references.extend(
            (
                f"{field_name} evidence_references for {label}",
                reference,
            )
            for reference in _recommendation_references(recommendation)
        )
    return references


def _reference_status(
    reference: dict[str, Any],
    *,
    inventory: dict[str, set[str]],
) -> tuple[str, str, bool]:
    recognized_ids: list[tuple[str, str, str]] = []
    for field_name, inventory_key in REFERENCE_ID_FIELD_KEYS.items():
        value = _string_value(reference.get(field_name))
        if value is not None:
            recognized_ids.append((field_name, inventory_key, value))
    if not recognized_ids:
        return "unverifiable", "has no selected-context target id", False

    missing = [
        f"{field_name}={value}"
        for field_name, inventory_key, value in recognized_ids
        if value not in inventory.get(inventory_key, set())
    ]
    if missing:
        return (
            "invalid",
            f"targets unavailable selected context ({', '.join(missing)})",
            False,
        )
    incompatible = _incompatible_reference_type_targets(
        reference,
        recognized_ids,
        inventory=inventory,
    )
    if incompatible:
        return (
            "invalid",
            f"has incompatible reference_type ({', '.join(incompatible)})",
            True,
        )
    return "valid", "targets selected context", False


def _incompatible_reference_type_targets(
    reference: dict[str, Any],
    recognized_ids: list[tuple[str, str, str]],
    *,
    inventory: dict[str, set[str]],
) -> list[str]:
    reference_type = _reference_type_key(reference.get("reference_type"))
    if reference_type is None:
        return []
    incompatible: list[str] = []
    allowed_fields = REFERENCE_TYPE_ALLOWED_ID_FIELDS.get(reference_type)
    if allowed_fields is not None:
        incompatible.extend(
            f"{reference_type} cannot target {field_name}={value}"
            for field_name, _inventory_key, value in recognized_ids
            if field_name not in allowed_fields
        )
    entity_inventory_key = REFERENCE_TYPE_ENTITY_INVENTORY_KEYS.get(reference_type)
    if entity_inventory_key is not None:
        entity_ids = inventory.get(entity_inventory_key, set())
        incompatible.extend(
            f"{reference_type} cannot target entity_id={value} with a different entity_type"
            for field_name, _inventory_key, value in recognized_ids
            if field_name == "entity_id" and value not in entity_ids
        )
    return incompatible


def _reference_label(reference: dict[str, Any]) -> str:
    parts: list[str] = []
    reference_type = _string_value(reference.get("reference_type"))
    if reference_type is not None:
        parts.append(f"reference_type={reference_type}")
    for field_name in REFERENCE_ID_FIELD_KEYS:
        value = _string_value(reference.get(field_name))
        if value is not None:
            parts.append(f"{field_name}={value}")
    label = _string_value(reference.get("label"))
    if label is not None:
        parts.append(f"label={label}")
    return ", ".join(parts) if parts else "unlabeled reference"


def _reference_inventory(context: dict[str, Any]) -> dict[str, set[str]]:
    inventory: dict[str, set[str]] = {
        key: set() for key in set(REFERENCE_ID_FIELD_KEYS.values())
    }
    _add_ids(inventory, "paper_ids", context.get("papers"), "paper_id")
    _add_ids(inventory, "source_ids", context.get("sources"), "source_id")
    _add_ids(inventory, "chunk_ids", context.get("chunks"), "chunk_id")
    _add_ids(
        inventory,
        "evidence_span_ids",
        context.get("evidence_spans"),
        "evidence_span_id",
    )
    _add_ids(inventory, "evidence_span_ids", context.get("evidence_spans"), "span_id")
    _add_ids(inventory, "evidence_span_ids", context.get("evidence_spans"), "evidence_id")
    _add_ids(inventory, "figure_ids", context.get("figures"), "figure_id")
    _add_ids(inventory, "table_ids", context.get("tables"), "table_id")
    _add_structured_entity_ids(inventory, context.get("structured_entities"))
    _add_result_evidence_ids(inventory, context.get("result_evidence"))
    intelligence_layers = context.get("intelligence_layers")
    if isinstance(intelligence_layers, dict):
        for layer_name in ("evidence_memory", "pattern_memory", "source_fact_memory"):
            _add_ids(
                inventory,
                "memory_record_ids",
                intelligence_layers.get(layer_name),
                "memory_record_id",
            )
        field_graph = intelligence_layers.get("field_graph")
        if isinstance(field_graph, dict):
            _add_ids(inventory, "graph_node_ids", field_graph.get("nodes"), "graph_node_id")
            _add_ids(inventory, "graph_edge_ids", field_graph.get("edges"), "graph_edge_id")
        study_brief = intelligence_layers.get("study_brief")
        if isinstance(study_brief, dict):
            study_brief_id = _string_value(study_brief.get("study_brief_id"))
            if study_brief_id is not None:
                inventory["study_brief_ids"].add(study_brief_id)
    return inventory


def _span_chunk_support_diagnostics(
    *,
    context: dict[str, Any],
    recommendations: list[dict[str, Any]],
    entailment_grader: Callable[..., Any] | None = None,
    count_unknown_entailment_checks: bool = True,
) -> dict[str, Any]:
    text_inventory = _span_chunk_support_text_inventory(context)
    checked_recommendations = 0
    weak_recommendations = 0
    entailment_checked_recommendations = 0
    entailment_unknown_recommendations = 0
    entailment_weak_recommendations = 0
    unavailable_references = 0
    missing_evidence: list[str] = []

    for recommendation in recommendations:
        support_items: list[tuple[str, str]] = []
        unavailable_labels: list[str] = []
        for reference in _recommendation_references(recommendation):
            for field_name in SPAN_CHUNK_SUPPORT_REFERENCE_FIELDS:
                value = _string_value(reference.get(field_name))
                if value is None:
                    continue
                support_text = _support_text_for_reference(
                    text_inventory=text_inventory,
                    field_name=field_name,
                    value=value,
                )
                if support_text is not None:
                    support_items.append((f"{field_name}={value}", support_text))
                elif _support_reference_exists(
                    text_inventory=text_inventory,
                    field_name=field_name,
                    value=value,
                ):
                    unavailable_labels.append(f"{field_name}={value}")

        if unavailable_labels:
            unavailable_references += len(unavailable_labels)
            missing_evidence.append(
                "Cited chunk/span evidence has no prompt-visible text for support "
                f"screening: {_recommendation_label(recommendation)}; "
                f"{', '.join(unavailable_labels)}"
            )
        if not support_items:
            continue

        checked_recommendations += 1
        recommendation_text = _recommendation_support_text(recommendation)
        recommendation_polarity = _comparator_polarity(
            recommendation_text,
            compact_conflict_policy="prose",
        )
        mismatched_comparator_labels: list[str] = []
        if recommendation_polarity is not None:
            recommendation_direction, recommendation_match = recommendation_polarity
            for label, support_text in support_items:
                support_polarity = _comparator_polarity(
                    support_text,
                    compact_conflict_policy="ambiguous",
                )
                if support_polarity is None:
                    continue
                support_direction, support_match = support_polarity
                if recommendation_direction == support_direction:
                    continue
                mismatched_comparator_labels.append(
                    f"{label} recommendation comparator {recommendation_match} "
                    f"contradicts cited comparator {support_match}"
                )
        if mismatched_comparator_labels:
            weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has chunk/span comparator polarity mismatch from "
                f"cited chunk/span evidence: {_recommendation_label(recommendation)}; "
                f"{', '.join(mismatched_comparator_labels)}"
            )
            continue

        negation_mismatch_labels: list[str] = []
        for label, support_text in support_items:
            negation_terms = _negation_mismatch_terms(
                recommendation_text,
                support_text,
            )
            if not negation_terms:
                continue
            negation_mismatch_labels.append(
                f"{label} negation terms {', '.join(sorted(negation_terms))}"
            )
        if negation_mismatch_labels:
            weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has chunk/span negation mismatch from cited "
                f"chunk/span evidence: {_recommendation_label(recommendation)}; "
                f"{', '.join(negation_mismatch_labels)}"
            )
            continue

        recommendation_tokens = _support_tokens(recommendation_text)
        weak_labels = [
            label
            for label, support_text in support_items
            if len(recommendation_tokens & _support_tokens(support_text))
            < MIN_SPAN_CHUNK_SUPPORT_TOKEN_OVERLAP
        ]
        if weak_labels:
            weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has weak lexical support from cited chunk/span evidence: "
                f"{_recommendation_label(recommendation)}; "
                f"{', '.join(weak_labels)}"
            )
            continue

        (
            entailment_labels,
            entailment_checked,
            entailment_unknown,
        ) = _entailment_mismatch_labels(
            entailment_grader=entailment_grader,
            recommendation=recommendation,
            recommendation_text=recommendation_text,
            support_items=support_items,
            evidence_family="chunk_span",
            count_unknown_checks=count_unknown_entailment_checks,
        )
        if entailment_checked:
            entailment_checked_recommendations += 1
        if entailment_unknown:
            entailment_unknown_recommendations += 1
        if entailment_labels:
            entailment_weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has semantic entailment mismatch from cited "
                f"chunk/span evidence: {_recommendation_label(recommendation)}; "
                f"{', '.join(entailment_labels)}"
            )

    return {
        "missing_evidence": missing_evidence,
        "report": {
            "span_chunk_support_checked_recommendation_count": checked_recommendations,
            "span_chunk_support_weak_recommendation_count": weak_recommendations,
            "span_chunk_support_unavailable_reference_count": unavailable_references,
        },
        "entailment_report": {
            "checked": entailment_checked_recommendations,
            "unknown": entailment_unknown_recommendations,
            "weak": entailment_weak_recommendations,
        },
    }


def _structured_evidence_support_diagnostics(
    *,
    context: dict[str, Any],
    recommendations: list[dict[str, Any]],
    entailment_grader: Callable[..., Any] | None = None,
    count_unknown_entailment_checks: bool = True,
) -> dict[str, Any]:
    text_inventory = _structured_evidence_support_text_inventory(context)
    checked_recommendations = 0
    weak_recommendations = 0
    entailment_checked_recommendations = 0
    entailment_unknown_recommendations = 0
    entailment_weak_recommendations = 0
    unavailable_references = 0
    missing_evidence: list[str] = []

    for recommendation in recommendations:
        support_items: list[tuple[str, str]] = []
        result_value_items: list[tuple[str, str, str]] = []
        result_comparator_items: list[tuple[str, str]] = []
        result_relation_items: list[tuple[str, str, str]] = []
        cited_relation_ids: dict[str, set[str]] = {
            field_name: set() for field_name in RESULT_RELATION_REFERENCE_FIELDS
        }
        unavailable_labels: list[str] = []
        for reference in _recommendation_references(recommendation):
            for field_name, value in _structured_support_reference_targets(reference):
                _add_cited_relation_id(
                    cited_relation_ids,
                    text_inventory=text_inventory,
                    field_name=field_name,
                    value=value,
                )
                if _structured_result_relation_reference_exists(
                    text_inventory=text_inventory,
                    field_name=field_name,
                    value=value,
                ):
                    result_relation_items.append(
                        (f"{field_name}={value}", field_name, value)
                    )
                support_text = _structured_support_text_for_reference(
                    text_inventory=text_inventory,
                    field_name=field_name,
                    value=value,
                )
                if support_text is not None:
                    support_items.append((f"{field_name}={value}", support_text))
                    if field_name in RESULT_VALUE_REFERENCE_FIELDS:
                        result_comparator_items.append(
                            (f"{field_name}={value}", support_text)
                        )
                    result_value_text = _structured_result_value_text_for_reference(
                        text_inventory=text_inventory,
                        field_name=field_name,
                        value=value,
                    )
                    if result_value_text is not None:
                        result_value_items.append(
                            (f"{field_name}={value}", result_value_text, support_text)
                        )
                elif _structured_support_reference_exists(
                    text_inventory=text_inventory,
                    field_name=field_name,
                    value=value,
                ):
                    unavailable_labels.append(f"{field_name}={value}")

        if unavailable_labels:
            unavailable_references += len(unavailable_labels)
            missing_evidence.append(
                "Cited structured evidence has no prompt-visible text for support "
                f"screening: {_recommendation_label(recommendation)}; "
                f"{', '.join(unavailable_labels)}"
            )
        has_relation_check = bool(result_relation_items) and any(
            cited_ids for cited_ids in cited_relation_ids.values()
        )
        if not support_items and not has_relation_check:
            continue

        checked_recommendations += 1
        recommendation_text = _recommendation_support_text(recommendation)
        mismatched_relation_labels = _result_relation_mismatch_labels(
            text_inventory=text_inventory,
            result_references=result_relation_items,
            cited_relation_ids=cited_relation_ids,
        )
        if mismatched_relation_labels:
            weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has result-row relation mismatch from cited structured "
                f"evidence: {_recommendation_label(recommendation)}; "
                f"{', '.join(mismatched_relation_labels)}"
            )
            continue
        if not support_items:
            continue

        mismatched_value_tokens: set[str] = set()
        mismatched_value_labels: list[str] = []
        for label, value_text, support_text in result_value_items:
            expected_value_tokens = _numeric_value_tokens(value_text)
            claimed_value_tokens = _result_value_claim_tokens(
                recommendation_text,
                result_value_text=value_text,
                result_support_text=support_text,
            )
            if (
                claimed_value_tokens
                and expected_value_tokens
                and not (claimed_value_tokens & expected_value_tokens)
            ):
                mismatched_value_tokens.update(claimed_value_tokens)
                mismatched_value_labels.append(f"{label} expected value {value_text}")
        if mismatched_value_labels:
            weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has result-row value mismatch from cited structured "
                f"evidence: {_recommendation_label(recommendation)}; "
                f"recommendation values "
                f"{', '.join(sorted(mismatched_value_tokens))}; "
                f"{', '.join(mismatched_value_labels)}"
            )
            continue

        recommendation_polarity = _comparator_polarity(
            recommendation_text,
            compact_conflict_policy="prose",
        )
        mismatched_comparator_labels: list[str] = []
        if recommendation_polarity is not None:
            recommendation_direction, recommendation_match = recommendation_polarity
            for label, support_text in result_comparator_items:
                support_polarity = _comparator_polarity(
                    support_text,
                    compact_conflict_policy="ambiguous",
                )
                if support_polarity is None:
                    continue
                support_direction, support_match = support_polarity
                if recommendation_direction == support_direction:
                    continue
                mismatched_comparator_labels.append(
                    f"{label} recommendation comparator {recommendation_match} "
                    f"contradicts cited comparator {support_match}"
                )
        if mismatched_comparator_labels:
            weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has result-row comparator polarity mismatch from "
                f"cited structured evidence: {_recommendation_label(recommendation)}; "
                f"{', '.join(mismatched_comparator_labels)}"
            )
            continue

        negation_mismatch_labels: list[str] = []
        for label, support_text in support_items:
            negation_terms = _negation_mismatch_terms(
                recommendation_text,
                support_text,
            )
            if not negation_terms:
                continue
            negation_mismatch_labels.append(
                f"{label} negation terms {', '.join(sorted(negation_terms))}"
            )
        if negation_mismatch_labels:
            weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has structured evidence negation mismatch from "
                f"cited structured evidence: {_recommendation_label(recommendation)}; "
                f"{', '.join(negation_mismatch_labels)}"
            )
            continue

        recommendation_tokens = _structured_support_tokens(recommendation_text)
        support_tokens: set[str] = set()
        for _label, support_text in support_items:
            support_tokens.update(_structured_support_tokens(support_text))
        if (
            len(recommendation_tokens & support_tokens)
            < MIN_STRUCTURED_EVIDENCE_SUPPORT_TOKEN_OVERLAP
        ):
            weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has weak lexical support from cited structured evidence: "
                f"{_recommendation_label(recommendation)}; "
                f"{', '.join(label for label, _support_text in support_items)}"
            )
            continue

        (
            entailment_labels,
            entailment_checked,
            entailment_unknown,
        ) = _entailment_mismatch_labels(
            entailment_grader=entailment_grader,
            recommendation=recommendation,
            recommendation_text=recommendation_text,
            support_items=support_items,
            evidence_family="structured_evidence",
            count_unknown_checks=count_unknown_entailment_checks,
        )
        if entailment_checked:
            entailment_checked_recommendations += 1
        if entailment_unknown:
            entailment_unknown_recommendations += 1
        if entailment_labels:
            entailment_weak_recommendations += 1
            missing_evidence.append(
                "Recommendation has semantic entailment mismatch from cited "
                f"structured evidence: {_recommendation_label(recommendation)}; "
                f"{', '.join(entailment_labels)}"
            )

    return {
        "missing_evidence": missing_evidence,
        "report": {
            "structured_evidence_support_checked_recommendation_count": (
                checked_recommendations
            ),
            "structured_evidence_support_weak_recommendation_count": weak_recommendations,
            "structured_evidence_support_unavailable_reference_count": (
                unavailable_references
            ),
        },
        "entailment_report": {
            "checked": entailment_checked_recommendations,
            "unknown": entailment_unknown_recommendations,
            "weak": entailment_weak_recommendations,
        },
    }


def _combined_entailment_report(
    *diagnostics: dict[str, Any],
) -> dict[str, int]:
    checked = 0
    unknown = 0
    weak = 0
    for diagnostic in diagnostics:
        report = diagnostic.get("entailment_report")
        if not isinstance(report, dict):
            continue
        checked += int(report.get("checked") or 0)
        unknown += int(report.get("unknown") or 0)
        weak += int(report.get("weak") or 0)
    return {
        "entailment_support_checked_recommendation_count": checked,
        "entailment_support_unknown_recommendation_count": unknown,
        "entailment_support_weak_recommendation_count": weak,
    }


def _entailment_mismatch_labels(
    *,
    entailment_grader: Callable[..., Any] | None,
    recommendation: dict[str, Any],
    recommendation_text: str,
    support_items: list[tuple[str, str]],
    evidence_family: str,
    count_unknown_checks: bool,
) -> tuple[list[str], bool, bool]:
    if entailment_grader is None:
        return [], False, False

    recommendation_label = _recommendation_label(recommendation)
    mismatch_labels: list[str] = []
    checked = False
    unknown = False
    for support_label, support_text in support_items:
        result = entailment_grader(
            recommendation_label=recommendation_label,
            support_label=support_label,
            recommendation_text=recommendation_text,
            support_text=support_text,
            evidence_family=evidence_family,
        )
        verdict = _entailment_verdict(result)
        if verdict in {"entailed", "not_entailed"}:
            checked = True
        elif count_unknown_checks:
            checked = True
            unknown = True
        if verdict != "not_entailed":
            continue
        reason = _entailment_reason(result)
        if reason:
            mismatch_labels.append(f"{support_label} verdict not_entailed: {reason}")
        else:
            mismatch_labels.append(f"{support_label} verdict not_entailed")
    return mismatch_labels, checked, unknown


def _active_entailment_grader(
    entailment_grader: Callable[..., Any] | None,
) -> Callable[..., Any]:
    if entailment_grader is None:
        return _default_entailment_grader

    def grade(**kwargs: Any) -> Any:
        custom_result = entailment_grader(**kwargs)
        if _entailment_verdict(custom_result) == "not_entailed":
            return custom_result
        default_result = _default_entailment_grader(**kwargs)
        if _entailment_verdict(default_result) == "not_entailed":
            return default_result
        return custom_result

    return grade


def _default_entailment_grader(
    *,
    recommendation_text: str,
    support_text: str,
    evidence_family: str,
    **_ignored: Any,
) -> dict[str, str]:
    if evidence_family not in {"chunk_span", "structured_evidence"}:
        return {"verdict": "unknown"}
    claim_family = _unsupported_scalar_overclaim_family(
        recommendation_text=recommendation_text,
        support_text=support_text,
    )
    if claim_family is None:
        causal_overclaim = _unsupported_causal_overclaim_reason(
            recommendation_text=recommendation_text,
            support_text=support_text,
        )
        if causal_overclaim is None:
            return {"verdict": "unknown"}
        return {
            "verdict": "not_entailed",
            "reason": causal_overclaim,
        }
    return {
        "verdict": "not_entailed",
        "reason": (
            "scalar metric/result evidence does not establish "
            f"{claim_family}"
        ),
    }


def _unsupported_causal_overclaim_reason(
    *,
    recommendation_text: str,
    support_text: str,
) -> str | None:
    support_tokens = set(_analysis_tokens(support_text))
    if not _has_explicit_causal_claim(recommendation_text):
        return None
    if not (support_tokens & ENTAILMENT_ASSOCIATIONAL_EVIDENCE_TOKENS):
        return None
    if _has_positive_causal_study_signal(support_text):
        return None
    return "associational evidence does not establish causal effect"


def _has_explicit_causal_claim(text: str) -> bool:
    for clause in NEGATION_CLAUSE_SPLIT_RE.split(text):
        tokens = list(_analysis_tokens(clause))
        for index, token in enumerate(tokens):
            if token not in ENTAILMENT_CAUSAL_CLAIM_TOKENS:
                continue
            following_tokens = ENTAILMENT_CAUSAL_CLAIM_FOLLOWING_TOKENS.get(token)
            if (
                following_tokens is not None
                and _next_token(tokens, index) not in following_tokens
            ):
                continue
            if _has_causal_claim_caution(tokens, index):
                continue
            return True
    return False


def _has_causal_claim_caution(tokens: list[str], index: int) -> bool:
    token = tokens[index]
    before_window = tokens[max(0, index - 8) : index]
    after_window = tokens[index + 1 : index + 8]
    before_tokens = set(before_window)
    after_tokens = set(after_window)
    if before_tokens & ENTAILMENT_CAUSAL_CLAIM_CAUTION_TOKENS:
        return True
    if _has_governing_causal_verb_pre_disclaimer(before_window):
        return True
    if _has_governing_causal_verb_post_disclaimer(after_window):
        return True
    if token in ENTAILMENT_CAUSAL_ADJECTIVE_TOKENS and {"rather", "than"} <= before_tokens:
        return True
    if token in ENTAILMENT_CAUSAL_ADJECTIVE_TOKENS:
        previous_token = tokens[index - 1] if index > 0 else None
        if previous_token in {"not", "noncausal"}:
            return True
        if after_tokens & ENTAILMENT_CAUSAL_POST_CAUTION_TOKENS:
            return True
    return False


def _has_governing_causal_verb_pre_disclaimer(tokens: list[str]) -> bool:
    for sequence in ENTAILMENT_CAUSAL_VERB_NEGATED_SUPPORT_SEQUENCES:
        for sequence_index in _token_sequence_start_indexes(tokens, sequence):
            governed_tokens = tokens[sequence_index + len(sequence) :]
            if _is_causal_disclaimer_governed_span(governed_tokens):
                return True
    return False


def _has_governing_causal_verb_post_disclaimer(tokens: list[str]) -> bool:
    for sequence in ENTAILMENT_CAUSAL_VERB_POST_DISCLAIMER_SEQUENCES:
        for sequence_index in _token_sequence_start_indexes(tokens, sequence):
            governed_tokens = tokens[:sequence_index]
            if _is_causal_disclaimer_governed_span(governed_tokens):
                return True
    return False


def _is_causal_disclaimer_governed_span(tokens: list[str]) -> bool:
    if len(tokens) > ENTAILMENT_CAUSAL_DISCLAIMER_GOVERNED_TOKEN_LIMIT:
        return False
    return not bool(
        set(tokens)
        & (
            ENTAILMENT_CAUSAL_DISCLAIMER_CONNECTOR_TOKENS
            | ENTAILMENT_CAUSAL_DISCLAIMER_TARGET_SWITCH_TOKENS
        )
    )


def _token_sequence_start_indexes(
    tokens: list[str], sequence: tuple[str, ...]
) -> list[int]:
    sequence_length = len(sequence)
    if not sequence_length or len(tokens) < sequence_length:
        return []
    return [
        index
        for index in range(len(tokens) - sequence_length + 1)
        if tuple(tokens[index : index + sequence_length]) == sequence
    ]


def _has_positive_causal_study_signal(text: str) -> bool:
    for clause in NEGATION_CLAUSE_SPLIT_RE.split(text):
        tokens = list(_analysis_tokens(clause))
        if not (set(tokens) & ENTAILMENT_CAUSAL_STUDY_SIGNAL_TOKENS):
            continue
        if _has_nonpositive_causal_study_context(tokens):
            continue
        return True
    return False


def _next_token(tokens: list[str], index: int) -> str | None:
    next_index = index + 1
    return tokens[next_index] if next_index < len(tokens) else None


def _has_nonpositive_causal_study_context(tokens: list[str]) -> bool:
    return bool(
        set(tokens)
        & (
            NEGATION_CUE_TOKENS
            | ENTAILMENT_CAUSAL_STUDY_NONPOSITIVE_TOKENS
            | {"not", "without"}
        )
    )


def _unsupported_scalar_overclaim_family(
    *,
    recommendation_text: str,
    support_text: str,
) -> str | None:
    recommendation_tokens = set(_analysis_tokens(recommendation_text))
    support_tokens = set(_analysis_tokens(support_text))
    if not (recommendation_tokens & ENTAILMENT_OVERCLAIM_CUE_TOKENS):
        return None
    if not _has_scalar_result_signal(support_text, support_tokens=support_tokens):
        return None
    for family, family_tokens in ENTAILMENT_CLAIM_FAMILY_TOKENS.items():
        if not (recommendation_tokens & family_tokens):
            continue
        if support_tokens & family_tokens:
            continue
        return family
    return None


def _has_scalar_result_signal(
    support_text: str,
    *,
    support_tokens: set[str],
) -> bool:
    return bool(
        _numeric_value_tokens(support_text)
        or support_tokens & ENTAILMENT_SCALAR_RESULT_TOKENS
    )


def _entailment_verdict(result: Any) -> str | None:
    if isinstance(result, str):
        return _entailment_verdict_key(result)
    if isinstance(result, dict):
        return _entailment_verdict_key(result.get("verdict"))
    return None


def _entailment_verdict_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    verdict = "_".join(value.strip().casefold().replace("-", "_").split())
    return verdict if verdict else None


def _entailment_reason(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    return _string_value(result.get("reason"))


def _source_fact_memory_support_diagnostics(
    *,
    context: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    text_inventory = _source_fact_memory_support_text_inventory(context)
    checked_recommendations = 0
    weak_recommendations = 0
    unavailable_references = 0
    missing_evidence: list[str] = []

    for recommendation in recommendations:
        status = _recommendation_support_status(recommendation)
        supporting_layers = set(_recommendation_supporting_layers(recommendation))
        if status != "user_provided" and "source_fact_memory" not in supporting_layers:
            continue

        support_items: list[tuple[str, str]] = []
        unavailable_labels: list[str] = []
        for reference in _recommendation_references(recommendation):
            for field_name in SOURCE_FACT_SUPPORT_REFERENCE_FIELDS:
                value = _string_value(reference.get(field_name))
                if value is None:
                    continue
                support_text = _source_fact_support_text_for_reference(
                    text_inventory=text_inventory,
                    field_name=field_name,
                    value=value,
                )
                if support_text is not None:
                    support_items.append((f"{field_name}={value}", support_text))
                elif _source_fact_support_reference_exists(
                    text_inventory=text_inventory,
                    field_name=field_name,
                    value=value,
                ):
                    unavailable_labels.append(f"{field_name}={value}")

        if unavailable_labels:
            unavailable_references += len(unavailable_labels)
            missing_evidence.append(
                "Cited source-fact memory has no prompt-visible text for support "
                f"screening: {_recommendation_label(recommendation)}; "
                f"{', '.join(unavailable_labels)}"
            )
        if not support_items:
            continue

        checked_recommendations += 1
        recommendation_tokens = _support_tokens(_recommendation_support_text(recommendation))
        weak_labels = [
            label
            for label, support_text in support_items
            if len(recommendation_tokens & _support_tokens(support_text))
            < MIN_SOURCE_FACT_SUPPORT_TOKEN_OVERLAP
        ]
        if not weak_labels:
            continue

        weak_recommendations += 1
        missing_evidence.append(
            "Recommendation has weak lexical support from cited source-fact memory: "
            f"{_recommendation_label(recommendation)}; "
            f"{', '.join(weak_labels)}"
        )

    return {
        "missing_evidence": missing_evidence,
        "report": {
            "source_fact_support_checked_recommendation_count": checked_recommendations,
            "source_fact_support_weak_recommendation_count": weak_recommendations,
            "source_fact_support_unavailable_reference_count": unavailable_references,
        },
    }


def _task_quality_diagnostics(*, output_payload: dict[str, Any]) -> dict[str, Any]:
    artifact_type = _task_artifact_type(output_payload)
    required_sections = (
        TASK_REQUIRED_SECTIONS_BY_ARTIFACT_TYPE.get(artifact_type, ())
        if artifact_type is not None
        else ()
    )
    missing_sections = [
        section
        for section in required_sections
        if not _section_has_content(output_payload.get(section))
    ]
    missing_evidence: list[str] = []
    if artifact_type is not None and missing_sections:
        missing_evidence.append(
            "Task artifact is missing required section(s) for "
            f"{artifact_type}: {', '.join(missing_sections)}"
        )

    return {
        "missing_evidence": missing_evidence,
        "report": {
            "task_quality_artifact_type": artifact_type,
            "task_quality_checked_artifact_count": 1 if required_sections else 0,
            "task_quality_required_section_count": len(required_sections),
            "task_quality_missing_required_section_count": len(missing_sections),
            "task_quality_required_sections": list(required_sections),
            "task_quality_missing_required_sections": missing_sections,
        },
    }


def _task_evidence_role_diagnostics(
    *,
    context: dict[str, Any],
    output_payload: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_type = _task_artifact_type(output_payload)
    role_requirements = (
        TASK_EVIDENCE_ROLE_REQUIREMENTS_BY_ARTIFACT_TYPE.get(artifact_type, {})
        if artifact_type is not None
        else {}
    )
    available_roles = _context_evidence_roles(context)
    available_groups = sorted(
        group_name
        for group_name, group_roles in role_requirements.items()
        if available_roles & set(group_roles)
    )
    reference_roles = _referenced_evidence_roles(
        context=context,
        output_payload=output_payload,
        recommendations=recommendations,
    )
    satisfied_groups = sorted(
        group_name
        for group_name, group_roles in role_requirements.items()
        if group_name in available_groups and reference_roles & set(group_roles)
    )
    missing_groups = [
        group_name
        for group_name in available_groups
        if group_name not in satisfied_groups
    ]
    missing_evidence: list[str] = []
    if artifact_type is not None and missing_groups:
        missing_evidence.append(
            "Task artifact is missing required evidence role citation(s) for "
            f"{artifact_type}: {', '.join(missing_groups)}"
        )
    return {
        "missing_evidence": missing_evidence,
        "report": {
            "task_evidence_role_checked_artifact_count": 1 if available_groups else 0,
            "task_evidence_role_available_groups": available_groups,
            "task_evidence_role_satisfied_groups": satisfied_groups,
            "task_evidence_role_missing_groups": missing_groups,
            "task_evidence_reference_roles": sorted(reference_roles),
            "task_evidence_available_roles": sorted(available_roles),
        },
    }


def _task_artifact_type(output_payload: dict[str, Any]) -> str | None:
    artifact_type = _task_quality_key(output_payload.get("artifact_type"))
    if artifact_type in TASK_REQUIRED_SECTIONS_BY_ARTIFACT_TYPE:
        return artifact_type
    skill_id = _task_quality_key(output_payload.get("skill_id"))
    if skill_id is None:
        return None
    return TASK_ARTIFACT_TYPES_BY_SKILL_ID.get(skill_id)


def _task_quality_key(value: Any) -> str | None:
    value = _string_value(value)
    if value is None:
        return None
    return "_".join(value.casefold().replace("-", "_").split())


def _section_has_content(value: Any) -> bool:
    return task_section_has_meaningful_content(value)


def _context_evidence_roles(context: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for item in context.get("chunks", []):
        if not isinstance(item, dict):
            continue
        roles.update(
            _text_evidence_roles(_first_text(item, ("text", "content", "section_text")))
        )
    for item in context.get("evidence_spans", []):
        if not isinstance(item, dict):
            continue
        roles.update(
            _text_evidence_roles(_first_text(item, ("quote_text", "text", "content")))
        )
    for item in context.get("structured_entities", []):
        if not isinstance(item, dict):
            continue
        roles.update(_structured_entity_roles(item))
    for item in context.get("result_evidence", []):
        if not isinstance(item, dict):
            continue
        roles.update(_result_evidence_roles(item))
    return roles


def _referenced_evidence_roles(
    *,
    context: dict[str, Any],
    output_payload: dict[str, Any],
    recommendations: list[dict[str, Any]],
) -> set[str]:
    inventory = _reference_inventory(context)
    roles: set[str] = set()
    for _location, reference in _output_references(
        output_payload=output_payload,
        recommendations=recommendations,
    ):
        status, _reason, _has_incompatible_type = _reference_status(
            reference,
            inventory=inventory,
        )
        if status != "valid":
            continue
        roles.update(_evidence_roles_for_reference(reference, context=context))
    return roles


def _evidence_roles_for_reference(
    reference: dict[str, Any],
    *,
    context: dict[str, Any],
) -> set[str]:
    roles: set[str] = set()
    chunk_id = _string_value(reference.get("chunk_id"))
    if chunk_id is not None:
        roles.update(_chunk_evidence_roles_by_id(context, chunk_id))
    span_id = _string_value(
        reference.get("evidence_span_id")
        or reference.get("span_id")
        or reference.get("evidence_id")
    )
    if span_id is not None:
        roles.update(_span_evidence_roles_by_id(context, span_id))
    if _string_value(reference.get("dataset_id")) is not None:
        roles.add("dataset")
    if _string_value(reference.get("method_id")) is not None:
        roles.add("method")
    if _string_value(reference.get("metric_id")) is not None:
        roles.add("metric")
    if _string_value(reference.get("result_id") or reference.get("result_row_id")) is not None:
        roles.add("result")
    entity_id = _string_value(reference.get("entity_id"))
    if entity_id is not None:
        roles.update(_structured_entity_roles_by_id(context, entity_id))
    result_id = _string_value(reference.get("result_id") or reference.get("result_row_id"))
    if result_id is not None:
        roles.update(_result_evidence_roles_by_id(context, result_id))
    return roles


def _structured_entity_roles_by_id(context: dict[str, Any], entity_id: str) -> set[str]:
    for item in context.get("structured_entities", []):
        if isinstance(item, dict) and _string_value(item.get("entity_id")) == entity_id:
            return _structured_entity_roles(item)
    return set()


def _result_evidence_roles_by_id(context: dict[str, Any], result_id: str) -> set[str]:
    for item in context.get("result_evidence", []):
        if not isinstance(item, dict):
            continue
        item_id = _string_value(item.get("result_row_id") or item.get("result_id"))
        if item_id == result_id:
            return _result_evidence_roles(item)
    return set()


def _chunk_evidence_roles_by_id(context: dict[str, Any], chunk_id: str) -> set[str]:
    for item in context.get("chunks", []):
        if not isinstance(item, dict):
            continue
        if _string_value(item.get("chunk_id")) == chunk_id:
            return _text_evidence_roles(
                _first_text(item, ("text", "content", "section_text"))
            )
    return set()


def _span_evidence_roles_by_id(context: dict[str, Any], span_id: str) -> set[str]:
    for item in context.get("evidence_spans", []):
        if not isinstance(item, dict):
            continue
        ids = {
            value
            for value in (
                _string_value(item.get("evidence_span_id")),
                _string_value(item.get("span_id")),
                _string_value(item.get("evidence_id")),
            )
            if value is not None
        }
        if span_id in ids:
            return _text_evidence_roles(
                _first_text(item, ("quote_text", "text", "content"))
            )
    return set()


def _text_evidence_roles(text: str | None) -> set[str]:
    if text is None:
        return set()
    analysis_tokens = _analysis_tokens(text)
    tokens = set(analysis_tokens)
    roles = {
        role
        for role, keywords in TEXT_EVIDENCE_ROLE_KEYWORDS.items()
        if tokens & keywords
    }
    if _has_control_role_context(analysis_tokens):
        roles.add("control")
    return roles


def _has_control_role_context(tokens: list[str]) -> bool:
    for index, token in enumerate(tokens):
        if token not in {"control", "controlled", "controls"}:
            continue
        window_start = max(0, index - 3)
        window_end = min(len(tokens), index + 4)
        nearby_tokens = set(tokens[window_start:index] + tokens[index + 1 : window_end])
        if nearby_tokens & CONTROL_ROLE_CONTEXT_TOKENS:
            return True
    return False


def _structured_entity_roles(item: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    entity_type = _task_quality_key(item.get("entity_type"))
    if entity_type in {"dataset", "method", "metric"}:
        roles.add(entity_type)
    return roles


def _result_evidence_roles(item: dict[str, Any]) -> set[str]:
    roles = {"result"}
    for field_name, role_name in (
        ("dataset_id", "dataset"),
        ("dataset_name", "dataset"),
        ("method_id", "method"),
        ("method_name", "method"),
        ("metric_id", "metric"),
        ("metric_name", "metric"),
    ):
        if _string_value(item.get(field_name)) is not None:
            roles.add(role_name)
    return roles


def _span_chunk_support_text_inventory(context: dict[str, Any]) -> dict[str, Any]:
    chunk_ids: set[str] = set()
    chunk_texts: dict[str, str] = {}
    for chunk in context.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        chunk_id = _string_value(chunk.get("chunk_id"))
        if chunk_id is None:
            continue
        chunk_ids.add(chunk_id)
        text = _first_text(chunk, ("text", "content", "section_text"))
        if text is not None:
            chunk_texts[chunk_id] = text

    evidence_span_ids: set[str] = set()
    evidence_span_texts: dict[str, str] = {}
    for span in context.get("evidence_spans", []):
        if not isinstance(span, dict):
            continue
        ids = [
            span_id
            for span_id in (
                _string_value(span.get("evidence_span_id")),
                _string_value(span.get("span_id")),
                _string_value(span.get("evidence_id")),
            )
            if span_id is not None
        ]
        if not ids:
            continue
        text = _first_text(span, ("quote_text", "text", "content"))
        for span_id in ids:
            evidence_span_ids.add(span_id)
            if text is not None:
                evidence_span_texts[span_id] = text

    return {
        "chunk_ids": chunk_ids,
        "chunk_texts": chunk_texts,
        "evidence_span_ids": evidence_span_ids,
        "evidence_span_texts": evidence_span_texts,
    }


def _structured_evidence_support_text_inventory(context: dict[str, Any]) -> dict[str, Any]:
    reference_ids: dict[str, set[str]] = {
        field_name: set() for field_name in STRUCTURED_SUPPORT_REFERENCE_FIELDS
    }
    reference_texts: dict[tuple[str, str], str] = {}
    result_value_texts: dict[tuple[str, str], str] = {}
    result_relation_ids: dict[tuple[str, str, str], str] = {}
    entity_relation_fields: dict[str, str] = {}
    entity_texts: dict[str, str] = {}

    for entity in context.get("structured_entities", []):
        if not isinstance(entity, dict):
            continue
        entity_id = _string_value(entity.get("entity_id"))
        if entity_id is None:
            continue
        entity_text = _structured_entity_support_text(entity)
        entity_texts[entity_id] = entity_text or ""
        reference_ids["entity_id"].add(entity_id)
        _append_structured_support_text(
            reference_texts,
            field_name="entity_id",
            value=entity_id,
            support_text=entity_text,
        )
        entity_type = _task_quality_key(entity.get("entity_type"))
        if entity_type in {"dataset", "method", "metric"}:
            field_name = f"{entity_type}_id"
            entity_relation_fields[entity_id] = field_name
            reference_ids[field_name].add(entity_id)
            _append_structured_support_text(
                reference_texts,
                field_name=field_name,
                value=entity_id,
                support_text=entity_text,
            )

    for result in context.get("result_evidence", []):
        if not isinstance(result, dict):
            continue
        result_text = _result_evidence_support_text(
            result,
            entity_texts=entity_texts,
        )
        result_id = _string_value(result.get("result_row_id") or result.get("result_id"))
        if result_id is not None:
            for field_name in ("result_id", "result_row_id"):
                reference_ids[field_name].add(result_id)
                _append_structured_support_text(
                    reference_texts,
                    field_name=field_name,
                    value=result_id,
                    support_text=result_text,
                )
                _append_structured_support_text(
                    result_value_texts,
                    field_name=field_name,
                    value=result_id,
                    support_text=_result_evidence_value_text(result),
                )
                _append_result_relation_ids(
                    result_relation_ids,
                    result=result,
                    field_name=field_name,
                    value=result_id,
                )
        for field_name in ("dataset_id", "method_id", "metric_id"):
            value = _string_value(result.get(field_name))
            if value is None:
                continue
            reference_ids[field_name].add(value)
            _append_structured_support_text(
                reference_texts,
                field_name=field_name,
                value=value,
                support_text=result_text,
            )

    return {
        "reference_ids": reference_ids,
        "reference_texts": reference_texts,
        "result_value_texts": result_value_texts,
        "result_relation_ids": result_relation_ids,
        "entity_relation_fields": entity_relation_fields,
    }


def _source_fact_memory_support_text_inventory(context: dict[str, Any]) -> dict[str, Any]:
    memory_record_ids: set[str] = set()
    memory_record_texts: dict[str, str] = {}
    intelligence_layers = context.get("intelligence_layers")
    if not isinstance(intelligence_layers, dict):
        return {
            "memory_record_ids": memory_record_ids,
            "memory_record_texts": memory_record_texts,
        }

    for memory_record in intelligence_layers.get("source_fact_memory", []):
        if not isinstance(memory_record, dict):
            continue
        memory_record_id = _string_value(memory_record.get("memory_record_id"))
        if memory_record_id is None:
            continue
        memory_record_ids.add(memory_record_id)
        support_text = _source_fact_memory_support_text(memory_record)
        if support_text is not None:
            memory_record_texts[memory_record_id] = support_text

    return {
        "memory_record_ids": memory_record_ids,
        "memory_record_texts": memory_record_texts,
    }


def _source_fact_memory_support_text(memory_record: dict[str, Any]) -> str | None:
    payload = memory_record.get("payload")
    payload_fact_text = (
        _string_value(payload.get("fact_text")) if isinstance(payload, dict) else None
    )
    return _joined_text_parts(
        (
            payload_fact_text,
            _string_value(memory_record.get("fact_text")),
            _string_value(memory_record.get("summary")),
            _string_value(memory_record.get("title")),
        )
    )


def _structured_entity_support_text(entity: dict[str, Any]) -> str | None:
    return _joined_text_parts(
        _string_value(entity.get(field_name))
        for field_name in (
            "display_name",
            "name",
            "normalized_name",
            "label",
            "description",
            "summary",
        )
    )


def _result_evidence_support_text(
    result: dict[str, Any],
    *,
    entity_texts: dict[str, str],
) -> str | None:
    direct_parts = [
        _string_value(result.get(field_name))
        for field_name in (
            "dataset",
            "dataset_name",
            "method",
            "method_name",
            "metric",
            "metric_name",
            "split_name",
            "value_text",
            "comparator_text",
            "finding",
            "summary",
            "notes",
        )
    ]
    joined_entity_parts = [
        entity_texts.get(entity_id, "")
        for entity_id in (
            _string_value(result.get("dataset_id")),
            _string_value(result.get("method_id")),
            _string_value(result.get("metric_id")),
        )
        if entity_id is not None
    ]
    return _joined_text_parts([*direct_parts, *joined_entity_parts])


def _result_evidence_value_text(result: dict[str, Any]) -> str | None:
    for field_name in ("value_numeric", "numeric_value", "value"):
        scalar_value = _scalar_numeric_value_text(result.get(field_name))
        if scalar_value is not None:
            return scalar_value
    value_text = result.get("value_text")
    if isinstance(value_text, str) and value_text.strip():
        return value_text.strip()
    return None


def _scalar_numeric_value_text(value: Any) -> str | None:
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str) and NUMERIC_VALUE_RE.fullmatch(value.strip()):
        return value.strip()
    return None


def _append_structured_support_text(
    reference_texts: dict[tuple[str, str], str],
    *,
    field_name: str,
    value: str,
    support_text: str | None,
) -> None:
    if support_text is None:
        return
    key = (field_name, value)
    existing_text = reference_texts.get(key)
    if existing_text is None:
        reference_texts[key] = support_text
    elif support_text not in existing_text:
        reference_texts[key] = f"{existing_text} {support_text}"


def _append_result_relation_ids(
    result_relation_ids: dict[tuple[str, str, str], str],
    *,
    result: dict[str, Any],
    field_name: str,
    value: str,
) -> None:
    for relation_field in RESULT_RELATION_REFERENCE_FIELDS:
        relation_id = _string_value(result.get(relation_field))
        if relation_id is not None:
            result_relation_ids[(field_name, value, relation_field)] = relation_id


def _structured_support_reference_targets(
    reference: dict[str, Any],
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for field_name in STRUCTURED_SUPPORT_REFERENCE_FIELDS:
        value = _string_value(reference.get(field_name))
        if value is not None:
            targets.append((field_name, value))
    return targets


def _add_cited_relation_id(
    cited_relation_ids: dict[str, set[str]],
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> None:
    if field_name in RESULT_RELATION_REFERENCE_FIELDS:
        cited_relation_ids[field_name].add(value)
        return
    if field_name != "entity_id":
        return
    relation_field = _structured_entity_relation_field_for_reference(
        text_inventory=text_inventory,
        value=value,
    )
    if relation_field is not None:
        cited_relation_ids[relation_field].add(value)


def _structured_entity_relation_field_for_reference(
    *,
    text_inventory: dict[str, Any],
    value: str,
) -> str | None:
    entity_relation_fields = text_inventory.get("entity_relation_fields")
    if not isinstance(entity_relation_fields, dict):
        return None
    relation_field = entity_relation_fields.get(value)
    return relation_field if relation_field in RESULT_RELATION_REFERENCE_FIELDS else None


def _result_relation_mismatch_labels(
    *,
    text_inventory: dict[str, Any],
    result_references: list[tuple[str, str, str]],
    cited_relation_ids: dict[str, set[str]],
) -> list[str]:
    expected_relation_ids: dict[str, set[str]] = {
        field_name: set() for field_name in RESULT_RELATION_REFERENCE_FIELDS
    }
    expected_relation_labels: dict[str, list[str]] = {
        field_name: [] for field_name in RESULT_RELATION_REFERENCE_FIELDS
    }
    mismatches: list[str] = []
    for result_label, field_name, value in result_references:
        for relation_field in RESULT_RELATION_REFERENCE_FIELDS:
            expected_relation_id = _structured_result_relation_id_for_reference(
                text_inventory=text_inventory,
                field_name=field_name,
                value=value,
                relation_field=relation_field,
            )
            if expected_relation_id is not None:
                expected_relation_ids[relation_field].add(expected_relation_id)
                expected_relation_labels[relation_field].append(
                    f"{result_label} expected {relation_field}={expected_relation_id}"
                )

    for relation_field in RESULT_RELATION_REFERENCE_FIELDS:
        expected_ids = expected_relation_ids[relation_field]
        cited_ids = cited_relation_ids.get(relation_field, set())
        if not expected_ids or not cited_ids:
            continue
        unexpected_ids = cited_ids - expected_ids
        if not unexpected_ids:
            continue
        mismatches.append(
            f"{', '.join(expected_relation_labels[relation_field])}; "
            f"cited {relation_field}={', '.join(sorted(unexpected_ids))}"
        )
    return mismatches


def _structured_result_relation_reference_exists(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> bool:
    return any(
        _structured_result_relation_id_for_reference(
            text_inventory=text_inventory,
            field_name=field_name,
            value=value,
            relation_field=relation_field,
        )
        is not None
        for relation_field in RESULT_RELATION_REFERENCE_FIELDS
    )


def _structured_support_text_for_reference(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> str | None:
    return text_inventory["reference_texts"].get((field_name, value))


def _structured_support_reference_exists(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> bool:
    return value in text_inventory["reference_ids"].get(field_name, set())


def _structured_result_relation_id_for_reference(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
    relation_field: str,
) -> str | None:
    if field_name not in RESULT_VALUE_REFERENCE_FIELDS:
        return None
    if relation_field not in RESULT_RELATION_REFERENCE_FIELDS:
        return None
    result_relation_ids = text_inventory.get("result_relation_ids")
    if not isinstance(result_relation_ids, dict):
        return None
    relation_id = result_relation_ids.get((field_name, value, relation_field))
    return relation_id if isinstance(relation_id, str) else None


def _source_fact_support_text_for_reference(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> str | None:
    if field_name != "memory_record_id":
        return None
    memory_record_texts = text_inventory.get("memory_record_texts")
    if not isinstance(memory_record_texts, dict):
        return None
    support_text = memory_record_texts.get(value)
    return support_text if isinstance(support_text, str) else None


def _source_fact_support_reference_exists(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> bool:
    if field_name != "memory_record_id":
        return False
    memory_record_ids = text_inventory.get("memory_record_ids")
    return isinstance(memory_record_ids, set) and value in memory_record_ids


def _structured_result_value_text_for_reference(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> str | None:
    if field_name not in RESULT_VALUE_REFERENCE_FIELDS:
        return None
    return text_inventory["result_value_texts"].get((field_name, value))


def _support_text_for_reference(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> str | None:
    if field_name == "chunk_id":
        return text_inventory["chunk_texts"].get(value)
    return text_inventory["evidence_span_texts"].get(value)


def _support_reference_exists(
    *,
    text_inventory: dict[str, Any],
    field_name: str,
    value: str,
) -> bool:
    if field_name == "chunk_id":
        return value in text_inventory["chunk_ids"]
    return value in text_inventory["evidence_span_ids"]


def _recommendation_support_text(recommendation: dict[str, Any]) -> str:
    return " ".join(
        value.strip()
        for field_name in RECOMMENDATION_SUPPORT_TEXT_FIELDS
        if isinstance(value := recommendation.get(field_name), str) and value.strip()
    )


def _structured_support_tokens(text: str) -> set[str]:
    return _support_tokens(text, allow_short_alphanumeric=True)


def _numeric_value_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_value in NUMERIC_VALUE_RE.findall(text.casefold()):
        tokens.update(_numeric_tokens_from_raw_value(raw_value))
    return tokens


def _result_value_claim_tokens(
    recommendation_text: str,
    *,
    result_value_text: str,
    result_support_text: str,
) -> set[str]:
    numeric_positions = _numeric_value_token_positions(recommendation_text)
    if not numeric_positions:
        return set()

    anchor_positions = _result_value_anchor_positions(
        recommendation_text,
        result_value_text=result_value_text,
        result_support_text=result_support_text,
    )
    if not anchor_positions:
        return set()

    return {
        value_token
        for value_index, value_token in numeric_positions
        if any(
            abs(value_index - anchor_index) <= RESULT_VALUE_CLAIM_TOKEN_WINDOW
            for anchor_index in anchor_positions
        )
    }


def _comparator_polarity(
    text: str,
    *,
    compact_conflict_policy: Literal["ambiguous", "prose"] = "ambiguous",
) -> tuple[str, str] | None:
    text_lower = text.casefold()
    direction_matches = list(RESULT_COMPARATOR_DIRECTION_RE.finditer(text_lower))
    compact_polarity: tuple[str, str] | None = None
    compact_is_ambiguous = False
    masked_text = text_lower
    if direction_matches:
        compact_directions = {
            "positive" if match.group("direction") == "higher" else "negative"
            for match in direction_matches
        }
        if len(compact_directions) == 1:
            compact_polarity = (compact_directions.pop(), direction_matches[0].group(0))
        else:
            compact_is_ambiguous = True
        masked_chars = list(text_lower)
        for direction_match in direction_matches:
            start, end = direction_match.span()
            masked_chars[start:end] = " " * (end - start)
        masked_text = "".join(masked_chars)

    prose_polarity = _prose_comparator_polarity(masked_text)
    if compact_is_ambiguous and compact_conflict_policy == "ambiguous":
        return None
    if compact_polarity is None:
        return prose_polarity
    if prose_polarity is None:
        return compact_polarity
    if prose_polarity[0] == compact_polarity[0]:
        return compact_polarity
    if compact_conflict_policy == "prose":
        return prose_polarity
    return None


def _prose_comparator_polarity(text_lower: str) -> tuple[str, str] | None:
    negative_matches = list(RESULT_COMPARATOR_NEGATIVE_RE.finditer(text_lower))
    masked_text = text_lower
    if negative_matches:
        masked_chars = list(text_lower)
        for negative_match in negative_matches:
            start, end = negative_match.span()
            masked_chars[start:end] = " " * (end - start)
        masked_text = "".join(masked_chars)
    positive_match = RESULT_COMPARATOR_POSITIVE_RE.search(masked_text)
    if negative_matches and positive_match is not None:
        return None
    if negative_matches:
        return "negative", negative_matches[0].group(0)
    if positive_match is not None:
        return "positive", positive_match.group(0)
    return None


def _negation_mismatch_terms(
    recommendation_text: str,
    support_text: str,
) -> set[str]:
    return _directional_negation_mismatch_terms(
        negated_text=support_text,
        positive_text=recommendation_text,
    ) | _directional_negation_mismatch_terms(
        negated_text=recommendation_text,
        positive_text=support_text,
    )


def _directional_negation_mismatch_terms(
    *,
    negated_text: str,
    positive_text: str,
) -> set[str]:
    positive_terms = _positive_evidence_terms(positive_text)
    if not positive_terms:
        return set()

    negated_clauses = _negation_clauses(negated_text)
    positive_negated_terms = _negated_evidence_terms(positive_text)
    mismatch_terms: set[str] = set()
    for clause in negated_clauses:
        clause_negated_terms = _negated_evidence_terms(clause)
        aligned_negated_terms = clause_negated_terms & positive_negated_terms
        candidate_terms = (clause_negated_terms - positive_negated_terms) & positive_terms
        for term in candidate_terms:
            if aligned_negated_terms and not _has_independent_positive_clause(
                positive_text,
                term=term,
                aligned_negated_terms=aligned_negated_terms,
            ):
                continue
            positive_anchor_terms = positive_terms - {term}
            if _has_matching_positive_clause(
                clauses=negated_clauses,
                term=term,
                positive_anchor_terms=positive_anchor_terms,
            ):
                continue
            mismatch_terms.add(term)
    return mismatch_terms


def _has_independent_positive_clause(
    text: str,
    *,
    term: str,
    aligned_negated_terms: set[str],
) -> bool:
    for clause in _negation_clauses(text):
        if term not in _positive_evidence_terms(clause):
            continue
        if not (_negated_evidence_terms(clause) & aligned_negated_terms):
            return True
    return False


def _has_matching_positive_clause(
    *,
    clauses: list[str],
    term: str,
    positive_anchor_terms: set[str],
) -> bool:
    for clause in clauses:
        clause_positive_terms = _positive_evidence_terms(clause)
        if term not in clause_positive_terms:
            continue
        if not positive_anchor_terms or clause_positive_terms & positive_anchor_terms:
            return True
    return False


def _positive_evidence_terms(text: str) -> set[str]:
    terms = {
        token
        for token in _analysis_tokens(text)
        if _is_negation_evidence_term(token)
    }
    return terms - _negated_evidence_terms(text)


def _negation_clauses(text: str) -> list[str]:
    clauses = [
        comma_clause.strip()
        for clause in NEGATION_CLAUSE_SPLIT_RE.split(text)
        for comma_clause in NEGATION_COMMA_CLAUSE_ACTION_RE.split(clause)
        if clause.strip()
        if comma_clause.strip()
    ]
    return clauses or [text]


def _negated_evidence_terms(text: str) -> set[str]:
    negated_terms: set[str] = set()
    positive_terms: set[str] = set()
    for clause in _negation_clauses(text):
        clause_negated_terms, clause_positive_terms = _negation_term_sets_for_clause(
            clause
        )
        negated_terms.update(clause_negated_terms)
        positive_terms.update(clause_positive_terms)
    return negated_terms - positive_terms


def _negation_term_sets_for_clause(text: str) -> tuple[set[str], set[str]]:
    tokens = _analysis_tokens(text)
    negated_positions: set[int] = set()
    for index, token in enumerate(tokens):
        if token in NEGATION_CUE_TOKENS:
            if _is_negated_cue(tokens, index):
                continue
            start_index = _negation_scope_start_after_cue(tokens, index)
            negated_positions.update(
                _negated_term_positions_in_scope(tokens, start_index=start_index)
            )
            if token in NEGATION_POSTPOSITIVE_CUE_TOKENS:
                negated_positions.update(
                    _negated_term_positions_before(tokens, before_index=index)
                )
        elif token == "not" and index + 1 < len(tokens):
            next_token = tokens[index + 1]
            if next_token not in NEGATION_ACTION_TOKENS:
                continue
            negated_positions.update(
                _negated_term_positions_before(tokens, before_index=index)
            )
            negated_positions.update(
                _negated_term_positions_in_scope(
                    tokens,
                    start_index=index + 2,
                    fallback_terms=False,
                )
            )

    negated_terms = {
        tokens[index]
        for index in negated_positions
        if _is_negation_evidence_term(tokens[index])
    }
    positive_terms = {
        token
        for index, token in enumerate(tokens)
        if index not in negated_positions and _is_negation_evidence_term(token)
    }
    comma_list_terms = _comma_negated_list_terms(text)
    negated_terms.update(comma_list_terms)
    positive_terms -= comma_list_terms
    return negated_terms, positive_terms


def _is_negated_cue(tokens: list[str], cue_index: int) -> bool:
    return cue_index > 0 and tokens[cue_index - 1] == "not"


def _comma_negated_list_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in NEGATION_COMMA_LIST_RE.finditer(text.casefold()):
        terms.update(
            token
            for token in _analysis_tokens(match.group("items"))
            if _is_negation_evidence_term(token)
        )
    return terms


def _negation_scope_start_after_cue(tokens: list[str], cue_index: int) -> int:
    start_index = cue_index + 1
    if start_index < len(tokens) and tokens[start_index] in NEGATION_DENIAL_HEAD_TOKENS:
        start_index += 1
        if (
            start_index < len(tokens)
            and tokens[start_index] in NEGATION_DENIAL_LINK_TOKENS
        ):
            start_index += 1
    return start_index


def _negated_term_positions_in_scope(
    tokens: list[str],
    *,
    start_index: int,
    fallback_terms: bool = True,
) -> set[int]:
    scope_indices: list[int] = []
    stop_index = min(len(tokens), start_index + NEGATION_SCOPE_TOKEN_WINDOW)
    for index in range(start_index, stop_index):
        token = tokens[index]
        if token in NEGATION_SCOPE_BOUNDARY_TOKENS and token != "for":
            break
        if token in NEGATION_ACTION_TOKENS:
            continue
        scope_indices.append(index)

    focused_positions = _focused_negation_head_positions(tokens, scope_indices)
    if focused_positions:
        return focused_positions
    if not fallback_terms:
        return set()
    positions: set[int] = set()
    for index in scope_indices:
        token = tokens[index]
        if token in NEGATION_SCOPE_BOUNDARY_TOKENS:
            break
        if _is_negation_evidence_term(token):
            positions.add(index)
    return positions


def _focused_negation_head_positions(
    tokens: list[str],
    scope_indices: list[int],
) -> set[int]:
    positions: set[int] = set()
    head_positions = [
        position
        for position, token_index in enumerate(scope_indices)
        if tokens[token_index] in NEGATION_EVIDENCE_HEAD_TOKENS
    ]
    for head_position in head_positions:
        expect_term = True
        crossed_coordinator = False
        for position in range(head_position - 1, -1, -1):
            token_index = scope_indices[position]
            token = tokens[token_index]
            if expect_term and _is_negation_evidence_term(token):
                if crossed_coordinator and _has_previous_evidence_term(
                    tokens,
                    scope_indices=scope_indices,
                    position=position,
                ):
                    break
                positions.add(token_index)
                expect_term = False
                crossed_coordinator = False
                continue
            if not expect_term and token in NEGATION_COORDINATOR_TOKENS:
                expect_term = True
                crossed_coordinator = True
                continue
            break
    return positions


def _has_previous_evidence_term(
    tokens: list[str],
    *,
    scope_indices: list[int],
    position: int,
) -> bool:
    if position <= 0:
        return False
    previous_token = tokens[scope_indices[position - 1]]
    return _is_negation_evidence_term(previous_token)


def _negated_term_positions_before(
    tokens: list[str],
    *,
    before_index: int,
) -> set[int]:
    scope_indices: list[int] = []
    start_index = max(0, before_index - NEGATION_SCOPE_TOKEN_WINDOW)
    for index in range(start_index, before_index):
        token = tokens[index]
        if token in NEGATION_SCOPE_BOUNDARY_TOKENS and token != "for":
            scope_indices.clear()
            continue
        if token in NEGATION_ACTION_TOKENS:
            continue
        scope_indices.append(index)
    return _focused_negation_head_positions(tokens, scope_indices)


def _is_negation_evidence_term(token: str) -> bool:
    if token in NEGATION_TERM_STOPWORDS:
        return False
    return _is_meaningful_support_token(token, allow_short_alphanumeric=True)


def _result_value_anchor_positions(
    recommendation_text: str,
    *,
    result_value_text: str,
    result_support_text: str,
) -> set[int]:
    specific_anchor_tokens = _result_value_anchor_tokens(result_value_text)
    if not specific_anchor_tokens:
        specific_anchor_tokens = _result_value_anchor_tokens(result_support_text)
    anchor_tokens = specific_anchor_tokens | RESULT_VALUE_GENERIC_ANCHOR_TOKENS
    return {
        index
        for index, token in enumerate(_analysis_tokens(recommendation_text))
        if token in anchor_tokens
    }


def _result_value_anchor_tokens(text: str) -> set[str]:
    numeric_tokens = _numeric_value_tokens(text)
    return {
        token
        for token in _analysis_tokens(text)
        if token not in numeric_tokens
        and _is_meaningful_support_token(token, allow_short_alphanumeric=True)
    }


def _numeric_value_token_positions(text: str) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    for index, raw_token in enumerate(_raw_analysis_tokens(text)):
        for value_token in _numeric_tokens_from_raw_value(raw_token):
            positions.append((index, value_token))
    return positions


def _analysis_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in _raw_analysis_tokens(text):
        value_tokens = _numeric_tokens_from_raw_value(raw_token)
        if value_tokens:
            tokens.append(sorted(value_tokens)[0])
        else:
            tokens.append(raw_token.casefold())
    return tokens


def _raw_analysis_tokens(text: str) -> list[str]:
    return ANALYSIS_TOKEN_RE.findall(text.casefold())


def _numeric_tokens_from_raw_value(raw_value: str) -> set[str]:
    if NUMERIC_VALUE_RE.fullmatch(raw_value) is None:
        return set()
    tokens: set[str] = set()
    is_percent = raw_value.endswith("%")
    value = raw_value.removesuffix("%")
    normalized = _normalized_decimal_token(value)
    if normalized is not None:
        tokens.add(normalized)
    if is_percent and normalized is not None:
        try:
            tokens.add(_format_decimal_token(Decimal(normalized) / Decimal(100)))
        except InvalidOperation:
            pass
    return tokens


def _normalized_decimal_token(value: str) -> str | None:
    try:
        return _format_decimal_token(Decimal(value))
    except InvalidOperation:
        return None


def _format_decimal_token(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _support_tokens(
    text: str,
    *,
    allow_short_alphanumeric: bool = False,
) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_+-]*", text.casefold())
        if _is_meaningful_support_token(
            token,
            allow_short_alphanumeric=allow_short_alphanumeric,
        )
    }


def _is_meaningful_support_token(
    token: str,
    *,
    allow_short_alphanumeric: bool = False,
) -> bool:
    if token in SUPPORT_TOKEN_STOPWORDS:
        return False
    if len(token) > 2:
        return True
    if not allow_short_alphanumeric:
        return False
    return (
        len(token) == 2
        and any(char.isalpha() for char in token)
        and any(char.isdigit() for char in token)
    )


def _first_text(item: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field_name in fields:
        value = _string_value(item.get(field_name))
        if value is not None:
            return value
    return None


def _joined_text_parts(parts: Any) -> str | None:
    text_parts = [
        part.strip()
        for part in parts
        if isinstance(part, str) and part.strip()
    ]
    return " ".join(text_parts) if text_parts else None


def _add_ids(
    inventory: dict[str, set[str]],
    inventory_key: str,
    items: Any,
    id_field: str,
) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        value = _string_value(item.get(id_field))
        if value is not None:
            inventory[inventory_key].add(value)


def _add_structured_entity_ids(
    inventory: dict[str, set[str]],
    items: Any,
) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        entity_id = _string_value(item.get("entity_id"))
        if entity_id is None:
            continue
        inventory["entity_ids"].add(entity_id)
        entity_type = _task_quality_key(item.get("entity_type"))
        if entity_type == "dataset":
            inventory["dataset_ids"].add(entity_id)
        elif entity_type == "method":
            inventory["method_ids"].add(entity_id)
        elif entity_type == "metric":
            inventory["metric_ids"].add(entity_id)


def _add_result_evidence_ids(
    inventory: dict[str, set[str]],
    items: Any,
) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        result_id = _string_value(item.get("result_row_id") or item.get("result_id"))
        if result_id is not None:
            inventory["result_row_ids"].add(result_id)
        for item_field, inventory_key in (
            ("dataset_id", "dataset_ids"),
            ("method_id", "method_ids"),
            ("metric_id", "metric_ids"),
        ):
            value = _string_value(item.get(item_field))
            if value is not None:
                inventory[inventory_key].add(value)


def _string_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _reference_type_key(value: Any) -> str | None:
    value = _string_value(value)
    if value is None:
        return None
    return "_".join(value.casefold().replace("-", "_").split())


def _available_layers(context: dict[str, Any]) -> set[str]:
    layers: set[str] = set()
    if any(isinstance(paper, dict) for paper in context.get("papers", [])):
        layers.add("source_library")
    if any(isinstance(source, dict) for source in context.get("sources", [])):
        layers.add("source_library")

    intelligence_layers = context.get("intelligence_layers")
    if not isinstance(intelligence_layers, dict):
        return layers
    if any(isinstance(item, dict) for item in intelligence_layers.get("evidence_memory", [])):
        layers.add("evidence_memory")
    if any(isinstance(item, dict) for item in intelligence_layers.get("pattern_memory", [])):
        layers.add("pattern_memory")
    if any(isinstance(item, dict) for item in intelligence_layers.get("source_fact_memory", [])):
        layers.add("source_fact_memory")
    field_graph = intelligence_layers.get("field_graph")
    if isinstance(field_graph, dict) and (
        any(isinstance(item, dict) for item in field_graph.get("nodes", []))
        or any(isinstance(item, dict) for item in field_graph.get("edges", []))
    ):
        layers.add("field_graph")
    if isinstance(intelligence_layers.get("study_brief"), dict):
        layers.add("study_brief")
    return layers


def _recommendation_diagnostics(
    *,
    recommendations: list[dict[str, Any]],
    available_layers: set[str],
) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    missing_evidence: list[str] = []
    unsupported_claims: list[str] = []
    missing_support_status = 0
    missing_supporting_layers = 0
    invalid_support_status = 0
    missing_evidence_references = 0
    speculative_as_supported = 0
    unavailable_layer_recommendations = 0
    invalid_layer_recommendations = 0
    source_fact_supported_status = 0

    for recommendation in recommendations:
        label = _recommendation_label(recommendation)
        status = _recommendation_support_status(recommendation)
        if status is None:
            missing_support_status += 1
            missing_evidence.append(f"Recommendation is missing support_status: {label}")
        elif status not in SUPPORT_STATUS_VALUES:
            invalid_support_status += 1
            missing_evidence.append(
                f"Recommendation has invalid support_status '{status}': {label}"
            )
        else:
            status_counts[status] += 1

        layers = _recommendation_supporting_layers(recommendation)
        if status in SUPPORTING_LAYER_REQUIRED_STATUSES and not layers:
            missing_supporting_layers += 1
            missing_evidence.append(f"Recommendation is missing supporting_layers: {label}")
        invalid_layers = [layer for layer in layers if layer not in SUPPORTING_LAYER_VALUES]
        if invalid_layers:
            invalid_layer_recommendations += 1
            missing_evidence.append(
                "Recommendation cites invalid supporting layer(s) "
                f"{', '.join(sorted(set(invalid_layers)))}: {label}"
            )
        valid_layers = [layer for layer in layers if layer in SUPPORTING_LAYER_VALUES]
        layer_counts.update(valid_layers)
        unavailable_layers = sorted(
            {layer for layer in valid_layers if layer not in available_layers}
        )
        if unavailable_layers:
            unavailable_layer_recommendations += 1
            unsupported_claims.append(
                "Recommendation cites unavailable supporting layer(s) "
                f"{', '.join(unavailable_layers)}: {label}"
            )

        if _needs_recommendation_references(
            status,
            valid_layers,
        ) and not _recommendation_references(recommendation):
            missing_evidence_references += 1
            missing_evidence.append(f"Recommendation is missing evidence references: {label}")
        if status == "supported" and "source_fact_memory" in valid_layers:
            source_fact_supported_status += 1
            unsupported_claims.append(
                "Recommendation treats user-provided source_fact_memory as supported "
                f"paper evidence; use user_provided for source facts or mixed with "
                f"paper evidence: {label}"
            )
        if status == "supported" and _is_speculative_recommendation(recommendation):
            speculative_as_supported += 1
            unsupported_claims.append(
                f"Recommendation is speculative but marked supported: {label}"
            )

    return {
        "missing_evidence": missing_evidence,
        "unsupported_claims": unsupported_claims,
        "report": {
            "recommendation_count": len(recommendations),
            "recommendations_missing_support_status": missing_support_status,
            "recommendations_missing_supporting_layers": missing_supporting_layers,
            "recommendations_invalid_support_status": invalid_support_status,
            "recommendations_missing_evidence_references": missing_evidence_references,
            "recommendations_speculative_as_supported": speculative_as_supported,
            "recommendations_with_unavailable_layers": unavailable_layer_recommendations,
            "recommendations_with_invalid_layers": invalid_layer_recommendations,
            "recommendations_with_source_fact_supported_status": (
                source_fact_supported_status
            ),
            "recommendation_support_status_counts": {
                status: status_counts[status]
                for status in SUPPORT_STATUS_VALUES
                if status_counts[status]
            },
            "recommendation_supporting_layer_counts": {
                layer: layer_counts[layer]
                for layer in SUPPORTING_LAYER_VALUES
                if layer_counts[layer]
            },
            "available_supporting_layers": sorted(available_layers),
        },
    }


def _recommendation_supporting_layers(recommendation: dict[str, Any]) -> list[str]:
    layers = recommendation.get("supporting_layers")
    if isinstance(layers, str):
        layer = layers.strip()
        return [layer] if layer else []
    if not isinstance(layers, list):
        return []
    return [layer.strip() for layer in layers if isinstance(layer, str) and layer.strip()]


def _needs_recommendation_references(status: str | None, layers: list[str]) -> bool:
    if status in EVIDENCE_BACKED_STATUSES:
        return True
    return bool(set(layers) & REFERENCE_BACKED_LAYERS)


def _is_speculative_recommendation(recommendation: dict[str, Any]) -> bool:
    if recommendation.get("is_speculative") is True or recommendation.get("speculative") is True:
        return True
    for key in ("claim_type", "evidence_type"):
        value = recommendation.get(key)
        if isinstance(value, str) and value.casefold() == "speculative":
            return True
    text = " ".join(
        str(recommendation.get(key) or "")
        for key in ("title", "detail", "claim", "rationale")
    ).casefold()
    return "speculative" in text


def _recommendation_label(recommendation: dict[str, Any]) -> str:
    for key in ("title", "claim", "summary", "detail"):
        value = recommendation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:160]
    return "Untitled recommendation"
