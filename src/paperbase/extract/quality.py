"""Extraction quality summaries for collection readiness."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    CollectionPaper,
    Dataset,
    EngineeringTrick,
    EvidenceSpan,
    ExtractionRun,
    Finding,
    GlossaryTerm,
    Limitation,
    Method,
    Metric,
    Paper,
    ResearchDesignElement,
    ResultRow,
)
from paperbase.extract.freshness import (
    FRESHNESS_DIAGNOSTICS_KEY,
    build_extraction_freshness,
)

ENTITY_CATEGORIES = (
    "datasets",
    "methods",
    "metrics",
    "result_rows",
    "findings",
    "limitations",
    "glossary_terms",
    "engineering_tricks",
    "research_design_elements",
)

ENTITY_MODELS = {
    "datasets": Dataset,
    "methods": Method,
    "metrics": Metric,
    "result_rows": ResultRow,
    "findings": Finding,
    "limitations": Limitation,
    "glossary_terms": GlossaryTerm,
    "engineering_tricks": EngineeringTrick,
    "research_design_elements": ResearchDesignElement,
}
EVIDENCE_SPAN_ANCHORS_DIAGNOSTICS_KEY = "evidence_span_anchors"
MAX_RECOVERY_ACTION_UNRESOLVED_SAMPLES = 20
MAX_RECOVERY_ACTION_QUOTE_PREVIEW_CHARS = 160


@dataclass(frozen=True, slots=True)
class PaperExtractionQuality:
    paper_id: str
    paper_title: str
    structured_entity_count: int
    entity_counts: dict[str, int]
    evidence_span_count: int
    anchored_evidence_span_count: int
    unresolved_evidence_span_count: int
    evidence_span_anchor_diagnostics: dict[str, Any]
    latest_completed_extraction_run_id: str | None
    extraction_profile_id: str | None
    model_name: str | None
    prompt_version: str | None
    schema_version: str | None
    freshness_status: str
    stale_reasons: list[str]
    missing_structured_evidence: list[str]


@dataclass(frozen=True, slots=True)
class CollectionExtractionQuality:
    paper_count: int
    papers_with_completed_extraction_count: int
    fresh_paper_count: int
    stale_paper_count: int
    missing_extraction_paper_count: int
    total_structured_entity_count: int
    total_evidence_span_count: int
    anchored_evidence_span_count: int
    unresolved_evidence_span_count: int
    missing_structured_evidence: list[str]
    papers: list[PaperExtractionQuality]


@dataclass(frozen=True, slots=True)
class ExtractionRecoveryAction:
    action_id: str
    action_type: str
    can_queue_job: bool
    priority: int
    label: str
    description: str
    paper_count: int
    paper_ids: list[str]
    truncated: bool
    stale_reasons: list[str]
    missing_structured_evidence: list[str]
    unresolved_evidence_span_count: int
    unresolved_evidence_span_samples: list[dict[str, Any]]


def build_collection_extraction_quality(
    session: Session,
    *,
    collection_id: str,
    collection_extraction_profile_id: str | None = None,
    collection_schema_payload: dict[str, Any] | None = None,
) -> CollectionExtractionQuality:
    """Summarize extraction completeness and freshness for one collection."""

    paper_rows = session.execute(
        select(CollectionPaper.paper_id, Paper.canonical_title)
        .join(Paper, Paper.id == CollectionPaper.paper_id)
        .where(CollectionPaper.collection_id == collection_id)
        .order_by(
            CollectionPaper.position.asc().nullslast(),
            CollectionPaper.created_at.asc(),
            CollectionPaper.id.asc(),
        )
    ).all()
    paper_ids = [paper_id for paper_id, _title in paper_rows]
    entity_counts_by_category = {
        category: _count_by_paper(session, model, paper_ids)
        for category, model in ENTITY_MODELS.items()
    }
    evidence_span_counts = _count_by_paper(session, EvidenceSpan, paper_ids)
    anchored_evidence_span_counts = _anchored_evidence_span_counts(session, paper_ids)
    latest_completed_runs = _latest_completed_extraction_runs(session, paper_ids)

    paper_summaries: list[PaperExtractionQuality] = []
    for paper_id, paper_title in paper_rows:
        entity_counts = {
            category: int(entity_counts_by_category[category].get(paper_id, 0) or 0)
            for category in ENTITY_CATEGORIES
        }
        structured_entity_count = sum(entity_counts.values())
        evidence_span_count = int(evidence_span_counts.get(paper_id, 0) or 0)
        anchored_evidence_span_count = int(anchored_evidence_span_counts.get(paper_id, 0) or 0)
        unresolved_evidence_span_count = max(
            0,
            evidence_span_count - anchored_evidence_span_count,
        )
        latest_run = latest_completed_runs.get(paper_id)
        freshness_status, stale_reasons = _freshness_state(
            session,
            paper_id=paper_id,
            latest_run=latest_run,
            collection_extraction_profile_id=collection_extraction_profile_id,
            collection_schema_payload=collection_schema_payload,
        )
        paper_summaries.append(
            PaperExtractionQuality(
                paper_id=paper_id,
                paper_title=paper_title,
                structured_entity_count=structured_entity_count,
                entity_counts=entity_counts,
                evidence_span_count=evidence_span_count,
                anchored_evidence_span_count=anchored_evidence_span_count,
                unresolved_evidence_span_count=unresolved_evidence_span_count,
                evidence_span_anchor_diagnostics=_evidence_span_anchor_diagnostics(
                    latest_run
                ),
                latest_completed_extraction_run_id=latest_run.id if latest_run else None,
                extraction_profile_id=latest_run.extraction_profile_id if latest_run else None,
                model_name=latest_run.model_name if latest_run else None,
                prompt_version=latest_run.prompt_version if latest_run else None,
                schema_version=latest_run.schema_version if latest_run else None,
                freshness_status=freshness_status,
                stale_reasons=stale_reasons,
                missing_structured_evidence=[
                    category for category, count in entity_counts.items() if count == 0
                ],
            )
        )

    collection_entity_counts = {
        category: sum(summary.entity_counts[category] for summary in paper_summaries)
        for category in ENTITY_CATEGORIES
    }
    return CollectionExtractionQuality(
        paper_count=len(paper_summaries),
        papers_with_completed_extraction_count=sum(
            1
            for summary in paper_summaries
            if summary.latest_completed_extraction_run_id is not None
        ),
        fresh_paper_count=sum(
            1 for summary in paper_summaries if summary.freshness_status == "fresh"
        ),
        stale_paper_count=sum(
            1 for summary in paper_summaries if summary.freshness_status == "stale"
        ),
        missing_extraction_paper_count=sum(
            1
            for summary in paper_summaries
            if summary.freshness_status == "missing_extraction"
        ),
        total_structured_entity_count=sum(
            summary.structured_entity_count for summary in paper_summaries
        ),
        total_evidence_span_count=sum(summary.evidence_span_count for summary in paper_summaries),
        anchored_evidence_span_count=sum(
            summary.anchored_evidence_span_count for summary in paper_summaries
        ),
        unresolved_evidence_span_count=sum(
            summary.unresolved_evidence_span_count for summary in paper_summaries
        ),
        missing_structured_evidence=[
            category for category, count in collection_entity_counts.items() if count == 0
        ],
        papers=paper_summaries,
    )


def build_extraction_recovery_actions(
    quality: CollectionExtractionQuality,
    *,
    parsed_paper_ids: set[str],
    paper_id_limit: int = 20,
) -> list[ExtractionRecoveryAction]:
    """Build bounded parse/extract/review actions from extraction quality."""

    actions: list[ExtractionRecoveryAction] = []
    unparsed_papers = [
        paper_quality
        for paper_quality in quality.papers
        if paper_quality.paper_id not in parsed_paper_ids
    ]
    missing_extraction_papers = [
        paper_quality
        for paper_quality in quality.papers
        if paper_quality.paper_id in parsed_paper_ids
        and paper_quality.freshness_status == "missing_extraction"
    ]
    stale_extraction_papers = [
        paper_quality
        for paper_quality in quality.papers
        if paper_quality.paper_id in parsed_paper_ids
        and paper_quality.freshness_status == "stale"
    ]
    unresolved_span_papers = [
        paper_quality
        for paper_quality in quality.papers
        if paper_quality.unresolved_evidence_span_count > 0
    ]

    if unparsed_papers:
        actions.append(
            _recovery_action(
                action_id="parse_missing_text",
                action_type="parse",
                can_queue_job=True,
                priority=10,
                label=(
                    f"Parse {len(unparsed_papers)} "
                    f"{_plural('paper', len(unparsed_papers))} before extraction"
                ),
                description=(
                    "Full-text sections are missing, so extraction cannot run reliably."
                ),
                paper_qualities=unparsed_papers,
                paper_id_limit=paper_id_limit,
                include_stale_reasons=False,
                include_missing_structured_evidence=False,
            )
        )
    if missing_extraction_papers:
        actions.append(
            _recovery_action(
                action_id="extract_missing_structured_evidence",
                action_type="extract",
                can_queue_job=True,
                priority=20,
                label=(
                    "Extract structured evidence for "
                    f"{len(missing_extraction_papers)} parsed "
                    f"{_plural('paper', len(missing_extraction_papers))}"
                ),
                description=(
                    "Parsed papers are missing completed structured extraction runs."
                ),
                paper_qualities=missing_extraction_papers,
                paper_id_limit=paper_id_limit,
            )
        )
    if stale_extraction_papers:
        actions.append(
            _recovery_action(
                action_id="reextract_stale_structured_evidence",
                action_type="extract",
                can_queue_job=True,
                priority=30,
                label=(
                    f"Re-extract {len(stale_extraction_papers)} stale "
                    f"{_plural('paper', len(stale_extraction_papers))}"
                ),
                description=(
                    "Completed extraction is stale relative to the active profile, "
                    "schema, parsed text, or source file."
                ),
                paper_qualities=stale_extraction_papers,
                paper_id_limit=paper_id_limit,
            )
        )
    unresolved_evidence_span_count = sum(
        paper_quality.unresolved_evidence_span_count
        for paper_quality in unresolved_span_papers
    )
    if unresolved_evidence_span_count:
        actions.append(
            _recovery_action(
                action_id="review_unresolved_evidence_spans",
                action_type="review_evidence",
                can_queue_job=False,
                priority=40,
                label=(
                    f"Review {unresolved_evidence_span_count} unresolved evidence "
                    f"{_plural('span', unresolved_evidence_span_count)}"
                ),
                description=(
                    "Some evidence spans lack section or chunk anchors, so citation "
                    "grounding is weaker."
                ),
                paper_qualities=unresolved_span_papers,
                paper_id_limit=paper_id_limit,
                unresolved_evidence_span_count=unresolved_evidence_span_count,
                include_stale_reasons=False,
                include_missing_structured_evidence=False,
                include_unresolved_samples=True,
            )
        )
    return actions


def _recovery_action(
    *,
    action_id: str,
    action_type: str,
    can_queue_job: bool,
    priority: int,
    label: str,
    description: str,
    paper_qualities: list[PaperExtractionQuality],
    paper_id_limit: int,
    unresolved_evidence_span_count: int = 0,
    include_stale_reasons: bool = True,
    include_missing_structured_evidence: bool = True,
    include_unresolved_samples: bool = False,
) -> ExtractionRecoveryAction:
    paper_ids = [paper_quality.paper_id for paper_quality in paper_qualities]
    return ExtractionRecoveryAction(
        action_id=action_id,
        action_type=action_type,
        can_queue_job=can_queue_job,
        priority=priority,
        label=label,
        description=description,
        paper_count=len(paper_qualities),
        paper_ids=paper_ids[:paper_id_limit],
        truncated=len(paper_ids) > paper_id_limit,
        stale_reasons=_ordered_unique(
            reason
            for paper_quality in paper_qualities
            for reason in paper_quality.stale_reasons
        )
        if include_stale_reasons
        else [],
        missing_structured_evidence=_ordered_unique(
            category
            for paper_quality in paper_qualities
            for category in paper_quality.missing_structured_evidence
        )
        if include_missing_structured_evidence
        else [],
        unresolved_evidence_span_count=unresolved_evidence_span_count,
        unresolved_evidence_span_samples=_unresolved_evidence_span_samples(
            paper_qualities,
            limit=MAX_RECOVERY_ACTION_UNRESOLVED_SAMPLES,
        )
        if include_unresolved_samples
        else [],
    )


def _unresolved_evidence_span_samples(
    paper_qualities: list[PaperExtractionQuality],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for paper_quality in paper_qualities:
        diagnostics = paper_quality.evidence_span_anchor_diagnostics
        raw_samples = diagnostics.get("unresolved_samples")
        if not isinstance(raw_samples, list):
            continue
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict):
                continue
            samples.append(
                {
                    "paper_id": paper_quality.paper_id,
                    "paper_title": paper_quality.paper_title,
                    "mode": _optional_string(raw_sample.get("mode")),
                    "reason": _optional_string(raw_sample.get("reason")),
                    "target_type": _optional_string(raw_sample.get("target_type")),
                    "target_id": _optional_string(raw_sample.get("target_id")),
                    "page_number": _optional_int(raw_sample.get("page_number")),
                    "quote_preview": _optional_string(
                        raw_sample.get("quote_preview"),
                        max_chars=MAX_RECOVERY_ACTION_QUOTE_PREVIEW_CHARS,
                    ),
                }
            )
            if len(samples) >= limit:
                return samples
    return samples


def _optional_string(value: object, *, max_chars: int | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    if max_chars is None or len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _ordered_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _plural(singular: str, count: int) -> str:
    return singular if count == 1 else f"{singular}s"


def _count_by_paper(session: Session, model: type[Any], paper_ids: list[str]) -> dict[str, int]:
    if not paper_ids:
        return {}
    return {
        paper_id: int(count or 0)
        for paper_id, count in session.execute(
            select(model.paper_id, func.count(model.id))
            .where(model.paper_id.in_(paper_ids))
            .group_by(model.paper_id)
        ).all()
    }


def _anchored_evidence_span_counts(session: Session, paper_ids: list[str]) -> dict[str, int]:
    if not paper_ids:
        return {}
    return {
        paper_id: int(count or 0)
        for paper_id, count in session.execute(
            select(EvidenceSpan.paper_id, func.count(EvidenceSpan.id))
            .where(
                EvidenceSpan.paper_id.in_(paper_ids),
                or_(
                    EvidenceSpan.section_id.is_not(None),
                    EvidenceSpan.chunk_id.is_not(None),
                ),
            )
            .group_by(EvidenceSpan.paper_id)
        ).all()
    }


def _latest_completed_extraction_runs(
    session: Session,
    paper_ids: list[str],
) -> dict[str, ExtractionRun]:
    if not paper_ids:
        return {}
    runs = session.execute(
        select(ExtractionRun)
        .where(
            ExtractionRun.paper_id.in_(paper_ids),
            ExtractionRun.status == "completed",
        )
        .order_by(
            ExtractionRun.paper_id.asc(),
            ExtractionRun.created_at.desc(),
            ExtractionRun.id.desc(),
        )
    ).scalars()
    latest_by_paper_id: dict[str, ExtractionRun] = {}
    for run in runs:
        latest_by_paper_id.setdefault(run.paper_id, run)
    return latest_by_paper_id


def _evidence_span_anchor_diagnostics(
    latest_run: ExtractionRun | None,
) -> dict[str, Any]:
    if latest_run is None:
        return {}
    diagnostics = dict(latest_run.diagnostics_json or {}).get(
        EVIDENCE_SPAN_ANCHORS_DIAGNOSTICS_KEY
    )
    return dict(diagnostics) if isinstance(diagnostics, dict) else {}


def _freshness_state(
    session: Session,
    *,
    paper_id: str,
    latest_run: ExtractionRun | None,
    collection_extraction_profile_id: str | None,
    collection_schema_payload: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    if latest_run is None:
        return "missing_extraction", ["missing_extraction"]

    stale_reasons: list[str] = []
    if (
        collection_extraction_profile_id is not None
        and latest_run.extraction_profile_id != collection_extraction_profile_id
    ):
        stale_reasons.append("profile_mismatch")

    freshness = dict(latest_run.diagnostics_json or {}).get(FRESHNESS_DIAGNOSTICS_KEY)
    if not isinstance(freshness, dict):
        stale_reasons.append("missing_freshness_metadata")
        return "stale", stale_reasons

    if collection_extraction_profile_id is not None and collection_schema_payload is None:
        stale_reasons.append("profile_schema_missing")
        return "stale", stale_reasons

    try:
        current_freshness = build_extraction_freshness(
            session,
            paper_id=paper_id,
            schema_payload=collection_schema_payload or {},
        )
    except ValueError:
        stale_reasons.append("missing_parse")
        return "stale", stale_reasons

    if (
        collection_schema_payload is not None
        and freshness.get("schema_payload_digest") != current_freshness.schema_payload_digest
    ):
        stale_reasons.append("schema_payload_mismatch")
    if freshness.get("parsed_content_digest") != current_freshness.parsed_content_digest:
        stale_reasons.append("parsed_content_mismatch")
    if freshness.get("source_content_digest") != current_freshness.source_content_digest:
        stale_reasons.append("source_content_mismatch")

    return ("stale", stale_reasons) if stale_reasons else ("fresh", [])
