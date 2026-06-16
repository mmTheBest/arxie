"""Pipeline for persisting structured extraction outputs into Paperbase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import (
    Chunk,
    Dataset,
    EngineeringTrick,
    EvidenceSpan,
    ExtractionRun,
    Finding,
    GlossaryTerm,
    Limitation,
    Method,
    Metric,
    ResearchDesignElement,
    ResultRow,
    Section,
)
from paperbase.extract.contracts import StructuredExtractionBundle
from paperbase.extract.freshness import (
    ExtractionFreshness,
    build_extraction_freshness,
    load_parsed_paper_text,
    merge_freshness_diagnostics,
)
from paperbase.schemas.extraction import (
    DatasetExtraction,
    EvidenceSpanPayload,
    MethodExtraction,
    MetricExtraction,
)

T = TypeVar("T")
MAX_CROSS_CHUNK_ANCHOR_WINDOW = 2
MAX_EVIDENCE_ANCHOR_UNRESOLVED_SAMPLES = 20
MAX_EVIDENCE_ANCHOR_QUOTE_PREVIEW_CHARS = 160
ANCHOR_MODE_CHUNK_EXACT_QUOTE = "chunk_exact_quote"
ANCHOR_MODE_CHUNK_ADJACENT_QUOTE = "chunk_adjacent_quote"
ANCHOR_MODE_SECTION_QUOTE_ONLY = "section_quote_only"
ANCHOR_MODE_SECTION_PAGE_ONLY = "section_page_only"
ANCHOR_MODE_UNRESOLVED = "unresolved"
ANCHOR_MODE_ORDER = (
    ANCHOR_MODE_CHUNK_EXACT_QUOTE,
    ANCHOR_MODE_CHUNK_ADJACENT_QUOTE,
    ANCHOR_MODE_SECTION_QUOTE_ONLY,
    ANCHOR_MODE_SECTION_PAGE_ONLY,
    ANCHOR_MODE_UNRESOLVED,
)
ANCHOR_REASON_MISSING_QUOTE_AND_PAGE = "missing_quote_and_page"
ANCHOR_REASON_MISSING_QUOTE_TEXT = "missing_quote_text"
ANCHOR_REASON_NO_PARSED_SECTIONS = "no_parsed_sections"
ANCHOR_REASON_AMBIGUOUS_SECTION_QUOTE = "ambiguous_section_quote"
ANCHOR_REASON_QUOTE_NOT_FOUND_WITHOUT_PAGE = "quote_not_found_without_page"
ANCHOR_REASON_REPORTED_PAGE_OUT_OF_RANGE = "reported_page_out_of_range"
ANCHOR_REASON_AMBIGUOUS_REPORTED_PAGE = "ambiguous_reported_page"
ANCHOR_REASON_QUOTE_NOT_FOUND_ON_REPORTED_PAGE = "quote_not_found_on_reported_page"


class ExtractionClient(Protocol):
    model_name: str

    def extract(
        self,
        *,
        paper_text: str,
        schema_payload: dict[str, object],
    ) -> StructuredExtractionBundle: ...


@dataclass(frozen=True, slots=True)
class PaperExtractionResult:
    paper_id: str
    extraction_run_id: str
    run_status: str
    result_count: int


@dataclass(frozen=True, slots=True)
class EvidenceAnchor:
    section_id: str | None
    chunk_id: str | None
    mode: str
    reason: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    page_number: int | None = None
    quote_preview: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceAnchorIndex:
    sections: list[Section]
    chunks: list[Chunk]


def _normalize_name(value: str | None, fallback: str) -> str:
    if value is not None and value.strip():
        return value.strip()
    return fallback.strip().lower()


def _normalize_lookup_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _text_contains_quote(text: str | None, quote: str | None) -> bool:
    normalized_quote = _normalize_lookup_text(quote)
    return bool(normalized_quote and normalized_quote in _normalize_lookup_text(text))


def _unique_item(items: list[T]) -> T | None:
    return items[0] if len(items) == 1 else None


def _page_candidates(page_number: int | None) -> list[int]:
    if page_number is None:
        return []
    candidates: list[int] = []
    if page_number > 0:
        candidates.append(page_number - 1)
    candidates.append(page_number)
    return list(dict.fromkeys(candidates))


def _section_contains_page(section: Section, page_number: int) -> bool:
    if section.page_start is None:
        return False
    if section.page_end is None:
        return page_number == section.page_start
    return section.page_start <= page_number < section.page_end


def _quote_preview(quote_text: str | None) -> str:
    preview = " ".join((quote_text or "").split())
    if len(preview) <= MAX_EVIDENCE_ANCHOR_QUOTE_PREVIEW_CHARS:
        return preview
    return (
        preview[: MAX_EVIDENCE_ANCHOR_QUOTE_PREVIEW_CHARS - 3].rstrip()
        + "..."
    )


def _evidence_anchor_diagnostics(anchors: list[EvidenceAnchor]) -> dict[str, object]:
    unresolved_anchors = [
        anchor for anchor in anchors if anchor.mode == ANCHOR_MODE_UNRESOLVED
    ]
    mode_counts = {
        mode: sum(1 for anchor in anchors if anchor.mode == mode)
        for mode in ANCHOR_MODE_ORDER
    }
    diagnostics: dict[str, object] = {
        "total": len(anchors),
        "section_anchored": sum(1 for anchor in anchors if anchor.section_id is not None),
        "chunk_anchored": sum(1 for anchor in anchors if anchor.chunk_id is not None),
        "unresolved": sum(
            1
            for anchor in anchors
            if anchor.section_id is None and anchor.chunk_id is None
        ),
        "modes": {
            mode: count
            for mode, count in mode_counts.items()
            if count
        },
    }
    if unresolved_anchors:
        diagnostics["unresolved_samples"] = [
            {
                "mode": anchor.mode,
                "reason": anchor.reason,
                "target_type": anchor.target_type,
                "target_id": anchor.target_id,
                "page_number": anchor.page_number,
                "quote_preview": anchor.quote_preview,
            }
            for anchor in unresolved_anchors[:MAX_EVIDENCE_ANCHOR_UNRESOLVED_SAMPLES]
        ]
        diagnostics["unresolved_sample_limit"] = (
            MAX_EVIDENCE_ANCHOR_UNRESOLVED_SAMPLES
        )
        diagnostics["unresolved_samples_truncated"] = (
            len(unresolved_anchors) > MAX_EVIDENCE_ANCHOR_UNRESOLVED_SAMPLES
        )
    return diagnostics


class PaperExtractionPipeline:
    """Run a structured extractor over parsed paper text and persist the outputs."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        client: ExtractionClient,
    ) -> None:
        self.session_factory = session_factory
        self.client = client

    def extract_paper(
        self,
        *,
        paper_id: str,
        schema_payload: dict[str, object],
        prompt_version: str,
        schema_version: str,
        extraction_profile_id: str | None = None,
    ) -> PaperExtractionResult:
        paper_text, freshness = self._load_paper_text_and_freshness(
            paper_id=paper_id,
            schema_payload=schema_payload,
        )
        run_id = self._start_run(
            paper_id=paper_id,
            extraction_profile_id=extraction_profile_id,
            prompt_version=prompt_version,
            schema_version=schema_version,
            freshness=freshness,
        )

        try:
            bundle = self.client.extract(paper_text=paper_text, schema_payload=schema_payload)
            result = self._persist_bundle(
                paper_id=paper_id,
                extraction_run_id=run_id,
                bundle=bundle,
            )
        except Exception as exc:
            self._mark_run_failed(extraction_run_id=run_id, error_message=str(exc))
            raise

        return result

    def _load_paper_text(self, paper_id: str) -> str:
        with self.session_factory() as session:
            return load_parsed_paper_text(session, paper_id=paper_id)

    def _load_paper_text_and_freshness(
        self,
        *,
        paper_id: str,
        schema_payload: dict[str, object],
    ) -> tuple[str, ExtractionFreshness]:
        with self.session_factory() as session:
            paper_text = load_parsed_paper_text(session, paper_id=paper_id)
            freshness = build_extraction_freshness(
                session,
                paper_id=paper_id,
                schema_payload=schema_payload,
                paper_text=paper_text,
            )
            return paper_text, freshness

    def _start_run(
        self,
        *,
        paper_id: str,
        extraction_profile_id: str | None,
        prompt_version: str,
        schema_version: str,
        freshness: ExtractionFreshness,
    ) -> str:
        with self.session_factory() as session:
            run = ExtractionRun(
                paper_id=paper_id,
                extraction_profile_id=extraction_profile_id,
                model_name=getattr(self.client, "model_name", self.client.__class__.__name__),
                prompt_version=prompt_version,
                schema_version=schema_version,
                status="running",
                diagnostics_json=merge_freshness_diagnostics(
                    None,
                    freshness=freshness,
                ),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run.id

    def _mark_run_failed(self, *, extraction_run_id: str, error_message: str) -> None:
        with self.session_factory() as session:
            run = session.get(ExtractionRun, extraction_run_id)
            if run is None:
                return
            run.status = "failed"
            diagnostics = dict(run.diagnostics_json or {})
            diagnostics["error"] = error_message
            run.diagnostics_json = diagnostics
            session.commit()

    def _persist_bundle(
        self,
        *,
        paper_id: str,
        extraction_run_id: str,
        bundle: StructuredExtractionBundle,
    ) -> PaperExtractionResult:
        with self.session_factory() as session:
            self._clear_previous_entities(session, paper_id=paper_id)
            anchor_index = self._load_evidence_anchor_index(session, paper_id=paper_id)
            evidence_anchors: list[EvidenceAnchor] = []

            datasets = {
                item.display_name: self._persist_dataset(session, paper_id=paper_id, item=item)
                for item in bundle.datasets
            }
            methods = {
                item.display_name: self._persist_method(session, paper_id=paper_id, item=item)
                for item in bundle.methods
            }
            metrics = {
                item.display_name: self._persist_metric(session, paper_id=paper_id, item=item)
                for item in bundle.metrics
            }

            for item in bundle.datasets:
                dataset = datasets[item.display_name]
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="dataset",
                    target_id=dataset.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            for item in bundle.methods:
                method = methods[item.display_name]
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="method",
                    target_id=method.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            for item in bundle.metrics:
                metric = metrics[item.display_name]
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="metric",
                    target_id=metric.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            for item in bundle.results:
                dataset = datasets.get(item.dataset_name)
                method = methods.get(item.method_name)
                metric = metrics.get(item.metric_name)
                result_row = ResultRow(
                    paper_id=paper_id,
                    dataset_id=dataset.id if dataset else None,
                    method_id=method.id if method else None,
                    metric_id=metric.id if metric else None,
                    value_numeric=item.value_numeric,
                    value_text=item.value_text,
                    comparator_text=item.comparator_text,
                    notes=item.notes,
                )
                session.add(result_row)
                session.flush()
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="result_row",
                    target_id=result_row.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            for item in bundle.findings:
                finding = Finding(
                    paper_id=paper_id,
                    statement=item.statement,
                    polarity=item.polarity,
                )
                session.add(finding)
                session.flush()
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="finding",
                    target_id=finding.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            for item in bundle.limitations:
                limitation = Limitation(
                    paper_id=paper_id,
                    statement=item.statement,
                )
                session.add(limitation)
                session.flush()
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="limitation",
                    target_id=limitation.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            for item in bundle.glossary_terms:
                glossary_term = GlossaryTerm(
                    paper_id=paper_id,
                    term=item.term,
                    definition=item.definition,
                    metadata_json=dict(item.metadata),
                )
                session.add(glossary_term)
                session.flush()
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="glossary_term",
                    target_id=glossary_term.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            for item in bundle.engineering_tricks:
                trick = EngineeringTrick(
                    paper_id=paper_id,
                    title=item.title,
                    description=item.description,
                )
                session.add(trick)
                session.flush()
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="engineering_trick",
                    target_id=trick.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            for item in bundle.research_design_elements:
                design_element = ResearchDesignElement(
                    paper_id=paper_id,
                    element_type=item.element_type,
                    title=item.title,
                    description=item.description,
                    metadata_json=dict(item.metadata),
                )
                session.add(design_element)
                session.flush()
                self._persist_evidence_spans(
                    session,
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type="research_design_element",
                    target_id=design_element.id,
                    evidence_spans=item.evidence_spans,
                    anchor_index=anchor_index,
                    evidence_anchors=evidence_anchors,
                )

            run = session.get(ExtractionRun, extraction_run_id)
            if run is None:
                raise ValueError(f"Extraction run {extraction_run_id} no longer exists")
            run.status = "completed"
            run.diagnostics_json = {
                **dict(run.diagnostics_json or {}),
                "datasets": len(bundle.datasets),
                "methods": len(bundle.methods),
                "metrics": len(bundle.metrics),
                "results": len(bundle.results),
                "findings": len(bundle.findings),
                "limitations": len(bundle.limitations),
                "glossary_terms": len(bundle.glossary_terms),
                "engineering_tricks": len(bundle.engineering_tricks),
                "research_design_elements": len(bundle.research_design_elements),
                "evidence_span_anchors": _evidence_anchor_diagnostics(evidence_anchors),
            }
            session.commit()

        return PaperExtractionResult(
            paper_id=paper_id,
            extraction_run_id=extraction_run_id,
            run_status="completed",
            result_count=len(bundle.results),
        )

    def _clear_previous_entities(self, session: Session, *, paper_id: str) -> None:
        session.execute(delete(EvidenceSpan).where(EvidenceSpan.paper_id == paper_id))
        session.execute(delete(ResultRow).where(ResultRow.paper_id == paper_id))
        session.execute(delete(GlossaryTerm).where(GlossaryTerm.paper_id == paper_id))
        session.execute(delete(Finding).where(Finding.paper_id == paper_id))
        session.execute(delete(Limitation).where(Limitation.paper_id == paper_id))
        session.execute(delete(EngineeringTrick).where(EngineeringTrick.paper_id == paper_id))
        session.execute(
            delete(ResearchDesignElement).where(ResearchDesignElement.paper_id == paper_id)
        )
        session.execute(delete(Dataset).where(Dataset.paper_id == paper_id))
        session.execute(delete(Method).where(Method.paper_id == paper_id))
        session.execute(delete(Metric).where(Metric.paper_id == paper_id))

    def _persist_dataset(
        self,
        session: Session,
        *,
        paper_id: str,
        item: DatasetExtraction,
    ) -> Dataset:
        dataset = Dataset(
            paper_id=paper_id,
            display_name=item.display_name,
            normalized_name=_normalize_name(item.normalized_name, item.display_name),
            metadata_json=dict(item.metadata),
        )
        session.add(dataset)
        session.flush()
        return dataset

    def _persist_method(self, session: Session, *, paper_id: str, item: MethodExtraction) -> Method:
        method = Method(
            paper_id=paper_id,
            display_name=item.display_name,
            normalized_name=_normalize_name(item.normalized_name, item.display_name),
            metadata_json=dict(item.metadata),
        )
        session.add(method)
        session.flush()
        return method

    def _persist_metric(self, session: Session, *, paper_id: str, item: MetricExtraction) -> Metric:
        metric = Metric(
            paper_id=paper_id,
            display_name=item.display_name,
            normalized_name=_normalize_name(item.normalized_name, item.display_name),
            metadata_json=dict(item.metadata),
        )
        session.add(metric)
        session.flush()
        return metric

    def _load_evidence_anchor_index(
        self,
        session: Session,
        *,
        paper_id: str,
    ) -> EvidenceAnchorIndex:
        sections = session.execute(
            select(Section)
            .where(Section.paper_id == paper_id)
            .order_by(Section.ordinal.asc())
        ).scalars().all()
        chunks = session.execute(
            select(Chunk)
            .where(Chunk.paper_id == paper_id)
            .order_by(Chunk.ordinal.asc())
        ).scalars().all()
        return EvidenceAnchorIndex(sections=list(sections), chunks=list(chunks))

    def _resolve_evidence_anchor(
        self,
        evidence: EvidenceSpanPayload,
        anchor_index: EvidenceAnchorIndex,
    ) -> EvidenceAnchor:
        section = self._match_section_by_quote(evidence.quote_text, anchor_index)
        section_matched_by_quote = section is not None
        if section is None:
            section = self._match_section_by_page(evidence.page_number, anchor_index)

        candidate_chunks = [
            chunk
            for chunk in anchor_index.chunks
            if section is None or chunk.section_id == section.id
        ]
        chunk = self._match_chunk_by_quote(evidence.quote_text, candidate_chunks)
        mode: str | None = None
        if chunk is not None:
            mode = ANCHOR_MODE_CHUNK_EXACT_QUOTE
        if chunk is None and section_matched_by_quote:
            chunk = self._match_adjacent_chunks_by_quote(
                evidence.quote_text,
                candidate_chunks,
                section_text=section.text if section is not None else None,
            )
            if chunk is not None:
                mode = ANCHOR_MODE_CHUNK_ADJACENT_QUOTE
        if mode is None and section is not None:
            mode = (
                ANCHOR_MODE_SECTION_QUOTE_ONLY
                if section_matched_by_quote
                else ANCHOR_MODE_SECTION_PAGE_ONLY
            )
        reason: str | None = None
        if mode is None:
            mode = ANCHOR_MODE_UNRESOLVED
            reason = self._unresolved_evidence_anchor_reason(evidence, anchor_index)
        section_id = section.id if section is not None else chunk.section_id if chunk else None
        return EvidenceAnchor(
            section_id=section_id,
            chunk_id=chunk.id if chunk is not None else None,
            mode=mode,
            reason=reason,
        )

    def _match_section_by_quote(
        self,
        quote_text: str | None,
        anchor_index: EvidenceAnchorIndex,
    ) -> Section | None:
        matches = [
            section
            for section in anchor_index.sections
            if _text_contains_quote(section.text, quote_text)
        ]
        return _unique_item(matches)

    def _match_section_by_page(
        self,
        page_number: int | None,
        anchor_index: EvidenceAnchorIndex,
    ) -> Section | None:
        matched_sections = self._unique_page_candidate_sections(
            page_number,
            anchor_index,
        )
        unique_section_ids = {section.id for section in matched_sections}
        return matched_sections[0] if len(unique_section_ids) == 1 else None

    def _unique_page_candidate_sections(
        self,
        page_number: int | None,
        anchor_index: EvidenceAnchorIndex,
    ) -> list[Section]:
        matched_sections: list[Section] = []
        for candidate_page in _page_candidates(page_number):
            matches = [
                section
                for section in anchor_index.sections
                if _section_contains_page(section, candidate_page)
            ]
            match = _unique_item(matches)
            if isinstance(match, Section):
                matched_sections.append(match)
        return matched_sections

    def _unresolved_evidence_anchor_reason(
        self,
        evidence: EvidenceSpanPayload,
        anchor_index: EvidenceAnchorIndex,
    ) -> str:
        if not anchor_index.sections:
            return ANCHOR_REASON_NO_PARSED_SECTIONS
        if not _normalize_lookup_text(evidence.quote_text):
            return (
                ANCHOR_REASON_MISSING_QUOTE_AND_PAGE
                if evidence.page_number is None
                else ANCHOR_REASON_MISSING_QUOTE_TEXT
            )

        quote_section_matches = [
            section
            for section in anchor_index.sections
            if _text_contains_quote(section.text, evidence.quote_text)
        ]
        if len(quote_section_matches) > 1:
            return ANCHOR_REASON_AMBIGUOUS_SECTION_QUOTE
        if evidence.page_number is None:
            return ANCHOR_REASON_QUOTE_NOT_FOUND_WITHOUT_PAGE

        page_section_matches = self._unique_page_candidate_sections(
            evidence.page_number,
            anchor_index,
        )
        unique_page_section_ids = {section.id for section in page_section_matches}
        if not unique_page_section_ids:
            return ANCHOR_REASON_REPORTED_PAGE_OUT_OF_RANGE
        if len(unique_page_section_ids) > 1:
            return ANCHOR_REASON_AMBIGUOUS_REPORTED_PAGE
        return ANCHOR_REASON_QUOTE_NOT_FOUND_ON_REPORTED_PAGE

    def _match_chunk_by_quote(
        self,
        quote_text: str | None,
        candidate_chunks: list[Chunk],
    ) -> Chunk | None:
        matches = [
            chunk
            for chunk in candidate_chunks
            if _text_contains_quote(chunk.text, quote_text)
        ]
        match = _unique_item(matches)
        return match if isinstance(match, Chunk) else None

    def _match_adjacent_chunks_by_quote(
        self,
        quote_text: str | None,
        candidate_chunks: list[Chunk],
        section_text: str | None,
    ) -> Chunk | None:
        normalized_quote = _normalize_lookup_text(quote_text)
        if not normalized_quote or not _normalize_lookup_text(section_text):
            return None

        matches: list[Chunk] = []
        chunks_by_section: dict[str | None, list[Chunk]] = {}
        for chunk in sorted(candidate_chunks, key=lambda item: item.ordinal):
            chunks_by_section.setdefault(chunk.section_id, []).append(chunk)

        for chunks in chunks_by_section.values():
            for start_index, first_chunk in enumerate(chunks):
                window_text_parts: list[str] = []
                for chunk in chunks[
                    start_index : start_index + MAX_CROSS_CHUNK_ANCHOR_WINDOW
                ]:
                    window_text_parts.append(chunk.text)
                if any(
                    _text_contains_quote(section_text, window_text)
                    and _text_contains_quote(window_text, quote_text)
                    for window_text in (
                        " ".join(window_text_parts),
                        "".join(window_text_parts),
                    )
                ):
                    matches.append(first_chunk)

        match = _unique_item(matches)
        return match if isinstance(match, Chunk) else None

    def _persist_evidence_spans(
        self,
        session: Session,
        *,
        paper_id: str,
        extraction_run_id: str,
        target_type: str,
        target_id: str,
        evidence_spans: list[EvidenceSpanPayload],
        anchor_index: EvidenceAnchorIndex,
        evidence_anchors: list[EvidenceAnchor],
    ) -> None:
        for evidence in evidence_spans:
            anchor = self._resolve_evidence_anchor(evidence, anchor_index)
            evidence_anchors.append(
                EvidenceAnchor(
                    section_id=anchor.section_id,
                    chunk_id=anchor.chunk_id,
                    mode=anchor.mode,
                    reason=anchor.reason,
                    target_type=target_type,
                    target_id=target_id,
                    page_number=evidence.page_number,
                    quote_preview=_quote_preview(evidence.quote_text),
                )
            )
            session.add(
                EvidenceSpan(
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    section_id=anchor.section_id,
                    chunk_id=anchor.chunk_id,
                    target_type=target_type,
                    target_id=target_id,
                    page_number=evidence.page_number,
                    quote_text=evidence.quote_text,
                    start_char=evidence.start_char,
                    end_char=evidence.end_char,
                )
            )
