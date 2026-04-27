"""Pipeline for persisting structured extraction outputs into Paperbase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import (
    Dataset,
    EngineeringTrick,
    EvidenceSpan,
    ExtractionRun,
    Finding,
    GlossaryTerm,
    Limitation,
    Method,
    Metric,
    ResultRow,
    Section,
)
from paperbase.extract.contracts import StructuredExtractionBundle
from paperbase.schemas.extraction import (
    DatasetExtraction,
    EngineeringTrickExtraction,
    EvidenceSpanPayload,
    FindingExtraction,
    GlossaryTermExtraction,
    LimitationExtraction,
    MethodExtraction,
    MetricExtraction,
    ResultExtraction,
)


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


def _normalize_name(value: str | None, fallback: str) -> str:
    if value is not None and value.strip():
        return value.strip()
    return fallback.strip().lower()


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
        paper_text = self._load_paper_text(paper_id)
        run_id = self._start_run(
            paper_id=paper_id,
            extraction_profile_id=extraction_profile_id,
            prompt_version=prompt_version,
            schema_version=schema_version,
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
            sections = session.execute(
                select(Section)
                .where(Section.paper_id == paper_id)
                .order_by(Section.ordinal.asc())
            ).scalars().all()

        if not sections:
            raise ValueError(f"No parsed sections found for paper_id={paper_id}")

        return "\n\n".join(f"{section.title}\n{section.text}" for section in sections if section.text.strip())

    def _start_run(
        self,
        *,
        paper_id: str,
        extraction_profile_id: str | None,
        prompt_version: str,
        schema_version: str,
    ) -> str:
        with self.session_factory() as session:
            run = ExtractionRun(
                paper_id=paper_id,
                extraction_profile_id=extraction_profile_id,
                model_name=getattr(self.client, "model_name", self.client.__class__.__name__),
                prompt_version=prompt_version,
                schema_version=schema_version,
                status="running",
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
            run.diagnostics_json = {"error": error_message}
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
                )

            for item in bundle.results:
                result_row = ResultRow(
                    paper_id=paper_id,
                    dataset_id=datasets.get(item.dataset_name).id if item.dataset_name in datasets else None,
                    method_id=methods.get(item.method_name).id if item.method_name in methods else None,
                    metric_id=metrics.get(item.metric_name).id if item.metric_name in metrics else None,
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
                )

            run = session.get(ExtractionRun, extraction_run_id)
            if run is None:
                raise ValueError(f"Extraction run {extraction_run_id} no longer exists")
            run.status = "completed"
            run.diagnostics_json = {
                "datasets": len(bundle.datasets),
                "methods": len(bundle.methods),
                "metrics": len(bundle.metrics),
                "results": len(bundle.results),
                "findings": len(bundle.findings),
                "limitations": len(bundle.limitations),
                "glossary_terms": len(bundle.glossary_terms),
                "engineering_tricks": len(bundle.engineering_tricks),
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
        session.execute(delete(Dataset).where(Dataset.paper_id == paper_id))
        session.execute(delete(Method).where(Method.paper_id == paper_id))
        session.execute(delete(Metric).where(Metric.paper_id == paper_id))

    def _persist_dataset(self, session: Session, *, paper_id: str, item: DatasetExtraction) -> Dataset:
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

    def _persist_evidence_spans(
        self,
        session: Session,
        *,
        paper_id: str,
        extraction_run_id: str,
        target_type: str,
        target_id: str,
        evidence_spans: list[EvidenceSpanPayload],
    ) -> None:
        for evidence in evidence_spans:
            session.add(
                EvidenceSpan(
                    paper_id=paper_id,
                    extraction_run_id=extraction_run_id,
                    target_type=target_type,
                    target_id=target_id,
                    page_number=evidence.page_number,
                    quote_text=evidence.quote_text,
                    start_char=evidence.start_char,
                    end_char=evidence.end_char,
                )
            )
