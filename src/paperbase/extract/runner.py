"""Collection-level orchestration for local-first Paperbase extraction."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.models import CollectionPaper, ExtractionRun, Section
from paperbase.extract.freshness import (
    ExtractionFreshness,
    build_extraction_freshness,
    extraction_run_matches_freshness,
)
from paperbase.extract.pipeline import ExtractionClient, PaperExtractionPipeline
from paperbase.parsing.files import load_active_pdf_file
from paperbase.parsing.pipeline import PaperParsePipeline


@dataclass(frozen=True, slots=True)
class CollectionExtractionSummary:
    collection_id: str
    extracted_paper_count: int
    extraction_run_ids: list[str]
    skipped_paper_ids: list[str]


class CollectionExtractionRunner:
    """Run structured extraction across papers in one local collection."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        client: ExtractionClient,
        parse_pipeline: PaperParsePipeline | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.pipeline = PaperExtractionPipeline(
            session_factory=session_factory,
            client=client,
        )
        self.parse_pipeline = parse_pipeline or PaperParsePipeline(session_factory=session_factory)

    def extract_collection(
        self,
        *,
        collection_id: str,
        schema_payload: dict[str, object],
        prompt_version: str,
        schema_version: str,
        extraction_profile_id: str | None = None,
        limit: int | None = None,
        paper_ids: list[str] | None = None,
    ) -> CollectionExtractionSummary:
        target_paper_ids = self._list_collection_paper_ids(
            collection_id=collection_id,
            limit=limit,
            paper_ids=paper_ids,
        )
        run_ids: list[str] = []
        skipped: list[str] = []

        for paper_id in target_paper_ids:
            try:
                if not self._has_current_parse(paper_id):
                    self.parse_pipeline.parse_paper(paper_id)

                freshness = self._build_freshness(
                    paper_id=paper_id,
                    schema_payload=schema_payload,
                )
            except ValueError:
                skipped.append(paper_id)
                continue

            if self._has_completed_extraction(
                paper_id,
                extraction_profile_id=extraction_profile_id,
                model_name=self._model_name(),
                prompt_version=prompt_version,
                schema_version=schema_version,
                freshness=freshness,
            ):
                skipped.append(paper_id)
                continue

            try:
                result = self.pipeline.extract_paper(
                    paper_id=paper_id,
                    schema_payload=schema_payload,
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                    extraction_profile_id=extraction_profile_id,
                )
            except ValueError:
                skipped.append(paper_id)
                continue

            run_ids.append(result.extraction_run_id)

        return CollectionExtractionSummary(
            collection_id=collection_id,
            extracted_paper_count=len(run_ids),
            extraction_run_ids=run_ids,
            skipped_paper_ids=skipped,
        )

    def _list_collection_paper_ids(
        self,
        *,
        collection_id: str,
        limit: int | None,
        paper_ids: list[str] | None,
    ) -> list[str]:
        statement = (
            select(CollectionPaper.paper_id)
            .where(CollectionPaper.collection_id == collection_id)
            .order_by(CollectionPaper.position.asc(), CollectionPaper.created_at.asc())
        )
        if paper_ids is not None:
            statement = statement.where(CollectionPaper.paper_id.in_(paper_ids))
        if limit is not None:
            statement = statement.limit(max(0, limit))
        with self.session_factory() as session:
            return list(session.execute(statement).scalars().all())

    def _has_current_parse(self, paper_id: str) -> bool:
        with self.session_factory() as session:
            file_record = load_active_pdf_file(session, paper_id=paper_id)
            if file_record is None or file_record.parser_status != "parsed":
                return False

            section_id = session.execute(
                select(Section.id).where(Section.paper_id == paper_id).limit(1)
            ).scalar_one_or_none()
            return section_id is not None

    def _build_freshness(
        self,
        *,
        paper_id: str,
        schema_payload: dict[str, object],
    ) -> ExtractionFreshness:
        with self.session_factory() as session:
            return build_extraction_freshness(
                session,
                paper_id=paper_id,
                schema_payload=schema_payload,
            )

    def _has_completed_extraction(
        self,
        paper_id: str,
        *,
        extraction_profile_id: str | None,
        model_name: str,
        prompt_version: str,
        schema_version: str,
        freshness: ExtractionFreshness,
    ) -> bool:
        with self.session_factory() as session:
            statement = (
                select(ExtractionRun)
                .where(
                    ExtractionRun.paper_id == paper_id,
                    ExtractionRun.status == "completed",
                    ExtractionRun.model_name == model_name,
                    ExtractionRun.prompt_version == prompt_version,
                    ExtractionRun.schema_version == schema_version,
                )
            )
            if extraction_profile_id is None:
                statement = statement.where(ExtractionRun.extraction_profile_id.is_(None))
            else:
                statement = statement.where(
                    ExtractionRun.extraction_profile_id == extraction_profile_id
                )
            runs = session.execute(statement).scalars().all()
            return any(
                extraction_run_matches_freshness(run, freshness=freshness)
                for run in runs
            )

    def _model_name(self) -> str:
        return getattr(self.pipeline.client, "model_name", self.pipeline.client.__class__.__name__)
