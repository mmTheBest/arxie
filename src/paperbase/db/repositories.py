"""Repository interfaces for Paperbase persistence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Select, delete, select, update
from sqlalchemy.orm import Session

from paperbase.db.models import (
    Annotation,
    Author,
    BackgroundJob,
    Collection,
    CollectionPaper,
    ExtractionProfile,
    Paper,
    PaperAuthor,
    PaperFile,
    PaperResearchLabel,
    PaperSource,
    PaperTag,
    ResearchAgentRun,
    ResearchAgentStep,
    ResearchArtifact,
    ResearchMessage,
    ResearchThread,
    ResearchValidationReport,
    StudyContextPack,
    StudySource,
    Tag,
    Venue,
    Workspace,
)


def _normalize_entity_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped:
            continue
        normalized = _normalize_entity_name(stripped)
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(stripped)
    return cleaned


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PaperRepository:
    """Persistence helpers for canonical paper records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_provider_id(self, provider: str, external_id: str) -> Paper | None:
        statement: Select[tuple[Paper]] = select(Paper).where(
            Paper.provider == provider,
            Paper.external_id == external_id,
        )
        return self.session.execute(statement).scalar_one_or_none()

    def get_by_id(self, paper_id: str) -> Paper | None:
        return self.session.get(Paper, paper_id)

    def get_by_doi(self, doi: str) -> Paper | None:
        statement: Select[tuple[Paper]] = select(Paper).where(Paper.doi == doi)
        return self.session.execute(statement).scalar_one_or_none()

    def get_by_arxiv_id(self, arxiv_id: str) -> Paper | None:
        statement: Select[tuple[Paper]] = select(Paper).where(Paper.arxiv_id == arxiv_id)
        return self.session.execute(statement).scalar_one_or_none()

    def list_author_names(self, paper_id: str) -> list[str]:
        rows = self.session.execute(
            select(Author.display_name)
            .join(PaperAuthor, PaperAuthor.author_id == Author.id)
            .where(PaperAuthor.paper_id == paper_id)
            .order_by(PaperAuthor.ordinal.asc(), Author.display_name.asc())
        ).all()
        return [row[0] for row in rows]

    def list_author_names_by_paper_ids(self, paper_ids: Sequence[str]) -> dict[str, list[str]]:
        if not paper_ids:
            return {}

        rows = self.session.execute(
            select(PaperAuthor.paper_id, Author.display_name)
            .join(Author, Author.id == PaperAuthor.author_id)
            .where(PaperAuthor.paper_id.in_(paper_ids))
            .order_by(PaperAuthor.paper_id.asc(), PaperAuthor.ordinal.asc(), Author.display_name.asc())
        ).all()
        grouped: dict[str, list[str]] = {paper_id: [] for paper_id in paper_ids}
        for paper_id, display_name in rows:
            grouped.setdefault(paper_id, []).append(display_name)
        return grouped

    def list_tags(self, paper_id: str) -> list[str]:
        rows = self.session.execute(
            select(Tag.display_name)
            .join(PaperTag, PaperTag.tag_id == Tag.id)
            .where(PaperTag.paper_id == paper_id)
            .order_by(Tag.display_name.asc())
        ).all()
        return [row[0] for row in rows]

    def list_tags_by_paper_ids(self, paper_ids: Sequence[str]) -> dict[str, list[str]]:
        if not paper_ids:
            return {}

        rows = self.session.execute(
            select(PaperTag.paper_id, Tag.display_name)
            .join(Tag, Tag.id == PaperTag.tag_id)
            .where(PaperTag.paper_id.in_(paper_ids))
            .order_by(PaperTag.paper_id.asc(), Tag.display_name.asc())
        ).all()
        grouped: dict[str, list[str]] = {paper_id: [] for paper_id in paper_ids}
        for paper_id, display_name in rows:
            grouped.setdefault(paper_id, []).append(display_name)
        return grouped

    def upsert(
        self,
        *,
        provider: str,
        external_id: str,
        canonical_title: str,
        abstract: str | None = None,
        publication_year: int | None = None,
        venue: str | None = None,
        doi: str | None = None,
        arxiv_id: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
        authors: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> Paper:
        paper = self.get_by_provider_id(provider, external_id)
        venue_display_name, venue_id = self._sync_venue(venue)
        if paper is None:
            paper = Paper(
                provider=provider,
                external_id=external_id,
                canonical_title=canonical_title,
                abstract=abstract,
                publication_year=publication_year,
                venue_id=venue_id,
                venue=venue_display_name,
                doi=doi,
                arxiv_id=arxiv_id,
                raw_metadata=raw_metadata or {},
            )
            self.session.add(paper)
        else:
            paper.canonical_title = canonical_title
            paper.abstract = abstract
            paper.publication_year = publication_year
            paper.venue_id = venue_id
            paper.venue = venue_display_name
            paper.doi = doi
            paper.arxiv_id = arxiv_id
            paper.raw_metadata = raw_metadata or paper.raw_metadata

        self.session.flush()

        if authors is not None:
            self._sync_authors(paper_id=paper.id, author_names=authors)
        if tags is not None:
            self._sync_tags(paper_id=paper.id, tag_names=tags)

        self.session.commit()
        self.session.refresh(paper)
        return paper

    def merge_metadata(
        self,
        paper_id: str,
        *,
        canonical_title: str | None = None,
        abstract: str | None = None,
        publication_year: int | None = None,
        venue: str | None = None,
        doi: str | None = None,
        arxiv_id: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
        authors: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> Paper:
        paper = self.get_by_id(paper_id)
        if paper is None:
            raise ValueError(f"No paper found for id: {paper_id}")

        if canonical_title:
            paper.canonical_title = canonical_title
        if abstract:
            paper.abstract = abstract
        if publication_year is not None:
            paper.publication_year = publication_year
        if venue:
            venue_display_name, venue_id = self._sync_venue(venue)
            paper.venue_id = venue_id
            paper.venue = venue_display_name
        if doi:
            paper.doi = doi
        if arxiv_id:
            paper.arxiv_id = arxiv_id
        if raw_metadata:
            merged = dict(paper.raw_metadata or {})
            merged.update(raw_metadata)
            paper.raw_metadata = merged

        self.session.flush()

        if authors is not None:
            self._sync_authors(paper_id=paper.id, author_names=authors)
        if tags is not None:
            self._sync_tags(paper_id=paper.id, tag_names=tags)

        self.session.commit()
        self.session.refresh(paper)
        return paper

    def _sync_authors(self, *, paper_id: str, author_names: Sequence[str]) -> None:
        cleaned_names = _dedupe_preserve_order(author_names)
        self.session.execute(delete(PaperAuthor).where(PaperAuthor.paper_id == paper_id))
        self.session.flush()

        for ordinal, display_name in enumerate(cleaned_names, start=1):
            normalized_name = _normalize_entity_name(display_name)
            author = self.session.execute(
                select(Author).where(Author.normalized_name == normalized_name)
            ).scalar_one_or_none()
            if author is None:
                author = Author(normalized_name=normalized_name, display_name=display_name)
                self.session.add(author)
                self.session.flush()
            else:
                author.display_name = display_name

            self.session.add(
                PaperAuthor(
                    paper_id=paper_id,
                    author_id=author.id,
                    ordinal=ordinal,
                )
            )

    def _sync_tags(self, *, paper_id: str, tag_names: Sequence[str]) -> None:
        cleaned_tags = sorted(_dedupe_preserve_order(tag_names), key=str.casefold)
        self.session.execute(delete(PaperTag).where(PaperTag.paper_id == paper_id))
        self.session.flush()

        for display_name in cleaned_tags:
            normalized_name = _normalize_entity_name(display_name)
            tag = self.session.execute(
                select(Tag).where(Tag.normalized_name == normalized_name)
            ).scalar_one_or_none()
            if tag is None:
                tag = Tag(normalized_name=normalized_name, display_name=display_name)
                self.session.add(tag)
                self.session.flush()
            else:
                tag.display_name = display_name

            self.session.add(
                PaperTag(
                    paper_id=paper_id,
                    tag_id=tag.id,
                )
            )

    def _sync_venue(self, venue_name: str | None) -> tuple[str | None, str | None]:
        if venue_name is None:
            return None, None

        cleaned_name = " ".join(venue_name.strip().split())
        if not cleaned_name:
            return None, None

        normalized_name = _normalize_entity_name(cleaned_name)
        venue = self.session.execute(
            select(Venue).where(Venue.normalized_name == normalized_name)
        ).scalar_one_or_none()
        if venue is None:
            venue = Venue(normalized_name=normalized_name, display_name=cleaned_name)
            self.session.add(venue)
            self.session.flush()

        return venue.display_name, venue.id


class ExtractionProfileRepository:
    """Persistence helpers for collection-specific extraction profiles."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        owner_id: str,
        name: str,
        description: str | None = None,
        scope_type: str = "private",
        schema_payload: dict[str, Any] | None = None,
        active: bool = True,
    ) -> ExtractionProfile:
        profile = ExtractionProfile(
            owner_id=owner_id,
            name=name,
            description=description,
            scope_type=scope_type,
            schema_payload=schema_payload or {},
            active=active,
        )
        self.session.add(profile)
        self.session.commit()
        self.session.refresh(profile)
        return profile

    def get_by_id(self, profile_id: str) -> ExtractionProfile | None:
        return self.session.get(ExtractionProfile, profile_id)

    def list_profiles(self, *, owner_id: str | None = None) -> Sequence[ExtractionProfile]:
        statement = select(ExtractionProfile)
        if owner_id is not None:
            statement = statement.where(ExtractionProfile.owner_id == owner_id)
        statement = statement.order_by(ExtractionProfile.created_at.asc(), ExtractionProfile.name.asc())
        return self.session.execute(statement).scalars().all()


class CollectionRepository:
    """Persistence helpers for curated local-first paper collections."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_owner_title(self, owner_id: str, title: str) -> Collection | None:
        statement: Select[tuple[Collection]] = select(Collection).where(
            Collection.owner_id == owner_id,
            Collection.title == title,
        )
        return self.session.execute(statement).scalar_one_or_none()

    def get_by_id(self, collection_id: str) -> Collection | None:
        return self.session.get(Collection, collection_id)

    def list_collections(self, *, owner_id: str | None = None) -> Sequence[Collection]:
        statement = select(Collection)
        if owner_id is not None:
            statement = statement.where(Collection.owner_id == owner_id)
        statement = statement.order_by(Collection.created_at.asc(), Collection.title.asc())
        return self.session.execute(statement).scalars().all()

    def create(
        self,
        *,
        owner_id: str,
        title: str,
        description: str | None = None,
        scope_type: str = "private",
        tags: list[str] | None = None,
        extraction_profile_id: str | None = None,
    ) -> Collection:
        collection = Collection(
            owner_id=owner_id,
            scope_type=scope_type,
            title=title,
            description=description,
            tags_json=tags or [],
            extraction_profile_id=extraction_profile_id,
        )
        self.session.add(collection)
        self.session.commit()
        self.session.refresh(collection)
        return collection

    def create_or_get(
        self,
        *,
        owner_id: str,
        title: str,
        description: str | None = None,
        scope_type: str = "private",
        tags: list[str] | None = None,
        extraction_profile_id: str | None = None,
    ) -> Collection:
        collection = self.get_by_owner_title(owner_id, title)
        if collection is not None:
            return collection

        return self.create(
            owner_id=owner_id,
            title=title,
            description=description,
            scope_type=scope_type,
            tags=tags,
            extraction_profile_id=extraction_profile_id,
        )

    def add_paper(
        self,
        *,
        collection_id: str,
        paper_id: str,
        position: int | None = None,
        membership_note: str | None = None,
    ) -> CollectionPaper:
        statement: Select[tuple[CollectionPaper]] = select(CollectionPaper).where(
            CollectionPaper.collection_id == collection_id,
            CollectionPaper.paper_id == paper_id,
        )
        membership = self.session.execute(statement).scalar_one_or_none()
        if membership is None:
            membership = CollectionPaper(
                collection_id=collection_id,
                paper_id=paper_id,
                position=position,
                membership_note=membership_note,
            )
            self.session.add(membership)
        else:
            membership.position = position if position is not None else membership.position
            membership.membership_note = membership_note

        self.session.commit()
        self.session.refresh(membership)
        return membership

    def list_papers(self, collection_id: str) -> Sequence[CollectionPaper]:
        statement: Select[tuple[CollectionPaper]] = (
            select(CollectionPaper)
            .where(CollectionPaper.collection_id == collection_id)
            .order_by(CollectionPaper.position.asc(), CollectionPaper.created_at.asc())
        )
        return self.session.execute(statement).scalars().all()


class AnnotationRepository:
    """Persistence helpers for user-authored annotations separate from canonical facts."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        author_id: str,
        target_type: str,
        target_id: str,
        body: str,
        collection_id: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
    ) -> Annotation:
        annotation = Annotation(
            author_id=author_id,
            collection_id=collection_id,
            target_type=target_type,
            target_id=target_id,
            body=body,
            tags_json=tags or [],
            status=status,
        )
        self.session.add(annotation)
        self.session.commit()
        self.session.refresh(annotation)
        return annotation

    def list_for_target(self, *, target_type: str, target_id: str) -> Sequence[Annotation]:
        statement: Select[tuple[Annotation]] = (
            select(Annotation)
            .where(Annotation.target_type == target_type, Annotation.target_id == target_id)
            .order_by(Annotation.created_at.asc())
        )
        return self.session.execute(statement).scalars().all()


class WorkspaceRepository:
    """Persistence helpers for durable research workspaces."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_owner_title(self, owner_id: str, title: str) -> Workspace | None:
        statement: Select[tuple[Workspace]] = select(Workspace).where(
            Workspace.owner_id == owner_id,
            Workspace.title == title,
        )
        return self.session.execute(statement).scalar_one_or_none()

    def get_by_id(self, workspace_id: str) -> Workspace | None:
        return self.session.get(Workspace, workspace_id)

    def list_workspaces(self, *, owner_id: str | None = None) -> Sequence[Workspace]:
        statement = select(Workspace)
        if owner_id is not None:
            statement = statement.where(Workspace.owner_id == owner_id)
        statement = statement.order_by(Workspace.created_at.asc(), Workspace.title.asc())
        return self.session.execute(statement).scalars().all()

    def create(
        self,
        *,
        owner_id: str,
        title: str,
        description: str | None = None,
        collection_id: str | None = None,
        saved_query: str | None = None,
        focus_note: str | None = None,
        active_filters: dict[str, Any] | None = None,
        pinned_paper_ids: Sequence[str] | None = None,
    ) -> Workspace:
        workspace = Workspace(
            owner_id=owner_id,
            title=title,
            description=description,
            collection_id=collection_id,
            saved_query=saved_query,
            focus_note=focus_note,
            active_filters_json=active_filters or {},
            pinned_paper_ids_json=list(pinned_paper_ids or []),
        )
        self.session.add(workspace)
        self.session.commit()
        self.session.refresh(workspace)
        return workspace

    def update(
        self,
        workspace_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        collection_id: str | None = None,
        saved_query: str | None = None,
        focus_note: str | None = None,
        active_filters: dict[str, Any] | None = None,
        pinned_paper_ids: Sequence[str] | None = None,
    ) -> Workspace:
        workspace = self.get_by_id(workspace_id)
        if workspace is None:
            raise ValueError(f"No workspace found for id: {workspace_id}")

        if title is not None:
            workspace.title = title
        if description is not None:
            workspace.description = description
        workspace.collection_id = collection_id
        workspace.saved_query = saved_query
        workspace.focus_note = focus_note
        if active_filters is not None:
            workspace.active_filters_json = dict(active_filters)
        if pinned_paper_ids is not None:
            workspace.pinned_paper_ids_json = list(pinned_paper_ids)

        self.session.commit()
        self.session.refresh(workspace)
        return workspace

    def create_source(
        self,
        *,
        workspace_id: str,
        source_type: str,
        title: str,
        path: str | None = None,
        content: str | None = None,
        summary: str | None = None,
        read_status: str = "ready",
        error_message: str | None = None,
    ) -> StudySource:
        source = StudySource(
            workspace_id=workspace_id,
            source_type=source_type,
            title=title,
            path=path,
            content=content,
            summary=summary,
            read_status=read_status,
            error_message=error_message,
        )
        self.session.add(source)
        self.session.commit()
        self.session.refresh(source)
        return source

    def get_source(self, source_id: str) -> StudySource | None:
        return self.session.get(StudySource, source_id)

    def list_sources(self, *, workspace_id: str) -> Sequence[StudySource]:
        statement: Select[tuple[StudySource]] = (
            select(StudySource)
            .where(StudySource.workspace_id == workspace_id)
            .order_by(StudySource.created_at.asc(), StudySource.id.asc())
        )
        return self.session.execute(statement).scalars().all()

    def delete_source(self, source_id: str) -> bool:
        source = self.get_source(source_id)
        if source is None:
            return False
        self.session.delete(source)
        self.session.commit()
        return True


class ResearchRepository:
    """Persistence helpers for collection-grounded research agent state."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_thread(self, thread_id: str) -> ResearchThread | None:
        return self.session.get(ResearchThread, thread_id)

    def list_threads(self, *, collection_id: str | None = None) -> Sequence[ResearchThread]:
        statement = select(ResearchThread)
        if collection_id is not None:
            statement = statement.where(ResearchThread.collection_id == collection_id)
        statement = statement.order_by(ResearchThread.updated_at.desc(), ResearchThread.created_at.desc())
        return self.session.execute(statement).scalars().all()

    def create_thread(
        self,
        *,
        owner_id: str,
        title: str,
        collection_id: str,
        workspace_id: str | None = None,
        selected_paper_ids: Sequence[str] | None = None,
        status: str = "active",
    ) -> ResearchThread:
        thread = ResearchThread(
            owner_id=owner_id,
            title=title,
            collection_id=collection_id,
            workspace_id=workspace_id,
            selected_paper_ids_json=list(selected_paper_ids or []),
            status=status,
        )
        self.session.add(thread)
        self.session.commit()
        self.session.refresh(thread)
        return thread

    def create_message(
        self,
        *,
        thread_id: str,
        role: str,
        content: str,
        artifact_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchMessage:
        message = ResearchMessage(
            thread_id=thread_id,
            role=role,
            content=content,
            artifact_id=artifact_id,
            metadata_json=metadata or {},
        )
        self.session.add(message)
        self.session.commit()
        self.session.refresh(message)
        return message

    def list_messages(self, *, thread_id: str) -> Sequence[ResearchMessage]:
        statement = (
            select(ResearchMessage)
            .where(ResearchMessage.thread_id == thread_id)
            .order_by(ResearchMessage.created_at.asc(), ResearchMessage.id.asc())
        )
        return self.session.execute(statement).scalars().all()

    def create_artifact(
        self,
        *,
        collection_id: str,
        thread_id: str | None,
        artifact_type: str,
        title: str,
        status: str = "pending",
        input_payload: dict[str, Any] | None = None,
        output_payload: dict[str, Any] | None = None,
        evidence_payload: dict[str, Any] | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
        error_message: str | None = None,
    ) -> ResearchArtifact:
        artifact = ResearchArtifact(
            collection_id=collection_id,
            thread_id=thread_id,
            artifact_type=artifact_type,
            title=title,
            status=status,
            input_payload_json=input_payload or {},
            output_payload_json=output_payload or {},
            evidence_payload_json=evidence_payload or {},
            model_name=model_name,
            prompt_version=prompt_version,
            error_message=error_message,
        )
        self.session.add(artifact)
        self.session.commit()
        self.session.refresh(artifact)
        return artifact

    def get_artifact(self, artifact_id: str) -> ResearchArtifact | None:
        return self.session.get(ResearchArtifact, artifact_id)

    def list_artifacts(self, *, collection_id: str | None = None) -> Sequence[ResearchArtifact]:
        statement = select(ResearchArtifact)
        if collection_id is not None:
            statement = statement.where(ResearchArtifact.collection_id == collection_id)
        statement = statement.order_by(ResearchArtifact.updated_at.desc(), ResearchArtifact.created_at.desc())
        return self.session.execute(statement).scalars().all()

    def update_artifact(
        self,
        artifact_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
        output_payload: dict[str, Any] | None = None,
        evidence_payload: dict[str, Any] | None = None,
        model_name: str | None = None,
        prompt_version: str | None = None,
        error_message: str | None = None,
    ) -> ResearchArtifact:
        artifact = self.get_artifact(artifact_id)
        if artifact is None:
            raise ValueError(f"No research artifact found for id: {artifact_id}")

        if title is not None:
            artifact.title = title
        if status is not None:
            artifact.status = status
        if output_payload is not None:
            artifact.output_payload_json = dict(output_payload)
        if evidence_payload is not None:
            artifact.evidence_payload_json = dict(evidence_payload)
        if model_name is not None:
            artifact.model_name = model_name
        if prompt_version is not None:
            artifact.prompt_version = prompt_version
        artifact.error_message = error_message

        self.session.commit()
        self.session.refresh(artifact)
        return artifact

    def get_label(self, *, collection_id: str, paper_id: str) -> PaperResearchLabel | None:
        statement = select(PaperResearchLabel).where(
            PaperResearchLabel.collection_id == collection_id,
            PaperResearchLabel.paper_id == paper_id,
        )
        return self.session.execute(statement).scalar_one_or_none()

    def list_labels(self, *, collection_id: str) -> Sequence[PaperResearchLabel]:
        statement = (
            select(PaperResearchLabel)
            .where(PaperResearchLabel.collection_id == collection_id)
            .order_by(PaperResearchLabel.updated_at.desc(), PaperResearchLabel.created_at.desc())
        )
        return self.session.execute(statement).scalars().all()

    def upsert_label(
        self,
        *,
        collection_id: str,
        paper_id: str,
        user_label: str,
        inferred_label: str | None = None,
        inferred_signals: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> PaperResearchLabel:
        label = self.get_label(collection_id=collection_id, paper_id=paper_id)
        if label is None:
            label = PaperResearchLabel(
                collection_id=collection_id,
                paper_id=paper_id,
                user_label=user_label,
                inferred_label=inferred_label,
                inferred_signals_json=inferred_signals or {},
                notes=notes,
            )
            self.session.add(label)
        else:
            label.user_label = user_label
            label.inferred_label = inferred_label
            label.inferred_signals_json = inferred_signals or {}
            label.notes = notes

        self.session.commit()
        self.session.refresh(label)
        return label


class BackgroundJobRepository:
    """Persistence helpers for queued local-first background jobs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        job_type: str,
        payload_json: dict[str, Any] | None = None,
    ) -> BackgroundJob:
        job = BackgroundJob(
            job_type=job_type,
            status="pending",
            payload_json=payload_json or {},
        )
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job

    def get_by_id(self, job_id: str) -> BackgroundJob | None:
        return self.session.get(BackgroundJob, job_id)

    def list_recent(self, *, limit: int = 20) -> Sequence[BackgroundJob]:
        statement: Select[tuple[BackgroundJob]] = (
            select(BackgroundJob)
            .order_by(BackgroundJob.created_at.desc())
            .limit(max(1, limit))
        )
        return self.session.execute(statement).scalars().all()

    def claim_next(self, *, job_types: Sequence[str] | None = None) -> BackgroundJob | None:
        statement: Select[tuple[BackgroundJob]] = (
            select(BackgroundJob)
            .where(BackgroundJob.status == "pending")
            .order_by(BackgroundJob.created_at.asc())
            .limit(1)
        )
        if job_types is not None:
            statement = statement.where(BackgroundJob.job_type.in_(job_types))
        job = self.session.execute(statement).scalar_one_or_none()
        if job is None:
            return None

        job.status = "running"
        job.attempt_count += 1
        job.started_at = _utc_now()
        job.finished_at = None
        job.error_message = None
        self.session.commit()
        self.session.refresh(job)
        return job

    def claim_by_id(self, job_id: str) -> BackgroundJob | None:
        started_at = _utc_now()
        result = self.session.execute(
            update(BackgroundJob)
            .where(BackgroundJob.id == job_id, BackgroundJob.status == "pending")
            .values(
                status="running",
                attempt_count=BackgroundJob.attempt_count + 1,
                started_at=started_at,
                finished_at=None,
                error_message=None,
            )
        )
        if result.rowcount == 0:
            self.session.rollback()
            return None
        self.session.commit()
        return self.session.get(BackgroundJob, job_id)

    def mark_completed(self, job_id: str, *, result_json: dict[str, Any] | None = None) -> BackgroundJob:
        job = self.session.get(BackgroundJob, job_id)
        if job is None:
            raise ValueError(f"No background job found for id: {job_id}")

        job.status = "completed"
        job.result_json = result_json or {}
        job.error_message = None
        job.finished_at = _utc_now()
        self.session.commit()
        self.session.refresh(job)
        return job

    def mark_failed(self, job_id: str, *, error_message: str) -> BackgroundJob:
        job = self.session.get(BackgroundJob, job_id)
        if job is None:
            raise ValueError(f"No background job found for id: {job_id}")

        job.status = "failed"
        job.error_message = error_message
        job.finished_at = _utc_now()
        self.session.commit()
        self.session.refresh(job)
        return job

    def mark_pending(self, job_id: str, *, error_message: str | None = None) -> BackgroundJob:
        job = self.session.get(BackgroundJob, job_id)
        if job is None:
            raise ValueError(f"No background job found for id: {job_id}")

        job.status = "pending"
        job.error_message = error_message
        job.finished_at = None
        self.session.commit()
        self.session.refresh(job)
        return job

    def list_stale_running(self, *, older_than_seconds: float) -> Sequence[BackgroundJob]:
        cutoff = _utc_now() - timedelta(seconds=older_than_seconds)
        statement: Select[tuple[BackgroundJob]] = (
            select(BackgroundJob)
            .where(
                BackgroundJob.status == "running",
                BackgroundJob.started_at.is_not(None),
                BackgroundJob.started_at <= cutoff,
            )
            .order_by(BackgroundJob.started_at.asc())
        )
        return self.session.execute(statement).scalars().all()


class ResearchAgentRunRepository:
    """Persistence helpers for traceable Paperbase research-agent runs."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(
        self,
        *,
        thread_id: str | None,
        artifact_id: str,
        collection_id: str,
        workspace_id: str | None,
        skill_id: str,
        artifact_type: str,
        model_policy: str,
        input_json: dict[str, Any] | None = None,
    ) -> ResearchAgentRun:
        run = ResearchAgentRun(
            thread_id=thread_id,
            artifact_id=artifact_id,
            collection_id=collection_id,
            workspace_id=workspace_id,
            skill_id=skill_id,
            artifact_type=artifact_type,
            model_policy=model_policy,
            status="pending",
            input_json=input_json or {},
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def get_run(self, run_id: str) -> ResearchAgentRun | None:
        return self.session.get(ResearchAgentRun, run_id)

    def get_run_for_artifact(self, artifact_id: str) -> ResearchAgentRun | None:
        statement = (
            select(ResearchAgentRun)
            .where(ResearchAgentRun.artifact_id == artifact_id)
            .order_by(ResearchAgentRun.created_at.desc(), ResearchAgentRun.id.desc())
            .limit(1)
        )
        return self.session.execute(statement).scalar_one_or_none()

    def mark_running(self, run_id: str) -> ResearchAgentRun:
        run = self._require_run(run_id)
        run.status = "running"
        run.started_at = _utc_now()
        run.finished_at = None
        run.error_message = None
        self.session.commit()
        self.session.refresh(run)
        return run

    def mark_finished(
        self,
        run_id: str,
        *,
        status: str,
        model_name: str | None = None,
        error_message: str | None = None,
    ) -> ResearchAgentRun:
        run = self._require_run(run_id)
        run.status = status
        if model_name is not None:
            run.model_name = model_name
        run.error_message = error_message
        run.finished_at = _utc_now()
        self.session.commit()
        self.session.refresh(run)
        return run

    def append_step(
        self,
        *,
        run_id: str,
        step_type: str,
        label: str,
        status: str = "completed",
        input_json: dict[str, Any] | None = None,
        output_json: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> ResearchAgentStep:
        ordinal = self._next_step_ordinal(run_id)
        step = ResearchAgentStep(
            run_id=run_id,
            ordinal=ordinal,
            step_type=step_type,
            label=label,
            status=status,
            input_json=input_json or {},
            output_json=output_json or {},
            error_message=error_message,
        )
        self.session.add(step)
        self.session.commit()
        self.session.refresh(step)
        return step

    def list_steps(self, *, run_id: str) -> Sequence[ResearchAgentStep]:
        statement = (
            select(ResearchAgentStep)
            .where(ResearchAgentStep.run_id == run_id)
            .order_by(ResearchAgentStep.ordinal.asc(), ResearchAgentStep.created_at.asc())
        )
        return self.session.execute(statement).scalars().all()

    def create_context_pack(
        self,
        *,
        run_id: str,
        collection_id: str,
        workspace_id: str | None,
        task_type: str,
        context_json: dict[str, Any],
        selected_item_counts: dict[str, Any],
        readiness_warnings: list[str],
        cache_key: str | None = None,
    ) -> StudyContextPack:
        context_pack = StudyContextPack(
            run_id=run_id,
            collection_id=collection_id,
            workspace_id=workspace_id,
            task_type=task_type,
            cache_key=cache_key,
            context_json=context_json,
            selected_item_counts_json=selected_item_counts,
            readiness_warnings_json=readiness_warnings,
        )
        self.session.add(context_pack)
        self.session.commit()
        self.session.refresh(context_pack)
        return context_pack

    def get_context_pack(self, *, run_id: str) -> StudyContextPack | None:
        statement = (
            select(StudyContextPack)
            .where(StudyContextPack.run_id == run_id)
            .order_by(StudyContextPack.created_at.desc(), StudyContextPack.id.desc())
            .limit(1)
        )
        return self.session.execute(statement).scalar_one_or_none()

    def create_validation_report(
        self,
        *,
        run_id: str,
        artifact_id: str,
        harness_status: str,
        missing_evidence: list[str],
        unsupported_claims: list[str],
        readiness_blockers: list[str],
        report_json: dict[str, Any],
    ) -> ResearchValidationReport:
        report = ResearchValidationReport(
            run_id=run_id,
            artifact_id=artifact_id,
            harness_status=harness_status,
            missing_evidence_json=missing_evidence,
            unsupported_claims_json=unsupported_claims,
            readiness_blockers_json=readiness_blockers,
            report_json=report_json,
        )
        self.session.add(report)
        self.session.commit()
        self.session.refresh(report)
        return report

    def get_validation_report(self, *, run_id: str) -> ResearchValidationReport | None:
        statement = (
            select(ResearchValidationReport)
            .where(ResearchValidationReport.run_id == run_id)
            .order_by(ResearchValidationReport.created_at.desc(), ResearchValidationReport.id.desc())
            .limit(1)
        )
        return self.session.execute(statement).scalar_one_or_none()

    def _require_run(self, run_id: str) -> ResearchAgentRun:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"No research agent run found for id: {run_id}")
        return run

    def _next_step_ordinal(self, run_id: str) -> int:
        steps = self.session.execute(
            select(ResearchAgentStep.ordinal).where(ResearchAgentStep.run_id == run_id)
        ).scalars().all()
        return (max(steps) + 1) if steps else 0


class PaperSourceRepository:
    """Persistence helpers for provider-specific source records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_provider_record(self, *, provider: str, provider_record_id: str) -> PaperSource | None:
        statement: Select[tuple[PaperSource]] = select(PaperSource).where(
            PaperSource.provider == provider,
            PaperSource.provider_record_id == provider_record_id,
        )
        return self.session.execute(statement).scalar_one_or_none()

    def list_for_paper(self, paper_id: str) -> Sequence[PaperSource]:
        statement: Select[tuple[PaperSource]] = (
            select(PaperSource)
            .where(PaperSource.paper_id == paper_id)
            .order_by(PaperSource.created_at.asc(), PaperSource.provider.asc())
        )
        return self.session.execute(statement).scalars().all()

    def upsert(
        self,
        *,
        paper_id: str,
        provider: str,
        provider_record_id: str,
        source_payload: dict[str, Any] | None = None,
        is_primary: bool = False,
    ) -> PaperSource:
        source = self.get_by_provider_record(
            provider=provider,
            provider_record_id=provider_record_id,
        )
        if source is None:
            source = PaperSource(
                paper_id=paper_id,
                provider=provider,
                provider_record_id=provider_record_id,
                is_primary=is_primary,
                source_payload=source_payload or {},
            )
            self.session.add(source)
        else:
            source.paper_id = paper_id
            source.is_primary = is_primary
            source.source_payload = source_payload or source.source_payload

        self.session.commit()
        self.session.refresh(source)
        return source


class PaperFileRepository:
    """Persistence helpers for local and remote paper file records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_storage_uri(self, *, paper_id: str, storage_uri: str) -> PaperFile | None:
        statement: Select[tuple[PaperFile]] = select(PaperFile).where(
            PaperFile.paper_id == paper_id,
            PaperFile.storage_uri == storage_uri,
        )
        return self.session.execute(statement).scalar_one_or_none()

    def list_for_paper(self, *, paper_id: str, file_kind: str | None = None) -> Sequence[PaperFile]:
        statement = select(PaperFile).where(PaperFile.paper_id == paper_id)
        if file_kind is not None:
            statement = statement.where(PaperFile.file_kind == file_kind)
        statement = statement.order_by(PaperFile.created_at.asc())
        return self.session.execute(statement).scalars().all()

    def upsert(
        self,
        *,
        paper_id: str,
        storage_uri: str,
        file_kind: str = "pdf",
        content_hash: str | None = None,
        mime_type: str | None = None,
        parser_status: str | None = None,
    ) -> PaperFile:
        file_record = self.get_by_storage_uri(paper_id=paper_id, storage_uri=storage_uri)
        if file_record is None:
            file_record = PaperFile(
                paper_id=paper_id,
                storage_uri=storage_uri,
                file_kind=file_kind,
                content_hash=content_hash,
                mime_type=mime_type,
                parser_status=parser_status,
            )
            self.session.add(file_record)
        else:
            file_record.file_kind = file_kind
            file_record.content_hash = content_hash
            file_record.mime_type = mime_type
            file_record.parser_status = parser_status

        self.session.commit()
        self.session.refresh(file_record)
        return file_record
