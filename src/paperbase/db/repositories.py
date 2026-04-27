"""Repository interfaces for Paperbase persistence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Select, delete, select
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
    PaperSource,
    PaperTag,
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
