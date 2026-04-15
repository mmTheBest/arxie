"""Repository interfaces for Paperbase persistence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import Select, delete, select
from sqlalchemy.orm import Session

from paperbase.db.models import (
    Annotation,
    Author,
    Collection,
    CollectionPaper,
    ExtractionProfile,
    Paper,
    PaperAuthor,
    PaperFile,
    PaperTag,
    Tag,
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
        if paper is None:
            paper = Paper(
                provider=provider,
                external_id=external_id,
                canonical_title=canonical_title,
                abstract=abstract,
                publication_year=publication_year,
                venue=venue,
                doi=doi,
                arxiv_id=arxiv_id,
                raw_metadata=raw_metadata or {},
            )
            self.session.add(paper)
        else:
            paper.canonical_title = canonical_title
            paper.abstract = abstract
            paper.publication_year = publication_year
            paper.venue = venue
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
