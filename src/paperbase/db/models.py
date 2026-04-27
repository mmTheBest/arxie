"""Canonical SQLAlchemy models for the Paperbase platform."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid_str() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base for Paperbase models."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=_utc_now,
        onupdate=_utc_now,
        nullable=False,
    )


class Paper(Base, TimestampMixin):
    __tablename__ = "papers"
    __table_args__ = (UniqueConstraint("provider", "external_id", name="uq_papers_provider_external"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    canonical_title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text)
    publication_year: Mapped[int | None] = mapped_column(Integer)
    venue_id: Mapped[str | None] = mapped_column(ForeignKey("venues.id"))
    venue: Mapped[str | None] = mapped_column(String(255))
    doi: Mapped[str | None] = mapped_column(String(255))
    arxiv_id: Mapped[str | None] = mapped_column(String(128))
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Venue(Base, TimestampMixin):
    __tablename__ = "venues"
    __table_args__ = (UniqueConstraint("normalized_name", name="uq_venues_normalized_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)


class Author(Base, TimestampMixin):
    __tablename__ = "authors"
    __table_args__ = (UniqueConstraint("normalized_name", name="uq_authors_normalized_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)


class PaperAuthor(Base, TimestampMixin):
    __tablename__ = "paper_authors"
    __table_args__ = (UniqueConstraint("paper_id", "author_id", name="uq_paper_authors_paper_author"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    author_id: Mapped[str] = mapped_column(ForeignKey("authors.id"), nullable=False, index=True)
    ordinal: Mapped[int | None] = mapped_column(Integer)


class Tag(Base, TimestampMixin):
    __tablename__ = "tags"
    __table_args__ = (UniqueConstraint("normalized_name", name="uq_tags_normalized_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)


class PaperTag(Base, TimestampMixin):
    __tablename__ = "paper_tags"
    __table_args__ = (UniqueConstraint("paper_id", "tag_id", name="uq_paper_tags_paper_tag"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    tag_id: Mapped[str] = mapped_column(ForeignKey("tags.id"), nullable=False, index=True)


class PaperSource(Base, TimestampMixin):
    __tablename__ = "paper_sources"
    __table_args__ = (
        UniqueConstraint("provider", "provider_record_id", name="uq_paper_sources_provider_record"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_record_id: Mapped[str] = mapped_column(String(255), nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class PaperFile(Base, TimestampMixin):
    __tablename__ = "paper_files"
    __table_args__ = (UniqueConstraint("paper_id", "storage_uri", name="uq_paper_files_paper_storage"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    file_kind: Mapped[str] = mapped_column(String(64), nullable=False, default="pdf")
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(128))
    mime_type: Mapped[str | None] = mapped_column(String(128))
    parser_status: Mapped[str | None] = mapped_column(String(64))


class Section(Base, TimestampMixin):
    __tablename__ = "sections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)


class Chunk(Base, TimestampMixin):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    section_id: Mapped[str | None] = mapped_column(ForeignKey("sections.id"))
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding_status: Mapped[str | None] = mapped_column(String(64))


class Figure(Base, TimestampMixin):
    __tablename__ = "figures"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    page_number: Mapped[int | None] = mapped_column(Integer)
    figure_label: Mapped[str | None] = mapped_column(String(128))
    caption: Mapped[str | None] = mapped_column(Text)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    bbox_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class TableArtifact(Base, TimestampMixin):
    __tablename__ = "tables"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    page_number: Mapped[int | None] = mapped_column(Integer)
    table_label: Mapped[str | None] = mapped_column(String(128))
    caption: Mapped[str | None] = mapped_column(Text)
    storage_uri: Mapped[str | None] = mapped_column(Text)
    bbox_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    structured_payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Method(Base, TimestampMixin):
    __tablename__ = "methods"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Metric(Base, TimestampMixin):
    __tablename__ = "metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class ResultRow(Base, TimestampMixin):
    __tablename__ = "result_rows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.id"))
    method_id: Mapped[str | None] = mapped_column(ForeignKey("methods.id"))
    metric_id: Mapped[str | None] = mapped_column(ForeignKey("metrics.id"))
    split_name: Mapped[str | None] = mapped_column(String(128))
    value_numeric: Mapped[float | None] = mapped_column(Float)
    value_text: Mapped[str | None] = mapped_column(String(128))
    comparator_text: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)


class GlossaryTerm(Base, TimestampMixin):
    __tablename__ = "glossary_terms"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    term: Mapped[str] = mapped_column(String(255), nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Finding(Base, TimestampMixin):
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    polarity: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class Limitation(Base, TimestampMixin):
    __tablename__ = "limitations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class EngineeringTrick(Base, TimestampMixin):
    __tablename__ = "engineering_tricks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class ExtractionProfile(Base, TimestampMixin):
    __tablename__ = "extraction_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, default="local-user")
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    scope_type: Mapped[str] = mapped_column(String(64), nullable=False, default="private")
    schema_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class ExtractionRun(Base, TimestampMixin):
    __tablename__ = "extraction_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    extraction_profile_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_profiles.id"))
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="pending")
    diagnostics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class EvidenceSpan(Base, TimestampMixin):
    __tablename__ = "evidence_spans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    extraction_run_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_runs.id"))
    section_id: Mapped[str | None] = mapped_column(ForeignKey("sections.id"))
    chunk_id: Mapped[str | None] = mapped_column(ForeignKey("chunks.id"))
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(36))
    page_number: Mapped[int | None] = mapped_column(Integer)
    quote_text: Mapped[str | None] = mapped_column(Text)
    start_char: Mapped[int | None] = mapped_column(Integer)
    end_char: Mapped[int | None] = mapped_column(Integer)


class Collection(Base, TimestampMixin):
    __tablename__ = "collections"
    __table_args__ = (UniqueConstraint("owner_id", "title", name="uq_collections_owner_title"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, default="local-user")
    scope_type: Mapped[str] = mapped_column(String(64), nullable=False, default="private")
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    extraction_profile_id: Mapped[str | None] = mapped_column(ForeignKey("extraction_profiles.id"))
    tags_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)


class CollectionPaper(Base, TimestampMixin):
    __tablename__ = "collection_papers"
    __table_args__ = (
        UniqueConstraint("collection_id", "paper_id", name="uq_collection_papers_collection_paper"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"), nullable=False, index=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), nullable=False, index=True)
    position: Mapped[int | None] = mapped_column(Integer)
    membership_note: Mapped[str | None] = mapped_column(Text)


class Workspace(Base, TimestampMixin):
    __tablename__ = "workspaces"
    __table_args__ = (UniqueConstraint("owner_id", "title", name="uq_workspaces_owner_title"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, default="local-user")
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    collection_id: Mapped[str | None] = mapped_column(ForeignKey("collections.id"))
    saved_query: Mapped[str | None] = mapped_column(Text)
    focus_note: Mapped[str | None] = mapped_column(Text)
    active_filters_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    pinned_paper_ids_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)


class Annotation(Base, TimestampMixin):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    author_id: Mapped[str] = mapped_column(String(128), nullable=False, default="local-user")
    collection_id: Mapped[str | None] = mapped_column(ForeignKey("collections.id"))
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    tags_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str | None] = mapped_column(String(64))


class BackgroundJob(Base, TimestampMixin):
    __tablename__ = "background_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="pending", index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
