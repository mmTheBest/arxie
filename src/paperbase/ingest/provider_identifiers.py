"""Provider-backed ingest by DOI, arXiv ID, and OpenAlex identifier."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

import httpx
from sqlalchemy.orm import Session, sessionmaker

from paperbase.db.repositories import (
    CollectionRepository,
    PaperFileRepository,
    PaperRepository,
    PaperSourceRepository,
)
from paperbase.db.models import Collection, Paper
from paperbase.ingest.arxiv_seed import seed_from_arxiv_paper
from paperbase.ingest.crossref import seed_from_crossref_work
from paperbase.ingest.models import CanonicalPaperSeed
from paperbase.ingest.openalex import seed_from_openalex_work
from ra.retrieval.arxiv import ArxivClient
from ra.utils.security import validate_public_http_url


@dataclass(frozen=True, slots=True)
class IdentifierInput:
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class ProviderIdentifierIngestResult:
    requested_count: int
    imported_papers: int
    reused_papers: int
    paper_ids: list[str]
    skipped_identifiers: list[str]
    collection_id: str | None = None
    collection_title: str | None = None


@dataclass(frozen=True, slots=True)
class PaperMetadataRefreshResult:
    requested_count: int
    refreshed_papers: int
    skipped_paper_ids: list[str]


class ProviderResolver(Protocol):
    def fetch_identifier(self, *, kind: str, value: str) -> CanonicalPaperSeed:
        """Return a canonical ingest seed for a provider-backed identifier."""


class DefaultProviderResolver:
    """Fetch provider metadata from public scholarly endpoints."""

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def fetch_identifier(self, *, kind: str, value: str) -> CanonicalPaperSeed:
        if kind == "doi":
            return self._fetch_doi(value)
        if kind == "arxiv":
            return self._fetch_arxiv(value)
        if kind == "openalex":
            return self._fetch_openalex(value)
        raise ValueError(f"Unsupported identifier kind: {kind}")

    def _fetch_doi(self, doi: str) -> CanonicalPaperSeed:
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": "academic-research-assistant/1.0"}) as client:
            response = client.get(f"https://api.crossref.org/works/{quote(doi, safe='')}")
            response.raise_for_status()
            payload = response.json().get("message") or {}
        return seed_from_crossref_work(payload)

    def _fetch_openalex(self, openalex_id: str) -> CanonicalPaperSeed:
        normalized_id = openalex_id.strip()
        lookup_id = normalized_id.rstrip("/").rsplit("/", 1)[-1]
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": "academic-research-assistant/1.0"}) as client:
            response = client.get(f"https://api.openalex.org/works/{quote(lookup_id, safe='')}")
            response.raise_for_status()
            payload = response.json()
        return seed_from_openalex_work(payload)

    def _fetch_arxiv(self, arxiv_id: str) -> CanonicalPaperSeed:
        async def _load() -> CanonicalPaperSeed:
            client = ArxivClient()
            try:
                paper = await client.get_paper(arxiv_id)
            finally:
                await client.close()
            if paper is None:
                raise LookupError(f"No arXiv paper found for identifier: {arxiv_id}")
            return seed_from_arxiv_paper(paper)

        return asyncio.run(_load())


def _default_pdf_fetcher(storage_url: str) -> bytes:
    with httpx.Client(timeout=30.0, headers={"User-Agent": "academic-research-assistant/1.0"}) as client:
        response = client.get(storage_url)
        response.raise_for_status()
        return response.content


def _paper_object_key(*, paper_id: str, provider: str, content_hash: str) -> str:
    return f"papers/{paper_id}/{provider}-{content_hash}.pdf"


def ingest_provider_identifiers(
    *,
    identifiers: Sequence[IdentifierInput],
    session_factory: sessionmaker[Session],
    resolver: ProviderResolver | None = None,
    owner_id: str = "local-user",
    collection_id: str | None = None,
    collection_title: str | None = None,
    collection_description: str | None = None,
    object_store: object | None = None,
    pdf_fetcher=None,
) -> ProviderIdentifierIngestResult:
    """Ingest scholarly records by identifier and optionally attach them to a collection."""

    effective_resolver = resolver or DefaultProviderResolver()
    requested = list(identifiers)
    if not requested:
        return ProviderIdentifierIngestResult(
            requested_count=0,
            imported_papers=0,
            reused_papers=0,
            paper_ids=[],
            skipped_identifiers=[],
            collection_id=collection_id,
            collection_title=collection_title,
        )

    imported_papers = 0
    reused_papers = 0
    paper_ids: list[str] = []
    skipped_identifiers: list[str] = []

    with session_factory() as session:
        paper_repository = PaperRepository(session)
        paper_source_repository = PaperSourceRepository(session)
        paper_file_repository = PaperFileRepository(session)
        collection_repository = CollectionRepository(session)
        target_collection = _resolve_target_collection(
            collection_repository=collection_repository,
            collection_id=collection_id,
            collection_title=collection_title,
            collection_description=collection_description,
            owner_id=owner_id,
        )

        for position, identifier in enumerate(requested, start=1):
            identifier_key = f"{identifier.kind}:{identifier.value}"
            try:
                seed = effective_resolver.fetch_identifier(kind=identifier.kind, value=identifier.value)
                paper, created = _upsert_seed(
                    paper_repository=paper_repository,
                    paper_source_repository=paper_source_repository,
                    paper_file_repository=paper_file_repository,
                    seed=seed,
                    object_store=object_store,
                    pdf_fetcher=pdf_fetcher,
                )
                if target_collection is not None:
                    collection_repository.add_paper(
                        collection_id=target_collection.id,
                        paper_id=paper.id,
                        position=position,
                    )
            except Exception:  # noqa: BLE001
                skipped_identifiers.append(identifier_key)
                continue

            paper_ids.append(paper.id)
            if created:
                imported_papers += 1
            else:
                reused_papers += 1

        resolved_collection_id = target_collection.id if target_collection is not None else None
        resolved_collection_title = target_collection.title if target_collection is not None else None

    return ProviderIdentifierIngestResult(
        requested_count=len(requested),
        imported_papers=imported_papers,
        reused_papers=reused_papers,
        paper_ids=paper_ids,
        skipped_identifiers=skipped_identifiers,
        collection_id=resolved_collection_id,
        collection_title=resolved_collection_title,
    )


def refresh_paper_metadata(
    *,
    paper_ids: Sequence[str],
    session_factory: sessionmaker[Session],
    resolver: ProviderResolver | None = None,
    object_store: object | None = None,
    pdf_fetcher=None,
) -> PaperMetadataRefreshResult:
    """Refresh stored paper metadata from provider-backed identifiers."""

    effective_resolver = resolver or DefaultProviderResolver()
    requested_ids = list(paper_ids)
    refreshed_papers = 0
    skipped_paper_ids: list[str] = []

    with session_factory() as session:
        paper_repository = PaperRepository(session)
        paper_source_repository = PaperSourceRepository(session)
        paper_file_repository = PaperFileRepository(session)

        for paper_id in requested_ids:
            try:
                paper = paper_repository.get_by_id(paper_id)
                if paper is None:
                    raise ValueError(f"No paper found for id: {paper_id}")
                seed = _resolve_refresh_seed(
                    paper=paper,
                    paper_source_repository=paper_source_repository,
                    resolver=effective_resolver,
                )
                _upsert_seed(
                    paper_repository=paper_repository,
                    paper_source_repository=paper_source_repository,
                    paper_file_repository=paper_file_repository,
                    seed=seed,
                    object_store=object_store,
                    pdf_fetcher=pdf_fetcher,
                )
            except Exception:  # noqa: BLE001
                skipped_paper_ids.append(paper_id)
                continue

            refreshed_papers += 1

    return PaperMetadataRefreshResult(
        requested_count=len(requested_ids),
        refreshed_papers=refreshed_papers,
        skipped_paper_ids=skipped_paper_ids,
    )


def _resolve_target_collection(
    *,
    collection_repository: CollectionRepository,
    collection_id: str | None,
    collection_title: str | None,
    collection_description: str | None,
    owner_id: str,
) -> Collection | None:
    if collection_id is not None:
        collection = collection_repository.get_by_id(collection_id)
        if collection is None:
            raise ValueError(f"No collection found for id: {collection_id}")
        return collection
    if collection_title is not None:
        return collection_repository.create_or_get(
            owner_id=owner_id,
            title=collection_title,
            description=collection_description,
            tags=["provider-ingest"],
        )
    return None


def _upsert_seed(
    *,
    paper_repository: PaperRepository,
    paper_source_repository: PaperSourceRepository,
    paper_file_repository: PaperFileRepository,
    seed: CanonicalPaperSeed,
    object_store: object | None = None,
    pdf_fetcher=None,
) -> tuple[Paper, bool]:
    existing_source = paper_source_repository.get_by_provider_record(
        provider=seed.provider,
        provider_record_id=seed.external_id,
    )
    paper = None
    created = False

    if existing_source is not None:
        paper = paper_repository.get_by_id(existing_source.paper_id)
    if paper is None and seed.doi:
        paper = paper_repository.get_by_doi(seed.doi)
    if paper is None and seed.arxiv_id:
        paper = paper_repository.get_by_arxiv_id(seed.arxiv_id)
    if paper is None:
        paper = paper_repository.get_by_provider_id(seed.provider, seed.external_id)

    if paper is None:
        paper = paper_repository.upsert(
            provider=seed.provider,
            external_id=seed.external_id,
            canonical_title=seed.canonical_title,
            abstract=seed.abstract,
            publication_year=seed.publication_year,
            venue=seed.venue,
            doi=seed.doi,
            arxiv_id=seed.arxiv_id,
            raw_metadata=dict(seed.raw_metadata or {}),
            authors=list(seed.authors),
        )
        created = True
    else:
        paper = paper_repository.merge_metadata(
            paper.id,
            canonical_title=seed.canonical_title,
            abstract=seed.abstract,
            publication_year=seed.publication_year,
            venue=seed.venue,
            doi=seed.doi,
            arxiv_id=seed.arxiv_id,
            raw_metadata=dict(seed.raw_metadata or {}),
            authors=list(seed.authors) if seed.authors else None,
        )

    paper_source_repository.upsert(
        paper_id=paper.id,
        provider=seed.provider,
        provider_record_id=seed.external_id,
        source_payload=dict(seed.source_payload or {}),
        is_primary=(paper.provider == seed.provider and paper.external_id == seed.external_id),
    )

    if seed.pdf_url:
        validated_pdf_url = validate_public_http_url(seed.pdf_url, field_name="pdf_url")
        storage_uri = validated_pdf_url
        content_hash = None
        if object_store is not None:
            fetcher = pdf_fetcher or _default_pdf_fetcher
            pdf_bytes = fetcher(validated_pdf_url)
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            storage_uri = object_store.put_bytes(
                key=_paper_object_key(
                    paper_id=paper.id,
                    provider=seed.provider,
                    content_hash=content_hash,
                ),
                content=pdf_bytes,
                content_type="application/pdf",
            )

        paper_file_repository.upsert(
            paper_id=paper.id,
            storage_uri=storage_uri,
            file_kind="pdf",
            content_hash=content_hash,
            mime_type="application/pdf",
            parser_status="pending",
        )

    return paper, created


def _resolve_refresh_seed(
    *,
    paper: Paper,
    paper_source_repository: PaperSourceRepository,
    resolver: ProviderResolver,
) -> CanonicalPaperSeed:
    if paper.doi:
        return resolver.fetch_identifier(kind="doi", value=paper.doi)
    if paper.arxiv_id:
        return resolver.fetch_identifier(kind="arxiv", value=paper.arxiv_id)

    sources = list(paper_source_repository.list_for_paper(paper.id))
    for source in sources:
        if source.provider == "openalex":
            return resolver.fetch_identifier(kind="openalex", value=source.provider_record_id)
        if source.provider == "crossref":
            return resolver.fetch_identifier(kind="doi", value=source.provider_record_id)
        if source.provider == "arxiv":
            return resolver.fetch_identifier(kind="arxiv", value=source.provider_record_id)

    raise ValueError(f"No refreshable provider identifier found for paper_id={paper.id}")
