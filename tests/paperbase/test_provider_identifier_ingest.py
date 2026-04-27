from __future__ import annotations

from pathlib import Path

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import Paper, PaperFile, PaperSource
from paperbase.db.repositories import CollectionRepository
from paperbase.db.session import make_session_factory
from paperbase.ingest.models import CanonicalPaperSeed
from paperbase.ingest.provider_identifiers import IdentifierInput, ingest_provider_identifiers


class FakeProviderResolver:
    def __init__(self, seeds: dict[tuple[str, str], CanonicalPaperSeed]) -> None:
        self.seeds = seeds

    def fetch_identifier(self, *, kind: str, value: str) -> CanonicalPaperSeed:
        return self.seeds[(kind, value)]


def test_provider_identifier_ingest_merges_sources_by_doi_and_attaches_collection(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    resolver = FakeProviderResolver(
        {
            (
                "doi",
                "10.1038/s41467-025-58699-1",
            ): CanonicalPaperSeed(
                provider="crossref",
                external_id="10.1038/s41467-025-58699-1",
                canonical_title="scRegNet Crossref record",
                abstract="Crossref abstract",
                publication_year=2025,
                venue="Nature Communications",
                doi="10.1038/s41467-025-58699-1",
                pdf_url="https://example.org/crossref.pdf",
                authors=["Alice Smith", "Bob Jones"],
                source_payload={"source": "crossref"},
                raw_metadata={"publisher": "Nature"},
            ),
            (
                "openalex",
                "https://openalex.org/W123",
            ): CanonicalPaperSeed(
                provider="openalex",
                external_id="https://openalex.org/W123",
                canonical_title="scRegNet OpenAlex record",
                abstract="OpenAlex abstract",
                publication_year=2025,
                venue="Nature Communications",
                doi="10.1038/s41467-025-58699-1",
                pdf_url="https://example.org/openalex.pdf",
                authors=["Alice Smith", "Bob Jones"],
                source_payload={"source": "openalex"},
                raw_metadata={"cited_by_count": 42},
            ),
        }
    )

    result = ingest_provider_identifiers(
        identifiers=[
            IdentifierInput(kind="doi", value="10.1038/s41467-025-58699-1"),
            IdentifierInput(kind="openalex", value="https://openalex.org/W123"),
        ],
        session_factory=session_factory,
        resolver=resolver,
        owner_id="local-user",
        collection_title="scRegNet provider import",
        collection_description="Imported from provider identifiers.",
    )

    assert result.collection_title == "scRegNet provider import"
    assert result.requested_count == 2
    assert result.imported_papers == 1
    assert result.reused_papers == 1
    assert len(result.paper_ids) == 2
    assert result.paper_ids[0] == result.paper_ids[1]
    assert result.skipped_identifiers == []

    with session_factory() as session:
        papers = session.query(Paper).all()
        sources = session.query(PaperSource).all()
        files = session.query(PaperFile).all()
        collections = CollectionRepository(session).list_collections()
        memberships = CollectionRepository(session).list_papers(collections[0].id)

    assert len(papers) == 1
    assert papers[0].canonical_title == "scRegNet OpenAlex record"
    assert papers[0].doi == "10.1038/s41467-025-58699-1"
    assert len(sources) == 2
    assert sorted(source.provider for source in sources) == ["crossref", "openalex"]
    assert len(files) == 2
    assert sorted(file.storage_uri for file in files) == [
        "https://example.org/crossref.pdf",
        "https://example.org/openalex.pdf",
    ]
    assert len(collections) == 1
    assert collections[0].title == "scRegNet provider import"
    assert [membership.paper_id for membership in memberships] == [papers[0].id]
