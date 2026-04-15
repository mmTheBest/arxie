from __future__ import annotations

from ra.retrieval.arxiv import ArxivPaper
from ra.retrieval.semantic_scholar import Author, Paper as SemanticScholarPaper

from paperbase.ingest.arxiv_seed import seed_from_arxiv_paper
from paperbase.ingest.crossref import seed_from_crossref_work
from paperbase.ingest.openalex import seed_from_openalex_work


def test_openalex_adapter_maps_work_to_canonical_seed() -> None:
    payload = {
        "id": "https://openalex.org/W12345",
        "title": "scLong for Gene Context Modeling",
        "abstract_inverted_index": {
            "scLong": [0],
            "models": [1],
            "gene": [2],
            "context": [3],
        },
        "publication_year": 2026,
        "primary_location": {
            "source": {
                "display_name": "Nature",
            }
        },
        "ids": {
            "doi": "https://doi.org/10.1234/sclong.2026.1",
        },
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Lee"}},
        ],
    }

    seed = seed_from_openalex_work(payload)

    assert seed.provider == "openalex"
    assert seed.external_id == "https://openalex.org/W12345"
    assert seed.canonical_title == "scLong for Gene Context Modeling"
    assert seed.abstract == "scLong models gene context"
    assert seed.publication_year == 2026
    assert seed.venue == "Nature"
    assert seed.doi == "10.1234/sclong.2026.1"
    assert seed.authors == ["Alice Smith", "Bob Lee"]


def test_crossref_adapter_maps_work_to_canonical_seed() -> None:
    payload = {
        "DOI": "10.5555/crossref.2026.7",
        "title": ["Gene Regulatory Networks at Scale"],
        "abstract": "<jats:p>Structured extraction for regulatory biology.</jats:p>",
        "published-print": {"date-parts": [[2026, 4, 1]]},
        "container-title": ["Nature Biotechnology"],
        "author": [
            {"given": "Mya", "family": "Zhang"},
            {"name": "Research Team"},
        ],
    }

    seed = seed_from_crossref_work(payload)

    assert seed.provider == "crossref"
    assert seed.external_id == "10.5555/crossref.2026.7"
    assert seed.canonical_title == "Gene Regulatory Networks at Scale"
    assert seed.abstract == "Structured extraction for regulatory biology."
    assert seed.publication_year == 2026
    assert seed.venue == "Nature Biotechnology"
    assert seed.doi == "10.5555/crossref.2026.7"
    assert seed.authors == ["Mya Zhang", "Research Team"]


def test_arxiv_and_semantic_scholar_adapters_expose_canonical_seed_shape() -> None:
    arxiv_paper = ArxivPaper(
        arxiv_id="2503.01682v1",
        title="scLong- a billion-parameter foundation model",
        abstract="Long-range gene context modeling for single-cell transcriptomics.",
        authors=["Alice Smith", "Bob Lee"],
        published="2025-03-03T00:00:00Z",
        updated="2025-03-04T00:00:00Z",
        categories=["q-bio.GN"],
        pdf_url="https://arxiv.org/pdf/2503.01682v1.pdf",
        doi=None,
    )
    semantic_paper = SemanticScholarPaper(
        paper_id="S2-123",
        title="scRegNetBench",
        abstract="Benchmarking gene-regulatory-network models.",
        year=2026,
        authors=[Author(author_id="A1", name="Alice Smith")],
        venue="bioRxiv",
        citation_count=12,
        is_open_access=True,
        pdf_url="https://example.com/paper.pdf",
        doi="10.1000/example",
        arxiv_id="2503.01682",
    )

    arxiv_seed = seed_from_arxiv_paper(arxiv_paper)
    semantic_payload = semantic_paper.to_seed_payload()

    assert arxiv_seed.provider == "arxiv"
    assert arxiv_seed.external_id == "2503.01682v1"
    assert arxiv_seed.publication_year == 2025
    assert arxiv_seed.authors == ["Alice Smith", "Bob Lee"]

    assert semantic_payload["provider"] == "semantic_scholar"
    assert semantic_payload["external_id"] == "S2-123"
    assert semantic_payload["canonical_title"] == "scRegNetBench"
    assert semantic_payload["doi"] == "10.1000/example"
