from __future__ import annotations

from ra.retrieval.arxiv import ArxivPaper
from ra.retrieval.semantic_scholar import Author as SemanticScholarAuthor
from ra.retrieval.semantic_scholar import Paper as SemanticScholarPaper

from paperbase.ingest.arxiv_seed import seed_from_arxiv_paper, seed_from_semantic_scholar_paper
from paperbase.ingest.crossref import seed_from_crossref_work
from paperbase.ingest.openalex import seed_from_openalex_work


def test_seed_from_arxiv_paper_maps_current_client_shape() -> None:
    paper = ArxivPaper(
        arxiv_id="2503.01682v1",
        title="scLong",
        abstract="Long-range gene context for single-cell transcriptomics.",
        authors=["Alice Smith", "Bob Jones"],
        published="2025-03-01T00:00:00Z",
        updated="2025-03-10T00:00:00Z",
        categories=["cs.LG", "q-bio.GN"],
        pdf_url="https://arxiv.org/pdf/2503.01682v1.pdf",
        doi="10.1234/scLong",
    )

    seed = seed_from_arxiv_paper(paper)

    assert seed.provider == "arxiv"
    assert seed.external_id == "2503.01682v1"
    assert seed.canonical_title == "scLong"
    assert seed.publication_year == 2025
    assert seed.doi == "10.1234/scLong"
    assert seed.authors == ["Alice Smith", "Bob Jones"]


def test_seed_from_semantic_scholar_paper_maps_existing_metadata_model() -> None:
    paper = SemanticScholarPaper(
        paper_id="S2-123",
        title="scRegNet paper",
        abstract="Structured extraction from single-cell GRN papers.",
        year=2026,
        authors=[SemanticScholarAuthor(author_id="A1", name="Alice Smith")],
        venue="Nature",
        citation_count=14,
        is_open_access=True,
        pdf_url="https://example.org/scRegNet.pdf",
        doi="10.5555/scregnet",
        arxiv_id=None,
        external_ids={"DOI": "10.5555/scregnet"},
    )

    seed = seed_from_semantic_scholar_paper(paper)

    assert seed.provider == "semantic_scholar"
    assert seed.external_id == "S2-123"
    assert seed.canonical_title == "scRegNet paper"
    assert seed.venue == "Nature"
    assert seed.pdf_url == "https://example.org/scRegNet.pdf"
    assert seed.authors == ["Alice Smith"]


def test_seed_from_openalex_work_normalizes_title_abstract_and_ids() -> None:
    work = {
        "id": "https://openalex.org/W123",
        "title": "Novae",
        "publication_year": 2025,
        "ids": {"doi": "https://doi.org/10.1038/s41467-025-58699-1"},
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Jones"}},
        ],
        "primary_location": {
            "source": {"display_name": "Nature Communications"},
            "pdf_url": "https://example.org/novae.pdf",
        },
        "abstract_inverted_index": {"Novae": [0], "models": [1], "space": [2]},
    }

    seed = seed_from_openalex_work(work)

    assert seed.provider == "openalex"
    assert seed.external_id == "https://openalex.org/W123"
    assert seed.canonical_title == "Novae"
    assert seed.abstract == "Novae models space"
    assert seed.venue == "Nature Communications"
    assert seed.doi == "10.1038/s41467-025-58699-1"


def test_seed_from_crossref_work_extracts_pdf_link_and_authors() -> None:
    work = {
        "DOI": "10.1038/s41586-024-08391-z",
        "title": ["scFoundation"],
        "abstract": "A Nature paper about foundation models.",
        "container-title": ["Nature"],
        "published-print": {"date-parts": [[2024, 12, 1]]},
        "author": [
            {"given": "Alice", "family": "Smith"},
            {"given": "Bob", "family": "Jones"},
        ],
        "link": [
            {"content-type": "application/pdf", "URL": "https://example.org/scfoundation.pdf"},
        ],
    }

    seed = seed_from_crossref_work(work)

    assert seed.provider == "crossref"
    assert seed.external_id == "10.1038/s41586-024-08391-z"
    assert seed.publication_year == 2024
    assert seed.venue == "Nature"
    assert seed.pdf_url == "https://example.org/scfoundation.pdf"
    assert seed.authors == ["Alice Smith", "Bob Jones"]
