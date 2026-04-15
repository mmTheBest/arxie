"""Ingest entrypoints for Paperbase."""

from paperbase.ingest.arxiv_seed import seed_from_arxiv_paper
from paperbase.ingest.crossref import seed_from_crossref_work
from paperbase.ingest.local_library import LocalLibraryImportResult, import_local_pdf_directory
from paperbase.ingest.models import CanonicalPaperSeed
from paperbase.ingest.openalex import seed_from_openalex_work

__all__ = [
    "CanonicalPaperSeed",
    "LocalLibraryImportResult",
    "import_local_pdf_directory",
    "seed_from_arxiv_paper",
    "seed_from_crossref_work",
    "seed_from_openalex_work",
]
