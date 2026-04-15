"""Table extraction contracts for Paperbase."""

from paperbase.tables.models import TableCandidate
from paperbase.tables.pipeline import TableExtractionPipeline, TableExtractionResult

__all__ = ["TableCandidate", "TableExtractionPipeline", "TableExtractionResult"]
