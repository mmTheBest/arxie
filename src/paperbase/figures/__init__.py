"""Figure extraction contracts for Paperbase."""

from paperbase.figures.models import FigureCandidate
from paperbase.figures.pipeline import FigureExtractionPipeline, FigureExtractionResult

__all__ = ["FigureCandidate", "FigureExtractionPipeline", "FigureExtractionResult"]
