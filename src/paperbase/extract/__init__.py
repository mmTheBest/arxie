"""Structured extraction interfaces and pipelines for Paperbase."""

from paperbase.extract.client import OpenAIExtractionClient
from paperbase.extract.contracts import GlossaryTermExtraction, StructuredExtractionBundle
from paperbase.extract.pipeline import PaperExtractionPipeline, PaperExtractionResult
from paperbase.extract.prompts import build_extraction_messages
from paperbase.extract.runner import CollectionExtractionRunner, CollectionExtractionSummary

__all__ = [
    "CollectionExtractionRunner",
    "CollectionExtractionSummary",
    "GlossaryTermExtraction",
    "OpenAIExtractionClient",
    "PaperExtractionPipeline",
    "PaperExtractionResult",
    "StructuredExtractionBundle",
    "build_extraction_messages",
]
