# Ingest And Extraction

## Pipeline

The ingest pipeline should turn a paper identifier or PDF into persistent structured records.

Stages:

1. seed metadata from providers
2. fetch/store PDF
3. parse sections and chunks
4. extract schema-constrained entities
5. attach provenance spans
6. index searchable read models

For local-first operation, the first seed path can be a user-owned PDF directory. That importer should register each filesystem paper in the canonical schema, attach file records, and group them into a curated collection before any deeper parsing or extraction runs happen.

## Parser Strategy

Use the current `src/ra/parsing/pdf_parser.py` heuristics as the phase-1 parser adapter.

Later parser upgrades such as GROBID or PDFFigures2 should plug into the same pipeline contract rather than forcing a redesign.

## Field-Specific Extraction

Extraction should support collection-specific profiles so different research fields can ask for different structured outputs.
