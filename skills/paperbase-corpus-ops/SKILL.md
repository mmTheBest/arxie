---
name: paperbase-corpus-ops
description: Use when importing a local paper collection, parsing PDFs, running structured extraction, or preparing a field-specific Paperbase corpus for Arxie workflows, and you need the current local-first operating sequence for collections, parses, and extraction runs.
---

# Paperbase Corpus Ops

## Overview

This skill covers the operator path for turning a folder of PDFs into a usable
Paperbase collection:

1. initialize the DB
2. import the local library into a collection
3. parse PDFs into sections and chunks
4. run collection extraction with a field-specific schema
5. verify the corpus before using it from Arxie

## When To Use

- onboarding a new paper collection
- refreshing a collection after adding PDFs
- parsing papers before extraction
- running domain-specific extraction over a curated corpus
- preparing a collection for proposal or lit-review workflows

## Quick Reference

Key modules:

- `paperbase.db.bootstrap.initialize_database`
- `paperbase.ingest.local_library.import_local_pdf_directory`
- `paperbase.parsing.pipeline.PaperParsePipeline`
- `paperbase.extract.runner.CollectionExtractionRunner`

Key tables to verify after each stage:

- import: `papers`, `paper_files`, `collection_papers`
- parse: `sections`, `chunks`
- extract: `extraction_runs`, `datasets`, `methods`, `metrics`, `result_rows`

## Operating Sequence

1. Initialize or migrate the DB before touching a collection.
2. Import the local PDF directory into a named collection.
3. Parse papers before extraction. The extraction runner can auto-parse missing
   papers, but explicit parse verification is easier to debug.
4. Run extraction with the collection’s field-specific schema.
5. Verify a few concrete papers and rows before treating the corpus as ready.

## Current Local-First Notes

- the default DB is `data/paperbase.db`
- local collections can be used directly by Arxie proposal and lit-review workflows
- current curated corpus examples should be treated as operator-owned, not shared
  multi-user state

## Common Mistakes

- importing PDFs and assuming extraction already happened
- running extraction without checking that sections exist
- changing schema payloads without recording the prompt/schema version used
- using ad hoc filesystem scans in assistant code instead of going through the
  collection and Paperbase DB
