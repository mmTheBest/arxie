# Arxie Platform PRD

Date: 2026-04-14
Status: Draft v1

## Product Summary

Arxie is a research workspace for scientific literature. The merged product combines:

- the existing **Arxie assistant layer** for search, Q&A, literature review, chat, and proposal workflows
- a new **Paperbase platform layer** for persistent corpus storage, parsing, extraction, indexing, and comparison

Users should experience one product. Internally, the product has two responsibilities:

- **assistant workflows**: ask, compare, explain, draft, and iterate
- **paperbase workflows**: ingest, normalize, parse, extract, index, and preserve evidence

## Problem

Current AI research assistants have a structural weakness: they operate at request time.

They can search a few sources, summarize some papers, and produce plausible answers, but they usually cannot:

- maintain a durable paper corpus
- compare papers through structured result rows
- store extraction outputs with provenance
- expose figures, tables, datasets, and methods as reusable entities
- support long-lived analyst or proposal workflows without repeated re-discovery

Current Arxie already solves important user-facing problems:

- grounded Q&A
- literature review mode
- interactive chat
- proposal workspace
- evidence mapping

But it still leans on live retrieval and lightweight local caching. The merged product extends Arxie into a persistent research operating system.

## Product Vision

Arxie becomes the main product surface for working with research.

Paperbase becomes the internal platform that powers:

- durable paper memory
- hybrid search
- structured extraction
- figure and table access
- result comparison
- Codex-operable research workflows

The user does not need to think in terms of "Arxie vs Paperbase." They should experience one coherent workflow from search to evidence to output.

## Target Users

- researchers exploring a technical domain
- startup teams doing technical diligence
- analysts creating literature-backed reports
- research engineers comparing models, datasets, and results
- agents such as Codex that need schema-aware access to paper corpora

## Goals

- preserve Arxie's current strengths in interactive assistant workflows
- add a persistent paperbase that stores and indexes literature over time
- extract research fundamentals into structured, queryable entities
- enable comparison workflows across papers, methods, datasets, metrics, and figures
- support proposal and report workflows with stronger evidence and reusable corpus memory

## Non-Goals for V1

- closed-access publisher ingestion beyond available/public content
- perfect OCR or chart digitization across all PDFs
- fully automated truth verification of scientific claims
- full multi-user collaboration, enterprise permissions, and billing
- replacing Arxie's current workflows before the Paperbase integration path is proven

## V1 Deployment Posture

V1 should be single-user and local-first.

That means:

- one user can run and curate their own Paperbase locally
- one user can maintain field-specific paper collections and annotations
- one user can run field-specific extraction over their own curated corpus

But the architecture should remain expandable to:

- multi-user ownership and sharing
- larger hosted scholarly databases
- additional provider-backed corpora

## Existing Arxie Capabilities To Preserve

Current Arxie already provides:

- search over Semantic Scholar and arXiv
- a unified retrieval abstraction
- PDF parsing with PyMuPDF and pdfplumber heuristics
- grounded Q&A with inline citation formatting
- literature review mode
- conversational mode
- proposal workspace, branching, export, and release-gate evaluation

These are not side features. They are the current product surface and should remain first-class.

## New Platform Capabilities To Add

### 1. Persistent corpus ingestion

- ingest papers by DOI, arXiv ID, OpenAlex ID, or batch source sync
- store canonical paper metadata and source provenance
- persist raw PDFs and parser artifacts
- deduplicate across multiple upstream providers

### 2. Structured paper understanding

For each paper, the system should be able to store:

- title, abstract, sections, and chunks
- figures, captions, and page positions
- methods
- datasets
- metrics
- benchmark results
- glossary terms
- key findings
- limitations
- engineering tricks or implementation details

### 3. Hybrid search and retrieval

- keyword search over titles, abstracts, chunks, and captions
- semantic retrieval over chunk and paper embeddings
- structured filters by date, venue, author, tag, method, dataset, metric, and extraction state

### 4. Evidence-backed comparison

- compare methods across paper sets
- compare results by dataset and metric
- summarize engineering tricks across a topic or method family
- fetch figures and tables for a comparison slice

### 5. Durable research workflows

- saved workspaces
- persistent evidence collections
- user-owned paper collections that behave like custom field-specific databases
- user annotations, notes, and tags on papers and derived artifacts
- collection-scoped extraction profiles for domain-specific structured fields
- proposal evidence reuse from the paperbase
- re-extraction and re-indexing when prompts, models, or schemas change

## Product Experience

From the user perspective, Arxie should support this sequence:

1. search or ask a question
2. inspect relevant papers
3. filter the corpus by metadata and extracted entities
4. compare papers through structured evidence
5. inspect figures, tables, and result rows
6. generate a lit review, memo, or proposal draft
7. return later and continue from the same research context

## Core User Stories

- As a researcher, I can search a persistent corpus and filter by dataset, method, and date.
- As an analyst, I can compare reported results across papers and inspect the evidence behind each row.
- As an engineer, I can extract figures and engineering tricks across a method family.
- As a proposal author, I can reuse paperbase evidence inside the Arxie proposal workflow.
- As a domain expert, I can maintain my own curated collection of papers and run analyses inside that collection.
- As a researcher, I can annotate papers, chunks, figures, and result rows with my own notes and tags.
- As a collection owner, I can define field-specific extraction expectations such as datasets, benchmark methods, engineering tricks, and experiment design.
- As Codex, I can query the paperbase schema safely without guessing the data model.

## Architecture Decision

This is the central decision:

- merge the **product**
- do not collapse everything into the current `src/ra/*` assistant package

Recommended architecture:

- **Arxie app layer** remains the user-facing CLI, API, chat, lit-review, and proposal surface
- **Paperbase platform layer** is added as a sibling set of services/packages in the same repo

That means:

- Arxie depends on Paperbase
- Paperbase does not become a submodule inside `src/ra`

## Repo Direction

Implementation should move into the existing `arxie` repository because:

- it already has the real git remote and collaboration path
- it is the active product surface
- the merged product decision makes the Arxie repo the natural planning home

However, the code layout inside that repo should be layered. The Paperbase code should live in top-level sibling packages/services, not inside the current assistant package tree.

## Recommended Technical Architecture

### Arxie layer

- `src/ra/agents/`
- `src/ra/api/`
- `src/ra/proposal/`
- `src/ra/cli.py`

Responsibilities:

- interactive search and question answering
- lit review generation
- proposal workflow
- chat and session UX
- orchestration over paperbase data

### Paperbase layer

Responsibilities:

- source ingestion
- canonical storage
- parser pipelines
- LLM extraction
- Elasticsearch indexing
- comparison services
- Codex-facing skill and query helpers

Recommended components:

- PostgreSQL for canonical and relational storage
- Elasticsearch for search/read models
- object storage for PDFs and figure crops
- Redis or queue broker for asynchronous jobs
- worker pipeline for parse/extract/index tasks

## Data Model Summary

Canonical entities should include:

- `papers`
- `paper_sources`
- `paper_files`
- `authors`
- `paper_authors`
- `venues`
- `tags`
- `sections`
- `chunks`
- `figures`
- `datasets`
- `methods`
- `metrics`
- `results`
- `glossary_terms`
- `engineering_tricks`
- `findings`
- `evidence_spans`
- `extraction_runs`
- `review_tasks`
- `collections`
- `collection_papers`
- `annotations`
- `annotation_targets`
- `extraction_profiles`

Use normalized tables for stable comparison entities and JSONB for versioned extraction payloads and raw parser outputs.

## Integration Strategy

### Phase 1: Side-by-side operation

Arxie continues using live provider retrieval and current parsing where necessary.

Paperbase is introduced for:

- persistent ingest
- stored paper metadata
- chunk indexing
- extraction outputs

### Phase 2: Retrieval abstraction

Add a new Arxie retrieval backend:

- `PaperbaseRetriever`

Arxie query flows should:

- search Paperbase first
- fall back to live external sources second when needed

### Phase 3: Workflow deepening

Proposal, lit review, and comparison flows should increasingly use:

- stored evidence spans
- normalized result rows
- extracted figures and tables
- durable workspace state linked to paperbase entities

## Codex and Agent Operations

The platform should expose safe, documented workflows for agents such as Codex:

- search corpus slices
- query structured comparison tables
- fetch paper figures and captions
- rerun extraction for selected papers
- summarize engineering tricks with provenance

Recommended project skills:

- `paperbase-postgres`
- `paperbase-elasticsearch`
- `paperbase-corpus-ops`

## Success Metrics

### User-facing

- time from research question to usable comparison table
- literature review quality and citation trust
- proposal evidence reuse rate

### Platform-facing

- ingest success rate
- parse success rate
- extraction schema validity rate
- evidence attachment rate
- search precision at top 10
- comparison accuracy on labeled benchmarks

## Delivery Phases

### Phase A

- planning and repo migration into Arxie docs
- layered repo structure
- infrastructure bootstrap

### Phase B

- canonical storage and ingestion
- basic chunk indexing
- paper lookup from Arxie into Paperbase

### Phase C

- structured extraction
- result comparison
- figure and table access
- Paperbase-backed Arxie workflows

### Phase D

- Codex skills
- evaluation dashboards
- stronger report/proposal integrations

## References

Local source repo examined for current product shape:

- `/Users/mm/Projects/academic-research-assistant/README.md`
- `/Users/mm/Projects/academic-research-assistant/API.md`
- `/Users/mm/Projects/academic-research-assistant/src/ra/retrieval/unified.py`
- `/Users/mm/Projects/academic-research-assistant/src/ra/agents/research_agent.py`
- `/Users/mm/Projects/academic-research-assistant/src/ra/parsing/pdf_parser.py`
- `/Users/mm/Projects/academic-research-assistant/src/ra/api/app.py`
