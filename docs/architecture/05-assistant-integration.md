# Assistant Integration

## Role Of `src/ra`

`src/ra` remains the assistant layer.

It queries Paperbase first and uses live provider retrieval second.

## Implemented Retrieval Path

- `src/ra/retrieval/paperbase_gateway.py` exposes a read-only adapter over the local
  Paperbase database.
- `src/ra/retrieval/runtime.py` provides the default runtime retriever builder.
- `UnifiedRetriever` now:
  - searches Paperbase before Semantic Scholar / arXiv
  - resolves paper ids from Paperbase before external lookups
  - returns stored Paperbase sections as full text before attempting PDF download
  - can resolve pinned Paperbase papers by id for workspace-scoped synthesis
  - can load saved workspace context from the Paperbase DB when the runtime
    gateway supports it
- `read_paper_fulltext` in `src/ra/tools/retrieval_tools.py` uses stored sections
  when available, then falls back to PDF download and parse.
- Runtime entry points now construct a Paperbase-aware retriever:
  - `ra.api.app`
  - `ra.agents.research_agent`
  - `ra.agents.lit_review_agent`
  - `ra.cli`

## Why This Split Exists

The gateway is intentionally read-only and tolerant of missing local state.

- If the local Paperbase DB exists and has matching papers, Arxie uses it.
- If the DB is missing, empty, or does not have a match, Arxie falls back to the
  live provider clients without failing closed.

This keeps the assistant usable during local-first development while still moving
the product toward a database-backed workflow.

## Current Workspace Integration

Saved workspaces are no longer only a UI concern.

- `LitReviewRequest` can now take `workspace_id` and reuse the workspace
  collection, saved query, focus note, and pinned papers
- `ProposalEvidenceQueryRequest` can now take `workspace_id` and reuse the
  workspace collection plus pinned-paper defaults for evidence bucketing
- `LitReviewAgent` can seed synthesis with pinned papers before broader
  collection or live search retrieval, so workspace state changes the actual
  evidence surface instead of only decorating the UI

This is the current V1 realization of “return later and continue from the same
research context.” The assistant still remains tolerant of missing Paperbase
state, but when a workspace exists it can now drive the retrieval defaults
directly.

## Remaining Integration Work

The biggest remaining assistant-facing gap is deeper structured-evidence use:

- proposal and lit-review generation should increasingly consume stored result
  rows, figures, tables, and evidence spans directly instead of only paper
  titles/abstracts/fulltext sections
- the chat and answer flows should eventually accept workspace-scoped execution
  context too, not just the lit-review and proposal evidence paths
