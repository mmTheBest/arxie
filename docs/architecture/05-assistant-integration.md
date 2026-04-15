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

## Remaining Integration Work

Task 11 covers retrieval and fulltext access. The next integration layer is Task 12:

- proposal evidence queries should consume Paperbase-backed structured facts
- lit-review synthesis should explicitly use Paperbase collections, annotations, and
  extraction profiles instead of only the generic retrieval tools

## Workspace Implication

The new saved-workspace UI is intentionally a product layer, not a second assistant
stack.

- workspaces store durable research context over Paperbase collections
- the assistant layer should eventually read that context directly when a user asks
  Arxie to continue a prior investigation
- pinned papers, saved queries, and focus notes are now part of the expected
  user-facing state model even though the CLI/agent layer does not consume them yet
