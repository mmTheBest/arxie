# 2026-04-15 Paperbase Arxie Gateway Implementation

## Goal

Complete Task 11 from the April 14 platform implementation plan: make Arxie query
Paperbase first and use live providers second.

## Implemented

- Added `src/ra/retrieval/paperbase_gateway.py` as the read-only adapter from
  Arxie runtime code into the local Paperbase SQLite corpus.
- Added `src/ra/retrieval/runtime.py` to centralize construction of the default
  Paperbase-aware `UnifiedRetriever`.
- Updated `src/ra/retrieval/unified.py` to:
  - search Paperbase before external providers
  - resolve papers from Paperbase before DOI / arXiv / Semantic Scholar fallback
  - return stored Paperbase sections as full text before PDF download
- Updated `src/ra/tools/retrieval_tools.py` so `read_paper_fulltext` uses stored
  Paperbase sections when present and retains backward compatibility with older
  retriever doubles that do not implement the new hook.
- Wired the Paperbase-aware retriever into:
  - `src/ra/api/app.py`
  - `src/ra/agents/research_agent.py`
  - `src/ra/agents/lit_review_agent.py`
  - `src/ra/cli.py`

## Verification

- `pytest tests/unit/test_unified_retriever_paperbase.py -q`
- `pytest tests/unit/test_runtime_paperbase_wiring.py -q`
- `pytest tests/unit/test_unified_retriever_paperbase.py tests/unit/test_runtime_paperbase_wiring.py tests/unit/test_fulltext_tool.py tests/unit/test_research_agent.py tests/unit/test_lit_review_agent.py tests/unit/test_api_rest.py -q`

## Result

Arxie now behaves like an agent over a local research database instead of a purely
live retrieval wrapper. If the local Paperbase DB contains the requested paper or
full text, Arxie uses that corpus first. If not, the assistant still falls back
to Semantic Scholar and arXiv.

## Remaining Gaps Against April 14 Design

- Task 12: Paperbase-backed proposal and lit-review workflows are still partial.
- Search/read models still need live indexing, not just query builders.
- Full collection extraction and field-specific schema/annotation workflows for the
  user corpus still need to be operationalized end-to-end.
