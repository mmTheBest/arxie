# 2026-04-15 Paperbase Proposal And Lit-Review Integration

## Goal

Complete Task 12 from the April 14 platform implementation plan: make proposal
evidence queries and lit-review generation consume Paperbase-backed collection
data instead of requiring only inline ad hoc paper payloads.

## Implemented

- Extended `UnifiedRetriever` and `PaperbaseGateway` with collection-backed paper
  access.
- Added collection-aware lookup support to `LitReviewAgent.arun(..., collection_id=...)`
  and `LitReviewAgent.run(..., collection_id=...)`.
- Extended `ProposalEvidenceQueryRequest` with optional
  `paperbase_collection_id`.
- Updated `/api/proposal/evidence/query` to load papers from the shared retriever
  when a Paperbase collection id is provided, then map evidence on top of that
  curated corpus.

## Verification

- `pytest tests/unit/test_api_proposal_paperbase.py tests/unit/test_lit_review_agent.py -q`
- `pytest tests/unit/test_api_proposal_paperbase.py tests/unit/test_api_proposal_evidence.py tests/unit/test_api_rest.py tests/unit/test_lit_review_agent.py tests/unit/test_unified_retriever_paperbase.py -q`

## Result

Two assistant workflows now read directly from curated Paperbase collections:

- proposal evidence mapping
- literature review generation

This closes the main “agent over the database” gap for collection-scoped work.

## Remaining Gaps Against April 14 Design

- collection annotations and extraction profiles are stored, but not yet exposed as
  first-class workflow controls in the assistant/API surface
- Paperbase search/read models still need live indexing and compare-backed views
- runbooks and Codex skills from Task 13 are still open
