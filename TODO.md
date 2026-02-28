# RA Development TODO

## Current: Phase 3 — Evaluation (IN PROGRESS)
- [x] Eval harness (tests/eval/harness.py)
- [x] 50-question dataset (tests/eval/dataset.json)
- [x] Eval CLI command (ra eval)
- [x] Unit tests for harness (tests/eval/test_eval_harness.py)
- [x] **Run baseline eval with GPT-4o-mini** (completed 2026-02-28; see docs/eval-baseline.md)
      Run: source ~/.zshrc && .venv/bin/python -c "from tests.eval.harness import EvalHarness; h = EvalHarness('tests/eval/dataset.json'); h.run(output_dir='results/')"
      Or use CLI: source ~/.zshrc && .venv/bin/python -m ra.cli eval --dataset tests/eval/dataset.json --output results/
      Save results to results/ and docs/eval-baseline.md
- [x] **Expand dataset to 100 questions** (done — 100 questions across 3 tiers)
- [x] **Researcher QA agent** (black-box tester in tests/eval/qa_agent.py)

## Phase 4 — Hardening (QUEUED)
- [x] Chroma vector store caching (src/ra/retrieval/chroma_cache.py)
- [x] Enhanced error handling + retries for retrieval clients
- [x] Rate limiting improvements (token bucket)
- [x] Structured logging with log levels
- [x] Security review (API key handling, input sanitization)

## Phase 5 — Production (QUEUED)
- [x] FastAPI REST layer (src/ra/api/)
- [x] Deployment config (Dockerfile, docker-compose)
- [x] API documentation (OpenAPI/Swagger)
- [x] Performance optimization (async batching, connection pooling)
- [x] README with setup instructions

## Rules
- Commit message: update on "YYYY-MM-DD"
- Run unit tests before committing: .venv/bin/python -m pytest tests/ -q --ignore=tests/integration
- Push to origin after every commit
- Source ~/.zshrc before any command needing OPENAI_API_KEY

## Post-Completion
- [ ] **Remotion demo video** — Motion graphics demo showing: query → agent tool calls → formatted answer with citations → metrics. Build with Remotion (React). Only after all phases complete and baseline eval passes.

## Critical Bug Fix
- [x] **Fix StructuredTool sync invocation error** — Tools throw `NotImplementedError: StructuredTool does not support sync invocation`. The tools in `src/ra/tools/retrieval_tools.py` use async functions but `create_agent` calls them synchronously. Fix: either make tools sync-compatible (add sync wrappers using `asyncio.run()` or `coroutine=` param) or ensure the agent uses `ainvoke()`. Then re-run baseline eval to get real metrics.

## Phase 6 — Differentiation Features
- [ ] **Full-text analysis in agent loop** — Wire the existing PDF parser into the agent's tool chain. Agent should be able to: download a paper's PDF, extract full text, and answer questions about methods/results/tables. Add a `read_paper_fulltext` tool that takes a paper ID, downloads PDF, parses it, and returns structured sections.
- [ ] **Multi-hop reasoning** — Enable iterative deep dives. Agent should: (1) do initial search, (2) identify promising papers, (3) read them in depth, (4) follow citations to related work, (5) synthesize across multiple papers. Increase max_iterations, add a "deep_search" mode that chains search → details → fulltext → citations.
- [ ] **Literature review generation** — Add a `lit_review` mode/command that produces structured multi-section output: Introduction, Thematic Grouping, Key Findings, Research Gaps, Future Directions. Should group papers by theme, not just list them. Add CLI: `ra lit-review "topic"` and API endpoint.
- [ ] **Citation graph exploration** — Add a `trace_influence` tool that maps how an idea evolved over time through citation chains. Output: chronological list of papers showing influence flow. Add CLI: `ra trace "paper or concept"`. Visualize as a simple text-based timeline or JSON structure for frontend rendering.
- [ ] **Confidence scoring** — For each claim in the answer, show evidence strength: number of supporting papers, contradicting papers, and overall confidence (high/medium/low). Add to the structured output format after each major claim.
- [ ] **Interactive refinement (conversational mode)** — Add a `ra chat` CLI mode and `/chat` API endpoint that maintains conversation state. User can ask follow-ups: "dig deeper into X", "compare paper A vs B", "find more recent work on Y". Use LangChain memory or message history.

## Phase 7 — Demo & Polish
- [ ] **Remotion demo video** — Motion graphics demo showing: (1) GPT-4o giving unverifiable answer, (2) RA giving same answer with real papers + citation graph. The "aha" moment. Build with Remotion (React).
- [ ] **Citation graph visualization** — Simple React component (or mermaid/d3) that renders paper relationships as a directed graph. For the demo and potential web UI.

## Critical: Citation Formatting
- [ ] **Fix inline citation formatting** — Agent retrieves papers successfully (tool_success=100%) but outputs 0 inline citations. The system prompt instructs (Author et al., Year) format but the agent ignores it. Debug: check if retrieved papers are included in agent context, verify the system prompt is being passed to create_agent, test with a stronger prompt that explicitly requires citations. Re-run eval after fix.
