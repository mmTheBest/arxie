# Arxie — Execution Plan

## SOP
- Each task: plan → test first (TDD) → implement → verify → commit → push
- Source `~/.zshrc` before any command needing `OPENAI_API_KEY`
- Run tests before committing: `.venv/bin/python -m pytest tests/ -q --ignore=tests/integration`
- Commit message: `update on "YYYY-MM-DD"`
- Skip any task marked BLOCKED

## Priority 0 — Critical Bug Fix (IN PROGRESS)
- [x] **Fix inline citation formatting** — Agent retrieves papers (tool_success=100%) but outputs 0 inline citations. Root cause: agent doesn't use (Author et al., Year) format in answers. Fix system prompt + verify tool output includes author/year data. Re-run eval.

## Priority 1 — Full-Text Analysis
- [ ] **Wire PDF parser into agent tool chain** — Add `read_paper_fulltext` tool to `src/ra/tools/retrieval_tools.py`. Takes paper_id → downloads PDF via `pdf_url` → parses with `src/ra/parsing/pdf_parser.py` → returns structured sections (abstract, methods, results, discussion). Add unit tests.
- [ ] **Integrate into agent loop** — Update system prompt to instruct agent to use `read_paper_fulltext` for detailed questions. Verify agent calls the tool when asked about methods/results.

## Priority 2 — Multi-Hop Reasoning
- [ ] **Deep search mode** — Add `deep_search` parameter to `ResearchAgent`. When enabled: (1) initial search, (2) read top-3 full text, (3) follow citations from those papers, (4) synthesize across all sources. Increase `max_iterations` for deep mode.
- [ ] **CLI + API** — Add `--deep` flag to `ra query` CLI. Add `deep` parameter to `/api/query` endpoint.

## Priority 3 — Literature Review Generation
- [ ] **Lit review agent mode** — New class or mode in `src/ra/agents/` that produces structured output: Introduction → Thematic Groups → Key Findings → Research Gaps → Future Directions. Groups papers by theme using LLM clustering.
- [ ] **CLI + API** — Add `ra lit-review "topic"` CLI command. Add `/api/lit-review` endpoint.
- [ ] **Tests** — Unit tests with mock agent for output structure validation.

## Priority 4 — Citation Graph Exploration
- [ ] **Trace influence tool** — Add `trace_influence` tool to `src/ra/tools/retrieval_tools.py`. Takes a paper title/ID → follows forward citations iteratively → builds chronological influence chain. Returns JSON timeline.
- [ ] **CLI** — Add `ra trace "paper or concept"` CLI command.
- [ ] **Text-based visualization** — Format timeline as readable text output (Year → Paper → cited by → Paper).

## Priority 5 — Confidence Scoring
- [ ] **Evidence scoring module** — New file `src/ra/citation/confidence.py`. For each claim: count supporting papers, contradicting papers, compute confidence (high/medium/low). Uses semantic similarity between claim and paper abstracts.
- [ ] **Wire into output** — Add confidence annotations to structured output after each major claim. Example: `[Confidence: HIGH — 8 supporting, 1 contradicting]`
- [ ] **Tests** — Unit tests for scoring logic.

## Priority 6 — Interactive Conversational Mode
- [ ] **Chat mode agent** — Add conversation memory to `ResearchAgent` using LangChain message history. Support follow-up queries that reference previous context.
- [ ] **CLI** — Add `ra chat` command that runs an interactive REPL with conversation state.
- [ ] **API** — Add `/api/chat` endpoint with session_id for stateful conversations.
- [ ] **Tests** — Test multi-turn conversations with mock agent.

## Priority 7 — Demo & Visualization
- [ ] **Remotion demo video** — Side-by-side: GPT-4o (hallucinated citations) vs Arxie (verified citations + citation graph). Build with Remotion (React).
- [ ] **Citation graph visualization** — React/mermaid/d3 component rendering paper relationships as directed graph.

## Completed
- [x] Phase 1: Foundation (retrieval clients, CLI, tests)
- [x] Phase 2: Core pipeline (agent, tools, citations, PDF parser)
- [x] Phase 3: Evaluation (harness, 100-question dataset, QA agent)
- [x] Phase 4: Hardening (cache, rate limiting, logging, security)
- [x] Phase 5: Production (FastAPI, Docker, OpenAPI, README)
- [x] Fix sync/async StructuredTool invocation
- [x] Rebrand to Arxie
- [x] PRD written (docs/PRD.md)
