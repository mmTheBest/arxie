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
- [ ] **Fix StructuredTool sync invocation error** — Tools throw `NotImplementedError: StructuredTool does not support sync invocation`. The tools in `src/ra/tools/retrieval_tools.py` use async functions but `create_agent` calls them synchronously. Fix: either make tools sync-compatible (add sync wrappers using `asyncio.run()` or `coroutine=` param) or ensure the agent uses `ainvoke()`. Then re-run baseline eval to get real metrics.
