# RA Development TODO

## Current: Phase 3 — Evaluation (IN PROGRESS)
- [x] Eval harness (tests/eval/harness.py)
- [x] 50-question dataset (tests/eval/dataset.json)
- [x] Eval CLI command (ra eval)
- [x] Unit tests for harness (tests/eval/test_eval_harness.py)
- [ ] **BLOCKED: Run baseline eval (needs V2Box proxy fix for api.openai.com) with GPT-4o-mini** (source ~/.zshrc first for API key)
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
- [ ] FastAPI REST layer (src/ra/api/)
- [ ] Deployment config (Dockerfile, docker-compose)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Performance optimization (async batching, connection pooling)
- [ ] README with setup instructions

## Rules
- Commit message: update on "YYYY-MM-DD"
- Run unit tests before committing: .venv/bin/python -m pytest tests/ -q --ignore=tests/integration
- Push to origin after every commit
- Source ~/.zshrc before any command needing OPENAI_API_KEY
