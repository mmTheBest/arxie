# Academic Research Assistant

## Stack
- Python 3.14, LangChain 1.2.x, httpx, PyMuPDF, pdfplumber
- Models: GPT-4o-mini (dev), GPT-4o (eval), configurable via RA_MODEL env var
- Tests: pytest, 26 unit tests + 3 E2E integration tests

## Architecture
- `src/ra/agents/` — LangChain create_agent with tool-calling loop
- `src/ra/retrieval/` — Semantic Scholar + arXiv clients, unified retriever
- `src/ra/citation/` — APA formatting, inline citations, claim extraction
- `src/ra/parsing/` — PDF parser (PyMuPDF + pdfplumber)
- `src/ra/tools/` — LangChain StructuredTool wrappers for retrieval
- `src/ra/utils/` — Config, JSONL usage logging

## Commands
```bash
# Run unit tests
.venv/bin/python -m pytest tests/ -q --ignore=tests/integration

# Run all tests (needs OPENAI_API_KEY)
source ~/.zshrc && .venv/bin/python -m pytest tests/ -v

# Run E2E only
source ~/.zshrc && .venv/bin/python -m pytest tests/integration/test_agent_e2e.py -v
```

## Commit Convention
- Use `update on "YYYY-MM-DD"` for regular commits
- Use `fix:` / `feat:` prefixes for specific changes

## Current Phase
Phase 2 (Core Pipeline) — remaining: structured output formatting, error handling, E2E validation.
Phase 3 next: evaluation harness + 100-question dataset.
