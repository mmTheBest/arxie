# Arxie — AI Research Assistant

## Stack
- Python 3.14, LangChain 1.2.x, FastAPI, ChromaDB, httpx, PyMuPDF, pdfplumber
- Tests: pytest (87+ unit tests)
- Model: configurable via `RA_MODEL` env var (default: gpt-4o-mini)

## Architecture
- `src/ra/agents/` — LangChain create_agent with tool-calling loop
- `src/ra/retrieval/` — Semantic Scholar + arXiv clients, unified retriever, Chroma cache
- `src/ra/citation/` — APA formatting, inline citations, claim extraction
- `src/ra/parsing/` — PDF parser (PyMuPDF + pdfplumber)
- `src/ra/tools/` — LangChain StructuredTool wrappers for retrieval
- `src/ra/api/` — FastAPI REST layer
- `src/ra/utils/` — Config, logging, rate limiting, security

## Commands
```bash
# Run unit tests
.venv/bin/python -m pytest tests/ -q --ignore=tests/integration

# Run all tests (needs OPENAI_API_KEY)
source ~/.zshrc && .venv/bin/python -m pytest tests/ -v

# Run eval
source ~/.zshrc && .venv/bin/python -m ra.cli eval --dataset tests/eval/dataset.json --output results/

# Single query
source ~/.zshrc && .venv/bin/python -c "from ra.agents.research_agent import ResearchAgent; print(ResearchAgent().run('your question'))"
```

## SOP
- TDD: write tests first, then implement
- Run tests before every commit
- Commit message: `update on "YYYY-MM-DD"`
- Source `~/.zshrc` before anything needing OPENAI_API_KEY

## Current Focus
Priority 0: Fix citation formatting (agent outputs 0 inline citations)
Then: Priorities 1-6 in TODO.md (full-text, multi-hop, lit review, citation graph, confidence, chat mode)
