# Academic Research Assistant (RA)

A production-ready AI agent that answers academic literature questions with verifiable citations.

## Project Status

**Phase:** 1 - Foundation  
**Target Quality:** Internal tool (robust error handling, logging, tests)  
**Estimated Completion:** 6-7 weeks from start

## Architecture Decisions

All technical decisions are documented in `docs/decisions/` with rationale for later book integration.

### Confirmed Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Deployment | CLI → API | Validate core logic before infrastructure |
| Model Backend | Configurable | Aligns with book's model-selection philosophy |
| Retrieval | Semantic Scholar + arXiv | Metadata + citations + full-text coverage |
| Vector Store | Chroma (MVP) | Simple, local, no external deps |
| Framework | LangChain | Rich ecosystem, faster development |
| PDF Parsing | PyMuPDF → GROBID | Start simple, migrate for quality |
| Citation Format | JSON + rendered | Programmatic + human readable |
| Semantic Scholar | Public tier | 100 req/sec, sufficient for dev |

## Project Structure

```
academic-research-assistant/
├── src/
│   ├── ra/                 # Main package
│   │   ├── agents/         # Agent implementations
│   │   ├── tools/          # Tool definitions (search, parse, etc.)
│   │   ├── retrieval/      # Retrieval backends
│   │   ├── models/         # Model adapters
│   │   └── utils/          # Utilities
│   └── cli.py              # CLI entry point
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/         # Evaluation harness
├── docs/
│   ├── decisions/          # ADRs (Architecture Decision Records)
│   └── book-notes/         # Notes for book chapters
├── data/
│   ├── eval/               # Evaluation datasets
│   └── cache/              # Local caches
├── .env.example
├── pyproject.toml
└── README.md
```

## Development

```bash
# Setup
cd /Users/mm/Projects/academic-research-assistant
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run CLI
ra query "What are the main critiques of BERT's tokenization?"

# Run tests
pytest tests/
```

## Team Structure

- **Claude (Lead):** Architecture, writing, documentation, code review
- **Codex (Dev):** Implementation, tests, refactoring

## Memory Management

Development state is tracked in:
- `docs/decisions/` — Technical decisions with rationale
- `docs/book-notes/` — Content for book chapters
- `DEVLOG.md` — Daily progress log
- `TODO.md` — Current task queue
