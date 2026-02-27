# RA Development Log

## 2026-02-24 (Day 1)

### Decisions Made
- **Repository:** Separate repo at `/Users/mm/Projects/academic-research-assistant/`
- **Framework:** LangChain for agent orchestration
- **Models:** Configurable; GPT-4o-mini for dev, GPT-4o for eval
- **Retrieval:** Semantic Scholar (metadata + citations) + arXiv (full-text)
- **Vector Store:** Chroma for MVP
- **PDF Parsing:** PyMuPDF + pdfplumber initially; GROBID for production
- **Quality Target:** Internal tool (~6-7 weeks)

### Progress
- Created project structure
- Set up pyproject.toml with dependencies
- Configured environment (.env.example, .gitignore)
- Verified OpenAI API key access
- ✅ Implemented Semantic Scholar API client (`src/ra/retrieval/semantic_scholar.py`)
- 🔄 arXiv API client (in progress - Codex sub-agent)
- 🔄 Unified retrieval interface (in progress - Codex sub-agent)

### Next Steps
1. ~~Implement Semantic Scholar API client~~ ✅
2. ~~Implement arXiv API client~~ (in progress)
3. ~~Create unified retrieval interface~~ (in progress)
4. Set up basic LangChain agent structure
5. Add tests for retrieval clients
6. Create CLI for testing

### Token Usage
- Today: 0 (setup only)
- Cumulative: 0

---

## Architecture Notes

### Agent Flow (Planned)
```
User Query
    ↓
Query Analysis (identify papers, topics, query type)
    ↓
Retrieval Planning (which sources, how many)
    ↓
┌──────────────────────────────────┐
│ Tool Loop                        │
│  - semantic_scholar_search       │
│  - arxiv_search                  │
│  - fetch_paper_metadata          │
│  - parse_pdf                     │
│  - extract_claims                │
└──────────────────────────────────┘
    ↓
Synthesis (with citations)
    ↓
Citation Verification
    ↓
Response
```

### Key Metrics
- Citation precision ≥ 0.85 (hard constraint)
- Claim support rate ≥ 0.80 (soft)
- Tool-call success ≥ 0.90 (hard)
- p95 latency ≤ 5s (hard)
- Cost/query ≤ $0.15 (hard)
