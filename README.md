<h1 align="center">Arxie</h1>

<p align="center">
  <strong>An AI research assistant that reads real papers, not its training data.</strong><br>
  <strong>Live retrieval. Verified citations. Full-text analysis. No hallucinations.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" /></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+" />
  <img src="https://img.shields.io/badge/tests-87%2B%20passing-brightgreen" alt="Tests" />
</p>

Ask a research question. Get an answer backed by real papers from Semantic Scholar and arXiv — with inline citations you can actually verify.

```
100-question eval suite · 5 retrieval tools · APA citations · Full-text PDF parsing · FastAPI · Docker-ready
```

### Features

- **Real Citations:** Every claim links to a real paper with a real DOI. No hallucinated references.
- **Live Retrieval:** Searches Semantic Scholar + arXiv in real-time. Finds papers published yesterday, not just what's in training data.
- **Full-Text Reading:** Downloads and parses PDFs (PyMuPDF + pdfplumber). Reads methods, results, and tables — not just abstracts.
- **Citation Chasing:** Follows forward citation chains to find follow-up work, validations, and contradictions.
- **Structured Output:** Every answer has `## Answer` with inline (Author et al., Year) citations and a formatted `## References` section.
- **Evaluation Built-In:** 100-question dataset across 3 difficulty tiers with automated metrics (citation precision, claim support, tool success rate).

### Why Arxie

- **Always up to date.** Arxie searches live databases — including papers published today. No training cutoff, no stale knowledge.
- **Abstracts aren't enough.** Most RAG tools only read titles and abstracts. Arxie reads full papers.
- **Trust but verify.** Every citation includes DOI/arXiv links. Click and read the source yourself.

### Quick Start

```bash
git clone https://github.com/mmTheBest/arxie.git
cd arxie

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Ask a question
ra query "What are the main approaches to neural machine translation?"

# Interactive mode
ra chat

# Generate a literature review
ra lit-review "attention mechanisms in computer vision"

# Trace citation influence
ra trace "Attention Is All You Need"

# Run evaluation
ra eval --dataset tests/eval/dataset.json --output results/
```

### Docker

```bash
docker build -t arxie .
docker run -e OPENAI_API_KEY="sk-..." arxie ra query "Your question here"
```

### API

```bash
# Start the server
uvicorn ra.api.main:app --host 0.0.0.0 --port 8000

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'
```

### Architecture

```
User Query
    │
    ▼
┌──────────────┐     ┌─────────────────────┐
│  ReAct Agent │────▶│  Tool Chain          │
│  (LangChain) │     │  ├─ search_papers    │
│              │◀────│  ├─ get_details      │
│  Synthesize  │     │  ├─ get_citations    │
│  + Cite      │     │  ├─ read_fulltext    │
└──────┬───────┘     │  └─ trace_influence  │
       │             └──────────┬───────────┘
       ▼                        │
┌──────────────┐     ┌──────────▼───────────┐
│  Structured  │     │  Retrieval Layer     │
│  Output      │     │  ├─ Semantic Scholar │
│  ├─ Answer   │     │  ├─ arXiv            │
│  ├─ Refs     │     │  ├─ Chroma cache     │
│  └─ Score    │     │  └─ PDF parser       │
└──────────────┘     └──────────────────────┘
```

### Evaluation

Arxie ships with a 100-question evaluation suite across three difficulty tiers:

| Tier | Questions | Description |
|------|-----------|-------------|
| Factual | 40 | Who/what/when about known papers and methods |
| Analytical | 40 | Compare methods, explain techniques, analyze tradeoffs |
| Synthesis | 20 | Cross-paper analysis, research gaps, emerging trends |

**Target metrics:**

| Metric | Target |
|--------|--------|
| Citation precision | ≥ 85% |
| Claim support ratio | ≥ 80% |
| Tool success rate | ≥ 90% |
| Latency p95 | ≤ 30s |

Run the eval:
```bash
ra eval --dataset tests/eval/dataset.json --output results/
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Custom endpoint (for proxies/relays). |
| `RA_MODEL` | `gpt-4o-mini` | Model for the agent. Any OpenAI-compatible model. |
| `RA_AGENT_MAX_RETRIES` | `3` | Retries on transient API errors. |
| `RA_AGENT_MAX_ITERATIONS` | `30` | Max tool-calling loop iterations. |

### Project Structure

```
src/ra/
├── agents/          # LangChain ReAct agent
├── api/             # FastAPI REST layer
├── citation/        # APA formatter, claim extraction
├── parsing/         # PDF parser (PyMuPDF + pdfplumber)
├── retrieval/       # Semantic Scholar, arXiv, Chroma cache
├── tools/           # Agent tool definitions
└── utils/           # Config, logging, rate limiting
tests/
├── unit/            # 87+ unit tests
├── integration/     # API integration tests
└── eval/            # Evaluation harness + 100-question dataset
```

### Roadmap

- [x] **Foundation:** Retrieval clients, unified search, CLI
- [x] **Core Pipeline:** ReAct agent, citation formatting, PDF parsing
- [x] **Evaluation:** 100-question dataset, eval harness, baseline metrics
- [x] **Hardening:** Caching, rate limiting, logging, security
- [x] **Production:** FastAPI, Docker, OpenAPI docs
- [ ] **Differentiation:** Full-text analysis, multi-hop reasoning, lit reviews, citation graphs, confidence scoring, conversational mode
- [ ] **Demo:** Remotion video — GPT-4o vs Arxie side-by-side

### License

MIT
