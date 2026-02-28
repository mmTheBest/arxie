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

- **Full-Text Analysis:** Downloads and reads entire papers — methods, results, tables, figures. Not just abstracts. The agent reasons over what researchers actually wrote.
- **Multi-Hop Reasoning:** Compares methodologies across papers, follows evidence chains, and synthesizes insights from multiple sources in a single query. "Compare LoRA vs QLoRA" reads both papers and gives you a real comparison.
- **Literature Review Generation:** Generates structured, multi-section literature reviews with thematic grouping, key findings, research gaps, and future directions. `ra lit-review "your topic"` and get a publication-ready draft.
- **Citation Graph Exploration:** Traces how ideas evolve through citation chains over time. "Show me how attention mechanisms developed from 2015 to 2024" maps the influence flow across papers.
- **Confidence Scoring:** Shows evidence strength per claim — how many papers support it, how many contradict it, and overall confidence level. You see the landscape, not just a summary.
- **Interactive Research Sessions:** Multi-turn conversations with memory. Ask a question, then refine: "dig deeper into the transformer variants", "find contradicting evidence", "compare with the 2023 papers". `ra chat` for conversational research.
- **Live Retrieval:** Searches Semantic Scholar + arXiv in real-time — including papers published today. No training cutoff.
- **Verified Citations:** Every claim links to a real paper with a real DOI. Inline (Author et al., Year) format with a formatted References section.

### Why Arxie

- **Reads full papers, not just metadata.** Most RAG tools stop at titles and abstracts. Arxie downloads PDFs and reads methods, results, and discussion sections. It knows what the paper actually says.
- **Compares papers for you.** Instead of reading 20 papers yourself, ask Arxie to compare methodologies, find contradictions, or identify which approach has the strongest evidence. Multi-hop reasoning across the literature.
- **Generates real literature reviews.** Not bullet-point summaries — structured reviews with thematic grouping, research gaps, and future directions. The kind you'd actually put in a paper.
- **Shows you the evidence, not just conclusions.** Confidence scoring tells you "12 papers support this, 2 contradict it" so you can judge the strength of any claim yourself.
- **Always current.** Searches live databases including papers published today. No training data cutoff.
- **Every citation is real.** DOI and arXiv links included. Click and read the source.

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
