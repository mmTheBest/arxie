# Arxie

*Research moves faster when evidence is easy to trust.*

Arxie is an AI research assistant that reads real papers (Semantic Scholar + arXiv), reasons across sources, and writes citation-grounded outputs you can verify.

---

## Version status

- **Current:** `v0.1.0` (released)
- **Next:** `v0.2.0` (in planning)

### What v0.1.0 ships
- Live paper retrieval and citation-grounded Q&A
- Deep search (multi-hop paper analysis)
- Full-text PDF parsing for methods/results-level reasoning
- Literature review generation (`ra lit-review`)
- Citation influence tracing (`ra trace`)
- Confidence annotations (supporting vs contradicting evidence)
- Conversational mode (`ra chat`)
- FastAPI + Docker support

### Planned for v0.2.0
- Dashboard-based proposal workspace (not terminal-first)
- Iterative research proposal co-creation workflow
- Visual artifacts (mindmap, evidence map, logical tree, method pipeline, outcome matrix)
- Cross-artifact sync when users revise hypotheses/assumptions

(See `docs/PRE-PRD-v0.2.md` for discussion draft.)

---

## Why Arxie

Most assistants stop at summaries. Arxie is designed for researchers who need a defensible reasoning trail:

- read full papers, not just abstracts
- compare methods and contradictions across papers
- keep citations tied to claims
- show confidence based on evidence landscape

---

## Quick start

```bash
git clone https://github.com/mmTheBest/arxie.git
cd arxie

python -m venv .venv
source .venv/bin/activate
pip install -e .

export OPENAI_API_KEY="sk-..."
```

### CLI examples

```bash
# Ask a question
ra query "What are recent approaches to long-context LLMs?"

# Deeper multi-hop analysis
ra query --deep "Compare LoRA vs QLoRA methodologies"

# Literature review draft
ra lit-review "attention mechanisms in computer vision"

# Citation timeline
ra trace "Attention Is All You Need"

# Interactive session
ra chat
```

---

## API

```bash
uvicorn ra.api.app:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are retrieval-augmented generation trade-offs?"}'
```

---

## Docker

```bash
docker build -t arxie .
docker run -e OPENAI_API_KEY="sk-..." arxie ra query "Your question here"
```

---

## Evaluation snapshot (internal benchmark)

Using Arxie’s internal 100-question benchmark with GPT-4o-mini:

| Metric | Result |
|---|---:|
| Citation precision | 86% |
| Claim support ratio | 100% |
| Tool success rate | 99.8% |

These are reported benchmark results, not a user quick-start workflow.

---

## Project structure

```text
src/ra/
├── agents/      # research, lit-review, chat behaviors
├── api/         # FastAPI app + request models
├── citation/    # citation formatting + confidence scoring
├── parsing/     # PDF parsing
├── retrieval/   # Semantic Scholar + arXiv + cache
├── tools/       # tool interfaces for the agent loop
└── utils/       # config, logging, rate limiting
```

---

## License

MIT
