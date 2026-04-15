# Arxie

*Research moves faster when papers are searchable, structured, and grounded.*

Arxie is an AI research workspace built for serious literature work. It does not just answer questions about papers. It builds a living paperbase behind every answer so you can search a corpus, inspect figures, compare methods, trace evidence, and return to the same research context later.

---

## Version status

- **Current:** `v0.1.0` (released)
- **Next:** `v0.2.0` (active branch development)

### What v0.1.0 ships
- Live paper retrieval and citation-grounded Q&A
- Deep search (multi-hop paper analysis)
- Full-text PDF parsing for methods/results-level reasoning
- Literature review generation (`ra lit-review`)
- Citation influence tracing (`ra trace`)
- Confidence annotations (supporting vs contradicting evidence)
- Conversational mode (`ra chat`)
- FastAPI + Docker support

### What is already implemented on the v0.2 branch
- Paperbase canonical storage for papers, sources, files, sections, chunks, figures, tables, datasets, methods, metrics, results, findings, glossary terms, engineering tricks, evidence spans, collections, annotations, workspaces, and background jobs
- Local-library ingest plus worker-backed parse, extraction, and reindex jobs
- Paper/chunk/artifact search surfaces with SQL fallback and backend-aware indexing contracts
- Collection summaries, comparison routes, and structured paper browse
- Field-specific extraction profiles, including the current `sc_regnet` preset
- Public Arxie homepage at `/` and a saved-workspace app at `/app`
- Paperbase-first retrieval inside Arxie before live provider fallback

### Still in progress for v0.2.0
- Dashboard-based proposal workspace (not terminal-first)
- Iterative research proposal co-creation workflow
- Visual artifacts (mindmap, evidence map, logical tree, method pipeline, outcome matrix)
- Figure/table extraction beyond the current placeholder pipeline
- Broader external scholarly sync and provider-backed corpus enrichment
- Deeper assistant/workspace integration over saved context and structured comparison state

(See `docs/PRE-PRD-v0.2.md` for discussion draft.)

---

## Why Arxie

Most assistants stop at summaries. Arxie is designed for researchers who need a defensible reasoning trail:

- read full papers, not just abstracts
- compare methods and contradictions across papers
- inspect extracted figures, tables, and evidence-backed result rows
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

Paperbase's local-first API is served separately from the legacy RA API in the
active feature branch. It now supports:

- queued local-library ingest, parse, extraction, and reindex jobs
- saved workspaces layered over collections, queries, focus notes, and pinned papers
- paper, chunk, and artifact search surfaces with SQL fallback
- structured paper browse for datasets, methods, metrics, evidence, figures,
  and tables
- collection summaries and comparison routes for results, methods,
  engineering tricks, figures, and tables
- a public homepage at `/`
- a build-free workspace app at `/app`

The current branch also includes:

- saved research workspaces over curated collections
- field-specific extraction profiles for custom local paper databases
- workerized long-running jobs instead of inline parse/extract/reindex calls
- comparison slices for results, methods, engineering tricks, figures, and tables

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

```text
src/paperbase/
├── db/          # canonical schema, repositories, bootstrap, migrations
├── ingest/      # local-library and provider-normalized ingest
├── parsing/     # PDF parse pipeline and collection runners
├── extract/     # structured extraction contracts and persistence
├── search/      # indexing contracts, runtime, query building
├── profiles/    # field-specific extraction presets
└── tables/      # table artifact pipeline
```

---

## License

MIT
