# Product Requirements Document: Academic Research Assistant

## 1. Product Overview

**Name:** Arxie

**One-liner:** An AI research assistant that reads real papers, not its training data.

**Problem:** Researchers spend 30-50% of their time on literature discovery and synthesis. LLMs can answer research questions fluently but hallucinate citations, reference nonexistent papers, miss recent work, and provide no way to verify claims. The gap between "sounds right" and "is right" costs researchers hours of manual verification.

**Solution:** A tool-augmented AI agent that searches live academic databases (Semantic Scholar, arXiv), retrieves and reads real papers, traces citation chains, and synthesizes answers with verifiable inline citations. Every claim links to a real paper with a real DOI.

**Core differentiation from "just calling OpenAI":**

| Capability | GPT-4o (direct) | RA |
|---|---|---|
| Fluent answers | Yes | Yes |
| Real citations | No (hallucinated) | Yes (live retrieval) |
| Recent papers | No (training cutoff) | Yes (live search) |
| Full-text reading | No | Yes (PDF parsing) |
| Citation verification | Manual | Automatic |
| Evidence strength | Unknown | Confidence scoring |
| Literature reviews | Shallow | Structured, multi-section |
| Citation tracing | No | Yes (forward/backward chains) |

## 2. Target Users

**Primary:** Academic researchers (PhD students, postdocs, faculty) who need to:
- Quickly survey a new research area
- Find relevant related work for a paper
- Verify claims across multiple sources
- Generate literature review drafts

**Secondary:** Industry ML/AI practitioners who need to:
- Stay current with SOTA methods
- Find papers supporting technical decisions
- Compare approaches with evidence

**Non-target:** Casual users, undergrad homework (tool assumes domain expertise to evaluate output).

## 3. User Journeys

### Journey 1: Quick Research Question
```
User: "What are the current best approaches for long-context LLMs?"
RA: [searches 3 databases] -> [finds 15 papers] -> [reads top 5] -> [synthesizes answer]
Output: Structured answer with 8 inline citations + References section
Time: ~30 seconds
```

### Journey 2: Literature Review
```
User: ra lit-review "attention mechanisms in computer vision"
RA: [broad search] -> [clusters papers by theme] -> [reads key papers] -> [identifies gaps]
Output: Multi-section review: Introduction, Thematic Groups, Key Findings, Research Gaps, Future Directions
Time: ~5 minutes
```

### Journey 3: Citation Tracing
```
User: ra trace "Attention Is All You Need"
RA: [finds paper] -> [follows forward citations] -> [builds influence map]
Output: Chronological timeline showing how transformer architecture influenced subsequent work
Time: ~2 minutes
```

### Journey 4: Interactive Research Session
```
User: ra chat
> "What methods exist for efficient fine-tuning of LLMs?"
RA: [answer with citations]
> "Compare LoRA vs QLoRA specifically"
RA: [deeper dive, reads both papers, compares methods/results]
> "Find the most cited follow-up work on LoRA"
RA: [citation chase, ranks by impact]
```

## 4. Feature Requirements

### Phase 1-5 (COMPLETE): Foundation through Production
- [x] Retrieval clients (Semantic Scholar + arXiv)
- [x] Unified retriever with dedup
- [x] LangChain ReAct agent with tool-calling
- [x] Citation formatting (APA, inline)
- [x] PDF parser (PyMuPDF + pdfplumber)
- [x] Structured output (## Answer + ## References)
- [x] Error handling with retries
- [x] Eval harness + 100-question dataset
- [x] FastAPI REST layer
- [x] Dockerfile + deployment config
- [x] 87+ unit tests

### Phase 6 (COMPLETE): Differentiation
- [x] Full-text analysis — Agent reads full paper PDFs, not just abstracts
- [x] Multi-hop reasoning — Iterative search -> read -> follow citations -> synthesize
- [x] Literature review mode — Structured multi-section output with thematic grouping
- [x] Citation graph — Trace idea evolution through citation chains
- [x] Confidence scoring — Per-claim evidence strength (supporting/contradicting paper counts)
- [x] Conversational mode — Multi-turn research sessions with memory

### Phase 7 (COMPLETE): Demo & Polish
- [x] Remotion demo video — Side-by-side: GPT-4o (hallucinated) vs RA (verified)
- [x] Citation graph visualization — Interactive graph component

## 5. Technical Architecture

```
+-----------+     +------------+     +-----------------+
| CLI / API |---->| ReAct Agent|---->| Tool Chain      |
| (FastAPI) |     | (LangChain)|     | - search        |
+-----------+     +-----+------+     | - get_details   |
                        |            | - get_citations  |
                  +-----v------+     | - read_fulltext  |
                  | Output     |     | - trace_chain    |
                  | Formatter  |     +--------+--------+
                  | - Citations|              |
                  | - Confidence|    +--------v--------+
                  | - Structure|    | Retrieval        |
                  +------------+    | - Semantic Sch.  |
                                    | - arXiv          |
                                    | - Chroma cache   |
                                    | - PDF parser     |
                                    +-----------------+
```

## 6. Success Metrics

### Quality (from eval harness)
| Metric | Target | Notes |
|---|---|---|
| Citation precision | >= 0.85 | % of citations that are real, relevant papers |
| Claim support ratio | >= 0.80 | % of claims backed by at least one citation |
| Tool success rate | >= 0.90 | % of tool calls that execute successfully |
| Latency p95 | <= 30s | For single-question mode |

### User Value (future measurement)
| Metric | Target |
|---|---|
| Time saved vs manual search | >= 60% |
| Citation accuracy (human eval) | >= 90% |
| User satisfaction (1-5) | >= 4.0 |

## 7. Constraints

- **API budget:** ~$15-30/day for development (GPT-4o-mini for dev, GPT-4o for eval)
- **Rate limits:** Semantic Scholar: 100 req/5min. ArXiv: 3 req/sec.
- **Model:** Configurable via RA_MODEL env var. Default: gpt-4o-mini.
- **Network:** OpenAI API routed through Vercel proxy (China network constraint).

## 8. Out of Scope (for now)

- Web UI / frontend (API + CLI only)
- User accounts / authentication
- Paper recommendation engine
- Writing assistance (generating paper drafts)
- Non-English paper support
- Real-time collaboration

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Semantic Scholar rate limiting | Slows agent, degrades UX | Token bucket rate limiter + Chroma cache |
| LLM hallucination despite RAG | Incorrect citations | Confidence scoring + citation verification |
| PDF parsing failures | Missing full-text data | Multi-parser fallback (PyMuPDF + pdfplumber) |
| API cost at scale | Budget overrun | Usage logger, configurable model, caching |
| OpenAI API unreachable (China) | Complete blocker | Vercel Edge Function proxy |
