# RA Development Task Queue

## Phase 1: Foundation ✅
- [x] Project scaffold and configuration
- [x] pyproject.toml configured
- [x] Environment setup (.env.example, .gitignore)
- [x] Semantic Scholar API client
- [x] arXiv API client
- [x] Unified retrieval interface (dedup, normalize)
- [x] CLI (ra search, ra get)
- [x] Unit tests (arXiv parsing, S2 parsing, dedup)
- [x] Integration smoke tests
- [x] Git repo initialized
- [x] Architecture diagrams (Mermaid)

## Phase 2: Core Pipeline (Current Sprint)

### Done
- [x] LangChain ReAct agent skeleton (research_agent.py)
- [x] Tool definitions (search_papers, get_paper_details, get_paper_citations)

### In Progress
- [ ] Citation extraction and formatting module
- [ ] PDF parsing pipeline (PyMuPDF + pdfplumber)
- [ ] Prompt engineering for citation accuracy
- [ ] End-to-end agent test (run query → get cited answer)

### Queued
- [ ] Basic generation with retrieved context
- [ ] Agent output structured formatting
- [ ] Usage/cost logging middleware
- [ ] Error handling in agent loop

## Phase 3: Evaluation (Next Sprint)
- [ ] Test harness implementation
- [ ] 100-question evaluation dataset (12 seed + expand)
- [ ] Metrics: citation precision, claim support, tool-call success
- [ ] Baseline measurements
- [ ] Researcher agent (black-box QA)

## Phase 4: Hardening
- [ ] Caching layer (Chroma vector store)
- [ ] Enhanced error handling and retries
- [ ] Rate limiting improvements
- [ ] Logging and observability
- [ ] Security review (API key handling, input validation)

## Phase 5: Production
- [ ] API layer (FastAPI)
- [ ] Deployment configuration
- [ ] Documentation (user guide, API docs)
- [ ] Performance optimization
