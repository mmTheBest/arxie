# RA Development Task Queue

## Phase 1: Foundation (Week 1)

### In Progress
- [ ] Project scaffold and configuration
- [ ] Semantic Scholar API client
- [ ] arXiv API client
- [ ] Basic retrieval interface

### Blocked
(none)

### Done
- [x] Project structure created
- [x] pyproject.toml configured
- [x] Environment setup (.env.example, .gitignore)

---

## Phase 2: Core Pipeline (Weeks 2-3)
- [ ] Agent loop implementation (LangChain)
- [ ] Tool definitions (search, parse_pdf, cite)
- [ ] Citation extraction and formatting
- [ ] Basic generation with retrieved context
- [ ] Prompt engineering for citation accuracy

## Phase 3: Evaluation (Week 4)
- [ ] Test harness implementation
- [ ] 100-question evaluation dataset
- [ ] Metrics: citation precision, claim support, tool-call success
- [ ] Baseline measurements

## Phase 4: Hardening (Weeks 5-6)
- [ ] Error handling and retries
- [ ] Caching layer (Chroma)
- [ ] Rate limiting for APIs
- [ ] Logging and observability
- [ ] Unit and integration tests

## Phase 5: Production (Week 7)
- [ ] API layer (FastAPI)
- [ ] Deployment configuration
- [ ] Documentation
- [ ] Performance optimization
