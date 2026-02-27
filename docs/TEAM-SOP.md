# RA Development Team SOP

## Team Roles

| Role | Agent | Responsibilities |
|------|-------|-----------------|
| **Tech Lead / PM** | Claude (main) | Product vision, task planning, code review, progress tracking, documentation, architecture decisions |
| **Architect** | Claude (main) | System design, diagrams, ADRs, API contracts, module boundaries |
| **Backend Engineer** | Codex (sub-agent) | Retrieval clients, agent loop, API layer, data models |
| **ML Engineer** | Codex (sub-agent) | LangChain agent, prompt engineering, model integration, evaluation |
| **Test Engineer** | Codex (sub-agent) | Unit tests, integration tests, smoke tests, test harness |
| **Security / DevOps** | Codex (sub-agent) | API key handling, rate limiting, error handling, logging, CI |
| **Researcher (QA)** | Codex (isolated) | Black-box RA usage, friction logging, UX feedback |

## Development Workflow (Review-Test-Edit Loop)

```
┌─────────────────────────────────────────────┐
│ 1. PLAN (Tech Lead)                         │
│    - Define task spec + acceptance criteria  │
│    - Assign to role/agent                    │
│    - Set priority + dependencies             │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 2. IMPLEMENT (Codex sub-agent)              │
│    - Write code per spec                    │
│    - Run local lint/compile check           │
│    - Commit with descriptive message        │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 3. REVIEW (Tech Lead / Claude)              │
│    - Read implementation                    │
│    - Check against spec + architecture      │
│    - Verify naming, patterns, error handling│
│    - If issues → EDIT (step 4)              │
│    - If OK → TEST (step 5)                  │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 4. EDIT (if review fails)                   │
│    - Spawn fix task with specific feedback   │
│    - Re-review after fix                    │
│    - Max 2 edit rounds before escalate      │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 5. TEST (Test Engineer / Codex)             │
│    - Run unit tests: pytest tests/unit/ -v  │
│    - Run integration: pytest -m integration │
│    - If fails → back to EDIT (step 4)       │
│    - If passes → MERGE                      │
└──────────────────┬──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────┐
│ 6. MERGE + LOG                              │
│    - Commit to main branch                  │
│    - Update DEVLOG.md + TODO.md             │
│    - Update architecture diagrams if needed │
│    - Log API usage if any                   │
└─────────────────────────────────────────────┘
```

## Task Dispatch Rules

1. **Parallelism**: Up to 3 Codex agents in parallel on independent tasks
2. **Dependencies**: Sequential tasks wait for predecessor to pass review
3. **Review gate**: Every implementation is reviewed by Tech Lead before merge
4. **Test gate**: All tests must pass before moving to next phase
5. **Escalation**: If a task fails review 2x, Tech Lead rewrites the spec

## Progress Tracking

- **TODO.md**: Task queue with phase/priority
- **DEVLOG.md**: Daily progress log with decisions + API usage
- **docs/decisions/**: ADRs for significant technical choices
- **Hourly Discord update**: Automated via cron

## API Usage Logging

Every live API call must log:
- Timestamp
- Endpoint (S2 / arXiv / OpenAI)
- Tokens used (for LLM calls)
- Cost estimate
- Response time

Log to: `DEVLOG.md` under daily entry + `data/api-usage.jsonl`
