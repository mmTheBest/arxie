# ADR-001: Model Selection Strategy

**Status:** Accepted  
**Date:** 2026-02-24  
**Book Chapter:** Chapter 1 (Model Selection)

## Context

The RA needs to select a foundation model that satisfies product constraints:
- Citation precision ≥ 0.85
- p95 latency ≤ 5s
- Cost/query ≤ $0.15
- Strong tool-use reliability

## Decision

Implement a **configurable model backend** with tiered defaults:

| Tier | Model | Use Case | Est. Cost/Query |
|------|-------|----------|-----------------|
| Development | GPT-4o-mini | Fast iteration, debugging | ~$0.01 |
| Integration | GPT-4o | Quality validation | ~$0.05 |
| Evaluation | GPT-4o + Claude Sonnet 4.5 | Benchmark comparison | ~$0.08 |
| Production | Configurable | Based on eval results | TBD |

## Rationale

1. **Configurable:** Aligns with book's model-agnostic philosophy; allows fair comparison.
2. **Tiered approach:** Minimizes dev costs while ensuring quality validation.
3. **GPT-4o-mini for dev:** 10-20x cheaper than flagship models; sufficient for pipeline testing.
4. **Multi-model eval:** Validates that pipeline works across providers; enables book's comparison tables.

## Consequences

- Must implement model-agnostic interfaces (via LangChain)
- Environment variables control model selection
- Evaluation harness must support multiple models
- Cost tracking per model tier

## Book Integration

This decision directly supports:
- Section 1.3.3 (Current model landscape)
- Section 1.4.2 (Baseline sweep results)
- Section 1.5.2 (Weighted scoring framework)

The RA implementation serves as the worked example for model selection methodology.
