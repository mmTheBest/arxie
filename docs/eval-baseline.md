# Phase 3 Baseline Eval (2026-02-28)

## Run Metadata

- Date (UTC): `2026-02-28T03:22:09.999037+00:00`
- Model: `gpt-4o-mini` (default `RA_MODEL`)
- Dataset: `tests/eval/dataset.json`
- Dataset size: `50` questions
- Artifacts:
  - `results/eval_results.json`
  - `results/eval_summary.md`

## Metrics vs Targets

| Metric | Result | Target | Status |
|---|---:|---:|---|
| `citation_precision` | `0.000000` | `>= 0.85` | FAIL |
| `claim_support_ratio` | `0.000000` | `>= 0.80` | FAIL |
| `tool_success_rate` | `0.000000` | `>= 0.90` | FAIL |
| `latency_p95` | `18.151169s` | n/a | n/a |

## Notable Failures / Patterns

- All 50 prompts produced unusable answers for scoring (`0` citation precision, `0` claim support, `0` keyword coverage).
- No tool calls were successfully executed (`tool_success_rate = 0.0`).
- Failure mode is consistent with model connectivity issues rather than retrieval logic quality:
  - Repeated `openai.APIConnectionError: Connection error`
  - Underlying TLS error: `SSL: UNEXPECTED_EOF_WHILE_READING`
- To complete the full 50-question pass in bounded time, eval was run with `RA_AGENT_MAX_RETRIES=0` (no extra per-question retry backoff).
