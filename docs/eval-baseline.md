# Baseline Eval: GPT-4o-mini

- Date (UTC): `2026-02-28`
- Generated at (UTC): `2026-02-28T11:35:56.023437+00:00`
- Model: `gpt-4o-mini` (`RA_MODEL` unset; default from `src/ra/utils/config.py`)
- Dataset: `tests/eval/dataset.json`
- Dataset size: `100`
- Artifacts:
  - `results/eval_summary.md`
  - `results/eval_results.json`

## Metrics vs Targets

| Metric | Target | Result | Status |
|---|---:|---:|---|
| `citation_precision` | `>= 0.85` | `0.8600` | PASS |
| `claim_support_ratio` | `>= 0.80` | `1.0000` | PASS |
| `tool_success_rate` | `>= 0.90` | `0.9982` | PASS |

Additional metric:
- `latency_p95`: `55.3008s`

## Notable Patterns and Failures

- Citation formatting fix is effective: `citation_precision=0.86` (`86/100` responses with valid inline citations).
- Claim support remains strong: `claim_support_ratio=1.0`.
- Tool execution stays robust: `tool_success_rate=0.9982` (minor misses tied to transient API 429/5xx failures).
- Latency improved versus prior run (`55.3008s` p95 vs `68.1868s`) but remains API-retry bound.
- Tier citation precision: `tier_1=0.80`, `tier_2=0.875`, `tier_3=0.95`.
