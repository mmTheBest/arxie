# Baseline Eval: GPT-4o-mini

- Date (UTC): `2026-02-28`
- Generated at (UTC): `2026-02-28T10:29:29.723278+00:00`
- Model: `gpt-4o-mini` (`RA_MODEL` unset; default from `src/ra/utils/config.py`)
- Dataset: `tests/eval/dataset.json`
- Dataset size: `100`
- Artifacts:
  - `results/eval_summary.md`
  - `results/eval_results.json`

## Metrics vs Targets

| Metric | Target | Result | Status |
|---|---:|---:|---|
| `citation_precision` | `>= 0.85` | `0.0000` | FAIL |
| `claim_support_ratio` | `>= 0.80` | `1.0000` | PASS |
| `tool_success_rate` | `>= 0.90` | `1.0000` | PASS |

Additional metric:
- `latency_p95`: `68.1868s`

## Notable Patterns and Failures

- Tool invocation is now healthy after the sync/async fix: `tool_success_rate=1.0`, with successful tool runs across all tiers.
- The prior `NotImplementedError: StructuredTool does not support sync invocation` failure mode was eliminated in this run.
- Citation behavior remains broken: `100/100` responses still have `0` inline citations (`citation_count_ok=false` across all rows).
- Latency is high (`p95=68.1868s`) and appears dominated by external API retries/rate limiting.
