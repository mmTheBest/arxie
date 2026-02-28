# Baseline Eval: GPT-4o-mini

- Date (UTC): `2026-02-28`
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
| `tool_success_rate` | `>= 0.90` | `0.0000` | FAIL |

Additional metric:
- `latency_p95`: `3.4758s`

## Notable Patterns and Failures

- Citation behavior is consistently broken: `100/100` responses had `0` inline citations, and `citation_count_ok` is `false` for all rows.
- Tool execution is failing across the run: `89` tool calls were attempted across `86` questions, with `0` successful tool calls.
- Runtime logs repeatedly show tool invocation failures (`NotImplementedError: StructuredTool does not support sync invocation`), and usage logs show repeated `tool:search_papers` status `500`.
- `claim_support_ratio=1.0` appears inflated relative to retrieval/citation failure and should not be treated as sufficient quality evidence by itself.
