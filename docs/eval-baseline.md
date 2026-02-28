# Baseline Eval (2026-02-28)

## Run Metadata

- Generated at (UTC): `2026-02-28T04:10:18.491048+00:00`
- Model: `gpt-4o-mini` (default `RA_MODEL`)
- Dataset: `tests/eval/dataset.json` (`50` questions)
- Command:
  - `source ~/.zshrc && .venv/bin/python -m ra.cli eval --dataset tests/eval/dataset.json --output results/ 2>&1 | tee results/baseline-run.log`
- Artifacts:
  - `results/eval_results.json`
  - `results/eval_summary.md`
  - `results/baseline-run.log`

## Scores

| Metric | Result |
|---|---:|
| `citation_precision` | `0.000000` |
| `claim_support_ratio` | `0.000000` |
| `tool_success_rate` | `0.000000` |
| `latency_p95` | `65.058724s` |

### Tier Breakdown

| Tier | Questions | Citation Precision | Claim Support | Tool Success | Latency p95 |
|---|---:|---:|---:|---:|---:|
| `tier_1` | `20` | `0.000000` | `0.000000` | `0.000000` | `65.058724s` |
| `tier_2` | `20` | `0.000000` | `0.000000` | `0.000000` | `58.303537s` |
| `tier_3` | `10` | `0.000000` | `0.000000` | `0.000000` | `59.420572s` |

## Pass Rates

Pass definitions used in this summary:
- `keyword pass`: `keyword_coverage == 1.0`
- `citation pass`: `citation_count_ok == true`
- `strict pass`: both keyword pass and citation pass

Overall:
- Questions completed by harness (`error == null`): `50/50` (`100%`)
- Keyword pass: `0/50` (`0%`)
- Citation pass: `0/50` (`0%`)
- Strict pass: `0/50` (`0%`)

By tier (strict pass):
- `tier_1`: `0/20` (`0%`)
- `tier_2`: `0/20` (`0%`)
- `tier_3`: `0/10` (`0%`)

## Notable Failures

- `results/baseline-run.log` shows repeated OpenAI transport failures across the entire run:
  - `ResearchAgent run failed.`: `50` occurrences
  - `APIConnectionError: Connection error.`: `50` occurrences
  - `SSL: UNEXPECTED_EOF_WHILE_READING`: `100` occurrences
  - `Transient agent invoke error on attempt ...`: `150` occurrences
- `ResearchAgent.run()` returns a fallback error response on exception, so `eval_results.json` records `error: null` even when upstream calls fail; this produced completed rows with zero keyword coverage, zero citations, and zero tool calls.
- Elevated latency (`p95 = 65.06s`) is consistent with retry/backoff behavior during repeated connection failures.
