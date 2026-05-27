# Migration Notes (`v0.1.0` stable -> `v0.2.0` feature set)

This guide covers migration from earlier Arxie builds to the current implemented
feature set, including the Paperbase Study/Library product, proposal workspace,
and release-gate evaluator.

## Compatibility snapshot

| Surface | Compatibility | Notes |
|---|---|---|
| `/search`, `/retrieve`, batch variants | Backward compatible | Top-level request/response shapes unchanged |
| `/query` | Backward compatible with additive field | `deep` flag is additive |
| `/answer` | Deprecated | Alias of `/query`; migrate new clients to `/query` |
| `/api/chat`, `/api/lit-review` | Additive | New endpoints/capabilities |
| `/api/proposal/*` | New surface | Session-first proposal workflow and export |
| `/api/studies/*` | New surface | Local study memory, user sources, deterministic study-agent task runs, and trace retrieval |
| `/api/v1/*` | New Paperbase product surface | Project-scoped libraries, ingestion, parse/extract jobs, Study sources, research threads, artifacts, and run traces |
| CLI `ra eval` (default `qa`) | Backward compatible | Existing QA harness flow preserved; JSON now includes `mode: "qa"` |
| CLI `ra eval --mode proposal_release_gate` | New mode | Emits release-gate JSON + Markdown artifacts (topline-metrics and case-results tables); dataset must be non-empty |
| `python -m ra.eval.mypy_gate` | New release policy command | Enforces v0.2 strict-scope + no-regression baseline checks |
| QA preflight/closure gates | New release workflow automation | Maintainer branches enforce PRD/reviewer/eval/release-gate checks before closure or release sync |

## Change summary by completed feature

| Priority | Change | Action required |
|---|---|---|
| 0 | Inline citation output normalized and parser hardened for apostrophe and diacritic surnames | Update strict citation regexes if they previously rejected names like `O'Neil` (or typographic-apostrophe variants) and `García`-style diacritic surnames |
| 1 | Full-text parsing integrated in reasoning flow | Expect higher latency on methods/results-heavy prompts |
| 2 | Deep mode (`deep` / `--deep`) | Opt in where multi-hop evidence synthesis is needed |
| 3 | Structured lit-review mode | Add integrations for `ra lit-review` or `POST /api/lit-review` |
| 4 | Citation influence tracing (`ra trace`) | Integrate timeline output if needed |
| 5 | Confidence annotations in claims | Update render/parsing logic if you post-process answer sentences |
| 6 | Stateful chat mode | Pass stable `session_id` for multi-turn continuity |
| 7 | Demo/visualization assets | No API migration; asset-only |
| 8 | Proposal workspace + release-gate evaluator | Add `/api/proposal/*` integration and eval mode as needed |
| Study agent | Local study memory and deterministic task runtime | Add `/api/studies/*` integration when building agent workflows that need a durable study brief, user sources, or run traces |
| REL-001 | Release-gate contract hardening | Treat empty datasets as hard failures and validate full-stage coverage semantics |
| REL-002/003 | Mypy gate unblock + executable typing policy | Add mypy policy check to release automation and monitor baseline regressions |
| REL-004 | Paperclip QA preflight and closure automation | Add equivalent PRD/reviewer/eval/release gates to maintainer merge/release workflow |

## Documentation cross-reference by completed feature

| Priority | Migration section in this file | API docs | Usage examples |
|---|---|---|---|
| 0 | [Change summary](#change-summary-by-completed-feature) | [Feature coverage + query output contract](API.md#feature-coverage) | [Priority 0](USAGE_EXAMPLES.md#priority-0-inline-citation-formatting) |
| 1 | [Change summary](#change-summary-by-completed-feature) | [Feature coverage](API.md#feature-coverage) | [Priority 1](USAGE_EXAMPLES.md#priority-1-full-text-analysis-in-the-tool-chain) |
| 2 | [Change summary](#change-summary-by-completed-feature) | [POST /query](API.md#post-query) | [Priority 2](USAGE_EXAMPLES.md#priority-2-deep-multi-hop-search) |
| 3 | [Change summary](#change-summary-by-completed-feature) | [POST /api/lit-review](API.md#post-apilit-review) | [Priority 3](USAGE_EXAMPLES.md#priority-3-structured-literature-review-mode) |
| 4 | [Change summary](#change-summary-by-completed-feature) | [Feature coverage](API.md#feature-coverage) | [Priority 4](USAGE_EXAMPLES.md#priority-4-citation-graph-exploration) |
| 5 | [Change summary](#change-summary-by-completed-feature) | [POST /query](API.md#post-query) | [Priority 5](USAGE_EXAMPLES.md#priority-5-confidence-scoring) |
| 6 | [Change summary](#change-summary-by-completed-feature) | [POST /api/chat](API.md#post-apichat) | [Priority 6](USAGE_EXAMPLES.md#priority-6-interactive-conversational-mode) |
| 7 | [Change summary](#change-summary-by-completed-feature) | [Feature coverage](API.md#feature-coverage) | [Priority 7](USAGE_EXAMPLES.md#priority-7-demo-and-visualization-artifacts) |
| 8 | [Proposal workflow migration](#proposal-workflow-migration-priority-8) | [Proposal workflow details](API.md#proposal-workflow-details) | [Priority 8](USAGE_EXAMPLES.md#priority-8-proposal-workspace-and-release-gate) |
| REL-001 | [Proposal release-gate eval outputs](#proposal-release-gate-eval-outputs) | [Release-gate evaluator contract](API.md#proposal-release-gate-evaluator-rel-001) | [8.8 release-gate evaluation](USAGE_EXAMPLES.md#88-proposal-release-gate-evaluation) |
| REL-002/003 | [Mypy release policy outputs](#mypy-release-policy-outputs-rel-003) | [Mypy release policy gate](API.md#mypy-release-policy-gate-rel-003) | [8.12 mypy policy](USAGE_EXAMPLES.md#812-validate-mypy-release-policy-rel-003) and [8.13 duplicate-module unblock](USAGE_EXAMPLES.md#813-confirm-duplicate-module-blocker-is-cleared-rel-002) |
| REL-004 | [Paperclip QA preflight and closure checks](#paperclip-qa-preflight-and-closure-checks-rel-004) | [Paperclip QA preflight and closure checks](API.md#paperclip-qa-preflight-and-closure-checks-rel-004) | [8.14 QA preflight + closure checks](USAGE_EXAMPLES.md#814-run-paperclip-qa-preflight-and-closure-checks-rel-004) |

## Core API migration details

### 1. Prefer `/query` over `/answer`

Old:

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?"}'
```

New:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","deep":false}'
```

### 2. Enforce Markdown output contract in downstream consumers

`answer` fields are Markdown with both sections:

```markdown
## Answer
...

## References
1. ...
2. ...
```

If your UI strips headers, keep both sections available to users.

### 3. Chat requires `session_id`

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Follow-up question","session_id":"session-1"}'
```

Missing `session_id` returns `422`.

## Proposal workflow migration (Priority 8)

### 1. Move to session-first state management

All proposal writes/reads are scoped to `session_id`:

- `POST /api/proposal/sessions`
- `GET /api/proposal/sessions/{session_id}`

### 2. Add optimistic concurrency on stage writes

Stage updates and stage advances require `expected_version`:

- `PATCH /api/proposal/sessions/{session_id}/stages/{stage}`
- `POST /api/proposal/sessions/{session_id}/advance`

Handle `409 version_conflict` by re-reading session state and retrying with latest version.

Behavioral notes:

- Stage payload updates can target any stage path directly.
- Version validation happens before stage-completeness transition checks.

### 3. Respect deterministic stage gates

Advance can fail with `409 stage_transition_rejected` and structured details:

- `reason` (`incomplete_stage`, `invalid_transition`, `final_stage_reached`)
- `missing_fields`
- `from_stage`, `to_stage`

Required fields per stage:

| Stage | Required fields |
|---|---|
| `idea_intake` | `problem`, `target_population`, `mechanism`, `expected_outcome` |
| `logic_refinement` | `problem_gap_chain`, `core_assumptions`, `testable_hypothesis` |
| `evidence_mapping` | `supporting_evidence`, `contradicting_evidence`, `landscape_summary` |
| `hypothesis_reshaping` | `candidate_hypotheses`, `falsification_criteria`, `selected_primary_hypothesis` |
| `data_feasibility_planning` | `data_options_table`, `feasibility_scorecard`, `selected_data_strategy` |
| `experiment_analysis_design` | `experiment_flow_diagram`, `analysis_plan_tree`, `outcome_comparison_matrix` |
| `proposal_assembly` | `background_and_gap`, `hypothesis_statement`, `method_summary`, `analysis_summary`, `expected_outcomes`, `risks_and_limitations` |

### 4. Integrate evidence mapping payload

Use:

- `POST /api/proposal/evidence/query`

Expect bucketed outputs (`supporting`, `contradicting`, `adjacent`) plus `landscape_summary`.

### 5. Integrate hypothesis branching operations

New endpoints:

- `POST /api/proposal/branches`
- `GET /api/proposal/branches/{session_id}`
- `GET /api/proposal/branches/{session_id}/{branch_id}`
- `POST /api/proposal/branches/compare`
- `POST /api/proposal/branches/{session_id}/{branch_id}/promote`

Branch contract notes:

- First branch in a session is auto-marked `is_primary=true`.
- Compare requires at least two unique `branch_ids`.
- Winner scoring is deterministic: `(evidence_support + feasibility + (1 - risk) + impact) / 4`.
- Promote sets target branch as primary and demotes all others.

### 6. Integrate cross-artifact synchronization

New endpoints:

- `PUT /api/proposal/artifacts/{session_id}/nodes/{artifact}/{node_id}`
- `POST /api/proposal/artifacts/{session_id}/dependencies`
- `POST /api/proposal/artifacts/{session_id}/edits`
- `GET /api/proposal/artifacts/{session_id}/dependencies/{artifact}/{node_id}`
- `GET /api/proposal/artifacts/{session_id}/provenance/{artifact}/{node_id}`

`/edits` marks downstream dependent nodes as `stale=true`; `/provenance/...` returns `307` redirect.

Dependency validation constraints:

- both endpoint nodes must exist before adding a dependency edge
- self-edge creation fails with `400 invalid_input`
- cycle-forming edges fail with `400 invalid_input`

### 7. Integrate export completeness contract

Use:

- `GET /api/proposal/sessions/{session_id}/export?format=markdown|pdf`

Response now includes completeness metadata:

- `completeness.complete`
- per-section checks for `framing`, `evidence`, `hypothesis`, `method`, `outcomes`, `risks`, `references`
- each section includes `section_id`, `title`, `complete`, `missing_fields`

Method completeness nuance:

- Stage advancement for `experiment_analysis_design` validates:
  - `experiment_flow_diagram`
  - `analysis_plan_tree`
  - `outcome_comparison_matrix`
- Export `method` completeness currently checks:
  - `proposal_assembly.method_summary`
  - `data_feasibility_planning.selected_data_strategy`
  - `experiment_analysis_design.experiment_flow_diagram`
  - `experiment_analysis_design.analysis_plan_tree`
  - `experiment_analysis_design.outcome_comparison_matrix`

### 8. Add proposal conversation and inspector contracts

- `POST /api/proposal/conversations/{session_id}/messages`
- `GET /api/proposal/conversations/{session_id}/messages`
- `GET /api/proposal/evidence/{session_id}/inspector`

Inspector currently returns placeholder payload shape; integrate schema now and treat content as evolving.

Conversation payload constraints:

- `role` must be one of: `user`, `assistant`, `system`
- `metadata` is persisted as a string map (`dict[str, str]`)

## Study-agent workflow migration

The study-agent surface is additive. Existing query/chat/lit-review/proposal clients do not need to change unless they want durable study memory and traceable research-agent tasks.

New endpoints:

- `POST /api/studies`
- `GET /api/studies/{study_id}`
- `PATCH /api/studies/{study_id}/brief`
- `POST /api/studies/{study_id}/sources`
- `POST /api/studies/{study_id}/runs`
- `GET /api/studies/{study_id}/runs`
- `GET /api/studies/{study_id}/runs/{run_id}`

Integration notes:

- `StudyBrief` writes use versioned state. Use `expected_version` on brief updates.
- `StudySource` stores pasted context and summaries, not arbitrary filesystem reads.
- Supported source types are `draft`, `note`, `code_summary`, and `result_summary`.
- Supported deterministic task types are `design_experiments`, `find_benchmarks`, and `review_draft_claims`.
- `StudyAgentRun` responses include trace steps, warnings, output, evidence references, and next actions.
- User-source facts are separate from paper-derived evidence in `EvidenceReference` records.

### 9. Handle provenance error differentiation

`GET /api/proposal/artifacts/{session_id}/provenance/{artifact}/{node_id}` can return:

- `404 artifact_node_not_found` (node does not exist)
- `404 provenance_not_found` (node exists but has no provenance URL)

Handle both as distinct user-facing states.

## CLI migration details

### Added/updated commands

- `ra query "..." --deep`
- `ra lit-review "<topic>" --max-papers N`
- `ra trace "<paper or concept>" --max-depth N --citations-per-paper N --max-papers N`
- `ra chat --session-id ID [--deep]`
- `ra eval --dataset <path> --output <dir>` (defaults to `qa`)
- `ra eval --dataset <path> --output <dir> --mode proposal_release_gate`

### QA eval mode (default) remains backward compatible

When `--mode` is omitted (or set to `qa`), CLI behavior matches the prior benchmark harness flow:

- uses `tests.eval.harness.EvalHarness`
- writes the existing QA artifacts under `<output>`
- returns JSON with `total_questions`, `metrics`, and `artifacts`

Additive output change:

- response now includes `mode` (value `qa`)

### Proposal release-gate eval outputs

When `mode=proposal_release_gate`, CLI writes:

- `<output>/proposal_release_gate_results.json`
- `<output>/proposal_release_gate_summary.md`

Markdown summary contract:

- `## Topline Metrics` table with:
  - `stage_completion_pass_rate`
  - `evidence_link_pass_rate`
  - `gate_pass_rate`
- `## Case Results` table with:
  - `Case ID`
  - `Stage Ratio`
  - `Link Coverage`
  - `Pass`

JSON includes:

- `total_cases`
- `metrics.stage_completion_pass_rate`
- `metrics.evidence_link_pass_rate`
- `metrics.gate_pass_rate`
- `overall_pass`

Release-gate contract notes:

- Dataset must be a non-empty JSON list; empty datasets fail with CLI exit code `1`.
- Stage completion is evaluated against the full canonical stage sequence from `ProposalStageEngine.stage_sequence` (currently 7 stages, including `proposal_assembly`).
- `overall_pass` is true only when every case passes both thresholds.

Dataset row schema (one case):

- `id`
- `session_id`
- `stage_payloads` (object keyed by stage)
- `min_stage_completion_ratio` (0-1)
- `min_evidence_link_coverage` (0-1)

Recommended release-automation behavior:

- runs a positive dataset check
- verifies `total_cases > 0`
- verifies empty-dataset fail-closed behavior

### Mypy release policy outputs (REL-003)

Run:

```bash
python -m ra.eval.mypy_gate
```

Output contract:

- top-level fields: `policy_id`, `overall_pass`, `checks`, `commands`, `python_bin`
- check entries include:
  - `name` (`strict_scope`, `repo_baseline`)
  - `max_errors`, `max_files_with_errors`
  - observed counts and `pass`

Current policy expectations:

- `strict_scope` must remain zero-error for `src/ra/proposal` + `src/ra/eval` (`--follow-imports=skip`)
- `repo_baseline` is no-regression gated against accepted full-repo debt thresholds

### Paperclip QA preflight and closure checks (REL-004)

Maintainer branches should run the PRD, reviewer, eval, and release-gate checks
as a single preflight before closure or release sync. Public releases expose the
runtime smoke workflow and install/runtime commands.

## Behavioral differences to plan for

### Citation auto-repair

When the model omits inline citations, Arxie may inject evidence lines from retrieved papers. Exact wording can differ from older runs.

### Confidence suffixes

Major claims may include:

```text
[Confidence: HIGH - 8 supporting, 1 contradicting]
```

Preserve or intentionally strip these suffixes in post-processing pipelines.

### Full-text and deep-mode cost

Deep mode and methods/results-focused prompts can trigger extra retrieval and PDF parsing calls. Plan for additional runtime.

## Migration checklist

1. Switch new query clients from `/answer` to `/query`.
2. Ensure query/chat markdown parsing keeps both `## Answer` and `## References`.
3. Pass `session_id` for chat clients.
4. Add proposal session/version handling if integrating `/api/proposal/*`.
5. Handle `409 version_conflict` and `409 stage_transition_rejected` in proposal clients.
6. If using proposal exports, consume completeness metadata before considering drafts release-ready.
7. Ensure proposal conversation clients enforce role values (`user`/`assistant`/`system`) and send string metadata values.
8. Handle `artifact_node_not_found` vs `provenance_not_found` distinctly for provenance click-through calls.
9. If using release automation, wire `ra eval --mode proposal_release_gate` and parse `overall_pass`.
10. Assert `total_cases > 0` for proposal release-gate runs; treat empty datasets as invalid releases.
11. Run an equivalent release-gate check that verifies positive and fail-closed behavior.
12. If using release automation, run `python -m ra.eval.mypy_gate` and enforce `overall_pass=true`.
13. Run PRD, reviewer, eval, and release-gate preflight checks before closure or release sync.

## Feature-level upgrade validation

Use this quick pass to confirm all completed priorities (`0`-`8`) are integrated in your downstream client:

1. Priority 0-2: verify query consumers accept inline citations, confidence suffixes, and optional `deep`.
2. Priority 3: verify lit-review renderers support fixed section headers.
3. Priority 4: verify `ra trace` timeline output is parsed as text + JSON payload.
4. Priority 5: verify confidence suffix handling is explicit (retain or strip intentionally).
5. Priority 6: verify chat clients always send stable `session_id`.
6. Priority 7: verify demo assets are treated as non-API build artifacts.
7. Priority 8: verify session/version flow, branch compare/promote behavior, artifact dependency validation, provenance error handling, and export completeness gating before release.
8. REL-001: verify proposal release-gate runs reject empty datasets and report full-stage completion metrics.
9. REL-002/003: verify development-branch type checking no longer fails with duplicate-module collision, and confirm `ra.eval.mypy_gate` reports both checks as passing.
10. REL-004: verify PRD/reviewer validation, eval checks, and release-gate checks all pass before release sync.
