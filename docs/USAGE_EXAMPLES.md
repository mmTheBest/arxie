# Usage Examples

Use the Paperbase product API and `/app` for the current Study/Library product.
Use the RA CLI/API examples for legacy assistant, proposal, and eval workflows.

For API-keyed commands in development, prefer the repo env wrapper so `.env`
loads consistently in non-interactive agents:

```bash
./scripts/with-env.sh .venv/bin/python -m ra.cli query "your question"
```

For legacy RA API examples, start the legacy service first:

```bash
uvicorn ra.api.app:app --host 0.0.0.0 --port 8000
```

For current product examples, start the Paperbase API and worker and open:

```text
http://localhost:8080/app
```

## Documentation cross-reference by completed feature

| Priority | Usage section in this file | API docs | Migration notes |
|---|---|---|---|
| 0 | [Priority 0](#priority-0-inline-citation-formatting) | [Feature coverage + query contract](API.md#feature-coverage) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) |
| 1 | [Priority 1](#priority-1-full-text-analysis-in-the-tool-chain) | [Feature coverage](API.md#feature-coverage) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) |
| 2 | [Priority 2](#priority-2-deep-multi-hop-search) | [POST /query](API.md#post-query) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) |
| 3 | [Priority 3](#priority-3-structured-literature-review-mode) | [POST /api/lit-review](API.md#post-apilit-review) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) |
| 4 | [Priority 4](#priority-4-citation-graph-exploration) | [Feature coverage](API.md#feature-coverage) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) |
| 5 | [Priority 5](#priority-5-confidence-scoring) | [POST /query](API.md#post-query) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) |
| 6 | [Priority 6](#priority-6-interactive-conversational-mode) | [POST /api/chat](API.md#post-apichat) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) |
| 7 | [Priority 7](#priority-7-demo-and-visualization-artifacts) | [Feature coverage](API.md#feature-coverage) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) |
| 8 | [Priority 8](#priority-8-proposal-workspace-and-release-gate) | [Proposal workflow details](API.md#proposal-workflow-details) | [Proposal workflow migration](MIGRATION.md#proposal-workflow-migration-priority-8) |
| Study agent | [Study-agent memory and task runs](#study-agent-memory-and-task-runs) | [Legacy RA study-agent workflow details](API.md#legacy-ra-study-agent-workflow-details) | [Study-agent workflow migration](MIGRATION.md#study-agent-workflow-migration) |
| REL-001 | [8.8 release-gate evaluation](#88-proposal-release-gate-evaluation) | [Release-gate evaluator contract](API.md#proposal-release-gate-evaluator-rel-001) | [Proposal release-gate outputs](MIGRATION.md#proposal-release-gate-eval-outputs) |
| REL-002/003 | [8.12 mypy policy](#812-validate-mypy-release-policy-rel-003) and [8.13 duplicate-module unblock](#813-confirm-duplicate-module-blocker-is-cleared-rel-002) | [Mypy release policy gate](API.md#mypy-release-policy-gate-rel-003) | [Mypy policy outputs](MIGRATION.md#mypy-release-policy-outputs-rel-003) |
| REL-004 | [8.14 QA preflight + closure checks](#814-run-paperclip-qa-preflight-and-closure-checks-rel-004) | [Paperclip QA preflight and closure checks](API.md#paperclip-qa-preflight-and-closure-checks-rel-004) | [QA preflight/closure migration](MIGRATION.md#paperclip-qa-preflight-and-closure-checks-rel-004) |

## Priority 0: Inline citation formatting

### CLI

```bash
ra query "What are the key limitations of retrieval-augmented generation?"
```

Expected output shape (truncated):

```markdown
## Answer
... (Lewis et al., 2020) ... [Confidence: MEDIUM - 2 supporting, 1 contradicting]

## References
1. ...
```

### API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key limitations of retrieval-augmented generation?","deep":false}'
```

Citation parser hardening accepts apostrophe surnames in both ASCII and typographic forms (e.g., `O'Neil`/`O’Neil`) and diacritic surnames (e.g., `García`) in inline citation format:

```text
(O'Neil et al., 2024)
(O’Neil et al., 2024)
(García et al., 2024)
```

## Priority 1: Full-text analysis in the tool chain

### Query flow (agent-triggered)

Use method/results-specific prompts so the agent calls `read_paper_fulltext` internally:

```bash
ra query "From the original LoRA paper, summarize the experimental setup and results section details."
```

### Direct tool call (Python)

```python
import asyncio
import json
from ra.retrieval.unified import UnifiedRetriever
from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.tools.retrieval_tools import make_retrieval_tools

async def main() -> None:
    retriever = UnifiedRetriever()
    s2 = SemanticScholarClient()
    tools = make_retrieval_tools(retriever=retriever, semantic_scholar=s2)
    tool = next(t for t in tools if t.name == "read_paper_fulltext")
    raw = await tool.ainvoke({"paper_id": "1706.03762"})
    print(json.loads(raw))
    await s2.close()

asyncio.run(main())
```

## Priority 2: Deep multi-hop search

### CLI

```bash
ra query --deep "Compare LoRA vs QLoRA trade-offs with supporting evidence across follow-up studies."
```

### API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare LoRA vs QLoRA trade-offs.","deep":true}'
```

## Priority 3: Structured literature review mode

### CLI

```bash
ra lit-review "graph neural networks for molecular property prediction" --max-papers 15
```

### API

```bash
curl -X POST http://localhost:8000/api/lit-review \
  -H "Content-Type: application/json" \
  -d '{"topic":"graph neural networks for molecular property prediction","max_papers":15}'
```

Expected section layout:

- `## Introduction`
- `## Thematic Groups`
- `## Key Findings`
- `## Research Gaps`
- `## Future Directions`

## Priority 4: Citation graph exploration

### CLI timeline output

```bash
ra trace "Attention Is All You Need" --max-depth 4 --citations-per-paper 10
```

Example line format:

```text
2017 -> Seed Paper -> cited by -> 2019: Follow-up Paper
```

### Direct tool call (Python)

```python
import asyncio
import json
from ra.retrieval.unified import UnifiedRetriever
from ra.retrieval.semantic_scholar import SemanticScholarClient
from ra.tools.retrieval_tools import make_retrieval_tools

async def main() -> None:
    retriever = UnifiedRetriever()
    s2 = SemanticScholarClient()
    tools = make_retrieval_tools(retriever=retriever, semantic_scholar=s2)
    tool = next(t for t in tools if t.name == "trace_influence")
    raw = await tool.ainvoke({"paper": "Attention Is All You Need", "max_depth": 2})
    print(json.loads(raw)["timeline"])
    await s2.close()

asyncio.run(main())
```

## Priority 5: Confidence scoring

Confidence annotations are attached to major claims in query/chat answers.

```bash
ra query "Do transformers improve machine translation quality across benchmarks?"
```

Look for suffixes like:

```text
[Confidence: HIGH - 4 supporting, 0 contradicting]
```

## Priority 6: Interactive conversational mode

### CLI REPL

```bash
ra chat --session-id thesis-session
```

Then enter multiple turns and exit with `/exit`.

### API multi-turn

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?","session_id":"session-42"}'

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"And what are its limitations?","session_id":"session-42"}'
```

Reuse the same `session_id` to preserve context.

## Priority 7: Demo and visualization artifacts

Priority 7 added demo/visualization artifacts. These are non-API assets and do
not change runtime commands, REST payloads, or CLI integration.

Use the API and CLI examples below for release-facing behavior.

## Priority 8: Proposal workspace and release gate

### 8.1 Create proposal session

```bash
curl -X POST http://localhost:8000/api/proposal/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id":"proposal-session-1"}'
```

### 8.2 Patch a stage payload with optimistic version

```bash
curl -X PATCH http://localhost:8000/api/proposal/sessions/proposal-session-1/stages/idea_intake \
  -H "Content-Type: application/json" \
  -d '{
    "expected_version": 0,
    "payload": {
      "problem": "Proposal drafts lose citation provenance.",
      "target_population": "Academic researchers",
      "mechanism": "Stage-gated drafting with evidence tagging",
      "expected_outcome": "Higher trust and traceability"
    }
  }'
```

### 8.2b Patch a non-current stage snapshot (allowed)

You can update any stage payload directly (with the correct `expected_version`), even if it is not the current active stage:

```bash
curl -X PATCH http://localhost:8000/api/proposal/sessions/proposal-session-1/stages/evidence_mapping \
  -H "Content-Type: application/json" \
  -d '{
    "expected_version": 1,
    "payload": {
      "supporting_evidence": ["paper-1"],
      "contradicting_evidence": [],
      "landscape_summary": "Early evidence is favorable."
    }
  }'
```

### 8.3 Advance to next stage

```bash
curl -X POST http://localhost:8000/api/proposal/sessions/proposal-session-1/advance \
  -H "Content-Type: application/json" \
  -d '{"expected_version":1}'
```

### 8.3b Handle deterministic gate rejection (`409`)

If required fields are missing, advance returns `stage_transition_rejected`:

```json
{
  "error": "stage_transition_rejected",
  "message": "Stage 'idea_intake' is incomplete. Fill missing fields: problem.",
  "details": [
    {
      "reason": "incomplete_stage",
      "from_stage": "idea_intake",
      "to_stage": "logic_refinement",
      "missing_fields": ["problem"]
    }
  ]
}
```

### 8.3c Handle optimistic concurrency conflicts (`409`)

If your `expected_version` is stale, stage writes return `version_conflict`:

```json
{
  "error": "version_conflict",
  "message": "Version conflict for session 'proposal-session-1': expected 2, current 3.",
  "details": [
    {
      "session_id": "proposal-session-1",
      "expected_version": 2,
      "current_version": 3
    }
  ]
}
```

Retry flow:

1. `GET /api/proposal/sessions/{session_id}` for latest `version`
2. Re-apply your stage payload with the fresh `expected_version`

### 8.4 Map evidence for a proposal claim

```bash
curl -X POST http://localhost:8000/api/proposal/evidence/query \
  -H "Content-Type: application/json" \
  -d '{
    "claim":"Transformers improve machine translation quality across benchmarks.",
    "pinned_paper_ids":["p1"],
    "papers":[
      {
        "paper_id":"p1",
        "title":"Positive evidence",
        "abstract":"Transformers improve machine translation quality.",
        "doi":"10.1000/xyz123"
      },
      {
        "paper_id":"p2",
        "title":"Adjacent evidence",
        "abstract":"Transformer deployment constraints and runtime trade-offs.",
        "pdf_url":"https://example.org/evidence.pdf"
      }
    ]
  }'
```

### 8.5 Create, compare, and promote branches

```bash
curl -X POST http://localhost:8000/api/proposal/branches \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"proposal-session-1",
    "branch_id":"branch-a",
    "name":"Primary mechanism",
    "hypothesis":"Mechanism A drives outcome B.",
    "scorecard":{
      "evidence_support":0.8,
      "feasibility":0.7,
      "risk":0.3,
      "impact":0.9
    }
  }'

curl -X POST http://localhost:8000/api/proposal/branches/compare \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"proposal-session-1",
    "branch_ids":["branch-a","branch-b"]
  }'

curl -X POST http://localhost:8000/api/proposal/branches/proposal-session-1/branch-a/promote
```

Example compare response fields to consume:

```json
{
  "session_id": "proposal-session-1",
  "winner_branch_id": "branch-a",
  "comparisons": [
    {
      "branch_id": "branch-a",
      "confidence_label": "high",
      "aggregate_score": 0.85,
      "is_primary": true
    }
  ]
}
```

`aggregate_score` is deterministic: `(evidence_support + feasibility + (1 - risk) + impact) / 4`.

### 8.6 Configure artifact dependencies and propagate stale markers

```bash
curl -X PUT http://localhost:8000/api/proposal/artifacts/proposal-session-1/nodes/logical_tree/logic-1 \
  -H "Content-Type: application/json" \
  -d '{"content":"Problem -> mechanism chain"}'

curl -X PUT http://localhost:8000/api/proposal/artifacts/proposal-session-1/nodes/evidence_map/evidence-1 \
  -H "Content-Type: application/json" \
  -d '{"content":"Evidence bucket baseline","provenance_link":"https://doi.org/10.1000/xyz123"}'

curl -X POST http://localhost:8000/api/proposal/artifacts/proposal-session-1/dependencies \
  -H "Content-Type: application/json" \
  -d '{
    "upstream_artifact":"logical_tree",
    "upstream_node_id":"logic-1",
    "downstream_artifact":"evidence_map",
    "downstream_node_id":"evidence-1"
  }'

curl -X POST http://localhost:8000/api/proposal/artifacts/proposal-session-1/edits \
  -H "Content-Type: application/json" \
  -d '{"artifact":"logical_tree","node_id":"logic-1","content":"Problem -> mechanism chain (revised)"}'
```

Check dependency snapshot:

```bash
curl http://localhost:8000/api/proposal/artifacts/proposal-session-1/dependencies/logical_tree/logic-1
```

Open provenance redirect:

```bash
curl -i http://localhost:8000/api/proposal/artifacts/proposal-session-1/provenance/evidence_map/evidence-1
```

### 8.6b Handle dependency validation errors

Self-edge rejection:

```bash
curl -X POST http://localhost:8000/api/proposal/artifacts/proposal-session-1/dependencies \
  -H "Content-Type: application/json" \
  -d '{
    "upstream_artifact":"logical_tree",
    "upstream_node_id":"logic-1",
    "downstream_artifact":"logical_tree",
    "downstream_node_id":"logic-1"
  }'
```

Expected response shape:

```json
{
  "error": "invalid_input",
  "message": "dependency edge must connect different nodes"
}
```

### 8.7 Export proposal draft with completeness checks

```bash
curl "http://localhost:8000/api/proposal/sessions/proposal-session-1/export?format=markdown"
curl "http://localhost:8000/api/proposal/sessions/proposal-session-1/export?format=pdf"
```

Example completeness payload fragment:

```json
{
  "completeness": {
    "complete": false,
    "sections": [
      {
        "section_id": "references",
        "title": "References",
        "complete": false,
        "missing_fields": [
          "evidence_mapping.supporting_evidence | evidence_mapping.contradicting_evidence | proposal_assembly.references"
        ]
      }
    ]
  }
}
```

Method completeness note:

- Stage advancement validates `experiment_flow_diagram`, `analysis_plan_tree`, and `outcome_comparison_matrix`.
- Export `method` completeness currently checks `proposal_assembly.method_summary`, `data_feasibility_planning.selected_data_strategy`, `experiment_analysis_design.experiment_flow_diagram`, `experiment_analysis_design.analysis_plan_tree`, and `experiment_analysis_design.outcome_comparison_matrix`.

### 8.8 Proposal release-gate evaluation

```bash
ra eval \
  --dataset path/to/proposal-release-gate-dataset.json \
  --output ./release-output/ \
  --mode proposal_release_gate
```

Expected artifacts:

- `release-output/proposal_release_gate_results.json`
- `release-output/proposal_release_gate_summary.md`

`release-output/proposal_release_gate_summary.md` contains:

- `## Topline Metrics` table:
  - `stage_completion_pass_rate`
  - `evidence_link_pass_rate`
  - `gate_pass_rate`
- `## Case Results` table with per-case:
  - `Case ID`
  - `Stage Ratio`
  - `Link Coverage`
  - `Pass`

### 8.8a QA benchmark eval mode (default)

If `--mode` is omitted, `ra eval` uses the QA benchmark harness:

```bash
ra eval \
  --dataset path/to/qa-dataset.json \
  --output ./release-output/
```

Expected payload keys:

- `mode` (`qa`)
- `total_questions`
- `metrics`
- `artifacts`

### 8.9 Validate release-gate metrics contract

Inspect the generated JSON:

```bash
jq '{mode, total_cases, metrics, overall_pass, artifacts}' results/proposal_release_gate_results.json
```

Minimum fields to enforce in automation:

- `total_cases` (must be `> 0`)
- `metrics.stage_completion_pass_rate`
- `metrics.evidence_link_pass_rate`
- `metrics.gate_pass_rate`
- `overall_pass`

Stage completion denominator in the current implementation is the full canonical stage sequence (including `proposal_assembly`), so each case row includes:

- `completed_stage_count`
- `expected_stage_count`
- `stage_completion_ratio`

### 8.10 Verify fail-closed behavior for empty datasets (REL-001 hardening)

```bash
printf '[]\n' >/tmp/arxie-empty-release-gate.json
python -m ra.cli eval \
  --dataset /tmp/arxie-empty-release-gate.json \
  --output /tmp/arxie-empty-out \
  --mode proposal_release_gate
echo "exit_code=$?"
```

Expected behavior:

- command exits with status `1`
- output is JSON with `type` set to `ValueError`

Example output shape:

```json
{
  "error": "Release-gate dataset must contain at least one case.",
  "type": "ValueError"
}
```

### 8.11 Proposal conversations and inspector contracts

Append a conversation message:

```bash
curl -X POST http://localhost:8000/api/proposal/conversations/proposal-session-1/messages \
  -H "Content-Type: application/json" \
  -d '{"role":"user","content":"Summarize current stage risks.","metadata":{"ui_source":"dashboard"}}'
```

Allowed `role` values: `user`, `assistant`, `system`.

List the thread:

```bash
curl http://localhost:8000/api/proposal/conversations/proposal-session-1/messages
```

Example thread response:

```json
{
  "session_id": "proposal-session-1",
  "count": 1,
  "messages": [
    {
      "message_id": "18fa65ed-bfef-4db5-bb36-fd6db315dc0f",
      "session_id": "proposal-session-1",
      "role": "user",
      "content": "Summarize current stage risks.",
      "metadata": {
        "ui_source": "dashboard"
      },
      "created_at": "2026-03-13T10:29:09Z"
    }
  ]
}
```

Inspector payload (currently placeholder shape):

```bash
curl http://localhost:8000/api/proposal/evidence/proposal-session-1/inspector
```

Example inspector response:

```json
{
  "session_id": "proposal-session-1",
  "count": 0,
  "items": []
}
```

### 8.12 Validate mypy release policy (REL-003)

Run the executable policy module:

```bash
python -m ra.eval.mypy_gate
```

Expected JSON fields:

- `policy_id` (`arxie-v0.2-mypy-release-gate`)
- `overall_pass` (boolean gate result)
- `checks[]` entries for:
  - `strict_scope`
  - `repo_baseline`

Example result fragment:

```json
{
  "policy_id": "arxie-v0.2-mypy-release-gate",
  "overall_pass": true,
  "checks": [
    {"name": "strict_scope", "pass": true},
    {"name": "repo_baseline", "pass": true}
  ]
}
```

### 8.13 Confirm duplicate-module blocker is cleared (REL-002)

Run the executable policy gate:

```bash
python -m ra.eval.mypy_gate
```

Expected behavior:

- command exits successfully when the strict scope and baseline checks pass
- it should no longer fail because a source file is discovered under duplicate module names

### 8.14 Run Paperclip QA preflight and closure checks (REL-004)

Maintainer branches should run PRD, reviewer, eval, and release-gate checks as
a single preflight before closure or public release sync. Public releases expose
the runtime smoke workflow and install/runtime commands.

Expected behavior:

- PRD schema check passes.
- Definition-of-Done checkbox strict check passes.
- Reviewer checkpoint contains all required fields plus `Release Verdict: PASS|FAIL`.
- Eval tests and release-gate checks pass before closure-ready output.
- Any failed gate exits non-zero and blocks closure.

## Study-agent memory and task runs

Create a durable study brief:

```bash
curl -X POST http://localhost:8000/api/studies \
  -H "Content-Type: application/json" \
  -d '{"study_id":"demo-study","title":"Benchmark review","research_goal":"Find stronger experiment designs for my study","collection_id":"demo-collection","current_method":"retrieval-augmented assistant","datasets":["internal benchmark"],"metrics":["citation precision"]}'
```

Attach user-provided context:

```bash
curl -X POST http://localhost:8000/api/studies/demo-study/sources \
  -H "Content-Type: application/json" \
  -d '{"source_type":"result_summary","title":"Current eval notes","content":"The current benchmark shows strong citation precision but weak coverage of ablations.","summary":"Current results need better ablation coverage.","extracted_facts":["citation precision is strong","ablation coverage is weak"]}'
```

Run a deterministic study-agent task:

```bash
curl -X POST http://localhost:8000/api/studies/demo-study/runs \
  -H "Content-Type: application/json" \
  -d '{"task_type":"design_experiments","query":"What else should I test before writing this up?","papers":[]}'
```

Inspect the run trace:

```bash
curl http://localhost:8000/api/studies/demo-study/runs
```

Supported task types:

- `design_experiments`
- `find_benchmarks`
- `review_draft_claims`

## Additional core API examples

### Batch search

```bash
curl -X POST http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -d '{"requests":[{"query":"rag benchmark","limit":5,"source":"semantic_scholar"},{"query":"long-context transformers","limit":5,"source":"both"}],"max_concurrency":4}'
```

### Batch retrieve

```bash
curl -X POST http://localhost:8000/retrieve/batch \
  -H "Content-Type: application/json" \
  -d '{"requests":[{"identifier":"10.5555/3295222.3295349"},{"identifier":"1706.03762"}],"max_concurrency":8}'
```
