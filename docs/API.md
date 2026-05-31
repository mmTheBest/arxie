# Arxie API Reference

Local API URLs depend on which service entrypoint is running:

- Paperbase product API and `/app`: usually `http://localhost:8080`
- legacy RA assistant API: usually `http://localhost:8000`

OpenAPI docs:

- `GET /docs`
- `GET /openapi.json`

`/openapi.json` is the source of truth for request/response schemas.

## Service Surfaces

Arxie currently exposes two compatible API surfaces:

- **Paperbase product API** under `/api/v1/*`, plus `/` and `/app`. This is the
  current Study/Library browser product, collection database, project-scoped
  runtime, background job surface, and research-agent run trace API.
- **Legacy RA assistant API** under top-level paths such as `/query`,
  `/api/chat`, `/api/lit-review`, and `/api/proposal/*`. This remains available
  for CLI, proposal, and compatibility workflows.

When a browser or client opens a project-scoped library, send
`X-Arxie-Project-Id: <project_id>` on subsequent Paperbase API requests. The
header routes database sessions and background jobs to that project's `.arxie/`
runtime instead of the default database.

## Feature coverage

| Priority | Completed feature | API surface |
|---|---|---|
| 0 | Inline citation formatting + citation parser hardening (including apostrophe and diacritic surname variants) | `POST /query`, `POST /api/chat` answer text |
| 1 | Full-text analysis in agent tool chain | Internal to query/lit-review pipelines |
| 2 | Deep multi-hop search mode | `POST /query` with `deep: true` |
| 3 | Structured literature review generation | `POST /api/lit-review` |
| 4 | Citation influence tracing | Tool-level (`trace_influence`), no dedicated REST route |
| 5 | Claim confidence scoring | Query/chat answer text annotations |
| 6 | Stateful conversational mode | `POST /api/chat` with `session_id` |
| 7 | Demo/visualization assets | Not a REST surface |
| 8 | Proposal workspace + export completeness + artifact sync + branching | `POST/GET/PATCH /api/proposal/*` |
| Paperbase Study/Library | Project-scoped libraries, local PDF upload/import, parse/extract jobs, Study sources, research threads, artifacts, and run traces | `/api/v1/projects`, `/api/v1/collections`, `/api/v1/studies`, `/api/v1/research/*`, `/api/v1/jobs` |
| Legacy study agent | Local study memory, user sources, deterministic task runs, and trace retrieval | `POST/GET/PATCH /api/studies/*` in the legacy RA API |
| REL-001 | Proposal release-gate evaluator contract hardening | `ra eval --mode proposal_release_gate` |
| REL-002 | Mypy duplicate-module gate unblock | Packaging/runtime fix; enables development-branch type checking |
| REL-003 | Executable v0.2 typing policy gate | `python -m ra.eval.mypy_gate` |
| REL-004 | Paperclip QA preflight and closure gating automation | Development-branch release gates; public runtime smoke checks use the release workflow |

## Documentation cross-reference by completed feature

| Priority | API section in this file | Migration notes | Usage examples |
|---|---|---|---|
| 0 | [Feature coverage](#feature-coverage) + [POST /query](#post-query) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) | [Priority 0](USAGE_EXAMPLES.md#priority-0-inline-citation-formatting) |
| 1 | [Feature coverage](#feature-coverage) + [POST /query](#post-query) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) | [Priority 1](USAGE_EXAMPLES.md#priority-1-full-text-analysis-in-the-tool-chain) |
| 2 | [POST /query](#post-query) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) | [Priority 2](USAGE_EXAMPLES.md#priority-2-deep-multi-hop-search) |
| 3 | [POST /api/lit-review](#post-apilit-review) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) | [Priority 3](USAGE_EXAMPLES.md#priority-3-structured-literature-review-mode) |
| 4 | [Feature coverage](#feature-coverage) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) | [Priority 4](USAGE_EXAMPLES.md#priority-4-citation-graph-exploration) |
| 5 | [POST /query](#post-query) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) | [Priority 5](USAGE_EXAMPLES.md#priority-5-confidence-scoring) |
| 6 | [POST /api/chat](#post-apichat) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) | [Priority 6](USAGE_EXAMPLES.md#priority-6-interactive-conversational-mode) |
| 7 | [Feature coverage](#feature-coverage) | [Change summary](MIGRATION.md#change-summary-by-completed-feature) | [Priority 7](USAGE_EXAMPLES.md#priority-7-demo-and-visualization-artifacts) |
| 8 | [Proposal workflow details](#proposal-workflow-details) | [Proposal workflow migration](MIGRATION.md#proposal-workflow-migration-priority-8) | [Priority 8](USAGE_EXAMPLES.md#priority-8-proposal-workspace-and-release-gate) |
| REL-001 | [Proposal release-gate evaluator (REL-001)](#proposal-release-gate-evaluator-rel-001) | [Proposal release-gate outputs](MIGRATION.md#proposal-release-gate-eval-outputs) | [8.8 release-gate evaluation](USAGE_EXAMPLES.md#88-proposal-release-gate-evaluation) |
| REL-002/003 | [Mypy release policy gate (REL-003)](#mypy-release-policy-gate-rel-003) | [Mypy policy outputs](MIGRATION.md#mypy-release-policy-outputs-rel-003) | [8.12 mypy policy](USAGE_EXAMPLES.md#812-validate-mypy-release-policy-rel-003) and [8.13 duplicate-module unblock](USAGE_EXAMPLES.md#813-confirm-duplicate-module-blocker-is-cleared-rel-002) |
| REL-004 | [Paperclip QA preflight and closure checks (REL-004)](#paperclip-qa-preflight-and-closure-checks-rel-004) | [QA preflight/closure migration](MIGRATION.md#paperclip-qa-preflight-and-closure-checks-rel-004) | [8.14 QA preflight + closure checks](USAGE_EXAMPLES.md#814-run-paperclip-qa-preflight-and-closure-checks-rel-004) |

## Endpoint summary

### Legacy RA core pipeline

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health and version |
| `POST` | `/search` | Search papers across providers |
| `POST` | `/search/batch` | Batch search with bounded concurrency |
| `POST` | `/retrieve` | Resolve one paper by identifier |
| `POST` | `/retrieve/batch` | Batch identifier resolution |
| `POST` | `/query` | Grounded Q&A endpoint |
| `POST` | `/api/chat` | Stateful multi-turn Q&A |
| `POST` | `/api/lit-review` | Structured literature review generation |
| `POST` | `/answer` | Deprecated alias of `/query` |

### Paperbase project and runtime

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Public Arxie landing page |
| `GET` | `/app` | Study/Library browser app |
| `GET` | `/health`, `/livez`, `/readyz` | Service and dependency probes |
| `GET` | `/api/v1/projects` | List known local projects |
| `POST` | `/api/v1/projects/open` | Open or register a project folder and return its project id |
| `GET` | `/api/v1/jobs` | List recent background jobs |
| `GET` | `/api/v1/jobs/{job_id}` | Inspect one background job |

### Paperbase library and preparation

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/collections` | List paper collections with readiness summaries |
| `POST` | `/api/v1/collections` | Create an empty collection |
| `GET` | `/api/v1/collections/{collection_id}` | Get one collection summary |
| `POST` | `/api/v1/collections/{collection_id}/papers` | Add an existing paper to a collection |
| `GET` | `/api/v1/collections/{collection_id}/papers` | List papers with parsed/extracted/job state |
| `GET` | `/api/v1/collections/{collection_id}/structured-summary` | Get collection-level extracted evidence and readiness counts |
| `POST` | `/api/v1/collections/{collection_id}/parse` | Queue parsing for all or selected papers |
| `POST` | `/api/v1/collections/{collection_id}/extract` | Queue structured extraction for all or selected parsed papers |
| `POST` | `/api/v1/ingest/local-library-upload` | Upload a browser-selected PDF folder and queue local-library ingest |
| `POST` | `/api/v1/ingest/local-library` | Queue ingest from a local filesystem directory path |
| `POST` | `/api/v1/ingest/providers` | Queue provider ingest by DOI, arXiv id, or OpenAlex id |
| `POST` | `/api/v1/ingest/refresh-metadata` | Queue metadata refresh for stored papers |
| `GET` | `/api/v1/extraction-profiles` | List extraction profiles |
| `POST` | `/api/v1/extraction-profiles` | Create an extraction profile |

Local-library upload uses a streaming multipart parser instead of FastAPI's
`UploadFile` materialization path and compares multipart media types
case-insensitively. The parser enforces `PAPERBASE_UPLOAD_MAX_FILE_COUNT`,
`PAPERBASE_UPLOAD_MAX_SINGLE_FILE_BYTES`, and `PAPERBASE_UPLOAD_MAX_TOTAL_BYTES`
while request bytes are read. The total limit applies to raw multipart request
bytes, including boundaries, headers, part bodies, and epilogue data. Multipart
header name, value, count, and per-part total header bytes are separately capped
so crafted part headers cannot grow unbounded in memory. All uploaded file parts
count toward the file limit, including non-PDF parts that are later discarded;
only PDFs are staged for ingest. The parser only accepts requests that reach the
terminal multipart boundary, so truncated bodies cannot enqueue partial staged
PDFs. Rejected, malformed, interrupted, metadata-invalid, or otherwise failed
uploads clean their staging directory before returning or propagating an error.
In hosted mode, project-open and local-path import requests are accepted only
when the submitted host path is under `PAPERBASE_LOCAL_PATH_IMPORT_ALLOWED_ROOTS`,
and the worker checks the source root before recursive scanning, then
revalidates each discovered PDF path so symlinks cannot escape configured roots.

### Paperbase search, paper browse, and compare

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/search/papers` | Search papers with collection and structured filters |
| `GET` | `/api/v1/search/chunks` | Search parsed full-text chunks |
| `GET` | `/api/v1/search/artifacts` | Search figure/table artifacts |
| `GET` | `/api/v1/search/status` | Inspect search backend configuration/readiness |
| `POST` | `/api/v1/search/reindex` | Queue backend read-model reindexing |
| `GET` | `/api/v1/papers/{paper_id}` | Fetch one paper |
| `GET` | `/api/v1/papers/{paper_id}/fulltext` | Fetch stored parsed sections |
| `GET` | `/api/v1/papers/{paper_id}/structured-data` | Fetch paper-level extracted evidence |
| `GET` | `/api/v1/papers/{paper_id}/figures` | Fetch paper figures |
| `GET` | `/api/v1/papers/{paper_id}/tables` | Fetch paper tables |
| `POST` | `/api/v1/compare/results` | Compare result rows across a collection or paper set |
| `POST` | `/api/v1/compare/methods` | Compare methods and best observed result rows |
| `POST` | `/api/v1/compare/engineering-tricks` | Compare recurring engineering tricks |
| `POST` | `/api/v1/compare/figures` | Compare figure artifacts |
| `POST` | `/api/v1/compare/tables` | Compare table artifacts |

### Paperbase Study and research agent

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/workspaces`, `/api/v1/studies` | List saved Study workspaces |
| `POST` | `/api/v1/workspaces`, `/api/v1/studies` | Create a saved Study workspace |
| `GET` | `/api/v1/workspaces/{workspace_id}`, `/api/v1/studies/{workspace_id}` | Fetch saved Study context |
| `PATCH` | `/api/v1/workspaces/{workspace_id}`, `/api/v1/studies/{workspace_id}` | Update title, linked collection, filters, focus note, or pinned papers |
| `GET` | `/api/v1/workspaces/{workspace_id}/study-brief`, `/api/v1/studies/{workspace_id}/brief` | Fetch an empty or stored versioned Study Brief |
| `PUT` | `/api/v1/workspaces/{workspace_id}/study-brief`, `/api/v1/studies/{workspace_id}/brief` | Replace the Study Brief JSON document and increment its version |
| `GET` | `/api/v1/studies/{workspace_id}/sources` | List explicit user sources for a Study |
| `POST` | `/api/v1/studies/{workspace_id}/sources` | Attach text, note, draft, code-summary, or result-summary context |
| `DELETE` | `/api/v1/studies/{workspace_id}/sources/{source_id}` | Remove one explicit Study source |
| `GET` | `/api/v1/research/suggestions` | Get readiness-aware suggested chat instructions |
| `GET` | `/api/v1/research/threads` | List collection research threads |
| `POST` | `/api/v1/research/threads` | Create a research thread linked to a collection and optional Study |
| `GET` | `/api/v1/research/threads/{thread_id}` | Fetch thread messages and artifacts |
| `POST` | `/api/v1/research/threads/{thread_id}/messages` | Queue a research-agent run for a user instruction |
| `GET` | `/api/v1/research/runs/{run_id}` | Fetch run status, steps, context pack, and validation report |
| `GET` | `/api/v1/research/artifacts` | List generated or saved artifacts |
| `GET` | `/api/v1/research/artifacts/{artifact_id}` | Fetch one artifact |
| `GET` | `/api/v1/research/artifacts/{artifact_id}/run` | Fetch the latest run trace for an artifact |
| `PATCH` | `/api/v1/research/artifacts/{artifact_id}` | Save, rename, or update an artifact |
| `GET` | `/api/v1/collections/{collection_id}/research-labels` | List paper research labels |
| `PATCH` | `/api/v1/collections/{collection_id}/papers/{paper_id}/research-label` | Update a paper research label |

Thread creation validates that an optional `workspace_id` exists and is either
collection-neutral or belongs to the requested `collection_id`. Artifact patch
requests are limited to `title`, `is_saved`, `saved_format`, and `saved_title`;
artifact `status` is server-owned and changes only through agent or worker
execution.

Study Brief payloads are bounded JSON objects. The API validates string length,
list length, object field count, key length, nesting depth, and finite numeric
values before storage. Validation responses omit raw input echoes so malformed
payloads cannot break JSON response serialization.

Research run responses include persisted trace steps, context-pack counts,
bounded `context_diagnostics`, and validation reports. Context diagnostics show
selected paper/source/memory/graph roles, ranking reasons, scores, and features
for debugging why a Study-agent artifact used a given context item.

### Proposal workflow (Priority 8)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/proposal/sessions` | Create proposal session |
| `GET` | `/api/proposal/sessions/{session_id}` | Get latest session snapshot |
| `PATCH` | `/api/proposal/sessions/{session_id}/stages/{stage}` | Merge stage payload with optimistic concurrency |
| `POST` | `/api/proposal/sessions/{session_id}/advance` | Advance one stage when current stage is complete |
| `GET` | `/api/proposal/sessions/{session_id}/export?format=markdown|pdf` | Export assembled proposal draft |
| `POST` | `/api/proposal/evidence/query` | Bucket evidence and return landscape summary |
| `PUT` | `/api/proposal/artifacts/{session_id}/nodes/{artifact}/{node_id}` | Upsert artifact node |
| `POST` | `/api/proposal/artifacts/{session_id}/dependencies` | Create artifact dependency edge |
| `POST` | `/api/proposal/artifacts/{session_id}/edits` | Record edit and propagate stale markers |
| `GET` | `/api/proposal/artifacts/{session_id}/dependencies/{artifact}/{node_id}` | Get downstream dependency snapshot |
| `GET` | `/api/proposal/artifacts/{session_id}/provenance/{artifact}/{node_id}` | 307 redirect to provenance URL |
| `POST` | `/api/proposal/conversations/{session_id}/messages` | Append conversation message |
| `GET` | `/api/proposal/conversations/{session_id}/messages` | List conversation thread |
| `GET` | `/api/proposal/evidence/{session_id}/inspector` | Evidence-inspector contract payload |
| `POST` | `/api/proposal/branches` | Create hypothesis branch |
| `GET` | `/api/proposal/branches/{session_id}` | List branches |
| `GET` | `/api/proposal/branches/{session_id}/{branch_id}` | Get branch |
| `POST` | `/api/proposal/branches/compare` | Compare branch scorecards |
| `POST` | `/api/proposal/branches/{session_id}/{branch_id}/promote` | Promote branch to primary |

### Study-agent workflow

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/studies` | Create durable local study memory |
| `GET` | `/api/studies/{study_id}` | Get one study brief |
| `PATCH` | `/api/studies/{study_id}/brief` | Update study brief with optimistic concurrency |
| `POST` | `/api/studies/{study_id}/sources` | Attach pasted source context such as draft, note, code summary, or result summary |
| `POST` | `/api/studies/{study_id}/runs` | Run a deterministic study-agent task and persist the trace |
| `GET` | `/api/studies/{study_id}/runs` | List study-agent runs |
| `GET` | `/api/studies/{study_id}/runs/{run_id}` | Get one persisted study-agent run |

## Core endpoint examples

### `GET /health`

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "service": "academic-research-assistant",
  "version": "0.1.0"
}
```

### `POST /search`

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer architecture for long-context summarization",
    "limit": 5,
    "source": "both"
  }'
```

`source` values: `semantic_scholar`, `arxiv`, `both`

### `POST /search/batch`

```bash
curl -X POST http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"query": "retrieval augmented generation benchmark survey", "limit": 5, "source": "semantic_scholar"},
      {"query": "long-context transformer memory mechanisms", "limit": 5, "source": "both"}
    ],
    "max_concurrency": 4
  }'
```

### `POST /retrieve`

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"identifier":"10.5555/3295222.3295349"}'
```

Identifier can be DOI, arXiv ID, or provider ID.

### `POST /retrieve/batch`

```bash
curl -X POST http://localhost:8000/retrieve/batch \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"identifier":"10.5555/3295222.3295349"},
      {"identifier":"1706.03762"}
    ],
    "max_concurrency": 8
  }'
```

Missing papers are returned as `"paper": null` in ordered results.

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query":"Compare LoRA and QLoRA training trade-offs with supporting evidence.",
    "deep": true
  }'
```

Behavior:

- `deep: true` enables multi-hop deep mode.
- `answer` is Markdown with `## Answer` and `## References`.
- Inline citations and confidence annotations are embedded in `answer`.
- Inline citation parsing accepts apostrophe surname variants in both ASCII and typographic forms (for example, `O'Neil` and `O’Neil`) and diacritic surnames (for example, `García`).

### `POST /api/chat`

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query":"How does this compare to what you said previously?",
    "session_id":"demo-session-1"
  }'
```

Behavior:

- `session_id` is required.
- Reuse `session_id` to preserve conversation context.

### `POST /api/lit-review`

```bash
curl -X POST http://localhost:8000/api/lit-review \
  -H "Content-Type: application/json" \
  -d '{
    "topic":"graph neural networks for molecular property prediction",
    "max_papers":20
  }'
```

Response `review` includes these sections:

- `## Introduction`
- `## Thematic Groups`
- `## Key Findings`
- `## Research Gaps`
- `## Future Directions`

## Legacy RA study-agent workflow details

The legacy RA study-agent API is local memory and deterministic task infrastructure for agent work that depends on a user's research context. The browser product uses the Paperbase `/api/v1/studies` and `/api/v1/research/*` endpoints described above.

### `POST /api/studies`

Creates a `StudyBrief`.

```bash
curl -X POST http://localhost:8000/api/studies \
  -H "Content-Type: application/json" \
  -d '{
    "study_id": "sc-regnet-review",
    "title": "scRegNet benchmark review",
    "research_goal": "Design stronger experiments for a single-cell GRN benchmark study.",
    "collection_id": "sample-papers",
    "domain": "single-cell network inference",
    "current_method": "graph neural network baseline",
    "datasets": ["PBMC"],
    "metrics": ["AUROC", "AUPRC"],
    "constraints": ["local-only data"]
  }'
```

### `PATCH /api/studies/{study_id}/brief`

Updates the study brief. `expected_version` is required for optimistic concurrency.

### `POST /api/studies/{study_id}/sources`

Attaches user-provided context. Supported source types are `draft`, `note`, `code_summary`, and `result_summary`.

### `POST /api/studies/{study_id}/runs`

Runs a deterministic study-agent task. Supported task types are `design_experiments`, `find_benchmarks`, and `review_draft_claims`.

```bash
curl -X POST http://localhost:8000/api/studies/sc-regnet-review/runs \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "design_experiments",
    "query": "What else should I test before writing the benchmark section?",
    "papers": []
  }'
```

Responses include the run id, status, context pack id, trace steps, warnings, recommendations, evidence references, and next actions.

### `POST /answer` (deprecated alias)

`/answer` mirrors `/query`. New clients should use `/query`.

## Proposal workflow details

### Stage identifiers

Canonical stage values for `/api/proposal/sessions/{session_id}/stages/{stage}`:

- `idea_intake`
- `logic_refinement`
- `evidence_mapping`
- `hypothesis_reshaping`
- `data_feasibility_planning`
- `experiment_analysis_design`
- `proposal_assembly`

### Stage payload requirements (for deterministic stage advancement)

Each stage must be complete before `POST /api/proposal/sessions/{session_id}/advance` can move forward.

| Stage | Required fields |
|---|---|
| `idea_intake` | `problem`, `target_population`, `mechanism`, `expected_outcome` |
| `logic_refinement` | `problem_gap_chain`, `core_assumptions`, `testable_hypothesis` |
| `evidence_mapping` | `supporting_evidence`, `contradicting_evidence`, `landscape_summary` |
| `hypothesis_reshaping` | `candidate_hypotheses`, `falsification_criteria`, `selected_primary_hypothesis` |
| `data_feasibility_planning` | `data_options_table`, `feasibility_scorecard`, `selected_data_strategy` |
| `experiment_analysis_design` | `experiment_flow_diagram`, `analysis_plan_tree`, `outcome_comparison_matrix` |
| `proposal_assembly` | `background_and_gap`, `hypothesis_statement`, `method_summary`, `analysis_summary`, `expected_outcomes`, `risks_and_limitations` |

### Stage write semantics

- `PATCH /api/proposal/sessions/{session_id}/stages/{stage}` can update any stage snapshot, not only the current stage.
- `expected_version` is validated before write/transition checks; stale clients receive `409 version_conflict`.
- `POST /api/proposal/sessions/{session_id}/advance` only evaluates the current stage and only allows the next canonical stage.

### Artifact identifiers

Artifact values for `/api/proposal/artifacts/.../{artifact}/...`:

- `logical_tree`
- `evidence_map`
- `hypothesis_tree`
- `data_options_table`
- `feasibility_scorecard`
- `experiment_flow_diagram`
- `analysis_plan_tree`
- `outcome_comparison_matrix`

### Create and advance a proposal session

```bash
curl -X POST http://localhost:8000/api/proposal/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id":"proposal-session-1"}'
```

```bash
curl -X PATCH http://localhost:8000/api/proposal/sessions/proposal-session-1/stages/idea_intake \
  -H "Content-Type: application/json" \
  -d '{
    "expected_version": 0,
    "payload": {
      "problem": "Citation traceability is inconsistent.",
      "target_population": "Academic researchers",
      "mechanism": "Stage-gated drafting with evidence tagging",
      "expected_outcome": "Higher trust and traceability"
    }
  }'
```

```bash
curl -X POST http://localhost:8000/api/proposal/sessions/proposal-session-1/advance \
  -H "Content-Type: application/json" \
  -d '{"expected_version":1}'
```

If required fields are missing, advance returns `409` with `error=stage_transition_rejected` and structured `details`.

### Export proposal draft

```bash
curl "http://localhost:8000/api/proposal/sessions/proposal-session-1/export?format=markdown"
```

Export response includes:

- `format`, `content_type`, `filename`, `content`
- `completeness.complete`
- per-section completeness checks:
  - `framing`, `evidence`, `hypothesis`, `method`, `outcomes`, `risks`, `references`

Completeness nuance:

- Stage advancement checks the stage-engine required fields.
- Export completeness checks section readiness for final draft output.
- For `method`, export completeness currently evaluates:
  - `proposal_assembly.method_summary`
  - `data_feasibility_planning.selected_data_strategy`
  - `experiment_analysis_design.experiment_flow_diagram`
  - `experiment_analysis_design.analysis_plan_tree`
  - `experiment_analysis_design.outcome_comparison_matrix`

Example `completeness` payload shape:

```json
{
  "completeness": {
    "complete": false,
    "sections": [
      {
        "section_id": "method",
        "title": "Method",
        "complete": false,
        "missing_fields": [
          "experiment_analysis_design.analysis_plan_tree"
        ]
      }
    ]
  }
}
```

### Evidence query

```bash
curl -X POST http://localhost:8000/api/proposal/evidence/query \
  -H "Content-Type: application/json" \
  -d '{
    "claim":"Transformers improve machine translation quality across benchmarks.",
    "pinned_paper_ids":["p1","p2"],
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

Response includes:

- `supporting`, `contradicting`, `adjacent`
- `bucket_counts`
- `representative_paper_ids`
- `landscape_summary` (`consensus`, `controversy`, `unknowns`)

### Artifact sync and provenance

```bash
curl -X PUT http://localhost:8000/api/proposal/artifacts/session-1/nodes/logical_tree/logic-1 \
  -H "Content-Type: application/json" \
  -d '{"content":"Problem -> mechanism chain"}'
```

```bash
curl -X PUT http://localhost:8000/api/proposal/artifacts/session-1/nodes/evidence_map/evidence-1 \
  -H "Content-Type: application/json" \
  -d '{"content":"Evidence bucket baseline","provenance_link":"https://doi.org/10.1000/xyz123"}'
```

```bash
curl -X POST http://localhost:8000/api/proposal/artifacts/session-1/dependencies \
  -H "Content-Type: application/json" \
  -d '{
    "upstream_artifact":"logical_tree",
    "upstream_node_id":"logic-1",
    "downstream_artifact":"evidence_map",
    "downstream_node_id":"evidence-1"
  }'
```

```bash
curl -X POST http://localhost:8000/api/proposal/artifacts/session-1/edits \
  -H "Content-Type: application/json" \
  -d '{"artifact":"logical_tree","node_id":"logic-1","content":"Problem -> mechanism chain (revised)"}'
```

`/edits` returns impacted downstream nodes and stale flags.

Dependency edge constraints:

- upstream and downstream nodes must both exist before edge creation
- self-edges are rejected (`400 invalid_input`)
- cycle-forming edges are rejected (`400 invalid_input`)

Provenance click-through:

```bash
curl -i http://localhost:8000/api/proposal/artifacts/session-1/provenance/evidence_map/evidence-1
```

Returns `307` redirect to the node provenance URL.

### Branching workflow

Create branch:

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
```

Compare branches:

```bash
curl -X POST http://localhost:8000/api/proposal/branches/compare \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"proposal-session-1",
    "branch_ids":["branch-a","branch-b"]
  }'
```

Promote branch:

```bash
curl -X POST http://localhost:8000/api/proposal/branches/proposal-session-1/branch-a/promote
```

Branch behavior notes:

- The first branch in a session is marked `is_primary=true`.
- `confidence_label` is derived from aggregate score buckets (`high`, `medium`, `low`).
- Aggregate score formula is deterministic: `(evidence_support + feasibility + (1 - risk) + impact) / 4`.
- `branch_ids` in compare requests must contain at least two unique branch IDs.
- Promoting a branch demotes all other branches in the same session.
- `parent_branch_id` and `metadata` are supported on branch creation (fork lineage + persisted labels).

### Conversation and inspector contracts

Append message:

```bash
curl -X POST http://localhost:8000/api/proposal/conversations/session-1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "role":"user",
    "content":"Draft a stage summary for idea intake.",
    "metadata":{"ui_source":"dashboard"}
  }'
```

`role` must be one of: `user`, `assistant`, `system`.

`metadata` keys/values are normalized to strings in persisted message responses.

List thread:

```bash
curl http://localhost:8000/api/proposal/conversations/session-1/messages
```

Example response shape:

```json
{
  "session_id": "session-1",
  "count": 1,
  "messages": [
    {
      "message_id": "18fa65ed-bfef-4db5-bb36-fd6db315dc0f",
      "session_id": "session-1",
      "role": "user",
      "content": "Draft a stage summary for idea intake.",
      "metadata": {
        "ui_source": "dashboard"
      },
      "created_at": "2026-03-13T10:29:09Z"
    }
  ]
}
```

Inspector payload (placeholder contract):

```bash
curl http://localhost:8000/api/proposal/evidence/session-1/inspector
```

Current placeholder response shape:

```json
{
  "session_id": "session-1",
  "count": 0,
  "items": []
}
```

## Release-readiness tooling (non-REST)

These completed release checks are CLI/runtime contracts, not HTTP endpoints.

### Proposal release-gate evaluator (REL-001)

Mode options:

- `--mode qa` (default): runs the benchmark harness.
- `--mode proposal_release_gate`: runs the v0.2 proposal release-gate evaluator.

Default QA mode:

```bash
python -m ra.cli eval \
  --dataset path/to/qa-dataset.json \
  --output path/to/results/
```

QA mode success output includes:

- `mode` (`qa`)
- `total_questions`
- `metrics`
- `artifacts`

Proposal release-gate mode:

```bash
python -m ra.cli eval \
  --dataset path/to/proposal-release-gate-dataset.json \
  --output path/to/results/ \
  --mode proposal_release_gate
```

Success output includes:

- `mode`
- `total_cases`
- `metrics.stage_completion_pass_rate`
- `metrics.draft_completeness_pass_rate`
- `metrics.evidence_link_pass_rate`
- `metrics.gate_pass_rate`
- `overall_pass`
- `artifacts.json`
- `artifacts.markdown`

Generated Markdown artifact contract (`proposal_release_gate_summary.md`):

- Section `## Topline Metrics` with rows for:
  - `stage_completion_pass_rate`
  - `draft_completeness_pass_rate`
  - `evidence_link_pass_rate`
  - `gate_pass_rate`
- Section `## Case Results` with columns:
  - `Case ID`
  - `Stage Ratio`
  - `Draft Complete`
  - `Link Coverage`
  - `Pass`

Contract semantics:

- Dataset must be a non-empty JSON list; empty datasets are rejected.
- Stage completion ratio is calculated across the full canonical stage sequence from `ProposalStageEngine.stage_sequence` (currently 7 stages, including `proposal_assembly`).
- `overall_pass` only becomes `true` when all cases pass stage completion, export draft completeness, and evidence-link coverage thresholds.

Failure output contract:

- CLI exits with status `1`.
- CLI emits machine-readable JSON: `{"error":"...", "type":"ValueError"}`.

Example failure payload (empty dataset):

```json
{
  "error": "Release-gate dataset must contain at least one case.",
  "type": "ValueError"
}
```

### Mypy release policy gate (REL-003)

Run the executable v0.2 mypy policy:

```bash
python -m ra.eval.mypy_gate
```

Expected JSON contract (shape):

```json
{
  "policy_id": "arxie-v0.2-mypy-release-gate",
  "overall_pass": true,
  "checks": [
    {
      "name": "strict_scope",
      "pass": true
    },
    {
      "name": "repo_baseline",
      "pass": true
    }
  ]
}
```

Policy semantics:

- `strict_scope`: must remain strict-clean for `src/ra/proposal` + `src/ra/eval` with `--follow-imports=skip`.
- `repo_baseline`: full-repo typing debt is currently no-regression gated against the accepted baseline threshold.

### Paperclip QA preflight and closure checks (REL-004)

Maintainer release branches run PRD, reviewer, eval, and release-gate checks
before syncing the public release branch. The public release branch exposes the
runtime smoke workflow and install/runtime commands instead of the development
gate bundle.

Contract semantics:

- `paperclip-quality-preflight.sh` exits non-zero on any PRD/reviewer/eval gate failure.
- `paperclip-task-close.sh` always runs preflight first; with `auto`/`yes`, it also runs release-gate checks before declaring closure-ready.
- CI `qa-gates` job must pass all quality-gate steps before merge.

## Error model

Error payload shape:

```json
{
  "error": "invalid_input",
  "message": "query must not be empty",
  "details": [{"loc": ["body", "query"], "msg": "Field required"}]
}
```

Common error codes:

- `400` invalid input
- `404` resource not found (`paper_not_found`, `session_not_found`, `branch_not_found`, `artifact_node_not_found`, `provenance_not_found`)
- `409` conflict or stage-transition rejection (`version_conflict`, `stage_transition_rejected`, `session_exists`, `branch_exists`)
- `422` schema validation errors
- `502` upstream provider failures
- `503` agent unavailable (query/chat/lit-review)

Common proposal `409` payloads:

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

`stage_transition_rejected` reasons include:

- `incomplete_stage`
- `invalid_transition`
- `final_stage_reached`
