# Arxie

**AI research support specialized in your field.**

Most AI assistants become shallow when the question depends on a specialized literature. They can summarize a paper, but they usually do not know which baselines your field expects, which datasets and metrics define a valid comparison, or how your draft and results compare with the papers reviewers already know. Arxie is built for that gap: give it the papers that matter to your project, and ask it to synthesize evidence, compare methods, design benchmarks, plan experiments, map assumptions, and critique revisions.

## Development Status

Arxie is in active development. The current release is useful for single-user local research workflows: importing papers, extracting research signals, creating Study context, and generating research artifacts grounded in selected papers and user-provided sources.

Expect the product surface, extraction quality, model-output contracts, evaluation checks, and deployment workflow to continue improving.

## What Arxie Helps You Do

Arxie helps with the parts of research where generic AI often loses the field context:

- **Find the field standard**: identify recurring methods, datasets, metrics, baselines, ablations, limitations, and validation patterns across your selected papers.
- **Compare papers at the research-design level**: compare methods, result rows, figures, tables, engineering tricks, and experimental assumptions instead of only reading summaries.
- **Plan stronger experiments**: turn prior work into benchmark plans, baseline choices, ablation lists, controls, and stress tests.
- **Check whether your claims are supported**: compare your draft, proposal, or notes against the paper evidence you selected.
- **Use your own project context**: add text notes, draft paths, code paths, and result-file paths so Arxie can reason over the literature plus your current work.
- **Keep useful outputs**: save generated literature reviews, comparisons, critiques, benchmark plans, revision plans, assumption maps, hypotheses, and experiment backlogs.

## The Core Workflow

```text
Import papers
→ Parse full text, figures, and tables
→ Extract research signals
→ Add your draft, code, results, or notes
→ Ask Arxie research-level questions
→ Save useful plans, critiques, comparisons, and revisions
```

Arxie is organized around two main surfaces:

- **Library**: upload papers, create collections, parse PDFs, extract evidence, and inspect processing readiness.
- **Study**: work on a specific question, pin key papers, add your own sources, ask Arxie for research artifacts, and save useful outputs.

## Example Questions

```text
Which datasets, metrics, and baselines recur across this collection?

Compare these papers by method, dataset, metric, reported result, and limitation.

Which papers make my proposed contribution look incremental?

Build a benchmark and ablation plan for this project.

What assumptions does my method rely on, and how should I stress-test them?

Here is my draft introduction. Which claims need stronger evidence?

Generate a revision plan using my draft, current results, and selected reference papers.

What limitations across this collection create realistic openings for a new study?
```

## What Arxie Can Generate

Arxie currently supports these research artifact types:

- **Literature review**: synthesize themes, consensus points, controversies, gaps, and future directions.
- **Paper comparison**: compare selected papers by methods, datasets, metrics, results, and limitations.
- **Field patterns**: identify recurring research patterns across a collection.
- **Benchmark plan**: recommend datasets, metrics, baselines, and ablations.
- **Experiment plan**: design experiments grounded in paper evidence and user-provided sources.
- **Hypotheses**: generate collection-grounded hypotheses and validation plans.
- **Critique**: check evidence coverage, unsupported claims, missing context, and reproducibility risks.
- **Assumption map**: identify assumptions worth challenging and propose tests.
- **Experiment backlog**: prioritize next experiments based on paper evidence.
- **Revision plan**: suggest draft or project revisions grounded in the selected literature.

## When Arxie Is Useful

Arxie is useful when your question depends on a specific body of literature, not broad internet knowledge.

Good use cases include:

- preparing a research proposal;
- designing experiments for a new method;
- checking whether a draft overclaims;
- planning baselines and ablations before implementation;
- comparing several related papers in depth;
- identifying gaps across a niche literature;
- preparing a revision after feedback;
- keeping a project-specific reading context across many sessions.

## Quick Start

### Requirements

- Python 3.10+
- Docker Desktop or Colima for the recommended local app workflow

### 1. Install

```bash
git clone https://github.com/mmTheBest/arxie.git
cd arxie

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional local embedding dependencies:

```bash
pip install -e .[local-embeddings]
```

## Start Arxie

### Recommended local launch

```bash
arxie-local run
```

This starts the local Arxie app, waits for readiness, and opens the Study app in your browser.

By default, the launcher starts PostgreSQL, MinIO, Redis, the API, and the worker. It skips the heavier backend search service so the app can run more reliably on a single machine.

To include Elasticsearch-backed search:

```bash
arxie-local run --with-search
```

Useful launcher commands:

```bash
arxie-local open
arxie-local down
arxie-local run --rebuild
arxie-local install-shortcut
```

`arxie-local install-shortcut` creates a double-clickable `Arxie.command` launcher on your Desktop by default.

### App URLs

After launch:

```text
Homepage:  http://localhost:8080/
App:       http://localhost:8080/app
Liveness:  http://localhost:8080/livez
Readiness: http://localhost:8080/readyz
```

## Manual Docker Stack

Start infrastructure:

```bash
docker compose -f infra/docker-compose.paperbase.yml up -d arxie-postgres arxie-elasticsearch arxie-minio arxie-redis
```

Apply schema migrations:

```bash
docker compose -f infra/docker-compose.paperbase.yml run --rm arxie-migrate
```

Start the API and worker:

```bash
docker compose -f infra/docker-compose.paperbase.yml up -d arxie-api arxie-worker
```

Open:

```text
http://localhost:8080/app
```

## Use Your Own Paper Collection

The browser app supports the main single-user paper workflow.

1. Open `http://localhost:8080/app`.
2. Go to **Library**.
3. Use **Upload PDF Folder**.
4. Select a local folder containing PDFs.
5. Optionally set a collection title.
6. Watch the processing log until the ingest job finishes.
7. Parse all unprocessed papers, or parse selected papers.
8. Extract all unextracted papers, or extract selected text-ready papers.
9. Switch to **Study**.
10. Search the collection and inspect extracted evidence.
11. Label important papers as exemplars, baselines, similar methods, or papers to ignore.
12. Save the Study.
13. Add explicit sources such as:
    - draft path;
    - code file path;
    - result file path;
    - text note.
14. Ask Arxie for experiment ideas, benchmark design, revision priorities, assumption checks, hypotheses, field patterns, comparisons, or critiques.
15. Save or export outputs that you want to keep.

For scripted or operator-driven ingestion, see:

```text
docs/runbooks/paperbase-ingest.md
```

## Study Sources

Arxie can use your own project material in addition to paper evidence.

Supported Study source types:

- `text`
- `draft_path`
- `code_path`
- `results_path`

This is useful when you want Arxie to critique or plan around your actual project state, rather than only the published literature.

Examples:

```text
Here is my current draft. Which claims need stronger paper support?

Use this result file and the selected papers to propose a stronger benchmark plan.

Given this code path and these papers, what ablations are missing?

Use my notes and the extracted limitations to build an assumption map.
```

## License

MIT
