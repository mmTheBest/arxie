# Legacy compatibility boundary

`src/ra/` is the original research-assistant implementation. It still supports
the CLI, proposal workflow, legacy REST API, evaluation gates, and compatibility
adapters that existing tests and scripts depend on.

Do not add new Arxie product code here. New Study, Library, Paperbase, worker,
and browser-product code should usually live under `src/paperbase/`,
`services/paperbase_api/`, or `services/paperbase_worker/`.

Use this folder only when maintaining an existing RA workflow or bridging that
workflow into Paperbase. Before deleting or moving anything in `src/ra/`, check
references with `rg`, update tests, and document the migration path in
`docs/architecture/05-assistant-integration.md` or a new ADR.
