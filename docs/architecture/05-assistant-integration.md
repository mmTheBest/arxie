# Assistant Integration

## Role Of `src/ra`

`src/ra` remains the assistant layer.

It should query Paperbase first and use live provider retrieval second.

## Integration Direction

Recommended path:

- add a Paperbase gateway under `src/ra/retrieval/`
- update the unified retriever to use database-first lookup
- keep external retrieval as fallback for discovery

## Workflow Integration

Over time, lit-review and proposal workflows should consume Paperbase-backed evidence, collections, and annotations.

