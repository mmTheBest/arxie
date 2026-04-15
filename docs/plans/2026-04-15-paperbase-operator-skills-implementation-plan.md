# 2026-04-15 Paperbase Operator Skills And Runbooks

## Goal

Complete Task 13 from the April 14 platform implementation plan: add Codex skills
and operator docs so collaborators can operate Paperbase safely without reverse
engineering the codebase every time.

## Implemented

- Added repo-local skills:
  - `skills/paperbase-postgres/SKILL.md`
  - `skills/paperbase-elasticsearch/SKILL.md`
  - `skills/paperbase-corpus-ops/SKILL.md`
- Added runbooks:
  - `docs/runbooks/paperbase-ingest.md`
  - `docs/runbooks/paperbase-reindex.md`
- Updated the architecture README to point collaborators at the runbooks.

## Verification

- `pytest tests/paperbase/test_skill_assets.py -q`

## Result

The branch now includes an operator surface alongside the code surface. New
contributors can discover:

- how to inspect the canonical relational store
- how to reason about the search/read-model layer
- how to import, parse, and extract a curated local corpus
- where the current system is still scaffolded versus fully operational
