# Collections And Annotations

## Product Requirement

Users must be able to maintain custom paper databases for a field of study.

In practice that means:

- curate a collection of selected papers
- run analysis against that collection
- attach manual annotations and tags
- apply field-specific extraction profiles

## V1 Assumption

Collections are single-user and local-first in v1.

But collection and annotation records should still include ownership-friendly structure so the model can expand later.

## Current Workflow Support

Collections are now visible to the assistant layer through the Paperbase gateway.

- proposal evidence queries can source papers from a `paperbase_collection_id`
- literature reviews can be scoped to a specific Paperbase collection
- full-text reads inside those workflows still reuse stored Paperbase sections before
  any PDF download fallback

This means curated field-specific corpora are no longer only a storage feature; they
are part of the active Arxie workflow surface.

## Current API Support

Paperbase now exposes first-class collection and annotation endpoints:

- create and list collections
- add papers to collections and list collection membership
- create annotations on papers and list annotations by target

This is still local-first and single-user by default, but the API and schema keep
ownership fields so the system can expand later without rewriting the core model.

Paperbase also now exposes extraction profile and collection extraction endpoints,
so users can store a field-specific schema once and execute it over a curated
collection through the API surface instead of only through internal runners.

## Collection Scope

A collection is a user-owned curated slice of the corpus.

At minimum it should support:

- title
- description
- owner identifier
- paper membership
- tags
- extraction profile

## Annotation Scope

Users should be able to annotate:

- papers
- sections
- chunks
- figures
- result rows

Manual annotations must stay separate from canonical extracted paper facts.

## Field-Specific Profiles

Collections should support domain-specific extraction targets such as:

- data used
- benchmark methods
- engineering tricks
- design of experiments
