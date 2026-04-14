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

