# 2026-04-15 scRegNet Field Profile

## Goal

Turn the `SamplePapers` corpus into a real field-specific Paperbase profile
instead of treating it as a generic PDF collection.

## Field Summary

The corpus points to the single-cell gene regulatory network inference field,
with scRegNet-style link prediction as the central pattern:

- infer TF-gene, TF-RE, or RE-gene relationships from `scRNA-seq`,
  `scATAC-seq`, and `multiome` data
- combine single-cell foundation-model embeddings with graph learning
- inject prior biological knowledge and validate against benchmark GRNs and
  independent biological resources

Representative papers in the current corpus include:

- `Prediction of gene regulatory connections with joint single-cell foundation models and graph-based learning.pdf`
- `Deep learning-based cell-specific gene regulatory networks inferred from single-cell multiome data.pdf`
- `scGRIP.pdf`
- `scNET.pdf`
- `s41467-025-58699-1.pdf`

## Implemented

- Added a built-in `sc_regnet` extraction profile preset under
  `src/paperbase/profiles/`
- Added `GET /api/v1/extraction-profile-presets`
- Extended `POST /api/v1/extraction-profiles` so a profile can be created from a
  `preset_name`
- Added `GET /api/v1/collections/{collection_id}/structured-summary` to expose a
  collection-level extracted snapshot for curated field databases

## scRegNet Schema Focus

The preset encodes the field-specific entities that recur across the corpus:

- `Study`
- `Method`
- `Dataset`
- `RegulatoryEntity`
- `PriorKnowledgeSource`
- `Benchmark`
- `ExperimentProtocol`
- `ValidationEvidence`
- `GlossaryTerm`

The preset also ships controlled vocabularies for assays, relation types,
metrics, benchmark families, model aliases, entity aliases, and split axes.

## Verification

- `pytest tests/paperbase/test_extraction_presets_api.py -q`
- `pytest tests/paperbase/test_collection_summary_api.py -q`
- `pytest tests/paperbase -q`

## Result

Paperbase now has a first-class field preset for the local scRegNet corpus and a
collection-level structured summary surface to inspect that field database once
extraction runs are available.
