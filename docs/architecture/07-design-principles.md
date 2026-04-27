# Design Principles

## 1. Database First

If research structure must persist and be queried repeatedly, it belongs in Paperbase.

## 2. Clear Layer Boundaries

- Paperbase owns data and indexing
- Arxie owns workflows and synthesis

## 3. Provenance Is Mandatory

Structured outputs must be traceable to source text, extraction runs, and user annotations when relevant.

## 4. Collections Are First-Class

The product is not only a global corpus. Curated field-specific collections are a core use case.

## 5. Single-User Local-First, Not Single-User Forever

Optimize v1 for one local operator, but do not hard-code assumptions that make future multi-user or larger hosted corpora expensive.
